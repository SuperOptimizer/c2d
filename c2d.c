// c2d implementation. See c2d.h for the public API and pipeline overview.
//
// Numeric policy:
//   - I/O dtype: user-selected (u8/u16/u32/s8/s16/s32/f32).
//   - Compute backend: pure float32 + int32 in all hot paths (DWT lifting,
//     quantize/dequantize, rANS encode/decode, color transform, normalize/
//     denormalize). c3d's policy.
//   - double is used only where float32 loses meaningful precision, all of
//     which are off the hot path:
//       (1) DC-offset mean accumulation over 65k pixels (catastrophic
//           cancellation if summed in f32).
//       (2) Entropy estimate (log2, log, exp) inside the encoder's
//           rate-control bisection.
//       (3) Clamp + round when storing reconstructed f32 into u32/s32
//           outputs whose range exceeds f32's mantissa.
//     These collectively account for <2% of encode time and 0% of decode time.
//
// Tile layout (256x256, N channels, interleaved):
//   pipeline:  pixels (dtype, interleaved) -> deinterleave -> per-channel
//              f32 normalize -> [optional YCoCg-R for n_ch==3] ->
//              per-plane 5-level CDF 9/7 DWT -> dead-zone quantize ->
//              65-symbol zigzag+escape -> 8-way interleaved rANS.
//
// Subbands per plane (16, coarse-to-fine):
//   0:        LL_5  (8x8)
//   1..3:     HL_5  LH_5  HH_5
//   4..6:     HL_4  LH_4  HH_4
//   ...
//   13..15:   HL_1  LH_1  HH_1  (each 128x128)
//
// Across N channels, subbands are stored in (band-outer, channel-inner) order
// so the LOD truncation prefix is coarse-to-fine across all channels.

#include "c2d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===========================================================================
// Panic
// ===========================================================================

static c2d_panic_fn g_panic = NULL;
void c2d_set_panic_hook(c2d_panic_fn hook) { g_panic = hook; }

void c2d_panic(const char *file, int line, const char *msg) {
    if (g_panic) { g_panic(file, line, msg); }
    fprintf(stderr, "c2d panic: %s:%d: %s\n", file, line, msg);
    abort();
}

// ===========================================================================
// Version & dtype
// ===========================================================================

const char *c2d_version(void)        { return C2D_VERSION_STRING; }
uint32_t    c2d_format_version(void) { return C2D_FORMAT_VERSION; }

static const size_t k_dtype_size[C2D_DTYPE__COUNT] = {
    [C2D_DTYPE_U8]  = 1, [C2D_DTYPE_U16] = 2, [C2D_DTYPE_U32] = 4,
    [C2D_DTYPE_S8]  = 1, [C2D_DTYPE_S16] = 2, [C2D_DTYPE_S32] = 4,
    [C2D_DTYPE_F32] = 4,
};

size_t c2d_dtype_size(c2d_dtype dt) {
    c2d_assert((unsigned)dt < C2D_DTYPE__COUNT);
    return k_dtype_size[dt];
}
size_t c2d_dtype_tile_bytes(c2d_dtype dt) {
    return C2D_PIXELS_PER_TILE * c2d_dtype_size(dt);
}
size_t c2d_dtype_tile_bytes_n(c2d_dtype dt, uint32_t n_channels) {
    return C2D_PIXELS_PER_TILE * c2d_dtype_size(dt) * (size_t)n_channels;
}

uint32_t c2d_side_per_lod(uint8_t lod) {
    c2d_assert(lod < C2D_N_LODS);
    return C2D_TILE_SIDE >> lod;
}
size_t c2d_pixels_per_lod(uint8_t lod) {
    uint32_t s = c2d_side_per_lod(lod);
    return (size_t)s * s;
}

// ===========================================================================
// Dtype <-> f32 (interleaved load/store)
// ===========================================================================

static inline float iload_f32(const void *in, c2d_dtype dt,
                              size_t i, uint32_t c, uint32_t nch) {
    size_t j = i * (size_t)nch + (size_t)c;
    switch (dt) {
        case C2D_DTYPE_U8:  return (float)((const uint8_t  *)in)[j];
        case C2D_DTYPE_U16: return (float)((const uint16_t *)in)[j];
        case C2D_DTYPE_U32: return (float)((const uint32_t *)in)[j];
        case C2D_DTYPE_S8:  return (float)((const int8_t   *)in)[j];
        case C2D_DTYPE_S16: return (float)((const int16_t  *)in)[j];
        case C2D_DTYPE_S32: return (float)((const int32_t  *)in)[j];
        case C2D_DTYPE_F32: return        ((const float    *)in)[j];
        default: c2d_panic(__FILE__, __LINE__, "bad dtype"); return 0;
    }
}

static inline void istore_f32(void *out, c2d_dtype dt,
                              size_t i, uint32_t c, uint32_t nch, float v) {
    size_t j = i * (size_t)nch + (size_t)c;
    switch (dt) {
        case C2D_DTYPE_U8: {
            float r = roundf(v); if (r < 0) r = 0; if (r > 255) r = 255;
            ((uint8_t *)out)[j] = (uint8_t)r; break;
        }
        case C2D_DTYPE_U16: {
            float r = roundf(v); if (r < 0) r = 0; if (r > 65535) r = 65535;
            ((uint16_t *)out)[j] = (uint16_t)r; break;
        }
        case C2D_DTYPE_U32: {
            double r = round((double)v);
            if (r < 0) r = 0; if (r > 4294967295.0) r = 4294967295.0;
            ((uint32_t *)out)[j] = (uint32_t)r; break;
        }
        case C2D_DTYPE_S8: {
            float r = roundf(v); if (r < -128) r = -128; if (r > 127) r = 127;
            ((int8_t *)out)[j] = (int8_t)r; break;
        }
        case C2D_DTYPE_S16: {
            float r = roundf(v); if (r < -32768) r = -32768; if (r > 32767) r = 32767;
            ((int16_t *)out)[j] = (int16_t)r; break;
        }
        case C2D_DTYPE_S32: {
            double r = round((double)v);
            if (r < -2147483648.0) r = -2147483648.0;
            if (r >  2147483647.0) r =  2147483647.0;
            ((int32_t *)out)[j] = (int32_t)r; break;
        }
        case C2D_DTYPE_F32:
            ((float *)out)[j] = v; break;
        default: c2d_panic(__FILE__, __LINE__, "bad dtype");
    }
}

static void compute_normalization_channel(const void *in, c2d_dtype dt,
                                          uint32_t c, uint32_t nch,
                                          float *out_dc, float *out_scale) {
    double sum = 0.0;
    for (size_t i = 0; i < C2D_PIXELS_PER_TILE; i++)
        sum += (double)iload_f32(in, dt, i, c, nch);
    float dc = (float)(sum / (double)C2D_PIXELS_PER_TILE);
    float maxmag = 0.0f;
    for (size_t i = 0; i < C2D_PIXELS_PER_TILE; i++) {
        float v = iload_f32(in, dt, i, c, nch) - dc;
        float a = fabsf(v);
        if (a > maxmag) maxmag = a;
    }
    *out_dc = dc;
    *out_scale = (maxmag < 1e-20f) ? 1.0f : (128.0f / maxmag);
}

// ===========================================================================
// YCoCg-R (reversible color decorrelation)
// ===========================================================================

static void ycocg_fwd(float *p0, float *p1, float *p2, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float R = p0[i], G = p1[i], B = p2[i];
        float Co = R - B;
        float t  = B + 0.5f * Co;
        float Cg = G - t;
        float Y  = t + 0.5f * Cg;
        p0[i] = Y; p1[i] = Co; p2[i] = Cg;
    }
}
static void ycocg_inv(float *p0, float *p1, float *p2, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float Y = p0[i], Co = p1[i], Cg = p2[i];
        float t = Y - 0.5f * Cg;
        float G = Cg + t;
        float B = t - 0.5f * Co;
        float R = B + Co;
        p0[i] = R; p1[i] = G; p2[i] = B;
    }
}

// ===========================================================================
// 2D CDF 9/7 DWT (separable lifting, whole-sample symmetric extension)
//
// Boundary handling: for the lifting equations
//   predict odd:   d[k]  -= a*(s[k-1]+s[k+1])  (treating evens as 's', odds as 'd')
//   update even:   s[k]  += b*(d[k-1]+d[k+1])
// at boundaries we mirror via whole-sample symmetry:  x[-i] = x[i], x[n+i] = x[n-2-i].
// This matches JPEG2000 (Annex F.4.8.1) for the irreversible 9/7.
// ===========================================================================

#define DWT_A  (-1.586134342f)
#define DWT_B  (-0.052980118f)
#define DWT_C  ( 0.882911075f)
#define DWT_D  ( 0.443506852f)
#define DWT_K  ( 1.230174105f)

// Forward CDF 9/7 lifting on length-n buffer (n even, >= 2).
// Whole-sample symmetric extension at boundaries: x[-1] -> x[1]; x[n] -> x[n-2].
// Interior loops are branch-free for vectorization; tiny epilogue/prologue
// handles the two boundary samples per pass.
static void dwt_fwd_1d(float *restrict x, int n) {
    if (n < 2) return;
    // Predict 1 (odds): d[k] += A*(s[k-1] + s[k+1])
    {
        // Interior odd k in [1, n-2]; right neighbor x[k+1] always valid.
        for (int k = 1; k < n - 1; k += 2) x[k] += DWT_A * (x[k-1] + x[k+1]);
        // Boundary at k=n-1: right neighbor x[n] mirrors to x[n-2].
        x[n-1] += DWT_A * (x[n-2] + x[n-2]);
    }
    // Update 1 (evens): s[k] += B*(d[k-1] + d[k+1])
    {
        // k=0: left x[-1] -> x[1]; combined with right x[1]: 2*x[1].
        x[0] += DWT_B * (x[1] + x[1]);
        for (int k = 2; k < n; k += 2) {
            int rk = (k + 1 < n) ? (k + 1) : (n - 1);
            x[k] += DWT_B * (x[k-1] + x[rk]);
        }
    }
    // Predict 2.
    {
        for (int k = 1; k < n - 1; k += 2) x[k] += DWT_C * (x[k-1] + x[k+1]);
        x[n-1] += DWT_C * (x[n-2] + x[n-2]);
    }
    // Update 2.
    {
        x[0] += DWT_D * (x[1] + x[1]);
        for (int k = 2; k < n; k += 2) {
            int rk = (k + 1 < n) ? (k + 1) : (n - 1);
            x[k] += DWT_D * (x[k-1] + x[rk]);
        }
    }
    // Scale: evens (low) *= 1/K; odds (high) *= K.
    const float invK = 1.0f / DWT_K;
    for (int k = 0; k < n; k += 2) x[k] *= invK;
    for (int k = 1; k < n; k += 2) x[k] *= DWT_K;
}

static void dwt_inv_1d(float *restrict x, int n) {
    if (n < 2) return;
    const float invK = 1.0f / DWT_K;
    for (int k = 0; k < n; k += 2) x[k] *= DWT_K;
    for (int k = 1; k < n; k += 2) x[k] *= invK;
    // Undo Update 2.
    {
        x[0] -= DWT_D * (x[1] + x[1]);
        for (int k = 2; k < n; k += 2) {
            int rk = (k + 1 < n) ? (k + 1) : (n - 1);
            x[k] -= DWT_D * (x[k-1] + x[rk]);
        }
    }
    // Undo Predict 2.
    {
        for (int k = 1; k < n - 1; k += 2) x[k] -= DWT_C * (x[k-1] + x[k+1]);
        x[n-1] -= DWT_C * (x[n-2] + x[n-2]);
    }
    // Undo Update 1.
    {
        x[0] -= DWT_B * (x[1] + x[1]);
        for (int k = 2; k < n; k += 2) {
            int rk = (k + 1 < n) ? (k + 1) : (n - 1);
            x[k] -= DWT_B * (x[k-1] + x[rk]);
        }
    }
    // Undo Predict 1.
    {
        for (int k = 1; k < n - 1; k += 2) x[k] -= DWT_A * (x[k-1] + x[k+1]);
        x[n-1] -= DWT_A * (x[n-2] + x[n-2]);
    }
}

// 2D one-level forward/inverse on the top-left `side x side` of a row-major
// buffer with the given stride. `tmp` holds `side` floats for column work.
// 8 columns at a time: load (256 rows) x (8 cols) into a contiguous 256x8
// block, lift along the long (256-element) axis, scatter back. The lifting
// operates on 8-float vectors per step so clang autovectorizes well.

// One-D forward lifting on 8 columns simultaneously; `x` has layout
// [row][col] with `rows` rows and 8 columns. Memory is contiguous, stride=8.
static void dwt_fwd_1d_x8(float *restrict x, int rows) {
    if (rows < 2) return;
    // Predict 1 on odd rows.
    for (int k = 1; k < rows - 1; k += 2) {
        float *xk = x + k * 8;
        const float *xl = x + (k - 1) * 8;
        const float *xr = x + (k + 1) * 8;
        for (int c = 0; c < 8; c++) xk[c] += DWT_A * (xl[c] + xr[c]);
    }
    { // boundary k = rows-1
        float *xk = x + (rows - 1) * 8;
        const float *xn = x + (rows - 2) * 8;
        for (int c = 0; c < 8; c++) xk[c] += DWT_A * (xn[c] + xn[c]);
    }
    // Update 1 on evens.
    {
        float *x0 = x;
        const float *x1 = x + 8;
        for (int c = 0; c < 8; c++) x0[c] += DWT_B * (x1[c] + x1[c]);
    }
    for (int k = 2; k < rows; k += 2) {
        float *xk = x + k * 8;
        const float *xl = x + (k - 1) * 8;
        const float *xr = (k + 1 < rows) ? x + (k + 1) * 8 : x + (rows - 1) * 8;
        for (int c = 0; c < 8; c++) xk[c] += DWT_B * (xl[c] + xr[c]);
    }
    // Predict 2.
    for (int k = 1; k < rows - 1; k += 2) {
        float *xk = x + k * 8;
        const float *xl = x + (k - 1) * 8;
        const float *xr = x + (k + 1) * 8;
        for (int c = 0; c < 8; c++) xk[c] += DWT_C * (xl[c] + xr[c]);
    }
    {
        float *xk = x + (rows - 1) * 8;
        const float *xn = x + (rows - 2) * 8;
        for (int c = 0; c < 8; c++) xk[c] += DWT_C * (xn[c] + xn[c]);
    }
    // Update 2.
    {
        float *x0 = x;
        const float *x1 = x + 8;
        for (int c = 0; c < 8; c++) x0[c] += DWT_D * (x1[c] + x1[c]);
    }
    for (int k = 2; k < rows; k += 2) {
        float *xk = x + k * 8;
        const float *xl = x + (k - 1) * 8;
        const float *xr = (k + 1 < rows) ? x + (k + 1) * 8 : x + (rows - 1) * 8;
        for (int c = 0; c < 8; c++) xk[c] += DWT_D * (xl[c] + xr[c]);
    }
    // Scale.
    const float invK = 1.0f / DWT_K;
    for (int k = 0; k < rows; k += 2) {
        float *xk = x + k * 8;
        for (int c = 0; c < 8; c++) xk[c] *= invK;
    }
    for (int k = 1; k < rows; k += 2) {
        float *xk = x + k * 8;
        for (int c = 0; c < 8; c++) xk[c] *= DWT_K;
    }
}

static void dwt_inv_1d_x8(float *restrict x, int rows) {
    if (rows < 2) return;
    const float invK = 1.0f / DWT_K;
    for (int k = 0; k < rows; k += 2) {
        float *xk = x + k * 8;
        for (int c = 0; c < 8; c++) xk[c] *= DWT_K;
    }
    for (int k = 1; k < rows; k += 2) {
        float *xk = x + k * 8;
        for (int c = 0; c < 8; c++) xk[c] *= invK;
    }
    // Undo Update 2.
    {
        float *x0 = x;
        const float *x1 = x + 8;
        for (int c = 0; c < 8; c++) x0[c] -= DWT_D * (x1[c] + x1[c]);
    }
    for (int k = 2; k < rows; k += 2) {
        float *xk = x + k * 8;
        const float *xl = x + (k - 1) * 8;
        const float *xr = (k + 1 < rows) ? x + (k + 1) * 8 : x + (rows - 1) * 8;
        for (int c = 0; c < 8; c++) xk[c] -= DWT_D * (xl[c] + xr[c]);
    }
    // Undo Predict 2.
    for (int k = 1; k < rows - 1; k += 2) {
        float *xk = x + k * 8;
        const float *xl = x + (k - 1) * 8;
        const float *xr = x + (k + 1) * 8;
        for (int c = 0; c < 8; c++) xk[c] -= DWT_C * (xl[c] + xr[c]);
    }
    {
        float *xk = x + (rows - 1) * 8;
        const float *xn = x + (rows - 2) * 8;
        for (int c = 0; c < 8; c++) xk[c] -= DWT_C * (xn[c] + xn[c]);
    }
    // Undo Update 1.
    {
        float *x0 = x;
        const float *x1 = x + 8;
        for (int c = 0; c < 8; c++) x0[c] -= DWT_B * (x1[c] + x1[c]);
    }
    for (int k = 2; k < rows; k += 2) {
        float *xk = x + k * 8;
        const float *xl = x + (k - 1) * 8;
        const float *xr = (k + 1 < rows) ? x + (k + 1) * 8 : x + (rows - 1) * 8;
        for (int c = 0; c < 8; c++) xk[c] -= DWT_B * (xl[c] + xr[c]);
    }
    // Undo Predict 1.
    for (int k = 1; k < rows - 1; k += 2) {
        float *xk = x + k * 8;
        const float *xl = x + (k - 1) * 8;
        const float *xr = x + (k + 1) * 8;
        for (int c = 0; c < 8; c++) xk[c] -= DWT_A * (xl[c] + xr[c]);
    }
    {
        float *xk = x + (rows - 1) * 8;
        const float *xn = x + (rows - 2) * 8;
        for (int c = 0; c < 8; c++) xk[c] -= DWT_A * (xn[c] + xn[c]);
    }
}

static void dwt_fwd_2d_level(float *buf, int side, int stride, float *tmp) {
    // Rows in place.
    for (int y = 0; y < side; y++) dwt_fwd_1d(buf + y * stride, side);
    // Columns in groups of 8 (side is always a power of 2 >= 8 for our DWT levels).
    if ((side & 7) != 0) {
        // Fallback to per-column scratch.
        for (int x = 0; x < side; x++) {
            for (int y = 0; y < side; y++) tmp[y] = buf[y * stride + x];
            dwt_fwd_1d(tmp, side);
            for (int y = 0; y < side; y++) buf[y * stride + x] = tmp[y];
        }
        return;
    }
    // `tmp` is sized for `side` floats; we need side*8. Use a heap alloc or
    // assume tmp is large enough. Our encoder/decoder provide a 256-float tmp;
    // we need side*8 floats. For side==256: 2048 floats = 8KB. We'll use the
    // existing `mallat` scratch which is 256*256 floats = 256KB, plenty.
    // But we only get `tmp` here. Switch to a small VLA bounded by stack.
    float block[256 * 8];   // 8 KB, fits comfortably on stack
    for (int x0 = 0; x0 < side; x0 += 8) {
        for (int y = 0; y < side; y++) {
            const float *src = buf + y * stride + x0;
            for (int c = 0; c < 8; c++) block[y * 8 + c] = src[c];
        }
        dwt_fwd_1d_x8(block, side);
        for (int y = 0; y < side; y++) {
            float *dst = buf + y * stride + x0;
            for (int c = 0; c < 8; c++) dst[c] = block[y * 8 + c];
        }
    }
}
static void dwt_inv_2d_level(float *buf, int side, int stride, float *tmp) {
    if ((side & 7) != 0) {
        for (int x = 0; x < side; x++) {
            for (int y = 0; y < side; y++) tmp[y] = buf[y * stride + x];
            dwt_inv_1d(tmp, side);
            for (int y = 0; y < side; y++) buf[y * stride + x] = tmp[y];
        }
        for (int y = 0; y < side; y++) dwt_inv_1d(buf + y * stride, side);
        return;
    }
    float block[256 * 8];
    for (int x0 = 0; x0 < side; x0 += 8) {
        for (int y = 0; y < side; y++) {
            const float *src = buf + y * stride + x0;
            for (int c = 0; c < 8; c++) block[y * 8 + c] = src[c];
        }
        dwt_inv_1d_x8(block, side);
        for (int y = 0; y < side; y++) {
            float *dst = buf + y * stride + x0;
            for (int c = 0; c < 8; c++) dst[c] = block[y * 8 + c];
        }
    }
    for (int y = 0; y < side; y++) dwt_inv_1d(buf + y * stride, side);
}

// Mallat de-interleave / re-interleave at one level on a `stride`-wide buffer.
static void mallat_pack(float *buf, int side, int stride, float *scratch) {
    int half = side / 2;
    for (int y = 0; y < side; y++)
        memcpy(scratch + y * side, buf + y * stride, sizeof(float) * (size_t)side);
    for (int y = 0; y < half; y++) {
        for (int x = 0; x < half; x++) {
            buf[y * stride + x]                   = scratch[(2*y)   * side + (2*x)  ];
            buf[y * stride + (x + half)]          = scratch[(2*y)   * side + (2*x+1)];
            buf[(y + half) * stride + x]          = scratch[(2*y+1) * side + (2*x)  ];
            buf[(y + half) * stride + (x + half)] = scratch[(2*y+1) * side + (2*x+1)];
        }
    }
}
static void mallat_unpack(float *buf, int side, int stride, float *scratch) {
    int half = side / 2;
    for (int y = 0; y < half; y++) {
        for (int x = 0; x < half; x++) {
            scratch[(2*y)   * side + (2*x)  ] = buf[y * stride + x];
            scratch[(2*y)   * side + (2*x+1)] = buf[y * stride + (x + half)];
            scratch[(2*y+1) * side + (2*x)  ] = buf[(y + half) * stride + x];
            scratch[(2*y+1) * side + (2*x+1)] = buf[(y + half) * stride + (x + half)];
        }
    }
    for (int y = 0; y < side; y++)
        memcpy(buf + y * stride, scratch + y * side, sizeof(float) * (size_t)side);
}

// 5-level forward DWT on a 256x256 plane. `scratch` reused across levels.
static void dwt_fwd(float *buf, int stride, float *tmp, float *scratch) {
    int side = C2D_TILE_SIDE;
    for (uint32_t lvl = 0; lvl < C2D_N_DWT_LEVELS; lvl++) {
        dwt_fwd_2d_level(buf, side, stride, tmp);
        mallat_pack(buf, side, stride, scratch);
        side /= 2;
    }
}

// Inverse DWT: apply `levels_to_apply` coarsest levels of inverse on the
// top-left `side x side` region, where `side` is the starting LL side after
// previous applied levels (e.g. for full inverse: starts at 16 with 5 levels).
static void dwt_inv_partial(float *buf, int stride, uint32_t levels_to_apply,
                            float *tmp, float *scratch) {
    int sides[C2D_N_DWT_LEVELS + 1];
    sides[0] = C2D_TILE_SIDE;
    for (uint32_t i = 1; i <= C2D_N_DWT_LEVELS; i++) sides[i] = sides[i-1] / 2;
    for (uint32_t step = 0; step < levels_to_apply; step++) {
        uint32_t lvl = C2D_N_DWT_LEVELS - 1 - step;
        int side = sides[lvl];
        mallat_unpack(buf, side, stride, scratch);
        dwt_inv_2d_level(buf, side, stride, tmp);
    }
}

// ===========================================================================
// Subband geometry
// ===========================================================================

typedef struct subband_rect {
    uint16_t x, y, w, h;
    uint8_t  level;
    uint8_t  kind;
} subband_rect;

static void enumerate_subbands(subband_rect out[C2D_N_SUBBANDS]) {
    int coarsest = C2D_TILE_SIDE >> C2D_N_DWT_LEVELS;
    out[0] = (subband_rect){ .x=0, .y=0, .w=(uint16_t)coarsest, .h=(uint16_t)coarsest,
                             .level=(uint8_t)C2D_N_DWT_LEVELS, .kind=0 };
    int idx = 1;
    for (int lvl = (int)C2D_N_DWT_LEVELS; lvl >= 1; lvl--) {
        int s = C2D_TILE_SIDE >> lvl;
        out[idx++] = (subband_rect){ .x=(uint16_t)s, .y=0,           .w=(uint16_t)s, .h=(uint16_t)s, .level=(uint8_t)lvl, .kind=1 };
        out[idx++] = (subband_rect){ .x=0,           .y=(uint16_t)s, .w=(uint16_t)s, .h=(uint16_t)s, .level=(uint8_t)lvl, .kind=2 };
        out[idx++] = (subband_rect){ .x=(uint16_t)s, .y=(uint16_t)s, .w=(uint16_t)s, .h=(uint16_t)s, .level=(uint8_t)lvl, .kind=3 };
    }
    c2d_assert(idx == (int)C2D_N_SUBBANDS);
}

// Per-subband relative quantizer step weighting. Larger = coarser quantization.
// PSNR-tuned baseline: empirically calibrated on Kodak (R3 #2 sweep), +1.2 dB
// across 0.5..4 bpp vs original w^0.5. Exponent 0.9 matches natural-image
// spectral falloff; HH factor accounts for the eye's lower sensitivity to
// diagonal detail; LL kept very accurate.
// LOW_BPP variant: SSIM-tuned — coarser finest-level detail (eye smooths it
// and SSIM's gaussian window weighs it less), slightly finer mid-frequency
// (where SSIM's structural term lives).
static float subband_baseline_flags(const subband_rect *sb, uint32_t flags) {
    if (flags & C2D_FLAG_LOW_BPP) {
        if (sb->kind == 0) return 0.10f;
        float w = (float)(1u << (C2D_N_DWT_LEVELS - sb->level));
        float b = powf(w, 0.92f);
        if (sb->level == 1)                              b *= 2.0f;   // finest = coarser
        else if (sb->level == 3 || sb->level == 4)       b *= 0.80f;  // mid = finer
        if (sb->kind == 3)                               b *= 2.5f;   // HH boost
        return b;
    }
    if (sb->kind == 0) return 0.12f;
    float w = (float)(1u << (C2D_N_DWT_LEVELS - sb->level));
    float b = powf(w, 0.9f);
    if (sb->kind == 3) b *= 2.3f;
    return b;
}
static float channel_weight_ycocg(uint32_t ch) {
    if (ch == 0) return 1.0f;
    return 2.0f;
}

// ===========================================================================
// Quantizer: dead-zone uniform, dead-zone width = 1.2 * step (Laplacian-ish).
// Reconstruction with +0.375 mid-tread bias matching c3d.
//   forward:   q = sign(x) * max(0, floor((|x| - 0.2*step) / step) + 1)  if |x| > 0.6*step
//              q = 0                                                      otherwise
//   inverse:   x_hat = sign(q) * (|q| + 0.375 - bias_correction) * step
// We use a simple form:
//   forward:   q = sign(x) * max(0, (int)((|x| - 0.2*step) / step + 0.5))  if |x| >= 0.6*step
//              q = 0 otherwise
//   inverse:   x_hat = sign(q) * (|q| + 0.375) * step
// ===========================================================================

// Uniform Reconstruction Quantizer with Dead-Zone (URQ-DZ), JPEG2000-style.
//   forward:   q = sign(x) * floor(|x| / step)    (deadzone width = step)
//   inverse:   x_hat = sign(q) * (|q| + 0.375) * step  for q != 0
// The +0.375 reconstruction bias is the Laplacian-optimal centroid offset
// at moderate bit rates (slightly biased toward zero from the bin midpoint).
//
static inline int32_t quantize_one(float v, float step) {
    if (step <= 0.0f) return 0;
    float a = fabsf(v);
    int32_t mag = (int32_t)(a / step);
    return (v >= 0.0f) ? mag : -mag;
}
static inline float dequantize_one(int32_t q, float step) {
    if (q == 0) return 0.0f;
    int32_t mag = q > 0 ? q : -q;
    float v = ((float)mag + 0.375f) * step;
    return q > 0 ? v : -v;
}

// ===========================================================================
// 65-symbol mapping (zigzag + escape with LEB128)
// ===========================================================================

static inline uint32_t zigzag_enc(int32_t v) {
    return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31);
}
static inline int32_t zigzag_dec(uint32_t z) {
    return (int32_t)((z >> 1) ^ -(z & 1));
}

#define C2D_ESCAPE_SYMBOL 64u
#define C2D_ALPHABET      65u

static inline void coef_to_symbol(int32_t qv, uint8_t *out_sym, uint32_t *out_esc) {
    uint32_t z = zigzag_enc(qv);
    if (z < 64) { *out_sym = (uint8_t)z; *out_esc = 0; }
    else        { *out_sym = (uint8_t)C2D_ESCAPE_SYMBOL; *out_esc = z - 63; }
}
static inline int32_t symbol_to_coef(uint8_t sym, uint32_t esc) {
    uint32_t z = (sym < 64) ? (uint32_t)sym : (esc + 63);
    return zigzag_dec(z);
}

// ===========================================================================
// rANS (8-way interleaved, 32-bit state, byte renorm, 12-bit precision)
// ===========================================================================

#define RANS_PROB_BITS  12u
#define RANS_M          (1u << RANS_PROB_BITS)
#define RANS_L          (1u << 23)
// rANS state count. 4-way interleaved (rygorous-style) gives ~30% decode
// speedup but costs +12 bytes/subband in bitstream overhead which translates
// to ~0.5 dB quality regression at low bpp on natural images. 1-way is the
// quality-optimal default; switching to 4 trades quality for decode speed.
#define RANS_NSTATES    1u

typedef struct rans_enc {
    uint8_t *buf;
    size_t   cap;
    size_t   pos;
    uint32_t state[RANS_NSTATES];
} rans_enc;

static void rans_enc_init(rans_enc *e, uint8_t *buf, size_t cap) {
    e->buf = buf; e->cap = cap; e->pos = 0;
    for (uint32_t i = 0; i < RANS_NSTATES; i++) e->state[i] = RANS_L;
}

static inline void rans_enc_put(rans_enc *e, uint32_t lane,
                                uint32_t freq, uint32_t cum) {
    c2d_assert(freq > 0);
    uint32_t x = e->state[lane];
    uint32_t x_max = ((RANS_L >> RANS_PROB_BITS) << 8) * freq;
    while (x >= x_max) {
        c2d_assert(e->pos < e->cap);
        e->buf[e->cap - 1 - e->pos] = (uint8_t)(x & 0xff);
        e->pos++;
        x >>= 8;
    }
    e->state[lane] = ((x / freq) << RANS_PROB_BITS) + (x % freq) + cum;
}

static void rans_enc_finish(rans_enc *e, uint8_t *final_states_out) {
    for (uint32_t i = 0; i < RANS_NSTATES; i++) {
        uint32_t s = e->state[i];
        final_states_out[i*4 + 0] = (uint8_t)(s      );
        final_states_out[i*4 + 1] = (uint8_t)(s >>  8);
        final_states_out[i*4 + 2] = (uint8_t)(s >> 16);
        final_states_out[i*4 + 3] = (uint8_t)(s >> 24);
    }
}
#define RANS_STATE_BYTES (4u * RANS_NSTATES)

// Helper: load 4 (or RANS_NSTATES) parallel rANS states from a packed
// little-endian byte array.
static inline void rans_states_load(uint32_t *out, const uint8_t *bytes) {
    for (uint32_t i = 0; i < RANS_NSTATES; i++) {
        uint32_t s = 0;
        s |= (uint32_t)bytes[i*4 + 0];
        s |= (uint32_t)bytes[i*4 + 1] << 8;
        s |= (uint32_t)bytes[i*4 + 2] << 16;
        s |= (uint32_t)bytes[i*4 + 3] << 24;
        out[i] = s;
    }
}

// Forward declarations of leb128 helpers used by EBCOT below.
static size_t leb128_write(uint32_t v, uint8_t *out);
static size_t leb128_read(const uint8_t *in, size_t avail, uint32_t *out);

// ===========================================================================
// Binary adaptive arithmetic coder (range coder).
//
// `low` is u64 to hold the carry naturally; `range` stays u32. Carry from
// addition lands in bit 32+ of `low`. When we shift out byte 24..31, we
// also look at bits 32+ — if any are set, they propagate into the byte
// being shifted out (carry) and the previous cached byte gets +1.
//
// Cache + pending FFs handle delayed carry across long FF runs.
// ===========================================================================

#define AC_BITS  12u
#define AC_M     (1u << AC_BITS)

typedef struct ac_enc {
    uint64_t low;          // bits 0..31 hold value; bits 32+ hold carry
    uint32_t range;
    uint8_t *buf;
    size_t cap;
    size_t pos;
    int64_t pending;       // count of pending 0xFF bytes awaiting carry decision
    int started;           // first byte cached?
    uint8_t cache;         // candidate first byte (may get +1 carry)
} ac_enc;

static void ac_enc_init(ac_enc *e, uint8_t *buf, size_t cap) {
    e->low = 0; e->range = 0xFFFFFFFFu;
    e->buf = buf; e->cap = cap; e->pos = 0;
    e->pending = 0; e->started = 0; e->cache = 0;
}

static inline void ac_emit(ac_enc *e, uint8_t b) {
    if (e->pos < e->cap) e->buf[e->pos++] = b;
}

static inline void ac_shift(ac_enc *e) {
    // The byte we want to emit is bits 24..31 of low.
    // But bits 32+ of low (if set) constitute a carry that must propagate
    // into bits 24..31 *first*, possibly bumping the cached byte too.
    uint64_t l = e->low;
    int carry = (int)((l >> 32) & 0xFF);    // can be 0 or 1 normally (more if many adds without renorm — impossible here since we renorm each call)
    uint8_t out_byte = (uint8_t)(l >> 24);  // truncated; if carry, this needs +carry
    // Actually `out_byte + carry` is the real byte that should be cached
    // BUT also propagated back to the previous cache+pending.
    if (!e->started) {
        e->cache = out_byte;   // no carry possible on the very first shift
        e->started = 1;
    } else {
        if (carry) {
            // Carry into the cached byte; pending FFs flip to 00s.
            ac_emit(e, (uint8_t)(e->cache + 1));
            while (e->pending > 0) { ac_emit(e, 0x00); e->pending--; }
            e->cache = out_byte;   // carry was absorbed; new cache = current byte
        } else if (out_byte == 0xFF) {
            e->pending++;
        } else {
            ac_emit(e, e->cache);
            while (e->pending > 0) { ac_emit(e, 0xFF); e->pending--; }
            e->cache = out_byte;
        }
    }
    // Shift low by 8 and clear the carry bit.
    e->low = (l & 0x00FFFFFFu) << 8;
}

static inline void ac_enc_renorm(ac_enc *e) {
    while (e->range < (1u << 24)) {
        ac_shift(e);
        e->range <<= 8;
    }
}

static inline void ac_enc_bit(ac_enc *e, uint32_t bit, uint32_t p0) {
    uint32_t r0 = (e->range >> AC_BITS) * p0;
    if (bit == 0) {
        e->range = r0;
    } else {
        e->low += r0;       // carry naturally goes into bit 32+
        e->range -= r0;
    }
    ac_enc_renorm(e);
}

static size_t ac_enc_finish(ac_enc *e) {
    // Flush low (5 bytes are enough to drain any carry-laden 32-bit value).
    for (int i = 0; i < 5; i++) ac_shift(e);
    if (e->started) {
        ac_emit(e, e->cache);
        while (e->pending > 0) { ac_emit(e, 0xFF); e->pending--; }
    }
    return e->pos;
}

// ===========================================================================
// EBCOT-lite bit-plane subband coding.
//
// For each subband with quantized int32 coefs q[i]:
//   K = max bit-plane needed (ceil(log2(max|q|+1)))
//   For p = K-1 .. 0:
//     significance pass: for each non-significant coef, encode "becomes
//       significant at this plane?" bit. If 1, encode sign bit.
//     refinement pass: for each already-significant coef, encode bit p of |q|.
// Context model: 5 contexts for sig-bit (by neighbor-sig count), 1 sign
// context, 1 refinement context. Adaptive probabilities (IIR rate-5 update).
//
// Per-subband output: K (varint, low bit reserved), then the AC bitstream.
// ===========================================================================

#define BP_SIG_CTX_COUNT    5u
// Sign-prediction contexts (JPEG2000-style). Indexed 0..4:
//   0: both H/V neighbors known with same sign (strong prediction)
//   1: both H/V known with opposite signs (weak/no prediction)
//   2: exactly one neighbor known positive (predict +)
//   3: exactly one neighbor known negative (predict -)
//   4: neither neighbor significant yet (no prediction)
#define BP_SIGN_CTX_COUNT   5u
#define BP_REFINE_CTX_COUNT 1u
// One zero-block skip context — single adaptive bit per 4x4 block in the sig
// pass when (a) all 16 coefs are still insignificant and (b) no sig neighbors
// (the "easy to predict all-zero" case). Bit=0 means "no new sig in this block";
// decoder then skips coding the 16 sig bits. Bit=1 means "code the 16 normally".
#define BP_ZSKIP_CTX_COUNT  1u
#define BP_TOTAL_CTX     (BP_SIG_CTX_COUNT + BP_SIGN_CTX_COUNT + BP_REFINE_CTX_COUNT + BP_ZSKIP_CTX_COUNT)
#define BP_SIGN_CTX      (BP_SIG_CTX_COUNT)
#define BP_REFINE_CTX    (BP_SIG_CTX_COUNT + BP_SIGN_CTX_COUNT)
#define BP_ZSKIP_CTX     (BP_SIG_CTX_COUNT + BP_SIGN_CTX_COUNT + BP_REFINE_CTX_COUNT)

// Initial p0 for each context (P(0) = 0.5 = AC_M/2). After adaptation
// converges, sig bits in higher planes lean toward 0 (most coefs not yet
// significant), refinement bits ~0.5.
static inline void bp_ctx_init(uint16_t *ctx /*[BP_TOTAL_CTX]*/) {
    for (uint32_t i = 0; i < BP_TOTAL_CTX; i++) ctx[i] = AC_M / 2;
}

// IIR update with rate 5 (lambda ~ 1/32). Clamped to [1, AC_M-1].
static inline void bp_ctx_update(uint16_t *ctx, uint32_t bit) {
    int32_t target = (bit == 0) ? (int32_t)AC_M : 0;
    int32_t delta = (target - (int32_t)*ctx) >> 4;
    int32_t v = (int32_t)*ctx + delta;
    if (v < 1) v = 1;
    if (v > (int32_t)AC_M - 1) v = (int32_t)AC_M - 1;
    *ctx = (uint16_t)v;
}

// Count of already-significant neighbors (W, NW, N, NE) for coef (y, x).
// Kind-aware: HL (horizontal LP, vertical HP -> horizontal edges) weights
// vertical neighbors more; LH (vertical LP, horizontal HP -> vertical edges)
// weights horizontal neighbors more; HH/LL use uniform weights.
// kind: 0=LL, 1=HL, 2=LH, 3=HH.
static inline uint32_t bp_neigh_sig_count(const uint8_t *sig_bits,
                                          uint32_t x, uint32_t y, uint32_t w,
                                          uint32_t kind) {
    uint32_t W  = (x > 0)              ? sig_bits[y * w + (x - 1)] : 0;
    uint32_t NW = (x > 0 && y > 0)     ? sig_bits[(y - 1) * w + (x - 1)] : 0;
    uint32_t N  = (y > 0)              ? sig_bits[(y - 1) * w + x] : 0;
    uint32_t NE = (y > 0 && x + 1 < w) ? sig_bits[(y - 1) * w + (x + 1)] : 0;
    uint32_t s;
    if (kind == 1) {
        // HL: horizontal edges -> vertical neighbor (N) most predictive.
        s = 2u * N + W + NW + NE;
    } else if (kind == 2) {
        // LH: vertical edges -> horizontal neighbor (W) most predictive.
        s = 2u * W + N + NW + NE;
    } else {
        s = W + NW + N + NE;
    }
    if (s > 4) s = 4;
    return s;
}

// Sign-prediction lookup. `signed_sig[i]` is 0 (not sig), +1 (sig +), -1 (sig -).
// Uses W, N, E, S neighbors (the latter two from prior bit-planes' sig state,
// available even in raster scan since they became sig in higher planes).
// Combine horizontal-pair and vertical-pair signs (sum each).
// Returns (ctx_idx, predicted_bit) packed: bit 0 = predicted sign bit, bits 4..6 = ctx.
static inline uint32_t bp_sign_pred(const int8_t *signed_sig,
                                    uint32_t x, uint32_t y,
                                    uint32_t w, uint32_t h) {
    int8_t W = (x > 0)       ? signed_sig[y * w + (x - 1)] : 0;
    int8_t E = (x + 1 < w)   ? signed_sig[y * w + (x + 1)] : 0;
    int8_t N = (y > 0)       ? signed_sig[(y - 1) * w + x] : 0;
    int8_t S = (y + 1 < h)   ? signed_sig[(y + 1) * w + x] : 0;
    int hsum = (int)W + (int)E;   // -2..+2
    int vsum = (int)N + (int)S;   // -2..+2
    int h_ind = (hsum > 0) ? 1 : (hsum < 0 ? -1 : 0);
    int v_ind = (vsum > 0) ? 1 : (vsum < 0 ? -1 : 0);
    uint32_t pred = 0, ctx = 4;
    if (h_ind != 0 && v_ind != 0) {
        if (h_ind == v_ind) { ctx = 0; pred = (h_ind < 0) ? 1u : 0u; }
        else                { ctx = 1; pred = 0u; }
    } else if (h_ind != 0 || v_ind != 0) {
        int k = h_ind != 0 ? h_ind : v_ind;
        if (k > 0) { ctx = 2; pred = 0u; }
        else       { ctx = 3; pred = 1u; }
    } else {
        ctx = 4; pred = 0u;
    }
    return (ctx << 4) | pred;
}

// Find max bit-plane K = ceil(log2(max|q|+1)). Returns 0 if all coefs are zero.
static uint32_t bp_max_bitplane(const int32_t *q, size_t n) {
    int32_t max_mag = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t a = q[i] >= 0 ? q[i] : -q[i];
        if (a > max_mag) max_mag = a;
    }
    if (max_mag == 0) return 0;
    uint32_t k = 0;
    while ((1 << k) <= max_mag) k++;
    return k;
}

// Encode one subband band of int32 coefs into the AC stream. Writes K (varint)
// + AC bitstream to `out`. Returns bytes written.
static size_t bp_encode_band(const int32_t *q, uint32_t w, uint32_t h,
                             uint32_t kind, uint32_t level,
                             uint8_t *out, size_t out_cap) {
    (void)level;
    size_t n = (size_t)w * h;
    uint32_t K = bp_max_bitplane(q, n);
    size_t pos = 0;
    if (pos + 10 > out_cap) return 0;
    // Encode K with low bit = "truncated" flag (always 0 now; reserved).
    uint32_t tag = K << 1;
    pos += leb128_write(tag, out + pos);
    if (K == 0) return pos;

    // Per-coef state: sig_bits (0 not yet significant), and |q[i]| for
    // refinement (use uint32_t to be safe).
    uint8_t sig_bits[C2D_PIXELS_PER_TILE];
    memset(sig_bits, 0, n);
    int8_t signed_sig[C2D_PIXELS_PER_TILE];   // 0/+1/-1 for sign prediction
    memset(signed_sig, 0, n);
    uint32_t mag[C2D_PIXELS_PER_TILE];
    for (size_t i = 0; i < n; i++) mag[i] = q[i] >= 0 ? (uint32_t)q[i] : (uint32_t)(-q[i]);
    uint8_t sgn[C2D_PIXELS_PER_TILE];
    for (size_t i = 0; i < n; i++) sgn[i] = q[i] < 0 ? 1u : 0u;
    uint8_t became_at[C2D_PIXELS_PER_TILE];   // plane at which coef became sig
    memset(became_at, 0xFF, n);

    uint16_t ctx[BP_TOTAL_CTX];
    bp_ctx_init(ctx);

    ac_enc enc;
    ac_enc_init(&enc, out + pos, out_cap - pos);

    for (int32_t p = (int32_t)K - 1; p >= 0; p--) {
        uint32_t mask = 1u << p;
        // Significance + sign pass. Process 4x4 blocks; emit a zero-block-skip
        // bit per block whose 16 coefs are all insig and have no sig neighbors.
        for (uint32_t by = 0; by < h; by += 4) {
            uint32_t yend = by + 4 < h ? by + 4 : h;
            for (uint32_t bx = 0; bx < w; bx += 4) {
                uint32_t xend = bx + 4 < w ? bx + 4 : w;
                // Check skip eligibility: all coefs insig AND none has sig neighbor.
                int eligible = 1;
                int any_will_sig = 0;
                for (uint32_t y = by; y < yend && eligible; y++) {
                    for (uint32_t xp = bx; xp < xend; xp++) {
                        size_t i = (size_t)y * w + xp;
                        if (sig_bits[i]) { eligible = 0; break; }
                        if (bp_neigh_sig_count(sig_bits, xp, y, w, kind)) { eligible = 0; break; }
                    }
                }
                if (eligible) {
                    for (uint32_t y = by; y < yend && !any_will_sig; y++) {
                        for (uint32_t xp = bx; xp < xend; xp++) {
                            if (mag[(size_t)y * w + xp] & mask) { any_will_sig = 1; break; }
                        }
                    }
                    ac_enc_bit(&enc, (uint32_t)any_will_sig, ctx[BP_ZSKIP_CTX]);
                    bp_ctx_update(&ctx[BP_ZSKIP_CTX], (uint32_t)any_will_sig);
                    if (!any_will_sig) continue;
                }
                // Code the 16 sig bits normally.
                for (uint32_t y = by; y < yend; y++) {
                    for (uint32_t xp = bx; xp < xend; xp++) {
                        size_t i = (size_t)y * w + xp;
                        if (sig_bits[i]) continue;
                        uint32_t becomes_sig = (mag[i] & mask) ? 1u : 0u;
                        uint32_t nbr = bp_neigh_sig_count(sig_bits, xp, y, w, kind);
                        ac_enc_bit(&enc, becomes_sig, ctx[nbr]);
                        bp_ctx_update(&ctx[nbr], becomes_sig);
                        if (becomes_sig) {
                            uint32_t sp = bp_sign_pred(signed_sig, xp, y, w, h);
                            uint32_t sc = sp >> 4;
                            uint32_t spred = sp & 1u;
                            uint32_t bit_to_code = (uint32_t)sgn[i] ^ spred;
                            ac_enc_bit(&enc, bit_to_code, ctx[BP_SIGN_CTX + sc]);
                            bp_ctx_update(&ctx[BP_SIGN_CTX + sc], bit_to_code);
                            sig_bits[i] = 1;
                            signed_sig[i] = sgn[i] ? -1 : 1;
                            became_at[i] = (uint8_t)p;
                        }
                    }
                }
            }
        }
        // Refinement pass.
        for (size_t i = 0; i < n; i++) {
            if (!sig_bits[i]) continue;
            if ((int32_t)became_at[i] == p) continue;
            uint32_t refine_bit = (mag[i] & mask) ? 1u : 0u;
            ac_enc_bit(&enc, refine_bit, ctx[BP_REFINE_CTX]);
            bp_ctx_update(&ctx[BP_REFINE_CTX], refine_bit);
        }
    }

    size_t ac_bytes = ac_enc_finish(&enc);
    return pos + ac_bytes;
}

// Decode the band; n is known by caller from rect.
static void bp_decode_band(const uint8_t *in, size_t in_cap,
                           uint32_t w, uint32_t h, uint32_t kind, uint32_t level,
                           float qstep, float *band) {
    (void)level;
    size_t n = (size_t)w * h;
    size_t pos = 0;
    uint32_t tag;
    pos += leb128_read(in + pos, in_cap - pos, &tag);
    uint32_t K = tag >> 1;
    // tag low bit reserved (was truncation flag); ignored.
    if (K == 0) {
        for (size_t i = 0; i < n; i++) band[i] = 0.0f;
        return;
    }

    uint8_t sig_bits[C2D_PIXELS_PER_TILE];
    memset(sig_bits, 0, n);
    int8_t signed_sig[C2D_PIXELS_PER_TILE];
    memset(signed_sig, 0, n);
    uint32_t mag[C2D_PIXELS_PER_TILE];
    memset(mag, 0, sizeof(uint32_t) * n);
    uint8_t sgn[C2D_PIXELS_PER_TILE];
    memset(sgn, 0, n);

    // Track which bit-plane is "just-significant" for each coef so we know
    // which bits are refinements vs sig pass.
    uint8_t became_at[C2D_PIXELS_PER_TILE];
    memset(became_at, 0xFF, n);

    uint16_t ctx[BP_TOTAL_CTX];
    bp_ctx_init(ctx);

    // Inline AC decode state for max perf.
    const uint8_t *rp = in + pos;
    const uint8_t *rp_end = in + in_cap;
    uint32_t ac_range = 0xFFFFFFFFu;
    uint32_t ac_code = 0;
    for (int i = 0; i < 4; i++)
        ac_code = (ac_code << 8) | (rp < rp_end ? *rp++ : 0);

#define AC_DEC_BIT(P0, OUT_BIT) do { \
    uint32_t _r0 = (ac_range >> AC_BITS) * (P0); \
    if (ac_code < _r0) { (OUT_BIT) = 0; ac_range = _r0; } \
    else               { (OUT_BIT) = 1; ac_code -= _r0; ac_range -= _r0; } \
    while (ac_range < (1u << 24)) { \
        ac_code = (ac_code << 8) | (rp < rp_end ? *rp++ : 0); \
        ac_range <<= 8; \
    } \
} while (0)

    for (int32_t p = (int32_t)K - 1; p >= 0; p--) {
        uint32_t mask = 1u << p;
        // Significance + sign pass. 4x4 blocks; skip bit may zero entire block.
        for (uint32_t by = 0; by < h; by += 4) {
            uint32_t yend = by + 4 < h ? by + 4 : h;
            for (uint32_t bx = 0; bx < w; bx += 4) {
                uint32_t xend = bx + 4 < w ? bx + 4 : w;
                int eligible = 1;
                for (uint32_t y = by; y < yend && eligible; y++) {
                    for (uint32_t xp = bx; xp < xend; xp++) {
                        size_t i = (size_t)y * w + xp;
                        if (sig_bits[i]) { eligible = 0; break; }
                        if (bp_neigh_sig_count(sig_bits, xp, y, w, kind)) { eligible = 0; break; }
                    }
                }
                if (eligible) {
                    uint32_t any_sig;
                    AC_DEC_BIT(ctx[BP_ZSKIP_CTX], any_sig);
                    bp_ctx_update(&ctx[BP_ZSKIP_CTX], any_sig);
                    if (!any_sig) continue;
                }
                for (uint32_t y = by; y < yend; y++) {
                    for (uint32_t xp = bx; xp < xend; xp++) {
                        size_t i = (size_t)y * w + xp;
                        if (sig_bits[i]) continue;
                        uint32_t nbr = bp_neigh_sig_count(sig_bits, xp, y, w, kind);
                        uint32_t becomes_sig;
                        AC_DEC_BIT(ctx[nbr], becomes_sig);
                        bp_ctx_update(&ctx[nbr], becomes_sig);
                        if (becomes_sig) {
                            uint32_t sp = bp_sign_pred(signed_sig, xp, y, w, h);
                            uint32_t sc = sp >> 4;
                            uint32_t spred = sp & 1u;
                            uint32_t coded_bit;
                            AC_DEC_BIT(ctx[BP_SIGN_CTX + sc], coded_bit);
                            bp_ctx_update(&ctx[BP_SIGN_CTX + sc], coded_bit);
                            uint32_t s = coded_bit ^ spred;
                            sgn[i] = (uint8_t)s;
                            sig_bits[i] = 1;
                            signed_sig[i] = s ? -1 : 1;
                            mag[i] |= mask;
                            became_at[i] = (uint8_t)p;
                        }
                    }
                }
            }
        }
        // Refinement pass.
        for (size_t i = 0; i < n; i++) {
            if (!sig_bits[i]) continue;
            if ((int32_t)became_at[i] == p) continue;
            uint32_t refine_bit;
            AC_DEC_BIT(ctx[BP_REFINE_CTX], refine_bit);
            bp_ctx_update(&ctx[BP_REFINE_CTX], refine_bit);
            if (refine_bit) mag[i] |= mask;
        }
    }
#undef AC_DEC_BIT

    // Dequantize.
    for (size_t i = 0; i < n; i++) {
        int32_t qv = sgn[i] ? -(int32_t)mag[i] : (int32_t)mag[i];
        band[i] = dequantize_one(qv, qstep);
    }
}

// ===========================================================================
// LEB128
// ===========================================================================

static size_t leb128_write(uint32_t v, uint8_t *out) {
    size_t n = 0;
    do {
        uint8_t b = v & 0x7f;
        v >>= 7;
        if (v) b |= 0x80;
        out[n++] = b;
    } while (v);
    return n;
}
static size_t leb128_read(const uint8_t *in, size_t avail, uint32_t *out) {
    uint32_t v = 0; uint32_t shift = 0; size_t n = 0;
    while (1) {
        c2d_assert(n < avail);
        uint8_t b = in[n++];
        v |= (uint32_t)(b & 0x7f) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
        c2d_assert(shift < 32);
    }
    *out = v;
    return n;
}
static inline uint32_t leb128_size(uint32_t v) {
    uint32_t n = 1;
    while (v >= 0x80) { v >>= 7; n++; }
    return n;
}

// ===========================================================================
// Frequency normalization
// ===========================================================================

static void normalize_freqs(const uint32_t *hist, uint32_t M,
                            uint16_t out_freq[C2D_ALPHABET]) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < C2D_ALPHABET; i++) total += hist[i];
    if (total == 0) {
        for (uint32_t i = 0; i < C2D_ALPHABET; i++) out_freq[i] = 0;
        out_freq[0] = (uint16_t)M;
        return;
    }
    uint32_t used = 0;
    for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
        if (hist[i] == 0) { out_freq[i] = 0; continue; }
        uint64_t f = ((uint64_t)hist[i] * M + total/2) / total;
        if (f == 0) f = 1;
        if (f > M)  f = M;
        out_freq[i] = (uint16_t)f;
        used += (uint32_t)f;
    }
    int32_t diff = (int32_t)M - (int32_t)used;
    if (diff != 0) {
        uint32_t maxi = 0; uint16_t maxf = 0;
        for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
            if (out_freq[i] > maxf) { maxf = out_freq[i]; maxi = i; }
        }
        int32_t nf = (int32_t)out_freq[maxi] + diff;
        c2d_assert(nf > 0 && nf <= (int32_t)M);
        out_freq[maxi] = (uint16_t)nf;
    }
}

// ===========================================================================
// Bitstream r/w helpers
// ===========================================================================

static inline void write_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
static inline void write_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline void write_f32(uint8_t *p, float f) {
    uint32_t u; memcpy(&u, &f, 4); write_u32(p, u);
}
static inline uint16_t read_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static inline uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline float read_f32(const uint8_t *p) {
    uint32_t u = read_u32(p); float f; memcpy(&f, &u, 4); return f;
}

#define HDR_FIXED_SIZE   (4 + 1 + 1 + 1 + 1 + 4 + 4 * C2D_N_LODS)
#define HDR_NORM_SIZE(n) ((size_t)(n) * 8u)
#define HDR_DIR_SIZE(n)  ((size_t)(n) * C2D_N_SUBBANDS * 4u * 2u)
#define HDR_TOTAL(n)     (HDR_FIXED_SIZE + HDR_NORM_SIZE(n) + HDR_DIR_SIZE(n))

// ===========================================================================
// Encoder context: pooled scratch buffers, persistent across calls.
//
// Memory layout:
//   planes      : nch_cap planes of 256*256 f32 (DWT working space)
//   dwt_tmp     : 256 f32  (1D column scratch)
//   mallat      : 256*256 f32 (mallat pack/unpack scratch)
//   coef_buf    : enough to hold all (band, channel) int32 coefficients
//                 (max nch_cap planes * 65536 = 65536 int32 per channel)
//   hist        : C2D_N_SUBBANDS * nch_cap * 65 uint32
//   blob_buf    : per-subband output scratch; sized worst case
//   renorm_buf  : per-subband rANS renorm buffer
//   escape_buf  : per-subband escape varint buffer
//   cum2sym     : 4096 uint16 (built per subband in decode)
// ===========================================================================

struct c2d_encoder {
    uint32_t nch_cap;
    float   *planes_mem;    // nch_cap * 256*256
    float    dwt_tmp[C2D_TILE_SIDE];
    float   *mallat;        // 256*256
    int32_t *coef_mem;      // nch_cap * 65536
    uint32_t *hist;         // nch_cap * 16 * 65
    uint8_t *blob_buf;
    uint8_t *renorm_buf;
    uint8_t *escape_buf;
    float   *dcs;
    float   *scales;
};
// Per-slot rANS decode table: packs (sym, freq, cum) into a single 8-byte
// load. Decoding one symbol becomes a single memory access into slot_tbl
// instead of two (cum2sym + syms[sym]).
typedef struct rans_slot {
    uint16_t sym;
    uint16_t freq;
    uint16_t cum;
    uint16_t _pad;
} rans_slot;

struct c2d_decoder {
    uint32_t nch_cap;
    float   *planes_mem;
    float    dwt_tmp[C2D_TILE_SIDE];
    float   *mallat;
    float   *dcs;
    float   *scales;
    float   *band_buf;     // 256*256 f32, holds one decoded subband at a time
    rans_slot slot_tbl[RANS_M];
};

static void encoder_grow(c2d_encoder *e, uint32_t nch) {
    if (nch <= e->nch_cap) return;
    free(e->planes_mem); free(e->coef_mem); free(e->hist);
    free(e->dcs); free(e->scales);
    e->nch_cap    = nch;
    e->planes_mem = (float    *)aligned_alloc(C2D_ALIGN, sizeof(float)    * (size_t)nch * C2D_PIXELS_PER_TILE);
    e->coef_mem   = (int32_t  *)aligned_alloc(C2D_ALIGN, sizeof(int32_t)  * (size_t)nch * C2D_PIXELS_PER_TILE);
    e->hist       = (uint32_t *)calloc((size_t)nch * C2D_N_SUBBANDS * C2D_ALPHABET, sizeof(uint32_t));
    e->dcs        = (float    *)calloc(nch, sizeof(float));
    e->scales     = (float    *)calloc(nch, sizeof(float));
}
static void decoder_grow(c2d_decoder *d, uint32_t nch) {
    if (nch <= d->nch_cap) return;
    free(d->planes_mem); free(d->dcs); free(d->scales);
    d->nch_cap    = nch;
    d->planes_mem = (float *)aligned_alloc(C2D_ALIGN, sizeof(float) * (size_t)nch * C2D_PIXELS_PER_TILE);
    d->dcs        = (float *)calloc(nch, sizeof(float));
    d->scales     = (float *)calloc(nch, sizeof(float));
}

c2d_encoder *c2d_encoder_new(void) {
    c2d_encoder *e = (c2d_encoder *)calloc(1, sizeof(c2d_encoder));
    e->mallat     = (float   *)aligned_alloc(C2D_ALIGN, sizeof(float) * C2D_PIXELS_PER_TILE);
    // Worst-case per-subband blob: 1 + 65*3 freq table + 4 + 4 + 32 + 5*N rans + 4 + 5*N escape.
    size_t per_sb = 1 + 65*3 + 4 + 4 + 32 + C2D_PIXELS_PER_TILE*5 + 4 + C2D_PIXELS_PER_TILE*5 + 64;
    e->blob_buf   = (uint8_t *)malloc(per_sb);
    e->renorm_buf = (uint8_t *)malloc(C2D_PIXELS_PER_TILE * 8 + 1024);
    e->escape_buf = (uint8_t *)malloc(C2D_PIXELS_PER_TILE * 5 + 64);
    encoder_grow(e, 4);
    return e;
}
void c2d_encoder_free(c2d_encoder *e) {
    if (!e) return;
    free(e->planes_mem); free(e->mallat); free(e->coef_mem); free(e->hist);
    free(e->blob_buf); free(e->renorm_buf); free(e->escape_buf);
    free(e->dcs); free(e->scales);
    free(e);
}
c2d_decoder *c2d_decoder_new(void) {
    c2d_decoder *d = (c2d_decoder *)calloc(1, sizeof(c2d_decoder));
    d->mallat   = (float *)aligned_alloc(C2D_ALIGN, sizeof(float) * C2D_PIXELS_PER_TILE);
    d->band_buf = (float *)aligned_alloc(C2D_ALIGN, sizeof(float) * C2D_PIXELS_PER_TILE);
    decoder_grow(d, 4);
    return d;
}
void c2d_decoder_free(c2d_decoder *d) {
    if (!d) return;
    free(d->planes_mem); free(d->mallat); free(d->band_buf);
    free(d->dcs); free(d->scales);
    free(d);
}

// Accessors into pooled buffers.
static inline float   *enc_plane(c2d_encoder *e, uint32_t c) { return e->planes_mem + (size_t)c * C2D_PIXELS_PER_TILE; }
static inline int32_t *enc_coefs(c2d_encoder *e, uint32_t c) { return e->coef_mem   + (size_t)c * C2D_PIXELS_PER_TILE; }
static inline uint32_t *enc_hist(c2d_encoder *e, uint32_t b, uint32_t c, uint32_t nch) {
    return e->hist + (((size_t)b * nch + c) * C2D_ALPHABET);
}
static inline float   *dec_plane(c2d_decoder *d, uint32_t c) { return d->planes_mem + (size_t)c * C2D_PIXELS_PER_TILE; }

// ===========================================================================
// Encode pass 1: dtype -> f32 -> ycocg -> DWT -> per-(band,channel) quantize
// into encoder's coef_mem at given q. Builds histograms in encoder->hist.
// Returns total escape varint bytes (data-dependent count of magnitudes > 63).
// ===========================================================================

static size_t encode_transform_and_quantize(c2d_encoder *e,
                                            const subband_rect *rects,
                                            uint32_t nch, float q, int do_ycocg,
                                            uint32_t flags) {
    // Per-plane DWT.
    for (uint32_t c = 0; c < nch; c++) {
        dwt_fwd(enc_plane(e, c), C2D_TILE_SIDE, e->dwt_tmp, e->mallat);
    }
    // Per-(band, channel) quantize into coef_mem; layout within a channel is
    // band-by-band contiguous, matching rect order.
    // Build histograms.
    size_t escape_bytes = 0;
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        const subband_rect *r = &rects[b];
        size_t npix = (size_t)r->w * (size_t)r->h;
        float qbase = q * subband_baseline_flags(r, flags);
        for (uint32_t c = 0; c < nch; c++) {
            float step = qbase;
            if (do_ycocg) step *= channel_weight_ycocg(c);
            if (step < 1e-20f) step = 1e-20f;
            const float *plane = enc_plane(e, c);
            int32_t *cbuf = enc_coefs(e, c);
            // Determine where this band's coefs go within the per-channel coef_mem.
            // We pack bands consecutively in the order of `rects`; compute offset.
            size_t band_off = 0;
            for (uint32_t bb = 0; bb < b; bb++) band_off += (size_t)rects[bb].w * rects[bb].h;
            int32_t *qout = cbuf + band_off;
            uint32_t *hist = enc_hist(e, b, c, nch);
            memset(hist, 0, sizeof(uint32_t) * C2D_ALPHABET);
            for (uint32_t y = 0; y < r->h; y++) {
                const float *row = plane + (size_t)(r->y + y) * C2D_TILE_SIDE + r->x;
                for (uint32_t x = 0; x < r->w; x++) {
                    int32_t qi = quantize_one(row[x], step);
                    qout[(size_t)y * r->w + x] = qi;
                    uint8_t sym; uint32_t esc;
                    coef_to_symbol(qi, &sym, &esc);
                    hist[sym]++;
                    if (sym == C2D_ESCAPE_SYMBOL) escape_bytes += leb128_size(esc);
                }
            }
            (void)npix;
        }
    }
    return escape_bytes;
}

// Like requantize_only but skips the coef_pool write (just histogram +
// escape bytes). Used during the tile API's bisection.
static size_t requantize_hist_only(c2d_encoder *e,
                                   const subband_rect *rects,
                                   uint32_t nch, float q, int do_ycocg,
                                   uint32_t flags) {
    size_t escape_bytes = 0;
    size_t band_off_tbl[C2D_N_SUBBANDS];
    band_off_tbl[0] = 0;
    for (uint32_t b = 1; b < C2D_N_SUBBANDS; b++)
        band_off_tbl[b] = band_off_tbl[b-1] + (size_t)rects[b-1].w * rects[b-1].h;
    (void)band_off_tbl;
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        const subband_rect *r = &rects[b];
        float qbase = q * subband_baseline_flags(r, flags);
        for (uint32_t c = 0; c < nch; c++) {
            float step = qbase;
            if (do_ycocg) step *= channel_weight_ycocg(c);
            if (step < 1e-20f) step = 1e-20f;
            float inv_step = 1.0f / step;
            const float *plane = enc_plane(e, c);
            uint32_t *hist = enc_hist(e, b, c, nch);
            memset(hist, 0, sizeof(uint32_t) * C2D_ALPHABET);
            for (uint32_t y = 0; y < r->h; y++) {
                const float *row = plane + (size_t)(r->y + y) * C2D_TILE_SIDE + r->x;
                for (uint32_t x = 0; x < r->w; x++) {
                    float v = row[x];
                    int32_t mag = (int32_t)(fabsf(v) * inv_step);
                    int32_t qi = (v >= 0.0f) ? mag : -mag;
                    uint32_t z = ((uint32_t)qi << 1) ^ (uint32_t)(qi >> 31);
                    if (z < 64) {
                        hist[z]++;
                    } else {
                        hist[C2D_ESCAPE_SYMBOL]++;
                        escape_bytes += leb128_size(z - 63);
                    }
                }
            }
        }
    }
    return escape_bytes;
}

// Re-quantize only (DWT already done). Used by bisection.
static size_t requantize_only(c2d_encoder *e,
                              const subband_rect *rects,
                              uint32_t nch, float q, int do_ycocg,
                              uint32_t flags) {
    size_t escape_bytes = 0;
    // Precompute per-band coef offsets within a channel's coef_mem layout.
    size_t band_off_tbl[C2D_N_SUBBANDS];
    band_off_tbl[0] = 0;
    for (uint32_t b = 1; b < C2D_N_SUBBANDS; b++)
        band_off_tbl[b] = band_off_tbl[b-1] + (size_t)rects[b-1].w * rects[b-1].h;
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        const subband_rect *r = &rects[b];
        float qbase = q * subband_baseline_flags(r, flags);
        for (uint32_t c = 0; c < nch; c++) {
            float step = qbase;
            if (do_ycocg) step *= channel_weight_ycocg(c);
            if (step < 1e-20f) step = 1e-20f;
            float inv_step = 1.0f / step;
            const float *plane = enc_plane(e, c);
            int32_t *qout = enc_coefs(e, c) + band_off_tbl[b];
            uint32_t *hist = enc_hist(e, b, c, nch);
            memset(hist, 0, sizeof(uint32_t) * C2D_ALPHABET);
            for (uint32_t y = 0; y < r->h; y++) {
                const float *row = plane + (size_t)(r->y + y) * C2D_TILE_SIDE + r->x;
                int32_t *qrow = qout + (size_t)y * r->w;
                for (uint32_t x = 0; x < r->w; x++) {
                    float v = row[x];
                    int32_t mag = (int32_t)(fabsf(v) * inv_step);
                    int32_t qi = (v >= 0.0f) ? mag : -mag;
                    qrow[x] = qi;
                    uint8_t sym; uint32_t esc;
                    coef_to_symbol(qi, &sym, &esc);
                    hist[sym]++;
                    if (sym == C2D_ESCAPE_SYMBOL) escape_bytes += leb128_size(esc);
                }
            }
        }
    }
    return escape_bytes;
}

// Shannon entropy estimate of the encoded bitstream size for current
// histograms in `e->hist`. Adds in escape bytes and a fixed per-subband
// header overhead (freq table + n_symbols + rans_size + final_states + esc_size).
static size_t estimate_bytes(c2d_encoder *e, uint32_t nch, size_t escape_bytes) {
    // Per-subband fixed overhead (R2-tight layout):
    //  1 (flags) + 1 (n_entries) + 3*n_nonzero (freq) + ~1-2 (varint rans_size)
    //  + RANS_STATE_BYTES + ~1-2 (varint esc_size when present)
    //  ~ 7 + 3*nnz bytes for non-all-zero bands.
    // Plus rANS body bytes ~= H * N / 8.
    double total_bits = 0.0;
    size_t total_overhead = 0;
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        for (uint32_t c = 0; c < nch; c++) {
            uint32_t *h = enc_hist(e, b, c, nch);
            uint64_t N = 0;
            uint32_t nnz = 0;
            for (uint32_t i = 0; i < C2D_ALPHABET; i++) { N += h[i]; if (h[i]) nnz++; }
            if (N == 0) { total_overhead += 1; continue; }   // all-zero band = 1 byte
            if (nnz == 1 && h[0] == N) { total_overhead += 1; continue; }  // all symbol 0 -> still 1 byte (all-zero flag will trigger)
            double bits = 0.0;
            for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
                if (!h[i]) continue;
                double p = (double)h[i] / (double)N;
                bits += -(double)h[i] * log2(p);
            }
            total_bits += bits;
            total_overhead += 7 + 3 * nnz + RANS_STATE_BYTES;
        }
    }
    size_t body = (size_t)((total_bits + 7.0) / 8.0);
    return HDR_TOTAL(nch) + total_overhead + body + escape_bytes;
}

// ===========================================================================
// Per-subband emit: write a single subband blob (freq table + n_symbols +
// rans body + escapes) into `out`. Uses encoder pooled scratch.
// ===========================================================================

// Per-subband bitstream layout (R2-tight):
//   flags:u8       bit0 = has_escapes, bit1 = all_zero
//   if !all_zero:
//     n_entries:u8 + entries(u8 sym, u16 freq) * n_entries
//     rans_size:varint
//     final_state:[RANS_STATE_BYTES]   (4 bytes for 1-way)
//     renorm:[rans_size]
//     if has_escapes:
//       esc_size:varint
//       escape:[esc_size]
//   if all_zero: just the flag byte (3 byte cost: flag + n_entries=1 + entry... no.
//   Actually all_zero means hist[0]=N -> freq[0]=M, single entry. We still need
//   the freq table so the decoder can run rANS or know the value. So all_zero
//   skips even the freq table; decoder fills band with zeros.

#define SB_FLAG_HAS_ESCAPES  0x01u
#define SB_FLAG_ALL_ZERO     0x02u

static size_t emit_subband(c2d_encoder *e,
                           uint32_t b, uint32_t c, uint32_t nch,
                           uint32_t w, uint32_t h,
                           const subband_rect *rects,
                           uint8_t *out, size_t out_cap) {
    size_t band_off = 0;
    for (uint32_t bb = 0; bb < b; bb++) band_off += (size_t)rects[bb].w * rects[bb].h;
    const int32_t *q = enc_coefs(e, c) + band_off;
    size_t n = (size_t)w * h;

    // Detect all-zero band from histogram.
    uint32_t *hist = enc_hist(e, b, c, nch);
    int all_zero = (hist[0] == n);
    if (all_zero) {
        if (out_cap < 1) return 0;
        out[0] = SB_FLAG_ALL_ZERO;
        return 1;
    }

    // Normalize freq table from precomputed histogram.
    uint16_t freq[C2D_ALPHABET];
    normalize_freqs(hist, RANS_M, freq);
    uint16_t cum[C2D_ALPHABET + 1];
    cum[0] = 0;
    for (uint32_t i = 0; i < C2D_ALPHABET; i++) cum[i+1] = cum[i] + freq[i];
    c2d_assert(cum[C2D_ALPHABET] == RANS_M);

    uint8_t n_entries = 0;
    for (uint32_t i = 0; i < C2D_ALPHABET; i++) if (freq[i]) n_entries++;

    // Write escapes forward, count their bytes.
    size_t escape_len = 0;
    int has_escapes = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t sym; uint32_t esc;
        coef_to_symbol(q[i], &sym, &esc);
        if (sym == C2D_ESCAPE_SYMBOL) {
            escape_len += leb128_write(esc, e->escape_buf + escape_len);
            has_escapes = 1;
        }
    }

    // rANS encode backward.
    size_t scratch_cap = C2D_PIXELS_PER_TILE * 8 + 1024;
    rans_enc enc;
    rans_enc_init(&enc, e->renorm_buf, scratch_cap);
    for (size_t i = n; i > 0; i--) {
        uint8_t sym; uint32_t esc;
        coef_to_symbol(q[i-1], &sym, &esc);
        (void)esc;
        rans_enc_put(&enc, (uint32_t)((i-1) & (RANS_NSTATES - 1)), freq[sym], cum[sym]);
    }
    uint8_t final_state[RANS_STATE_BYTES];
    rans_enc_finish(&enc, final_state);

    size_t pos = 0;
    uint8_t flags = has_escapes ? SB_FLAG_HAS_ESCAPES : 0;

    if (pos + 1 > out_cap) return 0;
    out[pos++] = flags;
    if (pos + 1 > out_cap) return 0;
    out[pos++] = n_entries;
    for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
        if (!freq[i]) continue;
        if (pos + 3 > out_cap) return 0;
        out[pos++] = (uint8_t)i;
        write_u16(out + pos, freq[i]); pos += 2;
    }
    // varint rans_size
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)enc.pos, out + pos);
    // final state (RANS_STATE_BYTES)
    if (pos + RANS_STATE_BYTES > out_cap) return 0;
    memcpy(out + pos, final_state, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    // renorm body
    if (pos + enc.pos > out_cap) return 0;
    memcpy(out + pos, e->renorm_buf + scratch_cap - enc.pos, enc.pos);
    pos += enc.pos;
    // escapes (only if has_escapes)
    if (has_escapes) {
        if (pos + 5 > out_cap) return 0;
        pos += leb128_write((uint32_t)escape_len, out + pos);
        if (pos + escape_len > out_cap) return 0;
        memcpy(out + pos, e->escape_buf, escape_len); pos += escape_len;
    }
    return pos;
}

static void decode_subband(c2d_decoder *d,
                           const uint8_t *in, size_t in_cap,
                           uint32_t w, uint32_t h, float qstep,
                           float *band) {
    size_t n = (size_t)w * h;
    size_t pos = 0;

    c2d_assert(pos + 1 <= in_cap);
    uint8_t flags = in[pos++];

    if (flags & SB_FLAG_ALL_ZERO) {
        // Decode as zeros (dequantize(0) = 0).
        for (size_t i = 0; i < n; i++) band[i] = 0.0f;
        return;
    }

    c2d_assert(pos + 1 <= in_cap);
    uint8_t n_entries = in[pos++];

    uint16_t freq[C2D_ALPHABET] = {0};
    for (uint32_t i = 0; i < n_entries; i++) {
        c2d_assert(pos + 3 <= in_cap);
        uint8_t sym = in[pos++];
        uint16_t f  = read_u16(in + pos); pos += 2;
        c2d_assert(sym < C2D_ALPHABET);
        freq[sym] = f;
    }

    uint32_t rans_block_size;
    pos += leb128_read(in + pos, in_cap - pos, &rans_block_size);

    c2d_assert(pos + RANS_STATE_BYTES <= in_cap);
    uint8_t final_state[RANS_STATE_BYTES];
    memcpy(final_state, in + pos, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;

    c2d_assert(pos + rans_block_size <= in_cap);
    const uint8_t *renorm = in + pos; pos += rans_block_size;

    uint32_t esc_len = 0;
    const uint8_t *esc_stream = NULL;
    if (flags & SB_FLAG_HAS_ESCAPES) {
        pos += leb128_read(in + pos, in_cap - pos, &esc_len);
        c2d_assert(pos + esc_len <= in_cap);
        esc_stream = in + pos;
    }

    // Build per-slot table: 1 load per symbol decode.
    {
        uint32_t c = 0;
        for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
            uint16_t f = freq[i];
            if (f == 0) continue;
            rans_slot s = { .sym = (uint16_t)i, .freq = f, .cum = (uint16_t)c, ._pad = 0 };
            for (uint32_t j = 0; j < f; j++) d->slot_tbl[c + j] = s;
            c += f;
        }
        c2d_assert(c == RANS_M);
    }

    // Inlined rANS decode loop, 4-way interleaved.
    uint32_t x[RANS_NSTATES];
    rans_states_load(x, final_state);
    const uint8_t *rp = renorm;
    const uint8_t *rp_end = renorm + rans_block_size;
    size_t esc_pos = 0;

    for (size_t i = 0; i < n; i++) {
        uint32_t lane = i & (RANS_NSTATES - 1);
        uint32_t slot = x[lane] & (RANS_M - 1);
        rans_slot s = d->slot_tbl[slot];
        x[lane] = (uint32_t)s.freq * (x[lane] >> RANS_PROB_BITS) + slot - (uint32_t)s.cum;
        while (x[lane] < RANS_L) {
            c2d_assert(rp < rp_end);
            x[lane] = (x[lane] << 8) | *rp++;
        }
        uint32_t esc_val = 0;
        if (s.sym == C2D_ESCAPE_SYMBOL) {
            esc_pos += leb128_read(esc_stream + esc_pos, esc_len - esc_pos, &esc_val);
        }
        int32_t qv = symbol_to_coef((uint8_t)s.sym, esc_val);
        band[i] = dequantize_one(qv, qstep);
    }
    (void)rp_end;
}

// ===========================================================================
// Max encoded size (very generous)
// ===========================================================================

size_t c2d_tile_encode_max_size(c2d_dtype dt, uint32_t n_channels) {
    (void)dt;
    c2d_assert(n_channels >= 1 && n_channels <= C2D_MAX_CHANNELS);
    size_t per_sb = 1 + 65*3 + 4 + 4 + 32 + C2D_PIXELS_PER_TILE*5 + 4 + C2D_PIXELS_PER_TILE*5 + 64;
    return HDR_TOTAL(n_channels) + (size_t)n_channels * C2D_N_SUBBANDS * per_sb;
}

// ===========================================================================
// Encode (context-based, with single-pass-via-entropy-estimate bisection)
// ===========================================================================

static size_t encoder_emit(c2d_encoder *e, c2d_dtype dt, uint32_t nch,
                           uint32_t flags, float q,
                           const subband_rect *rects,
                           uint8_t *out, size_t out_cap) {
    if (out_cap < HDR_TOTAL(nch)) return 0;

    // Emit per-(band, channel) blobs.
    size_t total = (size_t)nch * C2D_N_SUBBANDS;
    uint32_t *sb_off = (uint32_t *)calloc(total, sizeof(uint32_t));
    uint32_t *sb_len = (uint32_t *)calloc(total, sizeof(uint32_t));

    size_t pos = 0;
    memcpy(out + pos, C2D_TILE_MAGIC, 4); pos += 4;
    out[pos++] = (uint8_t)C2D_FORMAT_VERSION;
    out[pos++] = (uint8_t)dt;
    out[pos++] = (uint8_t)nch;
    out[pos++] = (uint8_t)(flags & 0xff);
    write_f32(out + pos, q); pos += 4;
    size_t lod_offs_pos = pos; pos += 4 * C2D_N_LODS;
    for (uint32_t c = 0; c < nch; c++) {
        write_f32(out + pos, e->dcs[c]);    pos += 4;
        write_f32(out + pos, e->scales[c]); pos += 4;
    }
    size_t dir_pos = pos; pos += HDR_DIR_SIZE(nch);

    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        const subband_rect *r = &rects[b];
        for (uint32_t c = 0; c < nch; c++) {
            size_t idx = (size_t)b * nch + c;
            size_t w = emit_subband(e, b, c, nch, r->w, r->h, rects,
                                    e->blob_buf, c2d_tile_encode_max_size(dt, nch));
            if (w == 0) { free(sb_off); free(sb_len); return 0; }
            if (pos + w > out_cap) { free(sb_off); free(sb_len); return 0; }
            sb_off[idx] = (uint32_t)pos;
            sb_len[idx] = (uint32_t)w;
            memcpy(out + pos, e->blob_buf, w);
            pos += w;
        }
    }
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        for (uint32_t c = 0; c < nch; c++) {
            size_t idx = (size_t)b * nch + c;
            uint8_t *p = out + dir_pos + idx * 8;
            write_u32(p,     sb_off[idx]);
            write_u32(p + 4, sb_len[idx]);
        }
    }
    static const uint32_t last_band_for_lod[C2D_N_LODS] = {15, 12, 9, 6, 3, 0};
    for (uint32_t lod = 0; lod < C2D_N_LODS; lod++) {
        uint32_t last_b = last_band_for_lod[lod];
        size_t last_idx = (size_t)last_b * nch + (nch - 1);
        uint32_t end = sb_off[last_idx] + sb_len[last_idx];
        write_u32(out + lod_offs_pos + 4*lod, end);
    }
    free(sb_off); free(sb_len);
    return pos;
}

size_t c2d_encoder_tile_encode_at_q(c2d_encoder *e, const void *in, c2d_dtype dt,
                                    uint32_t nch, float q, uint32_t flags,
                                    uint8_t *out, size_t out_cap) {
    c2d_assert(nch >= 1 && nch <= C2D_MAX_CHANNELS);
    encoder_grow(e, nch);
    int do_ycocg = (flags & C2D_FLAG_COLOR_YCOCG) && nch == 3;

    // Normalize + de-interleave into planes.
    for (uint32_t c = 0; c < nch; c++)
        compute_normalization_channel(in, dt, c, nch, &e->dcs[c], &e->scales[c]);
    for (uint32_t c = 0; c < nch; c++) {
        float dc = e->dcs[c], sc = e->scales[c];
        float *plane = enc_plane(e, c);
        for (size_t i = 0; i < C2D_PIXELS_PER_TILE; i++)
            plane[i] = (iload_f32(in, dt, i, c, nch) - dc) * sc;
    }
    if (do_ycocg) ycocg_fwd(enc_plane(e, 0), enc_plane(e, 1), enc_plane(e, 2),
                            C2D_PIXELS_PER_TILE);

    subband_rect rects[C2D_N_SUBBANDS];
    enumerate_subbands(rects);
    encode_transform_and_quantize(e, rects, nch, q, do_ycocg, flags);
    return encoder_emit(e, dt, nch, flags, q, rects, out, out_cap);
}

size_t c2d_encoder_tile_encode(c2d_encoder *e, const void *in, c2d_dtype dt,
                               uint32_t nch, float target_ratio, uint32_t flags,
                               uint8_t *out, size_t out_cap) {
    c2d_assert(nch >= 1 && nch <= C2D_MAX_CHANNELS);
    encoder_grow(e, nch);
    int do_ycocg = (flags & C2D_FLAG_COLOR_YCOCG) && nch == 3;
    size_t raw_bytes = c2d_dtype_tile_bytes_n(dt, nch);
    size_t target_bytes = (size_t)((double)raw_bytes / (double)target_ratio);
    size_t header_floor = HDR_TOTAL(nch) + 64;
    if (target_bytes < header_floor) target_bytes = header_floor;

    // 1) Normalize + de-interleave + DWT (once).
    for (uint32_t c = 0; c < nch; c++)
        compute_normalization_channel(in, dt, c, nch, &e->dcs[c], &e->scales[c]);
    for (uint32_t c = 0; c < nch; c++) {
        float dc = e->dcs[c], sc = e->scales[c];
        float *plane = enc_plane(e, c);
        for (size_t i = 0; i < C2D_PIXELS_PER_TILE; i++)
            plane[i] = (iload_f32(in, dt, i, c, nch) - dc) * sc;
    }
    if (do_ycocg) ycocg_fwd(enc_plane(e, 0), enc_plane(e, 1), enc_plane(e, 2),
                            C2D_PIXELS_PER_TILE);

    subband_rect rects[C2D_N_SUBBANDS];
    enumerate_subbands(rects);
    for (uint32_t c = 0; c < nch; c++)
        dwt_fwd(enc_plane(e, c), C2D_TILE_SIDE, e->dwt_tmp, e->mallat);

    // 2) Bisect q via cheap entropy estimate. Use histogram-only variant to
    //    skip int32 coef_pool writes; only the final quantize writes coefs.
    double lo = log(1.0 / 4096.0);
    double hi = log(4096.0);
    double best_mid = 0.5 * (lo + hi);
    int got = 0;
    for (int iter = 0; iter < 8; iter++) {
        double mid = 0.5 * (lo + hi);
        float q = (float)exp(mid);
        size_t esc_bytes = requantize_hist_only(e, rects, nch, q, do_ycocg, flags);
        size_t est = estimate_bytes(e, nch, esc_bytes);
        if (est > target_bytes) lo = mid;
        else { hi = mid; best_mid = mid; got = 1; }
        double rel = fabs((double)est - (double)target_bytes) / (double)target_bytes;
        if (rel < 0.02) { best_mid = mid; got = 1; break; }
    }
    float final_q = (float)exp(got ? best_mid : hi);

    // 3) Final quantize + emit (real rANS, once).
    requantize_only(e, rects, nch, final_q, do_ycocg, flags);
    return encoder_emit(e, dt, nch, flags, final_q, rects, out, out_cap);
}

// Stateless convenience wrappers (allocate a temporary context).
size_t c2d_tile_encode(const void *in, c2d_dtype dt, uint32_t nch,
                       float target_ratio, uint32_t flags,
                       uint8_t *out, size_t out_cap) {
    c2d_encoder *e = c2d_encoder_new();
    size_t r = c2d_encoder_tile_encode(e, in, dt, nch, target_ratio, flags, out, out_cap);
    c2d_encoder_free(e);
    return r;
}
size_t c2d_tile_encode_at_q(const void *in, c2d_dtype dt, uint32_t nch,
                            float q, uint32_t flags,
                            uint8_t *out, size_t out_cap) {
    c2d_encoder *e = c2d_encoder_new();
    size_t r = c2d_encoder_tile_encode_at_q(e, in, dt, nch, q, flags, out, out_cap);
    c2d_encoder_free(e);
    return r;
}

// ===========================================================================
// Inspect / decode
// ===========================================================================

bool c2d_is_tile(const uint8_t *in, size_t n) {
    return n >= 4 && memcmp(in, C2D_TILE_MAGIC, 4) == 0;
}
bool c2d_tile_validate(const uint8_t *in, size_t n) {
    if (!c2d_is_tile(in, n)) return false;
    if (n < HDR_FIXED_SIZE) return false;
    if (in[4] != C2D_FORMAT_VERSION) return false;
    if (in[5] >= C2D_DTYPE__COUNT) return false;
    uint32_t nch = in[6];
    if (nch < 1 || nch > C2D_MAX_CHANNELS) return false;
    if (n < HDR_TOTAL(nch)) return false;
    return true;
}
void c2d_tile_inspect(const uint8_t *in, size_t n, c2d_tile_info *info) {
    c2d_assert(c2d_tile_validate(in, n));
    info->dtype      = (c2d_dtype)in[5];
    info->n_channels = in[6];
    info->flags      = in[7];
    size_t lod_offs_pos = 4 + 1 + 1 + 1 + 1 + 4;
    for (uint32_t i = 0; i < C2D_N_LODS; i++)
        info->lod_offsets[i] = read_u32(in + lod_offs_pos + 4*i);
}

void c2d_decoder_tile_decode_lod(c2d_decoder *d, const uint8_t *in, size_t in_len,
                                 uint8_t requested_lod, void *out) {
    c2d_assert(c2d_tile_validate(in, in_len));
    c2d_assert(requested_lod < C2D_N_LODS);
    c2d_dtype dt    = (c2d_dtype)in[5];
    uint32_t  nch   = in[6];
    uint32_t  flags = in[7];
    float     q     = read_f32(in + 8);
    int do_ycocg = (flags & C2D_FLAG_COLOR_YCOCG) && nch == 3;

    decoder_grow(d, nch);

    size_t lod_offs_pos = 4 + 1 + 1 + 1 + 1 + 4;
    size_t norm_pos     = lod_offs_pos + 4 * C2D_N_LODS;
    size_t dir_pos      = norm_pos + 8 * (size_t)nch;

    for (uint32_t c = 0; c < nch; c++) {
        d->dcs[c]    = read_f32(in + norm_pos + 8*c);
        d->scales[c] = read_f32(in + norm_pos + 8*c + 4);
    }

    subband_rect rects[C2D_N_SUBBANDS];
    enumerate_subbands(rects);

    // Determine working side for this LOD. For LOD k the reconstructed plane
    // is `side x side` with side = 256 >> k. The stored coefficients we need
    // also fit in the top-left side x side region (only coarsest bands).
    uint32_t out_side = c2d_side_per_lod(requested_lod);
    uint32_t levels_to_apply = C2D_N_DWT_LEVELS - requested_lod;

    // Zero working planes within the side x side region.
    for (uint32_t c = 0; c < nch; c++) {
        float *plane = dec_plane(d, c);
        for (uint32_t y = 0; y < out_side; y++)
            memset(plane + y * C2D_TILE_SIDE, 0, sizeof(float) * out_side);
    }

    static const uint32_t last_band_for_lod[C2D_N_LODS] = {15, 12, 9, 6, 3, 0};
    uint32_t last_b = last_band_for_lod[requested_lod];

    // Decode coefficients into working planes (rect coords are in 256-coord
    // space; for LOD > 0 they fit inside side x side since coarse bands
    // occupy the top-left).
    for (uint32_t b = 0; b <= last_b; b++) {
        const subband_rect *r = &rects[b];
        for (uint32_t c = 0; c < nch; c++) {
            size_t idx = (size_t)b * nch + c;
            uint32_t off = read_u32(in + dir_pos + idx * 8);
            uint32_t len = read_u32(in + dir_pos + idx * 8 + 4);
            float qstep = q * subband_baseline_flags(r, flags);
            if (do_ycocg) qstep *= channel_weight_ycocg(c);
            if (qstep < 1e-20f) qstep = 1e-20f;
            decode_subband(d, in + off, len, r->w, r->h, qstep, d->band_buf);
            float *plane = dec_plane(d, c);
            for (uint32_t y = 0; y < r->h; y++) {
                memcpy(plane + (size_t)(r->y + y) * C2D_TILE_SIDE + r->x,
                       d->band_buf + (size_t)y * r->w,
                       sizeof(float) * r->w);
            }
        }
    }

    // Inverse DWT on side x side region (stride still 256).
    for (uint32_t c = 0; c < nch; c++)
        dwt_inv_partial(dec_plane(d, c), C2D_TILE_SIDE, levels_to_apply,
                        d->dwt_tmp, d->mallat);

    // Inverse color transform within reconstructed region.
    if (do_ycocg) {
        for (uint32_t y = 0; y < out_side; y++) {
            ycocg_inv(dec_plane(d, 0) + y * C2D_TILE_SIDE,
                      dec_plane(d, 1) + y * C2D_TILE_SIDE,
                      dec_plane(d, 2) + y * C2D_TILE_SIDE,
                      out_side);
        }
    }

    // Re-interleave + denormalize. Precompute per-channel inverse-scale so
    // the inner loop has no division.
    float inv_scale[C2D_MAX_CHANNELS];
    for (uint32_t c = 0; c < nch; c++) inv_scale[c] = 1.0f / d->scales[c];

    if (dt == C2D_DTYPE_U8) {
        // Fast path: u8 with clamp-and-round, no per-pixel switch.
        uint8_t *out8 = (uint8_t *)out;
        for (uint32_t y = 0; y < out_side; y++) {
            for (uint32_t c = 0; c < nch; c++) {
                const float *row = dec_plane(d, c) + y * C2D_TILE_SIDE;
                float is = inv_scale[c], dc = d->dcs[c];
                uint8_t *o = out8 + (size_t)y * out_side * nch + c;
                for (uint32_t x = 0; x < out_side; x++) {
                    float v = row[x] * is + dc + 0.5f;
                    int iv = (int)v;
                    if (v < 0) iv = 0;
                    if (iv > 255) iv = 255;
                    o[x * nch] = (uint8_t)iv;
                }
            }
        }
    } else {
        for (uint32_t y = 0; y < out_side; y++) {
            for (uint32_t x = 0; x < out_side; x++) {
                size_t i_out = (size_t)y * out_side + x;
                for (uint32_t c = 0; c < nch; c++) {
                    float v = dec_plane(d, c)[y * C2D_TILE_SIDE + x] * inv_scale[c] + d->dcs[c];
                    istore_f32(out, dt, i_out, c, nch, v);
                }
            }
        }
    }
}

void c2d_decoder_tile_decode(c2d_decoder *d, const uint8_t *in, size_t in_len, void *out) {
    c2d_decoder_tile_decode_lod(d, in, in_len, 0, out);
}
void c2d_tile_decode(const uint8_t *in, size_t in_len, void *out) {
    c2d_decoder *d = c2d_decoder_new();
    c2d_decoder_tile_decode(d, in, in_len, out);
    c2d_decoder_free(d);
}
void c2d_tile_decode_lod(const uint8_t *in, size_t in_len, uint8_t lod, void *out) {
    c2d_decoder *d = c2d_decoder_new();
    c2d_decoder_tile_decode_lod(d, in, in_len, lod, out);
    c2d_decoder_free(d);
}

// ===========================================================================
// Image container (C2DI): WxH array of pixels split into 256x256 tiles with
// shared per-channel normalization and shared per-(band, channel) frequency
// tables. Per-tile bodies omit the freq tables entirely, saving ~9 KB per
// tile on multi-tile images.
//
// Image header layout:
//   "C2DI" 4 + version 1 + dtype 1 + nch 1 + flags 1
//   width u32 + height u32
//   tile_side u32 (currently always 256)
//   global_q f32
//   per-channel norm: (dc f32, scale f32) * nch
//   per-(band, channel) shared freq tables:
//     for b in 0..16, c in 0..nch:
//       n_entries u8 + (sym u8, freq u16) * n_entries
//   n_tiles_x u32, n_tiles_y u32
//   tile directory: (offset u32, length u32) * n_tiles
//   tile bodies:
//     per-tile per-(band, channel) directory: (offset u32, length u32) * 16 * nch
//     subband bodies (no freq table):
//       flags u8 (bit0 has_escapes, bit1 all_zero)
//       if !all_zero:
//         rans_size varint
//         final_state [RANS_STATE_BYTES]
//         renorm [rans_size]
//         if has_escapes: esc_size varint + esc_stream
// ===========================================================================

// Image header: magic(4) + ver(1) + dtype(1) + nch(1) + flags(2) + reserved(1)
//              + w(4) + h(4) + tile_side(4) + q(4) = 26
// flags is u16 starting at offset 7 (v2). For v1 compat, only bits 0-7 mattered.
#define IMG_HDR_FIXED  (4 + 1 + 1 + 1 + 2 + 1 + 4 + 4 + 4 + 4)   // 26

static size_t img_norm_size(uint32_t nch)  { return 8u * nch; }

// Worst-case freq table size per (b,c): 1 (n_entries) + 65*3 = 196 B.
#define IMG_FREQ_MAX_PER_BC  196u

static size_t img_freq_max_size(uint32_t nch) {
    return (size_t)C2D_N_SUBBANDS * nch * IMG_FREQ_MAX_PER_BC;
}

size_t c2d_image_encode_max_size(uint32_t width, uint32_t height,
                                 c2d_dtype dt, uint32_t nch) {
    (void)dt;
    c2d_assert(nch >= 1 && nch <= C2D_MAX_CHANNELS);
    uint32_t ntx = (width  + C2D_TILE_SIDE - 1) / C2D_TILE_SIDE;
    uint32_t nty = (height + C2D_TILE_SIDE - 1) / C2D_TILE_SIDE;
    size_t n_tiles = (size_t)ntx * nty;
    size_t per_sb_worst = 1 + 5 + RANS_STATE_BYTES + C2D_PIXELS_PER_TILE * 5 + 5 + C2D_PIXELS_PER_TILE * 5;
    size_t per_tile_max = (size_t)C2D_N_SUBBANDS * nch * 8u + per_sb_worst * C2D_N_SUBBANDS * nch;
    return IMG_HDR_FIXED + img_norm_size(nch) + img_freq_max_size(nch)
           + 8 + n_tiles + n_tiles + n_tiles * 8 + n_tiles * per_tile_max + 4096;
}

bool c2d_is_image(const uint8_t *in, size_t n) {
    return n >= 4 && memcmp(in, C2D_IMAGE_MAGIC, 4) == 0;
}
bool c2d_image_validate(const uint8_t *in, size_t n) {
    if (!c2d_is_image(in, n)) return false;
    if (n < IMG_HDR_FIXED) return false;
    if (in[4] != C2D_FORMAT_VERSION) return false;
    if (in[5] >= C2D_DTYPE__COUNT) return false;
    if (in[6] < 1 || in[6] > C2D_MAX_CHANNELS) return false;
    return true;
}
void c2d_image_inspect(const uint8_t *in, size_t n, c2d_image_info *info) {
    c2d_assert(c2d_image_validate(in, n));
    info->dtype      = (c2d_dtype)in[5];
    info->n_channels = in[6];
    info->flags      = read_u16(in + 7);
    info->width      = read_u32(in + 10);
    info->height     = read_u32(in + 14);
}

// --- Encode helpers ---

// Read one tile out of the interleaved (w,h,nch) source, replicating edges.
static void gather_tile(const void *src, c2d_dtype dt, uint32_t w, uint32_t h,
                        uint32_t nch, uint32_t tx, uint32_t ty,
                        void *tile_buf) {
    size_t es = c2d_dtype_size(dt);
    int tw = (tx + C2D_TILE_SIDE > w) ? (int)(w - tx) : (int)C2D_TILE_SIDE;
    int th = (ty + C2D_TILE_SIDE > h) ? (int)(h - ty) : (int)C2D_TILE_SIDE;
    for (int y = 0; y < (int)C2D_TILE_SIDE; y++) {
        int sy = y < th ? y : th - 1;
        for (int x = 0; x < (int)C2D_TILE_SIDE; x++) {
            int sx = x < tw ? x : tw - 1;
            const uint8_t *sp = (const uint8_t *)src
                              + (size_t)(((ty + sy) * (size_t)w + (tx + sx)) * nch) * es;
            uint8_t *dp = (uint8_t *)tile_buf
                        + (size_t)((y * (int)C2D_TILE_SIDE + x) * nch) * es;
            memcpy(dp, sp, es * nch);
        }
    }
}

// Scatter a decoded tile back into the full image (cropping pad region).
static void scatter_tile(void *dst, c2d_dtype dt, uint32_t w, uint32_t h,
                         uint32_t nch, uint32_t tx, uint32_t ty,
                         const void *tile_buf) {
    size_t es = c2d_dtype_size(dt);
    int tw = (tx + C2D_TILE_SIDE > w) ? (int)(w - tx) : (int)C2D_TILE_SIDE;
    int th = (ty + C2D_TILE_SIDE > h) ? (int)(h - ty) : (int)C2D_TILE_SIDE;
    for (int y = 0; y < th; y++) {
        const uint8_t *sp = (const uint8_t *)tile_buf
                          + (size_t)((y * (int)C2D_TILE_SIDE) * nch) * es;
        uint8_t *dp = (uint8_t *)dst
                    + (size_t)((((ty + y) * (size_t)w + tx)) * nch) * es;
        memcpy(dp, sp, (size_t)tw * nch * es);
    }
}

// Compute per-channel normalization across the entire image (one global
// dc/scale per channel that all tiles share).
static void image_normalization(const void *src, c2d_dtype dt,
                                uint32_t w, uint32_t h, uint32_t nch,
                                float *out_dc, float *out_scale) {
    size_t total = (size_t)w * h;
    for (uint32_t c = 0; c < nch; c++) {
        double sum = 0.0;
        for (size_t i = 0; i < total; i++) sum += (double)iload_f32(src, dt, i, c, nch);
        float dc = (float)(sum / (double)total);
        float maxmag = 0.0f;
        for (size_t i = 0; i < total; i++) {
            float v = iload_f32(src, dt, i, c, nch) - dc;
            float a = fabsf(v);
            if (a > maxmag) maxmag = a;
        }
        out_dc[c]    = dc;
        out_scale[c] = (maxmag < 1e-20f) ? 1.0f : (128.0f / maxmag);
    }
}

// Image-context: precomputed per-tile DWT-transformed planes (kept around for
// the rate-control bisection so we don't re-DWT each iteration).
typedef struct img_tile_state {
    float *planes;     // nch * 256*256 f32, post-DWT (Mallat layout)
    uint8_t qmul_idx;  // per-tile q multiplier index for C2D_FLAG_LOW_BPP (else 128)
} img_tile_state;

// Map qmul_idx in [0,255] to multiplier in approx [0.25, 4.0] via
// mul = 2^((idx-128)/64). idx=128 -> 1.0 (neutral). Step ~1.011×.
static inline float qmul_decode(uint8_t idx) {
    return powf(2.0f, ((float)(int)idx - 128.0f) / 64.0f);
}
static inline uint8_t qmul_encode(float mul) {
    if (mul <= 0.0f) return 128;
    float v = 128.0f + 64.0f * log2f(mul);
    if (v < 0.0f) v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return (uint8_t)(v + 0.5f);
}

// Sigma filter (edge-preserving smoother) for a single 256x256 plane.
// Each output pixel is the average of those 3x3 neighbors within `sigma`
// of the center value. `sigma` controls smoothing strength (in plane units,
// which post-normalization are roughly [-1, +1]).
// strength_idx 0 = no-op, 1..15 = increasing sigma.
static void sigma_filter_plane(float *plane, int side, uint8_t strength_idx) {
    if (strength_idx == 0) return;
    // The plane is in post-DWT-inverse, pre-denormalize space. Values are
    // roughly in [-128, +128] (u8 mean-removed). Tuning sigma in this space.
    // strength 1..15 -> sigma 1.0 .. 30.0
    float sigma = 1.0f + 2.0f * (float)(int)(strength_idx - 1);
    float sigma2 = sigma * sigma * 4.0f;  // 2-sigma envelope squared
    int stride = side;
    // Need a scratch copy because the filter reads original neighbors.
    static __thread float *scratch = NULL;
    static __thread int scratch_side = 0;
    if (!scratch || scratch_side != side) {
        free(scratch);
        scratch = (float *)malloc(sizeof(float) * (size_t)side * side);
        scratch_side = side;
    }
    memcpy(scratch, plane, sizeof(float) * (size_t)side * side);
    for (int y = 0; y < side; y++) {
        int y0 = (y > 0) ? -1 : 0;
        int y1 = (y < side - 1) ? 1 : 0;
        for (int x = 0; x < side; x++) {
            int x0 = (x > 0) ? -1 : 0;
            int x1 = (x < side - 1) ? 1 : 0;
            float center = scratch[y * stride + x];
            float sum = center;
            float wsum = 1.0f;
            for (int dy = y0; dy <= y1; dy++) {
                for (int dx = x0; dx <= x1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    float v = scratch[(y + dy) * stride + (x + dx)];
                    float d = v - center;
                    if (d * d <= sigma2) {
                        sum += v;
                        wsum += 1.0f;
                    }
                }
            }
            plane[y * stride + x] = sum / wsum;
        }
    }
}

// Quantize all tiles at given q; accumulate global histograms in `g_hist`
// (size 16 * nch * 65) and return total escape bytes if rendered now.
// Like image_requantize_collect but skips the coef_pool write (just builds
// the histogram + counts escape bytes). Used during the bisection where we
// only need the entropy estimate; the real coef_pool gets written once at
// the chosen final_q.
static size_t image_hist_only(img_tile_state *tiles, size_t n_tiles,
                              uint32_t nch, int do_ycocg,
                              float q, const subband_rect *rects,
                              uint32_t *g_hist, uint32_t flags) {
    size_t escape_bytes = 0;
    size_t hist_total = (size_t)C2D_N_SUBBANDS * nch * C2D_ALPHABET;
    memset(g_hist, 0, sizeof(uint32_t) * hist_total);
    #pragma omp parallel
    {
        uint32_t *local_hist = (uint32_t *)calloc(hist_total, sizeof(uint32_t));
        size_t local_escape = 0;
        #pragma omp for schedule(static) nowait
        for (size_t t = 0; t < n_tiles; t++) {
            float tile_qmul = (flags & C2D_FLAG_LOW_BPP) ? qmul_decode(tiles[t].qmul_idx) : 1.0f;
            for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
                const subband_rect *r = &rects[b];
                float qbase = q * tile_qmul * subband_baseline_flags(r, flags);
                for (uint32_t c = 0; c < nch; c++) {
                    float step = qbase;
                    if (do_ycocg) step *= channel_weight_ycocg(c);
                    if (step < 1e-20f) step = 1e-20f;
                    float inv_step = 1.0f / step;
                    const float *plane = tiles[t].planes + (size_t)c * C2D_PIXELS_PER_TILE;
                    uint32_t *gh = local_hist + ((size_t)b * nch + c) * C2D_ALPHABET;
                    for (uint32_t y = 0; y < r->h; y++) {
                        const float *row = plane + (size_t)(r->y + y) * C2D_TILE_SIDE + r->x;
                        for (uint32_t x = 0; x < r->w; x++) {
                            float v = row[x];
                            int32_t mag = (int32_t)(fabsf(v) * inv_step);
                            int32_t qi = (v >= 0.0f) ? mag : -mag;
                            uint32_t z = ((uint32_t)qi << 1) ^ (uint32_t)(qi >> 31);
                            if (z < 64) {
                                gh[z]++;
                            } else {
                                gh[C2D_ESCAPE_SYMBOL]++;
                                local_escape += leb128_size(z - 63);
                            }
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            for (size_t i = 0; i < hist_total; i++) g_hist[i] += local_hist[i];
            escape_bytes += local_escape;
        }
        free(local_hist);
    }
    return escape_bytes;
}

static size_t image_requantize_collect(img_tile_state *tiles, size_t n_tiles,
                                       uint32_t nch, int do_ycocg,
                                       float q, const subband_rect *rects,
                                       int32_t *coef_pool,    // nch * n_tiles * 65536 ints
                                       uint32_t *g_hist, uint32_t flags) {
    size_t escape_bytes = 0;
    memset(g_hist, 0, sizeof(uint32_t) * C2D_N_SUBBANDS * nch * C2D_ALPHABET);
    size_t band_off_tbl[C2D_N_SUBBANDS];
    band_off_tbl[0] = 0;
    for (uint32_t b = 1; b < C2D_N_SUBBANDS; b++)
        band_off_tbl[b] = band_off_tbl[b-1] + (size_t)rects[b-1].w * rects[b-1].h;
    // Per-thread local histograms + reduce. Coef writes are independent
    // per tile so no races there.
    size_t hist_total = (size_t)C2D_N_SUBBANDS * nch * C2D_ALPHABET;
    #pragma omp parallel
    {
        uint32_t *local_hist = (uint32_t *)calloc(hist_total, sizeof(uint32_t));
        size_t local_escape = 0;
        #pragma omp for schedule(static) nowait
        for (size_t t = 0; t < n_tiles; t++) {
            float tile_qmul = (flags & C2D_FLAG_LOW_BPP) ? qmul_decode(tiles[t].qmul_idx) : 1.0f;
            for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
                const subband_rect *r = &rects[b];
                float qbase = q * tile_qmul * subband_baseline_flags(r, flags);
                for (uint32_t c = 0; c < nch; c++) {
                    float step = qbase;
                    if (do_ycocg) step *= channel_weight_ycocg(c);
                    if (step < 1e-20f) step = 1e-20f;
                    float inv_step = 1.0f / step;
                    const float *plane = tiles[t].planes + (size_t)c * C2D_PIXELS_PER_TILE;
                    int32_t *qout = coef_pool + ((size_t)t * nch + c) * C2D_PIXELS_PER_TILE + band_off_tbl[b];
                    uint32_t *gh = local_hist + ((size_t)b * nch + c) * C2D_ALPHABET;
                    for (uint32_t y = 0; y < r->h; y++) {
                        const float *row = plane + (size_t)(r->y + y) * C2D_TILE_SIDE + r->x;
                        int32_t *qrow = qout + (size_t)y * r->w;
                        for (uint32_t x = 0; x < r->w; x++) {
                            float v = row[x];
                            int32_t mag = (int32_t)(fabsf(v) * inv_step);
                            int32_t qi = (v >= 0.0f) ? mag : -mag;
                            qrow[x] = qi;
                            uint32_t z = ((uint32_t)qi << 1) ^ (uint32_t)(qi >> 31);
                            if (z < 64) {
                                gh[z]++;
                            } else {
                                gh[C2D_ESCAPE_SYMBOL]++;
                                local_escape += leb128_size(z - 63);
                            }
                        }
                    }
                }
            }
        }
        // Reduce local_hist into shared g_hist.
        #pragma omp critical
        {
            for (size_t i = 0; i < hist_total; i++) g_hist[i] += local_hist[i];
            escape_bytes += local_escape;
        }
        free(local_hist);
    }
    return escape_bytes;
}

// Shannon-entropy estimate using *global* per-(band, channel) histograms.
// Per-subband overhead is tiny (no freq table, just ~2 bytes flags+state).
static size_t image_estimate_bytes(const uint32_t *g_hist, uint32_t nch,
                                   size_t n_tiles, size_t escape_bytes,
                                   size_t header_overhead) {
    double total_bits = 0.0;
    int n_present = 0;
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        for (uint32_t c = 0; c < nch; c++) {
            const uint32_t *h = g_hist + ((size_t)b * nch + c) * C2D_ALPHABET;
            uint64_t N = 0;
            for (uint32_t i = 0; i < C2D_ALPHABET; i++) N += h[i];
            if (N == 0) continue;
            int empty = 1;
            for (uint32_t i = 1; i < C2D_ALPHABET; i++) if (h[i]) { empty = 0; break; }
            if (empty) continue;
            n_present++;
            for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
                if (!h[i]) continue;
                double p = (double)h[i] / (double)N;
                total_bits += -(double)h[i] * log2(p);
            }
        }
    }
    // Per-tile per-present-(b,c) overhead: 8 dir + 1 flag + RANS_STATE_BYTES
    // + ~2 varint rans_size.
    size_t per_sb_tile_overhead = 8 + 1 + RANS_STATE_BYTES + 2;
    size_t fixed = header_overhead + n_tiles * ((size_t)n_present * per_sb_tile_overhead);
    size_t body = (size_t)((total_bits + 7.0) / 8.0);
    return fixed + body + escape_bytes;
}

// Emit one subband within an image tile body (no freq table; freq is global).
static size_t image_emit_subband(const int32_t *q, size_t n,
                                 const uint16_t *freq, const uint16_t *cum,
                                 uint8_t *renorm_scratch, size_t renorm_cap,
                                 uint8_t *escape_scratch, size_t escape_cap,
                                 uint8_t *out, size_t out_cap) {
    int has_escapes = 0;
    size_t escape_len = 0;
    // Check if all coefficients are zero by scanning q (could also be passed in
    // but simpler this way).
    int all_zero = 1;
    for (size_t i = 0; i < n; i++) if (q[i] != 0) { all_zero = 0; break; }
    if (all_zero) {
        if (out_cap < 1) return 0;
        out[0] = SB_FLAG_ALL_ZERO;
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        uint8_t sym; uint32_t esc;
        coef_to_symbol(q[i], &sym, &esc);
        if (sym == C2D_ESCAPE_SYMBOL) {
            if (escape_len + 5 > escape_cap) return 0;
            escape_len += leb128_write(esc, escape_scratch + escape_len);
            has_escapes = 1;
        }
    }
    rans_enc enc;
    rans_enc_init(&enc, renorm_scratch, renorm_cap);
    for (size_t i = n; i > 0; i--) {
        uint8_t sym; uint32_t esc;
        coef_to_symbol(q[i-1], &sym, &esc);
        (void)esc;
        rans_enc_put(&enc, (uint32_t)((i-1) & (RANS_NSTATES - 1)), freq[sym], cum[sym]);
    }
    uint8_t final_state[RANS_STATE_BYTES];
    rans_enc_finish(&enc, final_state);

    size_t pos = 0;
    if (pos + 1 > out_cap) return 0;
    out[pos++] = has_escapes ? SB_FLAG_HAS_ESCAPES : 0;
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)enc.pos, out + pos);
    if (pos + RANS_STATE_BYTES > out_cap) return 0;
    memcpy(out + pos, final_state, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    if (pos + enc.pos > out_cap) return 0;
    memcpy(out + pos, renorm_scratch + renorm_cap - enc.pos, enc.pos);
    pos += enc.pos;
    if (has_escapes) {
        if (pos + 5 > out_cap) return 0;
        pos += leb128_write((uint32_t)escape_len, out + pos);
        if (pos + escape_len > out_cap) return 0;
        memcpy(out + pos, escape_scratch, escape_len);
        pos += escape_len;
    }
    return pos;
}

// ---------------------------------------------------------------------------
// Context-split coding: per (b, c) emit two rANS streams.
//   sig stream:  one symbol per coef in {0=zero, 1=nonzero}, 2-symbol alphabet.
//   mag stream:  one symbol per *nonzero* coef in the magnitude+sign alphabet
//                (same 65-symbol zigzag+escape as unified, but with sym=0 never
//                appearing — so the distribution is much more peaked).
//
// Why this helps: at low bpp the zero-probability dominates the unified
// distribution; the magnitude/sign part has its precision starved by the
// quantizer's 12-bit total. Splitting gives each stream full 12-bit precision.
// ---------------------------------------------------------------------------

// Encode an integer coef into the magnitude+sign symbol (caller has already
// verified qv != 0). Returns symbol; *out_esc is meaningful iff symbol == 64.
// Magnitude alphabet for nonzero coefs: zigzag-shifted by 1.
//   sym 0..62 = direct (adj 0..62)
//   sym 64    = escape (adj >= 63; esc payload = adj - 63)
// Symbol 63 is unused (to leave a slot for escape consistent with the
// unified alphabet layout).
static inline void mag_coef_to_symbol(int32_t qv, uint8_t *out_sym, uint32_t *out_esc) {
    uint32_t z = zigzag_enc(qv);
    c2d_assert(z >= 1);
    uint32_t adj = z - 1;
    if (adj < 63) { *out_sym = (uint8_t)adj; *out_esc = 0; }
    else          { *out_sym = (uint8_t)C2D_ESCAPE_SYMBOL; *out_esc = adj - 63; }
}
static inline int32_t mag_symbol_to_coef(uint8_t sym, uint32_t esc) {
    uint32_t adj = (sym < 63) ? (uint32_t)sym : (esc + 63);
    uint32_t z = adj + 1;
    return zigzag_dec(z);
}

// Build sig (2-symbol) histogram and mag (65-symbol) histogram from one
// per-(b,c) global histogram of the unified alphabet.
static void split_hists_from_unified(const uint32_t *g_hist,
                                     uint32_t *out_sig, uint32_t *out_mag) {
    // sig[0] = count of zeros = g_hist[0]
    // sig[1] = count of nonzeros = sum(g_hist[1..])
    // mag[i] = symbol distribution over nonzero coefs, *remapped*:
    //   for k in 1..63: mag_sym = k-1  (since sig already took out the zero)
    //   for k == 64 (escape): mag_sym = 63 + something? We keep escape at 64.
    //   Actually: mag_coef_to_symbol(qv) with qv != 0 produces:
    //     adj = zigzag(qv) - 1 in [0, ...]; adj < 64 -> sym = adj; else escape.
    //   So mag[i] = g_hist[zigzag(qv) where adj=i, ie zigzag=i+1].
    //   zigzag=i+1 corresponds to the unified-alphabet symbol value (i+1) for
    //   i+1 in [1..63], plus escape (unified sym 64) for adj >= 64.
    out_sig[0] = g_hist[0];
    uint64_t nz = 0;
    for (uint32_t i = 1; i < C2D_ALPHABET; i++) nz += g_hist[i];
    out_sig[1] = (uint32_t)nz;

    // Remap unified alphabet → magnitude alphabet:
    //   unified sym k in [1..62] → mag sym k-1
    //   unified sym 63 → mag escape with adj=62  -- but adj=62 < 63 is direct,
    //     so we need to treat unified sym 63 as: zigzag=63 -> adj=62 -> mag sym 62
    //     wait that's already in the loop below. The issue is unified sym 63
    //     corresponds to adj=62 in mag, NOT escape.
    //   unified sym 64 (escape, magnitude stored in esc) → mag escape
    for (uint32_t i = 0; i < C2D_ALPHABET; i++) out_mag[i] = 0;
    for (uint32_t k = 1; k < 64; k++) out_mag[k - 1] = g_hist[k];
    // ^^ k=1..63 -> mag idx 0..62 (direct values)
    out_mag[C2D_ESCAPE_SYMBOL] = g_hist[C2D_ESCAPE_SYMBOL];
    // mag idx 63 is intentionally unused.
}

// Per-(b,c) sig stream uses 2-symbol alphabet but we still use the full
// 12-bit precision rANS.  We treat it as a 2-entry freq table.
typedef struct sig_freq_pair { uint16_t f0; uint16_t f1; } sig_freq_pair;

// Number of contexts for neighbor-based sig coding:
// 4 already-decoded neighbors (W, NW, N, NE) each contribute 1 bit = 16
// contexts. Adding parent significance as a 5th bit (32 contexts) was tried
// and slightly regressed (extra header overhead outweighed entropy gain on
// natural images; parent-magnitude would help more than parent-sig but adds
// significant complexity).
#define SIG_CTX_COUNT 16u

// Number of contexts for neighbor-based magnitude coding (when CTX_MAG):
// sum of 4 neighbor magnitudes bucketed into {0, 1-2, 3-6, 7+} = 4 contexts.
#define MAG_CTX_COUNT 4u

// Compute the magnitude context for coef at (y,x) given already-decoded
// quantized magnitudes in the bitmap `mag_bits` (each byte = |q|, clipped to
// 255).
static inline uint32_t mag_context(const uint8_t *mag_bits,
                                   uint32_t x, uint32_t y, uint32_t w) {
    uint32_t W  = (x > 0)              ? mag_bits[y * w + (x - 1)] : 0;
    uint32_t NW = (x > 0 && y > 0)     ? mag_bits[(y - 1) * w + (x - 1)] : 0;
    uint32_t N  = (y > 0)              ? mag_bits[(y - 1) * w + x] : 0;
    uint32_t NE = (y > 0 && x + 1 < w) ? mag_bits[(y - 1) * w + (x + 1)] : 0;
    uint32_t sum = W + NW + N + NE;
    if (sum == 0) return 0;
    if (sum <= 2) return 1;
    if (sum <= 6) return 2;
    return 3;
}


// Compute the context index for the coef at (y, x) in a w*h band from the
// already-decoded neighbors' significance bitmap.
static inline uint32_t sig_context(const uint8_t *sig_bits,
                                   uint32_t x, uint32_t y, uint32_t w) {
    uint32_t W  = (x > 0)              ? sig_bits[y * w + (x - 1)] : 0;
    uint32_t NW = (x > 0 && y > 0)     ? sig_bits[(y - 1) * w + (x - 1)] : 0;
    uint32_t N  = (y > 0)              ? sig_bits[(y - 1) * w + x] : 0;
    uint32_t NE = (y > 0 && x + 1 < w) ? sig_bits[(y - 1) * w + (x + 1)] : 0;
    return W | (NW << 1) | (N << 2) | (NE << 3);
}

// (parent_sig_lookup removed: parent-sig context was a slight regression.)

static sig_freq_pair normalize_sig_freqs(const uint32_t *sig_hist) {
    uint64_t total = (uint64_t)sig_hist[0] + sig_hist[1];
    if (total == 0) return (sig_freq_pair){ RANS_M, 0 };
    if (sig_hist[0] == 0) return (sig_freq_pair){ 0, RANS_M };
    if (sig_hist[1] == 0) return (sig_freq_pair){ RANS_M, 0 };
    uint64_t f0 = ((uint64_t)sig_hist[0] * RANS_M + total/2) / total;
    if (f0 < 1) f0 = 1;
    if (f0 > RANS_M - 1) f0 = RANS_M - 1;
    return (sig_freq_pair){ (uint16_t)f0, (uint16_t)(RANS_M - f0) };
}

// ---------------------------------------------------------------------------
// Context-coded variant of split coding. Each significance bit is encoded
// with one of SIG_CTX_COUNT distinct freq tables, indexed by the
// already-decoded-neighbor context. Magnitude stream stays unconditional.
// ---------------------------------------------------------------------------

// Per-(b,c) per-context histograms — one 2-bin bucket per (band, channel, ctx).
// Filled by image_collect_ctx_sig_hists below from the per-(b,c) coef arrays.

// Build per-context sig histograms for one band by walking the coefs in
// row-major order, computing context from already-walked neighbors.
static void collect_ctx_sig_hist_band(const int32_t *q, uint32_t w, uint32_t h,
                                      uint8_t *scratch_sig_bits,
                                      uint32_t out_hist[SIG_CTX_COUNT][2]) {
    memset(out_hist, 0, sizeof(uint32_t) * SIG_CTX_COUNT * 2);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t ctx = sig_context(scratch_sig_bits, x, y, w);
            uint8_t bit = (q[y * w + x] != 0) ? 1 : 0;
            scratch_sig_bits[y * w + x] = bit;
            out_hist[ctx][bit]++;
        }
    }
}

// Collect per-(b,c,mag_ctx) magnitude histograms by walking coefs row-major.
// `out_hist`: [b][c][ctx][C2D_ALPHABET]
static void collect_ctx_mag_hist_band(const int32_t *q, uint32_t w, uint32_t h,
                                      uint8_t *scratch_mag_bits,
                                      uint32_t out_hist[MAG_CTX_COUNT][C2D_ALPHABET]) {
    memset(out_hist, 0, sizeof(uint32_t) * MAG_CTX_COUNT * C2D_ALPHABET);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            int32_t qv = q[y * w + x];
            uint32_t mag = qv >= 0 ? (uint32_t)qv : (uint32_t)(-qv);
            uint32_t cap_mag = mag > 255 ? 255 : mag;
            // Only nonzero coefs contribute to mag histogram.
            if (qv != 0) {
                uint32_t ctx = mag_context(scratch_mag_bits, x, y, w);
                uint8_t sym; uint32_t esc;
                mag_coef_to_symbol(qv, &sym, &esc);
                out_hist[ctx][sym]++;
                (void)esc;
            }
            scratch_mag_bits[y * w + x] = (uint8_t)cap_mag;
        }
    }
}

static void image_collect_ctx_mag_hists(const int32_t *coef_pool,
                                        size_t n_tiles, uint32_t nch,
                                        const subband_rect *rects,
                                        uint32_t *out_hist /* [b][c][ctx][ALPHABET] */) {
    memset(out_hist, 0, sizeof(uint32_t) * C2D_N_SUBBANDS * nch * MAG_CTX_COUNT * C2D_ALPHABET);
    uint8_t scratch_bits[C2D_PIXELS_PER_TILE];
    size_t band_off_tbl[C2D_N_SUBBANDS];
    band_off_tbl[0] = 0;
    for (uint32_t b = 1; b < C2D_N_SUBBANDS; b++)
        band_off_tbl[b] = band_off_tbl[b-1] + (size_t)rects[b-1].w * rects[b-1].h;
    for (size_t t = 0; t < n_tiles; t++) {
        for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
            const subband_rect *r = &rects[b];
            for (uint32_t c = 0; c < nch; c++) {
                const int32_t *q = coef_pool + ((size_t)t * nch + c) * C2D_PIXELS_PER_TILE + band_off_tbl[b];
                uint32_t tmp[MAG_CTX_COUNT][C2D_ALPHABET];
                collect_ctx_mag_hist_band(q, r->w, r->h, scratch_bits, tmp);
                uint32_t *dst = out_hist + (((size_t)b * nch + c) * MAG_CTX_COUNT) * C2D_ALPHABET;
                for (uint32_t k = 0; k < MAG_CTX_COUNT; k++) {
                    for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
                        dst[k * C2D_ALPHABET + i] += tmp[k][i];
                    }
                }
            }
        }
    }
}

// Sum per-(b,c,ctx) sig histograms over all tiles.
static void image_collect_ctx_sig_hists(const int32_t *coef_pool,
                                        size_t n_tiles, uint32_t nch,
                                        const subband_rect *rects,
                                        uint32_t *out_hist /* [b][c][ctx][2] */) {
    memset(out_hist, 0, sizeof(uint32_t) * C2D_N_SUBBANDS * nch * SIG_CTX_COUNT * 2);
    uint8_t scratch_bits[C2D_PIXELS_PER_TILE];
    size_t band_off_tbl[C2D_N_SUBBANDS];
    band_off_tbl[0] = 0;
    for (uint32_t b = 1; b < C2D_N_SUBBANDS; b++)
        band_off_tbl[b] = band_off_tbl[b-1] + (size_t)rects[b-1].w * rects[b-1].h;
    for (size_t t = 0; t < n_tiles; t++) {
        for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
            const subband_rect *r = &rects[b];
            for (uint32_t c = 0; c < nch; c++) {
                const int32_t *q = coef_pool + ((size_t)t * nch + c) * C2D_PIXELS_PER_TILE + band_off_tbl[b];
                uint32_t tmp[SIG_CTX_COUNT][2];
                collect_ctx_sig_hist_band(q, r->w, r->h, scratch_bits, tmp);
                uint32_t *dst = out_hist + (((size_t)b * nch + c) * SIG_CTX_COUNT) * 2;
                for (uint32_t k = 0; k < SIG_CTX_COUNT; k++) {
                    dst[k * 2 + 0] += tmp[k][0];
                    dst[k * 2 + 1] += tmp[k][1];
                }
            }
        }
    }
}

// Emit subband using split + context-coded significance (neighbor context).
static size_t image_emit_subband_split_ctx(const int32_t *q, uint32_t w, uint32_t h,
                                           const sig_freq_pair sig_ctx[SIG_CTX_COUNT],
                                           const uint16_t *mag_freq,
                                           const uint16_t *mag_cum,
                                           uint8_t *renorm_scratch1, size_t renorm_cap1,
                                           uint8_t *renorm_scratch2, size_t renorm_cap2,
                                           uint8_t *escape_scratch, size_t escape_cap,
                                           uint8_t *out, size_t out_cap) {
    size_t n = (size_t)w * h;
    int all_zero = 1;
    for (size_t i = 0; i < n; i++) if (q[i] != 0) { all_zero = 0; break; }
    if (all_zero) {
        if (out_cap < 1) return 0;
        out[0] = SB_FLAG_ALL_ZERO;
        return 1;
    }
    uint8_t  sig_bits[C2D_PIXELS_PER_TILE];
    uint8_t  sig_ctx_idx[C2D_PIXELS_PER_TILE];
    size_t   escape_len = 0;
    int      has_escapes = 0;
    size_t   n_nonzero = 0;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t ctx = sig_context(sig_bits, x, y, w);
            int32_t qv = q[y * w + x];
            uint8_t bit = (qv != 0) ? 1 : 0;
            sig_bits[y * w + x] = bit;
            sig_ctx_idx[y * w + x] = (uint8_t)ctx;
            if (qv != 0) {
                n_nonzero++;
                uint8_t sym; uint32_t esc;
                mag_coef_to_symbol(qv, &sym, &esc);
                if (sym == C2D_ESCAPE_SYMBOL) {
                    if (escape_len + 5 > escape_cap) return 0;
                    escape_len += leb128_write(esc, escape_scratch + escape_len);
                    has_escapes = 1;
                }
            }
        }
    }

    // Encode sig backward, per-context freq tables.
    rans_enc sig_enc;
    rans_enc_init(&sig_enc, renorm_scratch1, renorm_cap1);
    for (size_t i = n; i > 0; i--) {
        size_t idx = i - 1;
        uint32_t ctx = sig_ctx_idx[idx];
        sig_freq_pair p = sig_ctx[ctx];
        uint32_t lane = (uint32_t)(idx & (RANS_NSTATES - 1));
        if (sig_bits[idx] == 0) {
            c2d_assert(p.f0 > 0);
            rans_enc_put(&sig_enc, lane, p.f0, 0);
        } else {
            c2d_assert(p.f1 > 0);
            rans_enc_put(&sig_enc, lane, p.f1, p.f0);
        }
    }
    uint8_t sig_final[RANS_STATE_BYTES];
    rans_enc_finish(&sig_enc, sig_final);

    // Encode mag backward (only nonzeros). Lane indexed by forward-order
    // nonzero count so decoder's lane assignment matches.
    rans_enc mag_enc;
    rans_enc_init(&mag_enc, renorm_scratch2, renorm_cap2);
    // First count nonzeros to know the total nz count for lane indexing.
    size_t nz_total = 0;
    for (size_t i = 0; i < n; i++) if (sig_bits[i]) nz_total++;
    size_t nz_idx_backward = nz_total - 1;
    for (size_t i = n; i > 0; i--) {
        if (q[i-1] == 0) continue;
        uint32_t lane = (uint32_t)(nz_idx_backward & (RANS_NSTATES - 1));
        nz_idx_backward--;
        uint8_t sym; uint32_t esc;
        mag_coef_to_symbol(q[i-1], &sym, &esc);
        (void)esc;
        c2d_assert(mag_freq[sym] > 0);
        rans_enc_put(&mag_enc, lane, mag_freq[sym], mag_cum[sym]);
    }
    uint8_t mag_final[RANS_STATE_BYTES];
    rans_enc_finish(&mag_enc, mag_final);

    // Serialize: same layout as split, since ctx is implicit on decode.
    size_t pos = 0;
    if (pos + 1 > out_cap) return 0;
    out[pos++] = has_escapes ? SB_FLAG_HAS_ESCAPES : 0;
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)sig_enc.pos, out + pos);
    if (pos + RANS_STATE_BYTES > out_cap) return 0;
    memcpy(out + pos, sig_final, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    if (pos + sig_enc.pos > out_cap) return 0;
    memcpy(out + pos, renorm_scratch1 + renorm_cap1 - sig_enc.pos, sig_enc.pos);
    pos += sig_enc.pos;
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)n_nonzero, out + pos);
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)mag_enc.pos, out + pos);
    if (pos + RANS_STATE_BYTES > out_cap) return 0;
    memcpy(out + pos, mag_final, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    if (pos + mag_enc.pos > out_cap) return 0;
    memcpy(out + pos, renorm_scratch2 + renorm_cap2 - mag_enc.pos, mag_enc.pos);
    pos += mag_enc.pos;
    if (has_escapes) {
        if (pos + 5 > out_cap) return 0;
        pos += leb128_write((uint32_t)escape_len, out + pos);
        if (pos + escape_len > out_cap) return 0;
        memcpy(out + pos, escape_scratch, escape_len);
        pos += escape_len;
    }
    return pos;
}

// Emit subband body using split + neighbor-context sig + neighbor-context mag.
// mag_freq/mag_cum are MAG_CTX_COUNT-element arrays of full freq tables.
static size_t image_emit_subband_split_ctxmag(const int32_t *q, uint32_t w, uint32_t h,
                                              const sig_freq_pair sig_ctx[SIG_CTX_COUNT],
                                              const uint16_t (*mag_freq)[C2D_ALPHABET],
                                              const uint16_t (*mag_cum)[C2D_ALPHABET + 1],
                                              uint8_t *renorm_scratch1, size_t renorm_cap1,
                                              uint8_t *renorm_scratch2, size_t renorm_cap2,
                                              uint8_t *escape_scratch, size_t escape_cap,
                                              uint8_t *out, size_t out_cap) {
    size_t n = (size_t)w * h;
    int all_zero = 1;
    for (size_t i = 0; i < n; i++) if (q[i] != 0) { all_zero = 0; break; }
    if (all_zero) {
        if (out_cap < 1) return 0;
        out[0] = SB_FLAG_ALL_ZERO;
        return 1;
    }
    uint8_t  sig_bits[C2D_PIXELS_PER_TILE];
    uint8_t  mag_bits[C2D_PIXELS_PER_TILE];
    uint8_t  sig_ctx_idx[C2D_PIXELS_PER_TILE];
    uint8_t  mag_ctx_idx[C2D_PIXELS_PER_TILE];
    uint8_t  mag_sym_idx[C2D_PIXELS_PER_TILE];
    size_t   escape_len = 0;
    int      has_escapes = 0;
    size_t   n_nonzero = 0;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t sctx = sig_context(sig_bits, x, y, w);
            int32_t qv = q[y * w + x];
            uint8_t bit = (qv != 0) ? 1 : 0;
            sig_bits[y * w + x] = bit;
            sig_ctx_idx[y * w + x] = (uint8_t)sctx;
            uint32_t mag = qv >= 0 ? (uint32_t)qv : (uint32_t)(-qv);
            uint32_t cap_mag = mag > 255 ? 255 : mag;
            if (qv != 0) {
                n_nonzero++;
                uint32_t mctx = mag_context(mag_bits, x, y, w);
                mag_ctx_idx[y * w + x] = (uint8_t)mctx;
                uint8_t sym; uint32_t esc;
                mag_coef_to_symbol(qv, &sym, &esc);
                mag_sym_idx[y * w + x] = sym;
                if (sym == C2D_ESCAPE_SYMBOL) {
                    if (escape_len + 5 > escape_cap) return 0;
                    escape_len += leb128_write(esc, escape_scratch + escape_len);
                    has_escapes = 1;
                }
            }
            mag_bits[y * w + x] = (uint8_t)cap_mag;
        }
    }

    // Encode sig backward, per-context freq tables; 4-way interleaved.
    rans_enc sig_enc;
    rans_enc_init(&sig_enc, renorm_scratch1, renorm_cap1);
    for (size_t i = n; i > 0; i--) {
        size_t idx = i - 1;
        uint32_t lane = (uint32_t)(idx & (RANS_NSTATES - 1));
        uint32_t ctx = sig_ctx_idx[idx];
        sig_freq_pair p = sig_ctx[ctx];
        if (sig_bits[idx] == 0) {
            c2d_assert(p.f0 > 0);
            rans_enc_put(&sig_enc, lane, p.f0, 0);
        } else {
            c2d_assert(p.f1 > 0);
            rans_enc_put(&sig_enc, lane, p.f1, p.f0);
        }
    }
    uint8_t sig_final[RANS_STATE_BYTES];
    rans_enc_finish(&sig_enc, sig_final);

    // Encode mag backward, conditional on mag context; 4-way lanes indexed
    // by forward-order nonzero count.
    rans_enc mag_enc;
    rans_enc_init(&mag_enc, renorm_scratch2, renorm_cap2);
    size_t nz_total = 0;
    for (size_t i = 0; i < n; i++) if (sig_bits[i]) nz_total++;
    size_t nz_idx_backward = nz_total ? nz_total - 1 : 0;
    for (size_t i = n; i > 0; i--) {
        size_t idx = i - 1;
        if (sig_bits[idx] == 0) continue;
        uint32_t lane = (uint32_t)(nz_idx_backward & (RANS_NSTATES - 1));
        nz_idx_backward--;
        uint32_t mctx = mag_ctx_idx[idx];
        uint8_t sym = mag_sym_idx[idx];
        c2d_assert(mag_freq[mctx][sym] > 0);
        rans_enc_put(&mag_enc, lane, mag_freq[mctx][sym], mag_cum[mctx][sym]);
    }
    uint8_t mag_final[RANS_STATE_BYTES];
    rans_enc_finish(&mag_enc, mag_final);

    // Serialize (same wire format as plain split-ctx).
    size_t pos = 0;
    if (pos + 1 > out_cap) return 0;
    out[pos++] = has_escapes ? SB_FLAG_HAS_ESCAPES : 0;
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)sig_enc.pos, out + pos);
    if (pos + RANS_STATE_BYTES > out_cap) return 0;
    memcpy(out + pos, sig_final, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    if (pos + sig_enc.pos > out_cap) return 0;
    memcpy(out + pos, renorm_scratch1 + renorm_cap1 - sig_enc.pos, sig_enc.pos);
    pos += sig_enc.pos;
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)n_nonzero, out + pos);
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)mag_enc.pos, out + pos);
    if (pos + RANS_STATE_BYTES > out_cap) return 0;
    memcpy(out + pos, mag_final, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    if (pos + mag_enc.pos > out_cap) return 0;
    memcpy(out + pos, renorm_scratch2 + renorm_cap2 - mag_enc.pos, mag_enc.pos);
    pos += mag_enc.pos;
    if (has_escapes) {
        if (pos + 5 > out_cap) return 0;
        pos += leb128_write((uint32_t)escape_len, out + pos);
        if (pos + escape_len > out_cap) return 0;
        memcpy(out + pos, escape_scratch, escape_len);
        pos += escape_len;
    }
    return pos;
}

// Decode subband body with neighbor-context sig + neighbor-context mag.
// mag_slot_tbls: MAG_CTX_COUNT slot tables, RANS_M entries each.
static void image_decode_subband_split_ctxmag(const uint8_t *in, size_t in_cap,
                                              uint32_t w, uint32_t h, float qstep,
                                              const sig_freq_pair sig_ctx[SIG_CTX_COUNT],
                                              const rans_slot *mag_slot_tbls,  // [MAG_CTX_COUNT * RANS_M]
                                              float *band) {
    size_t n = (size_t)w * h;
    size_t pos = 0;
    c2d_assert(pos + 1 <= in_cap);
    uint8_t flags = in[pos++];
    if (flags & SB_FLAG_ALL_ZERO) {
        for (size_t i = 0; i < n; i++) band[i] = 0.0f;
        return;
    }
    uint32_t sig_rsz;
    pos += leb128_read(in + pos, in_cap - pos, &sig_rsz);
    c2d_assert(pos + RANS_STATE_BYTES <= in_cap);
    uint8_t sig_final[RANS_STATE_BYTES];
    memcpy(sig_final, in + pos, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    c2d_assert(pos + sig_rsz <= in_cap);
    const uint8_t *sig_renorm = in + pos; pos += sig_rsz;
    uint32_t n_nonzero;
    pos += leb128_read(in + pos, in_cap - pos, &n_nonzero);
    uint32_t mag_rsz;
    pos += leb128_read(in + pos, in_cap - pos, &mag_rsz);
    c2d_assert(pos + RANS_STATE_BYTES <= in_cap);
    uint8_t mag_final[RANS_STATE_BYTES];
    memcpy(mag_final, in + pos, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    c2d_assert(pos + mag_rsz <= in_cap);
    const uint8_t *mag_renorm = in + pos; pos += mag_rsz;
    uint32_t esc_len = 0;
    const uint8_t *esc_stream = NULL;
    if (flags & SB_FLAG_HAS_ESCAPES) {
        pos += leb128_read(in + pos, in_cap - pos, &esc_len);
        c2d_assert(pos + esc_len <= in_cap);
        esc_stream = in + pos;
    }

    // 1) Decode sig stream into bitmap, 4-way interleaved.
    uint8_t sig_bits[C2D_PIXELS_PER_TILE];
    {
        uint32_t xs[RANS_NSTATES];
        rans_states_load(xs, sig_final);
        const uint8_t *rp = sig_renorm;
        const uint8_t *rp_end = sig_renorm + sig_rsz;
        size_t i = 0;
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t xp = 0; xp < w; xp++, i++) {
                uint32_t lane = i & (RANS_NSTATES - 1);
                uint32_t ctx = sig_context(sig_bits, xp, y, w);
                sig_freq_pair p = sig_ctx[ctx];
                uint32_t slot = xs[lane] & (RANS_M - 1);
                uint32_t sym, fr, cum;
                if (slot < p.f0) { sym = 0; fr = p.f0; cum = 0; }
                else             { sym = 1; fr = p.f1; cum = p.f0; }
                xs[lane] = fr * (xs[lane] >> RANS_PROB_BITS) + slot - cum;
                while (xs[lane] < RANS_L) {
                    c2d_assert(rp < rp_end);
                    xs[lane] = (xs[lane] << 8) | *rp++;
                }
                sig_bits[y * w + xp] = (uint8_t)sym;
            }
        }
        (void)rp_end;
    }

    // 2) Decode mag using neighbor-mag context. Note: mag lane index counts
    // only nonzero positions (one rANS state advance per actually-coded mag).
    uint8_t mag_bits[C2D_PIXELS_PER_TILE];
    memset(mag_bits, 0, n);
    uint32_t xs[RANS_NSTATES];
    rans_states_load(xs, mag_final);
    const uint8_t *rp = mag_renorm;
    const uint8_t *rp_end = mag_renorm + mag_rsz;
    size_t esc_pos = 0;
    size_t decoded_nz = 0;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            size_t i = y * w + x;
            if (!sig_bits[i]) { band[i] = 0.0f; continue; }
            uint32_t lane = decoded_nz & (RANS_NSTATES - 1);
            uint32_t mctx = mag_context(mag_bits, x, y, w);
            const rans_slot *tbl = mag_slot_tbls + mctx * RANS_M;
            uint32_t slot = xs[lane] & (RANS_M - 1);
            rans_slot s = tbl[slot];
            xs[lane] = (uint32_t)s.freq * (xs[lane] >> RANS_PROB_BITS) + slot - (uint32_t)s.cum;
            while (xs[lane] < RANS_L) {
                c2d_assert(rp < rp_end);
                xs[lane] = (xs[lane] << 8) | *rp++;
            }
            uint32_t esc_val = 0;
            if (s.sym == C2D_ESCAPE_SYMBOL) {
                esc_pos += leb128_read(esc_stream + esc_pos, esc_len - esc_pos, &esc_val);
            }
            int32_t qv = mag_symbol_to_coef((uint8_t)s.sym, esc_val);
            band[i] = dequantize_one(qv, qstep);
            uint32_t mag = qv >= 0 ? (uint32_t)qv : (uint32_t)(-qv);
            mag_bits[i] = mag > 255 ? 255 : (uint8_t)mag;
            decoded_nz++;
        }
    }
    c2d_assert(decoded_nz == n_nonzero);
    (void)rp_end;
}

// Decode subband body with neighbor-context sig + unconditional mag.
static void image_decode_subband_split_ctx(const uint8_t *in, size_t in_cap,
                                           uint32_t w, uint32_t h, float qstep,
                                           const sig_freq_pair sig_ctx[SIG_CTX_COUNT],
                                           const rans_slot *mag_slot_tbl,
                                           float *band) {
    size_t n = (size_t)w * h;
    size_t pos = 0;
    c2d_assert(pos + 1 <= in_cap);
    uint8_t flags = in[pos++];
    if (flags & SB_FLAG_ALL_ZERO) {
        for (size_t i = 0; i < n; i++) band[i] = 0.0f;
        return;
    }
    uint32_t sig_rsz;
    pos += leb128_read(in + pos, in_cap - pos, &sig_rsz);
    c2d_assert(pos + RANS_STATE_BYTES <= in_cap);
    uint8_t sig_final[RANS_STATE_BYTES];
    memcpy(sig_final, in + pos, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    c2d_assert(pos + sig_rsz <= in_cap);
    const uint8_t *sig_renorm = in + pos; pos += sig_rsz;
    uint32_t n_nonzero;
    pos += leb128_read(in + pos, in_cap - pos, &n_nonzero);
    uint32_t mag_rsz;
    pos += leb128_read(in + pos, in_cap - pos, &mag_rsz);
    c2d_assert(pos + RANS_STATE_BYTES <= in_cap);
    uint8_t mag_final[RANS_STATE_BYTES];
    memcpy(mag_final, in + pos, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    c2d_assert(pos + mag_rsz <= in_cap);
    const uint8_t *mag_renorm = in + pos; pos += mag_rsz;
    uint32_t esc_len = 0;
    const uint8_t *esc_stream = NULL;
    if (flags & SB_FLAG_HAS_ESCAPES) {
        pos += leb128_read(in + pos, in_cap - pos, &esc_len);
        c2d_assert(pos + esc_len <= in_cap);
        esc_stream = in + pos;
    }
    // 1) Decode sig stream into bitmap, 4-way interleaved.
    uint8_t sig_bits[C2D_PIXELS_PER_TILE];
    {
        uint32_t xs[RANS_NSTATES];
        rans_states_load(xs, sig_final);
        const uint8_t *rp = sig_renorm;
        const uint8_t *rp_end = sig_renorm + sig_rsz;
        size_t i = 0;
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t xp = 0; xp < w; xp++, i++) {
                uint32_t lane = i & (RANS_NSTATES - 1);
                uint32_t ctx = sig_context(sig_bits, xp, y, w);
                sig_freq_pair p = sig_ctx[ctx];
                uint32_t slot = xs[lane] & (RANS_M - 1);
                uint32_t sym, fr, cum;
                if (slot < p.f0) { sym = 0; fr = p.f0; cum = 0; }
                else             { sym = 1; fr = p.f1; cum = p.f0; }
                xs[lane] = fr * (xs[lane] >> RANS_PROB_BITS) + slot - cum;
                while (xs[lane] < RANS_L) {
                    c2d_assert(rp < rp_end);
                    xs[lane] = (xs[lane] << 8) | *rp++;
                }
                sig_bits[y * w + xp] = (uint8_t)sym;
            }
        }
        (void)rp_end;
    }
    // 2) Decode mag at nonzero positions, 4-way; lane = decoded_nz & 3.
    uint32_t xs[RANS_NSTATES];
    rans_states_load(xs, mag_final);
    const uint8_t *rp = mag_renorm;
    const uint8_t *rp_end = mag_renorm + mag_rsz;
    size_t esc_pos = 0;
    size_t decoded_nz = 0;
    for (size_t i = 0; i < n; i++) {
        if (!sig_bits[i]) { band[i] = 0.0f; continue; }
        uint32_t lane = decoded_nz & (RANS_NSTATES - 1);
        uint32_t slot = xs[lane] & (RANS_M - 1);
        rans_slot s = mag_slot_tbl[slot];
        xs[lane] = (uint32_t)s.freq * (xs[lane] >> RANS_PROB_BITS) + slot - (uint32_t)s.cum;
        while (xs[lane] < RANS_L) {
            c2d_assert(rp < rp_end);
            xs[lane] = (xs[lane] << 8) | *rp++;
        }
        uint32_t esc_val = 0;
        if (s.sym == C2D_ESCAPE_SYMBOL) {
            esc_pos += leb128_read(esc_stream + esc_pos, esc_len - esc_pos, &esc_val);
        }
        int32_t qv = mag_symbol_to_coef((uint8_t)s.sym, esc_val);
        band[i] = dequantize_one(qv, qstep);
        decoded_nz++;
    }
    c2d_assert(decoded_nz == n_nonzero);
    (void)rp_end;
}

// Emit one subband body using two streams. Caller provides per-(b,c)
// global tables: sig_pair and mag_freq+mag_cum.
static size_t image_emit_subband_split(const int32_t *q, size_t n,
                                       sig_freq_pair sig,
                                       const uint16_t *mag_freq,
                                       const uint16_t *mag_cum,
                                       uint8_t *renorm_scratch1, size_t renorm_cap1,
                                       uint8_t *renorm_scratch2, size_t renorm_cap2,
                                       uint8_t *escape_scratch, size_t escape_cap,
                                       uint8_t *out, size_t out_cap) {
    // Detect all-zero.
    int all_zero = 1;
    for (size_t i = 0; i < n; i++) if (q[i] != 0) { all_zero = 0; break; }
    if (all_zero) {
        if (out_cap < 1) return 0;
        out[0] = SB_FLAG_ALL_ZERO;
        return 1;
    }
    // 1) Encode sig stream (2-symbol). sig.f0 is freq of zero, sig.f1 nonzero.
    rans_enc sig_enc;
    rans_enc_init(&sig_enc, renorm_scratch1, renorm_cap1);
    // Forward escape pre-pass over nonzeros.
    size_t escape_len = 0;
    int has_escapes = 0;
    size_t n_nonzero = 0;
    for (size_t i = 0; i < n; i++) {
        if (q[i] == 0) continue;
        n_nonzero++;
        uint8_t sym; uint32_t esc;
        mag_coef_to_symbol(q[i], &sym, &esc);
        if (sym == C2D_ESCAPE_SYMBOL) {
            if (escape_len + 5 > escape_cap) return 0;
            escape_len += leb128_write(esc, escape_scratch + escape_len);
            has_escapes = 1;
        }
    }
    // Encode sig backward; 4-way interleaved.
    for (size_t i = n; i > 0; i--) {
        uint32_t lane = (uint32_t)((i - 1) & (RANS_NSTATES - 1));
        if (q[i-1] == 0) {
            c2d_assert(sig.f0 > 0);
            rans_enc_put(&sig_enc, lane, sig.f0, 0);
        } else {
            c2d_assert(sig.f1 > 0);
            rans_enc_put(&sig_enc, lane, sig.f1, sig.f0);
        }
    }
    uint8_t sig_final[RANS_STATE_BYTES];
    rans_enc_finish(&sig_enc, sig_final);

    // 2) Encode mag stream (only nonzeros), backward, 4-way.
    rans_enc mag_enc;
    rans_enc_init(&mag_enc, renorm_scratch2, renorm_cap2);
    size_t nz_idx_backward = n_nonzero ? n_nonzero - 1 : 0;
    for (size_t i = n; i > 0; i--) {
        if (q[i-1] == 0) continue;
        uint32_t lane = (uint32_t)(nz_idx_backward & (RANS_NSTATES - 1));
        nz_idx_backward--;
        uint8_t sym; uint32_t esc;
        mag_coef_to_symbol(q[i-1], &sym, &esc);
        (void)esc;
        c2d_assert(mag_freq[sym] > 0);
        rans_enc_put(&mag_enc, lane, mag_freq[sym], mag_cum[sym]);
    }
    uint8_t mag_final[RANS_STATE_BYTES];
    rans_enc_finish(&mag_enc, mag_final);

    // 3) Serialize.
    size_t pos = 0;
    if (pos + 1 > out_cap) return 0;
    out[pos++] = has_escapes ? SB_FLAG_HAS_ESCAPES : 0;
    // sig stream
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)sig_enc.pos, out + pos);
    if (pos + RANS_STATE_BYTES > out_cap) return 0;
    memcpy(out + pos, sig_final, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    if (pos + sig_enc.pos > out_cap) return 0;
    memcpy(out + pos, renorm_scratch1 + renorm_cap1 - sig_enc.pos, sig_enc.pos);
    pos += sig_enc.pos;
    // mag stream
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)n_nonzero, out + pos);
    if (pos + 5 > out_cap) return 0;
    pos += leb128_write((uint32_t)mag_enc.pos, out + pos);
    if (pos + RANS_STATE_BYTES > out_cap) return 0;
    memcpy(out + pos, mag_final, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    if (pos + mag_enc.pos > out_cap) return 0;
    memcpy(out + pos, renorm_scratch2 + renorm_cap2 - mag_enc.pos, mag_enc.pos);
    pos += mag_enc.pos;
    // escapes
    if (has_escapes) {
        if (pos + 5 > out_cap) return 0;
        pos += leb128_write((uint32_t)escape_len, out + pos);
        if (pos + escape_len > out_cap) return 0;
        memcpy(out + pos, escape_scratch, escape_len);
        pos += escape_len;
    }
    return pos;
}

// Decode one split-stream subband body. `n` is the rect's total coef count
// (caller knows from rect geometry).
static void image_decode_subband_split(const uint8_t *in, size_t in_cap,
                                       size_t n, float qstep,
                                       sig_freq_pair sig,
                                       const rans_slot *mag_slot_tbl,
                                       float *band) {
    size_t pos = 0;
    c2d_assert(pos + 1 <= in_cap);
    uint8_t flags = in[pos++];
    if (flags & SB_FLAG_ALL_ZERO) {
        for (size_t i = 0; i < n; i++) band[i] = 0.0f;
        return;
    }
    // sig stream
    uint32_t sig_rsz;
    pos += leb128_read(in + pos, in_cap - pos, &sig_rsz);
    c2d_assert(pos + RANS_STATE_BYTES <= in_cap);
    uint8_t sig_final[RANS_STATE_BYTES];
    memcpy(sig_final, in + pos, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    c2d_assert(pos + sig_rsz <= in_cap);
    const uint8_t *sig_renorm = in + pos; pos += sig_rsz;
    // mag stream
    uint32_t n_nonzero;
    pos += leb128_read(in + pos, in_cap - pos, &n_nonzero);
    uint32_t mag_rsz;
    pos += leb128_read(in + pos, in_cap - pos, &mag_rsz);
    c2d_assert(pos + RANS_STATE_BYTES <= in_cap);
    uint8_t mag_final[RANS_STATE_BYTES];
    memcpy(mag_final, in + pos, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;
    c2d_assert(pos + mag_rsz <= in_cap);
    const uint8_t *mag_renorm = in + pos; pos += mag_rsz;
    uint32_t esc_len = 0;
    const uint8_t *esc_stream = NULL;
    if (flags & SB_FLAG_HAS_ESCAPES) {
        pos += leb128_read(in + pos, in_cap - pos, &esc_len);
        c2d_assert(pos + esc_len <= in_cap);
        esc_stream = in + pos;
    }

    // 1) Decode sig stream into a 1-bit-per-coef buffer on the stack.
    //    n is bounded by tile_side*tile_side = 65536, so we can stack-alloc
    //    a 65536-byte bitmap.
    uint8_t sig_bits[C2D_PIXELS_PER_TILE];
    {
        uint32_t xs[RANS_NSTATES];
        rans_states_load(xs, sig_final);
        const uint8_t *rp = sig_renorm;
        const uint8_t *rp_end = sig_renorm + sig_rsz;
        for (size_t i = 0; i < n; i++) {
            uint32_t lane = i & (RANS_NSTATES - 1);
            uint32_t slot = xs[lane] & (RANS_M - 1);
            uint32_t sym;
            uint32_t freq, cum;
            if (slot < sig.f0) { sym = 0; freq = sig.f0; cum = 0; }
            else               { sym = 1; freq = sig.f1; cum = sig.f0; }
            xs[lane] = freq * (xs[lane] >> RANS_PROB_BITS) + slot - cum;
            while (xs[lane] < RANS_L) {
                c2d_assert(rp < rp_end);
                xs[lane] = (xs[lane] << 8) | *rp++;
            }
            sig_bits[i] = (uint8_t)sym;
        }
        (void)rp_end;
    }

    // 2) Decode mag stream into band at nonzero positions; 4-way lanes.
    uint32_t xs[RANS_NSTATES];
    rans_states_load(xs, mag_final);
    const uint8_t *rp = mag_renorm;
    const uint8_t *rp_end = mag_renorm + mag_rsz;
    size_t esc_pos = 0;
    size_t decoded_nonzero = 0;
    for (size_t i = 0; i < n; i++) {
        if (!sig_bits[i]) { band[i] = 0.0f; continue; }
        uint32_t lane = decoded_nonzero & (RANS_NSTATES - 1);
        uint32_t slot = xs[lane] & (RANS_M - 1);
        rans_slot s = mag_slot_tbl[slot];
        xs[lane] = (uint32_t)s.freq * (xs[lane] >> RANS_PROB_BITS) + slot - (uint32_t)s.cum;
        while (xs[lane] < RANS_L) {
            c2d_assert(rp < rp_end);
            xs[lane] = (xs[lane] << 8) | *rp++;
        }
        uint32_t esc_val = 0;
        if (s.sym == C2D_ESCAPE_SYMBOL) {
            esc_pos += leb128_read(esc_stream + esc_pos, esc_len - esc_pos, &esc_val);
        }
        int32_t qv = mag_symbol_to_coef((uint8_t)s.sym, esc_val);
        band[i] = dequantize_one(qv, qstep);
        decoded_nonzero++;
    }
    c2d_assert(decoded_nonzero == n_nonzero);
    (void)rp_end;
}

size_t c2d_image_encode(const void *in, uint32_t width, uint32_t height,
                        c2d_dtype dt, uint32_t nch, float target_ratio,
                        uint32_t flags, uint8_t *out, size_t out_cap) {
    c2d_assert(width > 0 && height > 0);
    c2d_assert(nch >= 1 && nch <= C2D_MAX_CHANNELS);
    int do_ycocg = (flags & C2D_FLAG_COLOR_YCOCG) && nch == 3;

    uint32_t ntx = (width  + C2D_TILE_SIDE - 1) / C2D_TILE_SIDE;
    uint32_t nty = (height + C2D_TILE_SIDE - 1) / C2D_TILE_SIDE;
    size_t n_tiles = (size_t)ntx * nty;

    size_t target_bytes = (size_t)(c2d_dtype_tile_bytes_n(dt, nch) * n_tiles / target_ratio);
    // BITPLANE's entropy estimate is conservative; inflate target to account.
    if (flags & C2D_FLAG_BITPLANE) target_bytes = (size_t)(target_bytes * 1.35);

    // 1) Global normalization.
    float *dcs    = (float *)calloc(nch, sizeof(float));
    float *scales = (float *)calloc(nch, sizeof(float));
    image_normalization(in, dt, width, height, nch, dcs, scales);

    // 2) Allocate per-tile DWT'd planes.
    subband_rect rects[C2D_N_SUBBANDS];
    enumerate_subbands(rects);
    img_tile_state *tiles = (img_tile_state *)calloc(n_tiles, sizeof(img_tile_state));
    // Each thread gets its own scratch buffers to avoid races.
    #pragma omp parallel
    {
        float dwt_tmp[C2D_TILE_SIDE];
        float *mallat = (float *)aligned_alloc(C2D_ALIGN, sizeof(float) * C2D_PIXELS_PER_TILE);
        void  *tile_buf = malloc(c2d_dtype_tile_bytes_n(dt, nch));
        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_tiles; t++) {
            uint32_t tx = (uint32_t)((t % ntx) * C2D_TILE_SIDE);
            uint32_t ty = (uint32_t)((t / ntx) * C2D_TILE_SIDE);
            gather_tile(in, dt, width, height, nch, tx, ty, tile_buf);
            tiles[t].planes = (float *)aligned_alloc(C2D_ALIGN, sizeof(float) * (size_t)nch * C2D_PIXELS_PER_TILE);
            for (uint32_t c = 0; c < nch; c++) {
                float dc = dcs[c], sc = scales[c];
                float *plane = tiles[t].planes + (size_t)c * C2D_PIXELS_PER_TILE;
                for (size_t i = 0; i < C2D_PIXELS_PER_TILE; i++)
                    plane[i] = (iload_f32(tile_buf, dt, i, c, nch) - dc) * sc;
            }
            if (do_ycocg) ycocg_fwd(tiles[t].planes,
                                    tiles[t].planes + C2D_PIXELS_PER_TILE,
                                    tiles[t].planes + 2 * C2D_PIXELS_PER_TILE,
                                    C2D_PIXELS_PER_TILE);
            for (uint32_t c = 0; c < nch; c++)
                dwt_fwd(tiles[t].planes + (size_t)c * C2D_PIXELS_PER_TILE,
                        C2D_TILE_SIDE, dwt_tmp, mallat);
            tiles[t].qmul_idx = 128;  // neutral default
        }
        free(mallat); free(tile_buf);
    }

    // 2b) If C2D_FLAG_LOW_BPP: compute per-tile q-multiplier from per-tile
    // detail-band "activity" (sum of |coef| over all detail subbands of
    // channel 0). Smooth tiles -> high qmul (coarser). Active tiles -> low
    // qmul (finer). The multipliers are normalized to mean ~1.0 in log
    // space so the global q bisection still operates at the same target.
    if (flags & C2D_FLAG_LOW_BPP) {
        float *act = (float *)malloc(n_tiles * sizeof(float));
        // Compute activity per tile.
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < n_tiles; t++) {
            const float *plane = tiles[t].planes;  // channel 0 (luma if ycocg)
            double sum_abs = 0.0;
            size_t cnt = 0;
            // Sum |coef| across all detail bands (skip LL = subband 0).
            for (uint32_t b = 1; b < C2D_N_SUBBANDS; b++) {
                const subband_rect *r = &rects[b];
                for (uint32_t y = 0; y < r->h; y++) {
                    const float *row = plane + (size_t)(r->y + y) * C2D_TILE_SIDE + r->x;
                    for (uint32_t x = 0; x < r->w; x++) {
                        sum_abs += fabsf(row[x]);
                    }
                }
                cnt += (size_t)r->w * r->h;
            }
            act[t] = (cnt > 0) ? (float)(sum_abs / (double)cnt) : 1.0f;
        }
        // Compute log-mean activity for normalization.
        double log_sum = 0.0;
        size_t log_cnt = 0;
        for (size_t t = 0; t < n_tiles; t++) {
            if (act[t] > 1e-10f) { log_sum += log(act[t]); log_cnt++; }
        }
        float log_mean = (log_cnt > 0) ? (float)(log_sum / (double)log_cnt) : 0.0f;
        // Perceptual adaptive quantization: smooth tiles (low activity) get
        // FINER q (noise is visible on smooth surfaces), busy tiles (high
        // activity) get COARSER q (texture masks noise). This is the
        // classic HEVC/VVC AQ direction.
        // qmul = (act / mean_act)^0.3
        const float alpha = 0.3f;
        for (size_t t = 0; t < n_tiles; t++) {
            if (act[t] < 1e-10f) {
                tiles[t].qmul_idx = qmul_encode(0.5f);  // dead-flat tile: very fine
                continue;
            }
            float ratio = act[t] / expf(log_mean);
            float mul = powf(ratio, alpha);
            if (mul < 0.5f) mul = 0.5f;
            if (mul > 2.0f) mul = 2.0f;
            tiles[t].qmul_idx = qmul_encode(mul);
        }
        free(act);
    }

    // 3) Pool for quantized coefs across all tiles.
    int32_t *coef_pool = (int32_t *)aligned_alloc(C2D_ALIGN,
        sizeof(int32_t) * (size_t)n_tiles * nch * C2D_PIXELS_PER_TILE);

    // 4) Global histograms.
    uint32_t *g_hist = (uint32_t *)calloc((size_t)C2D_N_SUBBANDS * nch * C2D_ALPHABET, sizeof(uint32_t));

    // Header overhead estimate (constant during bisection):
    size_t hdr = IMG_HDR_FIXED + img_norm_size(nch);
    // shared freq tables: we estimate ~ 1 + 3*nnz per (b,c). After bisection
    // converges we know nnz. Approximate worst case for estimate.
    size_t freq_overhead_estimate = (size_t)C2D_N_SUBBANDS * nch * (1 + 3 * 32);  // ~half-full
    // Tile directory:
    size_t tile_dir = 8 + n_tiles * 8;
    // Per-tile per-subband directory:
    size_t per_tile_dir = (size_t)C2D_N_SUBBANDS * nch * 8;
    size_t varq_overhead = (flags & C2D_FLAG_LOW_BPP) ? n_tiles : 0;
    size_t filter_overhead = (flags & C2D_FLAG_LOW_BPP) ? n_tiles : 0;
    size_t header_overhead = hdr + freq_overhead_estimate + tile_dir + varq_overhead + filter_overhead + n_tiles * per_tile_dir;

    // 5) Bisect q via cheap entropy estimate. Use the histogram-only
    //    variant so we skip the int32 coef_pool writes during bisection.
    double lo = log(1.0 / 4096.0);
    double hi = log(4096.0);
    double best_mid = 0.5 * (lo + hi);
    int got = 0;
    for (int iter = 0; iter < 8; iter++) {
        double mid = 0.5 * (lo + hi);
        float q = (float)exp(mid);
        size_t esc_bytes = image_hist_only(tiles, n_tiles, nch, do_ycocg,
                                           q, rects, g_hist, flags);
        size_t est = image_estimate_bytes(g_hist, nch, n_tiles, esc_bytes, header_overhead);
        if (est > target_bytes) lo = mid;
        else { hi = mid; best_mid = mid; got = 1; }
        double rel = fabs((double)est - (double)target_bytes) / (double)target_bytes;
        if (rel < 0.02) { best_mid = mid; got = 1; break; }
    }
    float final_q = (float)exp(got ? best_mid : hi);

    // 6) Final quantize with chosen q.
    image_requantize_collect(tiles, n_tiles, nch, do_ycocg, final_q, rects, coef_pool, g_hist, flags);

    int ctx_split = (flags & C2D_FLAG_CTX_SPLIT) != 0;
    int ctx_neigh = ctx_split && (flags & C2D_FLAG_CTX_NEIGH);
    int ctx_mag   = ctx_neigh && (flags & C2D_FLAG_CTX_MAG);
    int bitplane  = (flags & C2D_FLAG_BITPLANE) != 0;
    if (bitplane) { ctx_split = ctx_neigh = ctx_mag = 0; }

    // 7) Build per-(band, channel) freq + cum tables from global histograms.
    //    Unified mode: one 65-symbol table per (b,c).
    //    Split mode: one 2-symbol sig table + one 65-symbol magnitude table
    //                (constructed from the unified hist by remapping).
    uint16_t (*g_freq)[C2D_ALPHABET] = (uint16_t (*)[C2D_ALPHABET])
        calloc((size_t)C2D_N_SUBBANDS * nch, sizeof(*g_freq));
    uint16_t (*g_cum)[C2D_ALPHABET + 1] = (uint16_t (*)[C2D_ALPHABET + 1])
        calloc((size_t)C2D_N_SUBBANDS * nch, sizeof(*g_cum));
    sig_freq_pair *g_sig = NULL;
    sig_freq_pair *g_sig_ctx = NULL;   // [b * nch * SIG_CTX_COUNT + c * SIG_CTX_COUNT + ctx]
    uint16_t (*g_mag_freq)[C2D_ALPHABET] = NULL;
    uint16_t (*g_mag_cum)[C2D_ALPHABET + 1] = NULL;
    // When ctx_mag is set: g_mag_freq_ctx is [b*nch*MAG_CTX_COUNT + c*MAG_CTX_COUNT + mctx][ALPHABET]
    uint16_t (*g_mag_freq_ctx)[C2D_ALPHABET] = NULL;
    uint16_t (*g_mag_cum_ctx)[C2D_ALPHABET + 1] = NULL;
    if (bitplane) {
        // No global freq tables needed for bit-plane mode.
    } else if (!ctx_split) {
        for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
            for (uint32_t c = 0; c < nch; c++) {
                size_t idx = (size_t)b * nch + c;
                normalize_freqs(g_hist + idx * C2D_ALPHABET, RANS_M, g_freq[idx]);
                g_cum[idx][0] = 0;
                for (uint32_t i = 0; i < C2D_ALPHABET; i++)
                    g_cum[idx][i+1] = g_cum[idx][i] + g_freq[idx][i];
                c2d_assert(g_cum[idx][C2D_ALPHABET] == RANS_M);
            }
        }
    } else {
        g_sig      = (sig_freq_pair *)calloc((size_t)C2D_N_SUBBANDS * nch, sizeof(sig_freq_pair));
        g_mag_freq = (uint16_t (*)[C2D_ALPHABET]) calloc((size_t)C2D_N_SUBBANDS * nch, sizeof(*g_mag_freq));
        g_mag_cum  = (uint16_t (*)[C2D_ALPHABET + 1]) calloc((size_t)C2D_N_SUBBANDS * nch, sizeof(*g_mag_cum));
        for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
            for (uint32_t c = 0; c < nch; c++) {
                size_t idx = (size_t)b * nch + c;
                uint32_t sig_h[2]; uint32_t mag_h[C2D_ALPHABET];
                split_hists_from_unified(g_hist + idx * C2D_ALPHABET, sig_h, mag_h);
                g_sig[idx] = normalize_sig_freqs(sig_h);
                normalize_freqs(mag_h, RANS_M, g_mag_freq[idx]);
                g_mag_cum[idx][0] = 0;
                for (uint32_t i = 0; i < C2D_ALPHABET; i++)
                    g_mag_cum[idx][i+1] = g_mag_cum[idx][i] + g_mag_freq[idx][i];
                c2d_assert(g_mag_cum[idx][C2D_ALPHABET] == RANS_M);
            }
        }
        if (ctx_neigh) {
            // Collect per-(b,c,ctx) sig histograms from the final coef pool.
            uint32_t *ctx_hist = (uint32_t *)calloc(
                (size_t)C2D_N_SUBBANDS * nch * SIG_CTX_COUNT * 2, sizeof(uint32_t));
            image_collect_ctx_sig_hists(coef_pool, n_tiles, nch, rects, ctx_hist);
            g_sig_ctx = (sig_freq_pair *)calloc(
                (size_t)C2D_N_SUBBANDS * nch * SIG_CTX_COUNT, sizeof(sig_freq_pair));
            for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
                for (uint32_t c = 0; c < nch; c++) {
                    size_t base = ((size_t)b * nch + c) * SIG_CTX_COUNT;
                    for (uint32_t k = 0; k < SIG_CTX_COUNT; k++) {
                        uint32_t h[2] = {
                            ctx_hist[(base + k) * 2 + 0],
                            ctx_hist[(base + k) * 2 + 1]
                        };
                        g_sig_ctx[base + k] = normalize_sig_freqs(h);
                    }
                }
            }
            free(ctx_hist);
        }
        if (ctx_mag) {
            // Collect per-(b,c,mag_ctx,sym) mag histograms.
            uint32_t *mag_ctx_hist = (uint32_t *)calloc(
                (size_t)C2D_N_SUBBANDS * nch * MAG_CTX_COUNT * C2D_ALPHABET, sizeof(uint32_t));
            image_collect_ctx_mag_hists(coef_pool, n_tiles, nch, rects, mag_ctx_hist);
            g_mag_freq_ctx = (uint16_t (*)[C2D_ALPHABET]) calloc(
                (size_t)C2D_N_SUBBANDS * nch * MAG_CTX_COUNT, sizeof(*g_mag_freq_ctx));
            g_mag_cum_ctx  = (uint16_t (*)[C2D_ALPHABET + 1]) calloc(
                (size_t)C2D_N_SUBBANDS * nch * MAG_CTX_COUNT, sizeof(*g_mag_cum_ctx));
            for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
                for (uint32_t c = 0; c < nch; c++) {
                    size_t base = ((size_t)b * nch + c) * MAG_CTX_COUNT;
                    for (uint32_t k = 0; k < MAG_CTX_COUNT; k++) {
                        uint32_t *h = mag_ctx_hist + (base + k) * C2D_ALPHABET;
                        normalize_freqs(h, RANS_M, g_mag_freq_ctx[base + k]);
                        g_mag_cum_ctx[base + k][0] = 0;
                        for (uint32_t i = 0; i < C2D_ALPHABET; i++)
                            g_mag_cum_ctx[base + k][i+1] = g_mag_cum_ctx[base + k][i] + g_mag_freq_ctx[base + k][i];
                        c2d_assert(g_mag_cum_ctx[base + k][C2D_ALPHABET] == RANS_M);
                    }
                }
            }
            free(mag_ctx_hist);
        }
    }

    // 8) Emit image header.
    if (out_cap < IMG_HDR_FIXED + img_norm_size(nch)) goto oom;
    size_t pos = 0;
    memcpy(out + pos, C2D_IMAGE_MAGIC, 4); pos += 4;
    out[pos++] = (uint8_t)C2D_FORMAT_VERSION;
    out[pos++] = (uint8_t)dt;
    out[pos++] = (uint8_t)nch;
    write_u16(out + pos, (uint16_t)(flags & 0xffff)); pos += 2;
    out[pos++] = 0;  // reserved for future
    write_u32(out + pos, width);  pos += 4;
    write_u32(out + pos, height); pos += 4;
    write_u32(out + pos, C2D_TILE_SIDE); pos += 4;
    write_f32(out + pos, final_q); pos += 4;
    for (uint32_t c = 0; c < nch; c++) {
        write_f32(out + pos, dcs[c]);    pos += 4;
        write_f32(out + pos, scales[c]); pos += 4;
    }

    // 8b) Build per-(b,c) presence bitmap from global histograms. A (b,c)
    //     is "empty" iff every coef across all tiles is zero. Empty (b,c)
    //     omit their freq tables here AND their per-tile dir/body entries.
    size_t presence_bytes = ((size_t)C2D_N_SUBBANDS * nch + 7) / 8;
    if (pos + presence_bytes > out_cap) goto oom;
    uint8_t *presence = out + pos;
    memset(presence, 0, presence_bytes);
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        for (uint32_t c = 0; c < nch; c++) {
            size_t idx = (size_t)b * nch + c;
            const uint32_t *h = g_hist + idx * C2D_ALPHABET;
            int empty = 1;
            for (uint32_t i = 1; i < C2D_ALPHABET; i++) if (h[i]) { empty = 0; break; }
            if (!empty) {
                presence[idx / 8] |= (uint8_t)(1u << (idx & 7));
            }
        }
    }
    pos += presence_bytes;

    // 9) Emit shared per-(b, c) freq tables — only for present (b,c).
    //    Skipped entirely in bit-plane mode (AC adapts on its own).
    if (bitplane) goto skip_freq_emit;
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        for (uint32_t c = 0; c < nch; c++) {
            size_t idx = (size_t)b * nch + c;
            if (!(presence[idx / 8] & (1u << (idx & 7)))) continue;
            if (!ctx_split) {
                uint8_t nnz = 0;
                for (uint32_t i = 0; i < C2D_ALPHABET; i++) if (g_freq[idx][i]) nnz++;
                if (pos + 1 > out_cap) goto oom;
                out[pos++] = nnz;
                for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
                    if (!g_freq[idx][i]) continue;
                    if (pos + 3 > out_cap) goto oom;
                    out[pos++] = (uint8_t)i;
                    write_u16(out + pos, g_freq[idx][i]); pos += 2;
                }
            } else {
                if (!ctx_neigh) {
                    if (pos + 2 > out_cap) goto oom;
                    write_u16(out + pos, g_sig[idx].f0); pos += 2;
                } else {
                    if (pos + 2 * SIG_CTX_COUNT > out_cap) goto oom;
                    for (uint32_t k = 0; k < SIG_CTX_COUNT; k++) {
                        write_u16(out + pos, g_sig_ctx[idx * SIG_CTX_COUNT + k].f0);
                        pos += 2;
                    }
                }
                if (!ctx_mag) {
                    uint8_t nnz = 0;
                    for (uint32_t i = 0; i < C2D_ALPHABET; i++) if (g_mag_freq[idx][i]) nnz++;
                    if (pos + 1 > out_cap) goto oom;
                    out[pos++] = nnz;
                    for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
                        if (!g_mag_freq[idx][i]) continue;
                        if (pos + 3 > out_cap) goto oom;
                        out[pos++] = (uint8_t)i;
                        write_u16(out + pos, g_mag_freq[idx][i]); pos += 2;
                    }
                } else {
                    // Emit MAG_CTX_COUNT mag freq tables.
                    for (uint32_t mctx = 0; mctx < MAG_CTX_COUNT; mctx++) {
                        size_t midx = idx * MAG_CTX_COUNT + mctx;
                        uint8_t nnz = 0;
                        for (uint32_t i = 0; i < C2D_ALPHABET; i++) if (g_mag_freq_ctx[midx][i]) nnz++;
                        if (pos + 1 > out_cap) goto oom;
                        out[pos++] = nnz;
                        for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
                            if (!g_mag_freq_ctx[midx][i]) continue;
                            if (pos + 3 > out_cap) goto oom;
                            out[pos++] = (uint8_t)i;
                            write_u16(out + pos, g_mag_freq_ctx[midx][i]); pos += 2;
                        }
                    }
                }
            }
        }
    }

skip_freq_emit:
    // 10) Tile directory placeholder.
    size_t varq_block = (flags & C2D_FLAG_LOW_BPP) ? n_tiles : 0;
    size_t filter_block = (flags & C2D_FLAG_LOW_BPP) ? n_tiles : 0;
    if (pos + 8 + varq_block + filter_block + n_tiles * 8 > out_cap) goto oom;
    write_u32(out + pos, ntx); pos += 4;
    write_u32(out + pos, nty); pos += 4;
    if (flags & C2D_FLAG_LOW_BPP) {
        for (size_t t = 0; t < n_tiles; t++) out[pos + t] = tiles[t].qmul_idx;
        pos += n_tiles;
    }
    if (flags & C2D_FLAG_LOW_BPP) {
        // Encoder-chosen per-tile filter strength. Calibrated empirically:
        // small filter (sigma ~1-7 pixel units) gives best PSNR/SSIM at low
        // bpp; larger sigma over-smooths.
        uint8_t global_strength = 0;
        if (target_ratio >= 80.0f)      global_strength = 4;
        else if (target_ratio >= 50.0f) global_strength = 3;
        else if (target_ratio >= 30.0f) global_strength = 2;
        else                            global_strength = 0;  // hi-quality: filter hurts
        memset(out + pos, global_strength, n_tiles);
        pos += n_tiles;
    }
    size_t tile_dir_pos = pos;
    pos += n_tiles * 8;

    // 11) Emit each tile body.
    size_t renorm_cap = C2D_PIXELS_PER_TILE * 8 + 1024;
    uint8_t *renorm_scratch = (uint8_t *)malloc(renorm_cap);
    uint8_t *renorm_scratch2 = (uint8_t *)malloc(renorm_cap);
    size_t escape_cap = C2D_PIXELS_PER_TILE * 5 + 64;
    uint8_t *escape_scratch = (uint8_t *)malloc(escape_cap);

    size_t emit_band_off_tbl[C2D_N_SUBBANDS];
    emit_band_off_tbl[0] = 0;
    for (uint32_t b = 1; b < C2D_N_SUBBANDS; b++)
        emit_band_off_tbl[b] = emit_band_off_tbl[b-1] + (size_t)rects[b-1].w * rects[b-1].h;


    // Phase A: encode each tile body into its own scratch buffer in parallel.
    // Phase B: serial concat of (lengths-prefix + body) into `out`.
    size_t per_tile_scratch_cap = (size_t)C2D_N_SUBBANDS * nch *
        (1 + 65*3 + 4 + RANS_STATE_BYTES + C2D_PIXELS_PER_TILE*5 + 4 + C2D_PIXELS_PER_TILE*5 + 64);
    uint8_t **tile_scratch = (uint8_t **)calloc(n_tiles, sizeof(uint8_t *));
    size_t   *tile_scratch_used = (size_t *)calloc(n_tiles, sizeof(size_t));
    size_t   *tile_dir_idx = (size_t *)calloc(n_tiles, sizeof(size_t));
    size_t  **tile_body_lens = (size_t **)calloc(n_tiles, sizeof(size_t *));
    int oom_flag = 0;
    if (!tile_scratch || !tile_scratch_used || !tile_dir_idx || !tile_body_lens) oom_flag = 1;

    if (!oom_flag) {
    #pragma omp parallel
    {
        size_t local_renorm_cap = C2D_PIXELS_PER_TILE * 8 + 1024;
        uint8_t *local_renorm1 = (uint8_t *)malloc(local_renorm_cap);
        uint8_t *local_renorm2 = (uint8_t *)malloc(local_renorm_cap);
        size_t local_escape_cap = C2D_PIXELS_PER_TILE * 5 + 64;
        uint8_t *local_escape = (uint8_t *)malloc(local_escape_cap);
        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_tiles; t++) {
            uint8_t *scratch = (uint8_t *)malloc(per_tile_scratch_cap);
            size_t  *lens   = (size_t *)calloc((size_t)C2D_N_SUBBANDS * nch, sizeof(size_t));
            if (!scratch || !lens) {
                free(scratch); free(lens);
                #pragma omp atomic write
                oom_flag = 1;
                continue;
            }
            size_t scratch_pos = 0;
            size_t dir_idx = 0;
            int local_bad = 0;
            for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
                const subband_rect *r = &rects[b];
                for (uint32_t c = 0; c < nch; c++) {
                    size_t fidx = (size_t)b * nch + c;
                    if (!(presence[fidx / 8] & (1u << (fidx & 7)))) continue;
                    const int32_t *q = coef_pool
                                     + ((size_t)t * nch + c) * C2D_PIXELS_PER_TILE
                                     + emit_band_off_tbl[b];
                    uint8_t *dst = scratch + scratch_pos;
                    size_t cap_left = per_tile_scratch_cap - scratch_pos;
                    size_t w;
                    if (bitplane) {
                        w = bp_encode_band(q, r->w, r->h, r->kind, r->level, dst, cap_left);
                    } else if (!ctx_split) {
                        w = image_emit_subband(q, (size_t)r->w * r->h,
                                               g_freq[fidx], g_cum[fidx],
                                               local_renorm1, local_renorm_cap,
                                               local_escape, local_escape_cap,
                                               dst, cap_left);
                    } else if (!ctx_neigh) {
                        w = image_emit_subband_split(q, (size_t)r->w * r->h,
                                                     g_sig[fidx],
                                                     g_mag_freq[fidx], g_mag_cum[fidx],
                                                     local_renorm1, local_renorm_cap,
                                                     local_renorm2, local_renorm_cap,
                                                     local_escape, local_escape_cap,
                                                     dst, cap_left);
                    } else if (!ctx_mag) {
                        w = image_emit_subband_split_ctx(q, r->w, r->h,
                                                         g_sig_ctx + fidx * SIG_CTX_COUNT,
                                                         g_mag_freq[fidx], g_mag_cum[fidx],
                                                         local_renorm1, local_renorm_cap,
                                                         local_renorm2, local_renorm_cap,
                                                         local_escape, local_escape_cap,
                                                         dst, cap_left);
                    } else {
                        w = image_emit_subband_split_ctxmag(q, r->w, r->h,
                                                           g_sig_ctx + fidx * SIG_CTX_COUNT,
                                                           (const uint16_t (*)[C2D_ALPHABET])(g_mag_freq_ctx + fidx * MAG_CTX_COUNT),
                                                           (const uint16_t (*)[C2D_ALPHABET + 1])(g_mag_cum_ctx + fidx * MAG_CTX_COUNT),
                                                           local_renorm1, local_renorm_cap,
                                                           local_renorm2, local_renorm_cap,
                                                           local_escape, local_escape_cap,
                                                           dst, cap_left);
                    }
                    if (w == 0) { local_bad = 1; break; }
                    lens[dir_idx] = w;
                    scratch_pos += w;
                    dir_idx++;
                }
                if (local_bad) break;
            }
            if (local_bad) {
                free(scratch); free(lens);
                #pragma omp atomic write
                oom_flag = 1;
            } else {
                tile_scratch[t] = scratch;
                tile_scratch_used[t] = scratch_pos;
                tile_dir_idx[t] = dir_idx;
                tile_body_lens[t] = lens;
            }
        }
        free(local_renorm1); free(local_renorm2); free(local_escape);
    }
    }

    if (oom_flag) {
        for (size_t t = 0; t < n_tiles; t++) {
            free(tile_scratch[t]); free(tile_body_lens[t]);
        }
        free(tile_scratch); free(tile_scratch_used); free(tile_dir_idx); free(tile_body_lens);
        goto oom2;
    }

    // Serial concat phase: write per-tile dir + length-prefix + body.
    for (size_t t = 0; t < n_tiles; t++) {
        size_t tile_start = pos;
        for (size_t i = 0; i < tile_dir_idx[t]; i++) {
            if (pos + 5 > out_cap) {
                for (size_t tt = 0; tt < n_tiles; tt++) {
                    free(tile_scratch[tt]); free(tile_body_lens[tt]);
                }
                free(tile_scratch); free(tile_scratch_used); free(tile_dir_idx); free(tile_body_lens);
                goto oom2;
            }
            pos += leb128_write((uint32_t)tile_body_lens[t][i], out + pos);
        }
        if (pos + tile_scratch_used[t] > out_cap) {
            for (size_t tt = 0; tt < n_tiles; tt++) {
                free(tile_scratch[tt]); free(tile_body_lens[tt]);
            }
            free(tile_scratch); free(tile_scratch_used); free(tile_dir_idx); free(tile_body_lens);
            goto oom2;
        }
        memcpy(out + pos, tile_scratch[t], tile_scratch_used[t]);
        pos += tile_scratch_used[t];

        size_t tile_len = pos - tile_start;
        uint8_t *tp = out + tile_dir_pos + t * 8;
        write_u32(tp,     (uint32_t)tile_start);
        write_u32(tp + 4, (uint32_t)tile_len);
    }
    for (size_t t = 0; t < n_tiles; t++) {
        free(tile_scratch[t]); free(tile_body_lens[t]);
    }
    free(tile_scratch); free(tile_scratch_used); free(tile_dir_idx); free(tile_body_lens);

    free(renorm_scratch); free(renorm_scratch2); free(escape_scratch);
    for (size_t t = 0; t < n_tiles; t++) { free(tiles[t].planes); }
    free(tiles); free(coef_pool); free(g_hist); free(g_freq); free(g_cum);
    free(g_sig); free(g_sig_ctx); free(g_mag_freq); free(g_mag_cum);
    free(g_mag_freq_ctx); free(g_mag_cum_ctx);
    free(dcs); free(scales);
    return pos;

oom2:
    free(renorm_scratch); free(renorm_scratch2); free(escape_scratch);
oom:
    for (size_t t = 0; t < n_tiles; t++) { free(tiles[t].planes); }
    free(tiles); free(coef_pool); free(g_hist);
    free(g_freq); free(g_cum);
    free(g_sig); free(g_sig_ctx); free(g_mag_freq); free(g_mag_cum);
    free(g_mag_freq_ctx); free(g_mag_cum_ctx);
    free(dcs); free(scales);
    return 0;
}

// --- Decode ---

// Decode one subband body (no freq table) using externally-supplied freq+cum
// and write the dequantized coefficients into `band`.
static void image_decode_subband(const uint8_t *in, size_t in_cap,
                                 size_t n, float qstep,
                                 const rans_slot *slot_tbl,
                                 float *band) {
    size_t pos = 0;
    c2d_assert(pos + 1 <= in_cap);
    uint8_t flags = in[pos++];

    if (flags & SB_FLAG_ALL_ZERO) {
        for (size_t i = 0; i < n; i++) band[i] = 0.0f;
        return;
    }

    uint32_t rans_block_size;
    pos += leb128_read(in + pos, in_cap - pos, &rans_block_size);

    c2d_assert(pos + RANS_STATE_BYTES <= in_cap);
    uint8_t final_state[RANS_STATE_BYTES];
    memcpy(final_state, in + pos, RANS_STATE_BYTES); pos += RANS_STATE_BYTES;

    c2d_assert(pos + rans_block_size <= in_cap);
    const uint8_t *renorm = in + pos; pos += rans_block_size;

    uint32_t esc_len = 0;
    const uint8_t *esc_stream = NULL;
    if (flags & SB_FLAG_HAS_ESCAPES) {
        pos += leb128_read(in + pos, in_cap - pos, &esc_len);
        c2d_assert(pos + esc_len <= in_cap);
        esc_stream = in + pos;
    }

    uint32_t xs[RANS_NSTATES];
    rans_states_load(xs, final_state);
    const uint8_t *rp = renorm;
    const uint8_t *rp_end = renorm + rans_block_size;
    size_t esc_pos = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t lane = i & (RANS_NSTATES - 1);
        uint32_t slot = xs[lane] & (RANS_M - 1);
        rans_slot s = slot_tbl[slot];
        xs[lane] = (uint32_t)s.freq * (xs[lane] >> RANS_PROB_BITS) + slot - (uint32_t)s.cum;
        while (xs[lane] < RANS_L) {
            c2d_assert(rp < rp_end);
            xs[lane] = (xs[lane] << 8) | *rp++;
        }
        uint32_t esc_val = 0;
        if (s.sym == C2D_ESCAPE_SYMBOL) {
            esc_pos += leb128_read(esc_stream + esc_pos, esc_len - esc_pos, &esc_val);
        }
        int32_t qv = symbol_to_coef((uint8_t)s.sym, esc_val);
        band[i] = dequantize_one(qv, qstep);
    }
    (void)rp_end;
}

void c2d_image_decode(const uint8_t *in, size_t in_len, void *out) {
    c2d_assert(c2d_image_validate(in, in_len));
    c2d_dtype dt    = (c2d_dtype)in[5];
    uint32_t  nch   = in[6];
    uint32_t  flags = read_u16(in + 7);
    // in[9] reserved
    uint32_t  width  = read_u32(in + 10);
    uint32_t  height = read_u32(in + 14);
    uint32_t  ts     = read_u32(in + 18);
    float     q      = read_f32(in + 22);
    c2d_assert(ts == C2D_TILE_SIDE);
    int do_ycocg = (flags & C2D_FLAG_COLOR_YCOCG) && nch == 3;

    size_t pos = IMG_HDR_FIXED;
    float *dcs    = (float *)calloc(nch, sizeof(float));
    float *scales = (float *)calloc(nch, sizeof(float));
    for (uint32_t c = 0; c < nch; c++) {
        dcs[c]    = read_f32(in + pos); pos += 4;
        scales[c] = read_f32(in + pos); pos += 4;
    }
    float inv_scale[C2D_MAX_CHANNELS];
    for (uint32_t c = 0; c < nch; c++) inv_scale[c] = 1.0f / scales[c];

    int ctx_split = (flags & C2D_FLAG_CTX_SPLIT) != 0;
    int ctx_neigh = ctx_split && (flags & C2D_FLAG_CTX_NEIGH);
    int ctx_mag   = ctx_neigh && (flags & C2D_FLAG_CTX_MAG);
    int bitplane  = (flags & C2D_FLAG_BITPLANE) != 0;
    if (bitplane) { ctx_split = ctx_neigh = ctx_mag = 0; }

    // Read presence bitmap.
    size_t presence_bytes = ((size_t)C2D_N_SUBBANDS * nch + 7) / 8;
    const uint8_t *presence = in + pos;
    pos += presence_bytes;

    // Read shared freq tables and build slot_tbls per present (b, c).
    // For ctx_mag, slot_tbls expands to MAG_CTX_COUNT tables per (b,c).
    size_t slot_tbls_count = (size_t)C2D_N_SUBBANDS * nch * (ctx_mag ? MAG_CTX_COUNT : 1);
    rans_slot *slot_tbls = (rans_slot *)calloc(slot_tbls_count * RANS_M, sizeof(rans_slot));
    sig_freq_pair *sig_pairs = NULL;
    sig_freq_pair *sig_ctx   = NULL;
    if (ctx_split && !ctx_neigh) {
        sig_pairs = (sig_freq_pair *)calloc((size_t)C2D_N_SUBBANDS * nch, sizeof(sig_freq_pair));
    } else if (ctx_neigh) {
        sig_ctx = (sig_freq_pair *)calloc((size_t)C2D_N_SUBBANDS * nch * SIG_CTX_COUNT, sizeof(sig_freq_pair));
    }

    int n_present = 0;
    for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
        for (uint32_t c = 0; c < nch; c++) {
            size_t fidx = (size_t)b * nch + c;
            if (!(presence[fidx / 8] & (1u << (fidx & 7)))) continue;
            n_present++;
            if (bitplane) continue;  // EBCOT has no per-(b,c) freq tables on the wire
            if (ctx_split && !ctx_neigh) {
                uint16_t f0 = read_u16(in + pos); pos += 2;
                sig_pairs[fidx].f0 = f0;
                sig_pairs[fidx].f1 = (uint16_t)(RANS_M - f0);
            } else if (ctx_neigh) {
                for (uint32_t k = 0; k < SIG_CTX_COUNT; k++) {
                    uint16_t f0 = read_u16(in + pos); pos += 2;
                    sig_ctx[fidx * SIG_CTX_COUNT + k].f0 = f0;
                    sig_ctx[fidx * SIG_CTX_COUNT + k].f1 = (uint16_t)(RANS_M - f0);
                }
            }
            // Read 1 mag table (or MAG_CTX_COUNT if ctx_mag).
            uint32_t n_mag_tables = ctx_mag ? MAG_CTX_COUNT : 1;
            for (uint32_t tt = 0; tt < n_mag_tables; tt++) {
                uint16_t freq[C2D_ALPHABET] = {0};
                uint8_t nnz = in[pos++];
                for (uint32_t i = 0; i < nnz; i++) {
                    uint8_t sym = in[pos++];
                    uint16_t f  = read_u16(in + pos); pos += 2;
                    c2d_assert(sym < C2D_ALPHABET);
                    freq[sym] = f;
                }
                rans_slot *tbl = slot_tbls + (fidx * (ctx_mag ? MAG_CTX_COUNT : 1) + tt) * RANS_M;
                uint32_t cum = 0;
                for (uint32_t i = 0; i < C2D_ALPHABET; i++) {
                    uint16_t f = freq[i];
                    if (f == 0) continue;
                    rans_slot s = { .sym = (uint16_t)i, .freq = f, .cum = (uint16_t)cum, ._pad = 0 };
                    for (uint32_t j = 0; j < f; j++) tbl[cum + j] = s;
                    cum += f;
                }
                c2d_assert(cum == RANS_M);
            }
        }
    }

    uint32_t ntx = read_u32(in + pos); pos += 4;
    uint32_t nty = read_u32(in + pos); pos += 4;
    size_t n_tiles = (size_t)ntx * nty;
    const uint8_t *tile_qmul = NULL;
    if (flags & C2D_FLAG_LOW_BPP) {
        tile_qmul = in + pos;
        pos += n_tiles;
    }
    const uint8_t *tile_filter = NULL;
    if (flags & C2D_FLAG_LOW_BPP) {
        tile_filter = in + pos;
        pos += n_tiles;
    }
    size_t tile_dir_pos = pos;
    pos += n_tiles * 8;

    // Per-tile working storage.
    subband_rect rects[C2D_N_SUBBANDS];
    enumerate_subbands(rects);

    #pragma omp parallel
    {
        float dwt_tmp[C2D_TILE_SIDE];
        float *mallat = (float *)aligned_alloc(C2D_ALIGN, sizeof(float) * C2D_PIXELS_PER_TILE);
        float **planes = (float **)calloc(nch, sizeof(float *));
        for (uint32_t c = 0; c < nch; c++)
            planes[c] = (float *)aligned_alloc(C2D_ALIGN, sizeof(float) * C2D_PIXELS_PER_TILE);
        float *band_buf = (float *)aligned_alloc(C2D_ALIGN, sizeof(float) * C2D_PIXELS_PER_TILE);
        void *tile_buf = malloc(c2d_dtype_tile_bytes_n(dt, nch));

    #pragma omp for schedule(static)
    for (size_t t = 0; t < n_tiles; t++) {
        const uint8_t *tp = in + tile_dir_pos + t * 8;
        uint32_t toff = read_u32(tp);
        uint32_t tlen = read_u32(tp + 4);
        const uint8_t *tbody = in + toff;

        size_t tpos = 0;
        // Read length-prefix array (varints) and compute body offsets.
        uint32_t body_lens[C2D_N_SUBBANDS * C2D_MAX_CHANNELS];
        for (int i = 0; i < n_present; i++) {
            uint32_t l;
            tpos += leb128_read(tbody + tpos, tlen - tpos, &l);
            body_lens[i] = l;
        }
        size_t bodies_start = tpos;

        // Zero planes — empty bands stay zero.
        for (uint32_t c = 0; c < nch; c++)
            memset(planes[c], 0, sizeof(float) * C2D_PIXELS_PER_TILE);

        size_t dir_idx = 0;
        size_t body_off = bodies_start;
        float tile_qmul_f = tile_qmul ? qmul_decode(tile_qmul[t]) : 1.0f;
        for (uint32_t b = 0; b < C2D_N_SUBBANDS; b++) {
            const subband_rect *r = &rects[b];
            float qbase = q * tile_qmul_f * subband_baseline_flags(r, flags);
            for (uint32_t c = 0; c < nch; c++) {
                size_t fidx = (size_t)b * nch + c;
                if (!(presence[fidx / 8] & (1u << (fidx & 7)))) continue;
                float step = qbase;
                if (do_ycocg) step *= channel_weight_ycocg(c);
                if (step < 1e-20f) step = 1e-20f;
                uint32_t len = body_lens[dir_idx];
                if (bitplane) {
                    bp_decode_band(tbody + body_off, len, r->w, r->h, r->kind, r->level, step, band_buf);
                } else if (!ctx_split) {
                    image_decode_subband(tbody + body_off, len,
                                         (size_t)r->w * r->h, step,
                                         slot_tbls + fidx * RANS_M, band_buf);
                } else if (!ctx_neigh) {
                    image_decode_subband_split(tbody + body_off, len,
                                               (size_t)r->w * r->h, step,
                                               sig_pairs[fidx],
                                               slot_tbls + fidx * RANS_M, band_buf);
                } else if (!ctx_mag) {
                    image_decode_subband_split_ctx(tbody + body_off, len, r->w, r->h, step,
                                                   sig_ctx + fidx * SIG_CTX_COUNT,
                                                   slot_tbls + fidx * RANS_M, band_buf);
                } else {
                    image_decode_subband_split_ctxmag(tbody + body_off, len, r->w, r->h, step,
                                                     sig_ctx + fidx * SIG_CTX_COUNT,
                                                     slot_tbls + fidx * MAG_CTX_COUNT * RANS_M, band_buf);
                }
                body_off += len;
                dir_idx++;
                for (uint32_t y = 0; y < r->h; y++) {
                    memcpy(planes[c] + (size_t)(r->y + y) * C2D_TILE_SIDE + r->x,
                           band_buf + (size_t)y * r->w,
                           sizeof(float) * r->w);
                }
            }
        }
        (void)tlen;

        // Inverse DWT (full).
        for (uint32_t c = 0; c < nch; c++)
            dwt_inv_partial(planes[c], C2D_TILE_SIDE, C2D_N_DWT_LEVELS, dwt_tmp, mallat);

        // Decoder-side adaptive filter (post-DWT, pre-YCoCg).
        if (tile_filter && tile_filter[t]) {
            for (uint32_t c = 0; c < nch; c++) {
                sigma_filter_plane(planes[c], C2D_TILE_SIDE, tile_filter[t]);
            }
        }

        if (do_ycocg) {
            for (uint32_t y = 0; y < C2D_TILE_SIDE; y++) {
                ycocg_inv(planes[0] + y * C2D_TILE_SIDE,
                          planes[1] + y * C2D_TILE_SIDE,
                          planes[2] + y * C2D_TILE_SIDE,
                          C2D_TILE_SIDE);
            }
        }

        // Re-interleave + denormalize into tile_buf, then scatter.
        if (dt == C2D_DTYPE_U8) {
            uint8_t *tb = (uint8_t *)tile_buf;
            for (uint32_t y = 0; y < C2D_TILE_SIDE; y++) {
                for (uint32_t c = 0; c < nch; c++) {
                    const float *row = planes[c] + y * C2D_TILE_SIDE;
                    float is = inv_scale[c], dc = dcs[c];
                    uint8_t *o = tb + (size_t)y * C2D_TILE_SIDE * nch + c;
                    for (uint32_t x = 0; x < C2D_TILE_SIDE; x++) {
                        float v = row[x] * is + dc + 0.5f;
                        int iv = (int)v;
                        if (v < 0) iv = 0;
                        if (iv > 255) iv = 255;
                        o[x * nch] = (uint8_t)iv;
                    }
                }
            }
        } else {
            for (uint32_t y = 0; y < C2D_TILE_SIDE; y++) {
                for (uint32_t x = 0; x < C2D_TILE_SIDE; x++) {
                    size_t i_out = (size_t)y * C2D_TILE_SIDE + x;
                    for (uint32_t c = 0; c < nch; c++) {
                        float v = planes[c][y * C2D_TILE_SIDE + x] * inv_scale[c] + dcs[c];
                        istore_f32(tile_buf, dt, i_out, c, nch, v);
                    }
                }
            }
        }

        uint32_t tx = (uint32_t)((t % ntx) * C2D_TILE_SIDE);
        uint32_t ty = (uint32_t)((t / ntx) * C2D_TILE_SIDE);
        scatter_tile(out, dt, width, height, nch, tx, ty, tile_buf);
        (void)tlen;
    }
        // End of per-thread tile loop; free per-thread scratch.
        for (uint32_t c = 0; c < nch; c++) free(planes[c]);
        free(planes); free(band_buf); free(tile_buf); free(mallat);
    }  // end #pragma omp parallel

    free(slot_tbls); free(sig_pairs); free(sig_ctx); free(dcs); free(scales);
}
