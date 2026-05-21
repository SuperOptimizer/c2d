// Round-trip + PSNR sanity tests for c2d, across all dtypes and channel counts.

#include "c2d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 0xc2dc2dc2u;
static uint32_t xrand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// Generate a "natural-ish" per-channel image with a channel-correlated offset.
static void gen_image_f32(float *out, float lo, float hi, uint32_t c, uint32_t nch) {
    float chan_bias = (nch > 1) ? ((float)c / (float)(nch - 1) - 0.5f) * (hi - lo) * 0.2f
                                : 0.0f;
    float phase_x = 0.3f * c;
    float phase_y = 0.7f * c;
    for (uint32_t y = 0; y < C2D_TILE_SIDE; y++) {
        for (uint32_t x = 0; x < C2D_TILE_SIDE; x++) {
            float fx = (float)x / (float)C2D_TILE_SIDE;
            float fy = (float)y / (float)C2D_TILE_SIDE;
            float grad = lo + (hi - lo) * (0.5f * (fx + fy)) + chan_bias;
            float wave = 10.0f * sinf(fx * 12.0f + phase_x)
                                 * cosf(fy * 9.0f + phase_y);
            float noise = ((float)(xrand() & 0xff) / 255.0f - 0.5f) * 4.0f;
            out[y * C2D_TILE_SIDE + x] = grad + wave + noise;
        }
    }
}

static void store_interleaved(const float *plane, c2d_dtype dt,
                              uint32_t c, uint32_t nch, void *out) {
    for (size_t i = 0; i < C2D_PIXELS_PER_TILE; i++) {
        float v = plane[i];
        size_t j = i * nch + c;
        switch (dt) {
            case C2D_DTYPE_U8:  { if (v<0)v=0; if (v>255)v=255; ((uint8_t  *)out)[j] = (uint8_t)(v+0.5f); break; }
            case C2D_DTYPE_U16: { if (v<0)v=0; if (v>65535)v=65535; ((uint16_t *)out)[j] = (uint16_t)(v+0.5f); break; }
            case C2D_DTYPE_U32: { if (v<0)v=0; ((uint32_t *)out)[j] = (uint32_t)(v+0.5f); break; }
            case C2D_DTYPE_S8:  { if (v<-128)v=-128; if (v>127)v=127; ((int8_t  *)out)[j] = (int8_t)(v+(v>=0?0.5f:-0.5f)); break; }
            case C2D_DTYPE_S16: { if (v<-32768)v=-32768; if (v>32767)v=32767; ((int16_t *)out)[j] = (int16_t)(v+(v>=0?0.5f:-0.5f)); break; }
            case C2D_DTYPE_S32: { ((int32_t *)out)[j] = (int32_t)(v+(v>=0?0.5f:-0.5f)); break; }
            case C2D_DTYPE_F32: { ((float   *)out)[j] = v; break; }
            default: abort();
        }
    }
}

static double load_d(const void *p, c2d_dtype dt, size_t i) {
    switch (dt) {
        case C2D_DTYPE_U8:  return (double)((const uint8_t  *)p)[i];
        case C2D_DTYPE_U16: return (double)((const uint16_t *)p)[i];
        case C2D_DTYPE_U32: return (double)((const uint32_t *)p)[i];
        case C2D_DTYPE_S8:  return (double)((const int8_t   *)p)[i];
        case C2D_DTYPE_S16: return (double)((const int16_t  *)p)[i];
        case C2D_DTYPE_S32: return (double)((const int32_t  *)p)[i];
        case C2D_DTYPE_F32: return (double)((const float    *)p)[i];
        default: abort();
    }
}

static double psnr_db(const void *a, const void *b, c2d_dtype dt,
                      size_t n_scalars, double peak) {
    double mse = 0.0;
    for (size_t i = 0; i < n_scalars; i++) {
        double d = load_d(a, dt, i) - load_d(b, dt, i);
        mse += d * d;
    }
    mse /= (double)n_scalars;
    if (mse < 1e-30) return 200.0;
    return 10.0 * log10((peak * peak) / mse);
}

static const char *dt_name(c2d_dtype dt) {
    switch (dt) {
        case C2D_DTYPE_U8:  return "u8";
        case C2D_DTYPE_U16: return "u16";
        case C2D_DTYPE_U32: return "u32";
        case C2D_DTYPE_S8:  return "s8";
        case C2D_DTYPE_S16: return "s16";
        case C2D_DTYPE_S32: return "s32";
        case C2D_DTYPE_F32: return "f32";
        default: return "?";
    }
}

static int test_one(c2d_dtype dt, uint32_t nch, uint32_t flags,
                    float ratio, double min_psnr) {
    double peak = 255.0; float lo = 0, hi = 0;
    switch (dt) {
        case C2D_DTYPE_U8:  lo=10;     hi=240;     peak=255.0;     break;
        case C2D_DTYPE_U16: lo=1000;   hi=50000;   peak=65535.0;   break;
        case C2D_DTYPE_U32: lo=100000; hi=4000000; peak=4000000.0; break;
        case C2D_DTYPE_S8:  lo=-100;   hi=100;     peak=255.0;     break;
        case C2D_DTYPE_S16: lo=-20000; hi=20000;   peak=65535.0;   break;
        case C2D_DTYPE_S32: lo=-1000000; hi=1000000; peak=2000000.0; break;
        case C2D_DTYPE_F32: lo=-1000;  hi=1000;    peak=2000.0;    break;
        default: abort();
    }

    size_t bytes = c2d_dtype_tile_bytes_n(dt, nch);
    void *src = malloc(bytes);
    void *dst = malloc(bytes);

    float *plane = (float *)malloc(sizeof(float) * C2D_PIXELS_PER_TILE);
    for (uint32_t c = 0; c < nch; c++) {
        gen_image_f32(plane, lo, hi, c, nch);
        store_interleaved(plane, dt, c, nch, src);
    }
    free(plane);

    size_t cap = c2d_tile_encode_max_size(dt, nch);
    uint8_t *bs = (uint8_t *)malloc(cap);

    size_t n = c2d_tile_encode(src, dt, nch, ratio, flags, bs, cap);
    if (n == 0) { fprintf(stderr, "encode failed dt=%s nch=%u\n", dt_name(dt), nch); return 1; }
    if (!c2d_tile_validate(bs, n)) { fprintf(stderr, "validate failed dt=%s nch=%u\n", dt_name(dt), nch); return 1; }

    c2d_tile_decode(bs, n, dst);

    double p = psnr_db(src, dst, dt, C2D_PIXELS_PER_TILE * nch, peak);
    double actual_ratio = (double)bytes / (double)n;
    const char *fl = (flags & C2D_FLAG_COLOR_YCOCG) ? " ycocg" : "";
    printf("  %-3s nch=%u%-6s  ratio=%.2f (t %.1f)  PSNR=%.2f dB  bytes=%zu\n",
           dt_name(dt), nch, fl, actual_ratio, (double)ratio, p, n);

    int ok = (p >= min_psnr);
    if (!ok) fprintf(stderr, "  -> FAIL: PSNR %.2f < %.2f\n", p, min_psnr);

    // LOD smoke.
    void *lod = malloc(8u * 8u * c2d_dtype_size(dt) * nch);
    c2d_tile_decode_lod(bs, n, 5, lod);
    free(lod);

    free(src); free(dst); free(bs);
    return ok ? 0 : 1;
}

int main(void) {
    int fail = 0;
    printf("c2d round-trip tests\n");

    // Grayscale across dtypes.
    printf("\ngrayscale (nch=1):\n");
    fail += test_one(C2D_DTYPE_U8,  1, 0, 10.0f, 30.0);
    fail += test_one(C2D_DTYPE_U16, 1, 0, 10.0f, 30.0);
    fail += test_one(C2D_DTYPE_U32, 1, 0, 10.0f, 30.0);
    fail += test_one(C2D_DTYPE_S8,  1, 0, 10.0f, 30.0);
    fail += test_one(C2D_DTYPE_S16, 1, 0, 10.0f, 30.0);
    fail += test_one(C2D_DTYPE_S32, 1, 0, 10.0f, 30.0);
    fail += test_one(C2D_DTYPE_F32, 1, 0, 10.0f, 30.0);

    // Multichannel.
    printf("\nmultichannel:\n");
    fail += test_one(C2D_DTYPE_U8, 3, 0,                       10.0f, 30.0);
    fail += test_one(C2D_DTYPE_U8, 3, C2D_FLAG_COLOR_YCOCG,    10.0f, 30.0);
    fail += test_one(C2D_DTYPE_U8, 4, 0,                       10.0f, 30.0);
    fail += test_one(C2D_DTYPE_U16, 3, C2D_FLAG_COLOR_YCOCG,   10.0f, 30.0);

    // High-quality regime.
    printf("\nhigh-quality (ratio 3.0):\n");
    fail += test_one(C2D_DTYPE_U8, 1, 0,                       3.0f, 40.0);
    fail += test_one(C2D_DTYPE_U8, 3, C2D_FLAG_COLOR_YCOCG,    3.0f, 40.0);

    // Image container: multi-tile round-trip.
    printf("\nimage container (multi-tile):\n");
    {
        uint32_t W = 600, H = 400;     // non-aligned, exercises edge replication
        uint32_t nch = 3;
        size_t bytes = (size_t)W * H * nch;
        uint8_t *src = (uint8_t *)malloc(bytes);
        uint8_t *dst = (uint8_t *)malloc(bytes);
        for (uint32_t y = 0; y < H; y++) for (uint32_t x = 0; x < W; x++) {
            src[(y*W + x)*3 + 0] = (uint8_t)(x ^ y);
            src[(y*W + x)*3 + 1] = (uint8_t)(x + y);
            src[(y*W + x)*3 + 2] = (uint8_t)((x*y) ^ (x+y));
        }
        size_t cap = c2d_image_encode_max_size(W, H, C2D_DTYPE_U8, nch);
        uint8_t *bs = (uint8_t *)malloc(cap);
        size_t n = c2d_image_encode(src, W, H, C2D_DTYPE_U8, nch, 10.0f,
                                    C2D_FLAG_COLOR_YCOCG, bs, cap);
        if (n == 0) { fprintf(stderr, "image encode failed\n"); fail++; }
        else if (!c2d_image_validate(bs, n)) { fprintf(stderr, "image validate failed\n"); fail++; }
        else {
            c2d_image_decode(bs, n, dst);
            double p = psnr_db(src, dst, C2D_DTYPE_U8, bytes, 255.0);
            double ratio = (double)bytes / (double)n;
            printf("  600x400 nch=3 ycocg: ratio=%.2f PSNR=%.2f dB bytes=%zu\n", ratio, p, n);
            // Source is XOR-noise (`x ^ y` etc) — essentially incompressible at
            // 10:1, so ~18 dB is the expected floor for this pattern. Lower than
            // that signals a real regression.
            if (p < 15.0) { fprintf(stderr, "image PSNR too low\n"); fail++; }
        }
        free(src); free(dst); free(bs);
    }

    if (fail) { fprintf(stderr, "\nFAILED: %d case(s)\n", fail); return 1; }
    printf("\nOK\n");
    return 0;
}
