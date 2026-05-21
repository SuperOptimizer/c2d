// c2d - lossy 2D image compression codec
// 2D analogue of c3d (https://github.com/SuperOptimizer/c3d)
//
// Pipeline: normalize -> 5-level 2D CDF 9/7 DWT -> dead-zone quantize ->
//           8-way interleaved rANS (65-symbol alphabet)
//
// Tile geometry: 256x256, 5 DWT levels, 6 LODs, 16 subbands.
// Supported dtypes: u8, u16, u32, s8, s16, s32, f32.

#ifndef C2D_H
#define C2D_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Version / format ------------------------------------------------------
//
// C2D_FORMAT_VERSION is the wire-format version. Bumped only on
// backward-incompatible bitstream changes. v1 is the first stable format;
// future v2+ decoders are expected to remain able to decode v1 streams.
//
// C2D_VERSION_MAJOR/MINOR/PATCH is the library version (semver) and may
// change independently of the format version.

#define C2D_FORMAT_VERSION  1u
#define C2D_VERSION_MAJOR   1u
#define C2D_VERSION_MINOR   0u
#define C2D_VERSION_PATCH   0u
#define C2D_VERSION_STRING  "1.0.0"

const char *c2d_version(void);
uint32_t    c2d_format_version(void);

// ---- Geometry --------------------------------------------------------------

#define C2D_TILE_SIDE       256u
#define C2D_PIXELS_PER_TILE (C2D_TILE_SIDE * C2D_TILE_SIDE)
#define C2D_N_LODS          6u
#define C2D_N_DWT_LEVELS    5u
// 1 LL at coarsest + 3 detail (HL,LH,HH) per level = 1 + 3*5 = 16
#define C2D_N_SUBBANDS      16u
#define C2D_ALIGN           32u

#define C2D_TILE_MAGIC      "C2DT"
#define C2D_IMAGE_MAGIC     "C2DI"

// Encode flags (bitmask passed to encode functions, also stored in bitstream).
// COLOR_YCOCG: apply reversible RGB->YCoCg-R decorrelation prior to DWT.
// Only valid when n_channels == 3. Ignored otherwise.
#define C2D_FLAG_COLOR_YCOCG  (1u << 0)
// Context-split coding (image container only): code significance (q==0 vs !=0)
// and magnitude/sign as two separate rANS streams per subband. Slightly larger
// header overhead. Without neighbor context this is roughly cost-neutral; it
// becomes a clear win when combined with CTX_NEIGH below.
#define C2D_FLAG_CTX_SPLIT    (1u << 1)
// Neighbor-context significance coding (requires CTX_SPLIT). For each coef in
// a subband, the significance bit is coded conditional on a 4-bit context
// formed from the four already-decoded neighbors' significance bits. The
// magnitude stream remains unconditional. Yields ~0.5-1.5 dB at low bpp on
// natural images.
#define C2D_FLAG_CTX_NEIGH    (1u << 2)
// Magnitude-context coding (requires CTX_SPLIT and CTX_NEIGH). The magnitude
// alphabet's distribution depends on a 2-bit context formed from the sum of
// the four already-decoded neighbors' magnitudes (bucketed). Each (b,c) gets
// 4 magnitude freq tables instead of 1. Yields ~0.3-0.5 dB on natural images.
#define C2D_FLAG_CTX_MAG      (1u << 3)
// Bit-plane EBCOT-lite encoding (replaces static rANS entirely for this
// image). Codes coefficients bit-plane by bit-plane using adaptive binary
// arithmetic coding with neighbor contexts. Has no per-(b,c) frequency tables
// on the wire — the AC adapts as it sees data — so header overhead is minimal
// and small-image quality wins. ~+0.5-1.5 dB at low bpp.
#define C2D_FLAG_BITPLANE     (1u << 4)

// Low-bpp perceptual mode (image container only). Bundles three things:
//   1. SSIM-tuned per-subband quantizer matrix (trades small PSNR for SSIM).
//   2. Per-tile activity-adaptive q multiplier (1 byte/tile).
//   3. Decoder-side sigma filter, strength chosen by encoder (1 byte/tile).
// Trades ~0.2 dB PSNR for ~+0.01 SSIM at <1 bpp. Particularly useful on
// photos with smooth gradients and structure. Recommended for ratio>=20.
#define C2D_FLAG_LOW_BPP      (1u << 5)

#define C2D_MAX_CHANNELS    255u

// ---- Dtype -----------------------------------------------------------------

typedef enum c2d_dtype {
    C2D_DTYPE_U8  = 0,
    C2D_DTYPE_U16 = 1,
    C2D_DTYPE_U32 = 2,
    C2D_DTYPE_S8  = 3,
    C2D_DTYPE_S16 = 4,
    C2D_DTYPE_S32 = 5,
    C2D_DTYPE_F32 = 6,
    C2D_DTYPE__COUNT
} c2d_dtype;

size_t c2d_dtype_size(c2d_dtype dt);          // bytes per scalar
size_t c2d_dtype_tile_bytes(c2d_dtype dt);    // 256*256*size, single channel
size_t c2d_dtype_tile_bytes_n(c2d_dtype dt, uint32_t n_channels);

// ---- Tile inspection -------------------------------------------------------

typedef struct c2d_tile_info {
    uint32_t  lod_offsets[C2D_N_LODS];  // byte prefix to decode LOD k
    c2d_dtype dtype;
    uint32_t  n_channels;
    uint32_t  flags;
} c2d_tile_info;

bool   c2d_is_tile(const uint8_t *in, size_t n);
bool   c2d_tile_validate(const uint8_t *in, size_t n);
void   c2d_tile_inspect(const uint8_t *in, size_t n, c2d_tile_info *info);

uint32_t c2d_side_per_lod(uint8_t lod);
size_t   c2d_pixels_per_lod(uint8_t lod);

// ---- Encode max size -------------------------------------------------------

size_t c2d_tile_encode_max_size(c2d_dtype dt, uint32_t n_channels);

// ---- Stateless encode/decode ----------------------------------------------
// `in` points to a 256x256 image of `dtype` with `n_channels` channels,
// pixel-interleaved (e.g. RGBRGBRGB... for 3-channel). n_channels in [1, 255].
// `target_ratio` is uncompressed_bytes / compressed_bytes (e.g. 20.0).
// `flags` is a bitmask of C2D_FLAG_*; pass 0 for defaults.
// Returns number of bytes written to `out`, or 0 on out-of-capacity.

size_t c2d_tile_encode(const void *in, c2d_dtype dt, uint32_t n_channels,
                       float target_ratio, uint32_t flags,
                       uint8_t *out, size_t out_cap);

size_t c2d_tile_encode_at_q(const void *in, c2d_dtype dt, uint32_t n_channels,
                            float q, uint32_t flags,
                            uint8_t *out, size_t out_cap);

// Output buffer must hold n_channels * pixels_per_lod(lod) * dtype_size
// interleaved scalars (same layout as input).
void c2d_tile_decode(const uint8_t *in, size_t in_len, void *out);
void c2d_tile_decode_lod(const uint8_t *in, size_t in_len,
                         uint8_t lod, void *out);

// ---- Encoder / decoder contexts (amortized allocation) --------------------

typedef struct c2d_encoder c2d_encoder;
typedef struct c2d_decoder c2d_decoder;

c2d_encoder *c2d_encoder_new(void);
void         c2d_encoder_free(c2d_encoder *);

size_t c2d_encoder_tile_encode(c2d_encoder *, const void *in, c2d_dtype dt,
                               uint32_t n_channels, float target_ratio,
                               uint32_t flags, uint8_t *out, size_t out_cap);
size_t c2d_encoder_tile_encode_at_q(c2d_encoder *, const void *in, c2d_dtype dt,
                                    uint32_t n_channels, float q,
                                    uint32_t flags, uint8_t *out, size_t out_cap);

c2d_decoder *c2d_decoder_new(void);
void         c2d_decoder_free(c2d_decoder *);

void c2d_decoder_tile_decode(c2d_decoder *, const uint8_t *in, size_t in_len,
                             void *out);
void c2d_decoder_tile_decode_lod(c2d_decoder *, const uint8_t *in, size_t in_len,
                                 uint8_t lod, void *out);

// ---- Image container API --------------------------------------------------
// An image is an arbitrary WxH array of `dtype` pixels with `n_channels`
// channels (interleaved). It is encoded as 256x256 tiles with edge replication
// for tiles that don't align. Per-image globals (normalization, frequency
// models) live in the header so per-tile overhead is small.

typedef struct c2d_image_info {
    uint32_t width;
    uint32_t height;
    c2d_dtype dtype;
    uint32_t n_channels;
    uint32_t flags;
} c2d_image_info;

bool   c2d_is_image(const uint8_t *in, size_t n);
bool   c2d_image_validate(const uint8_t *in, size_t n);
void   c2d_image_inspect(const uint8_t *in, size_t n, c2d_image_info *info);

size_t c2d_image_encode_max_size(uint32_t width, uint32_t height,
                                 c2d_dtype dt, uint32_t n_channels);

// `in` is row-major, channel-interleaved (w*h*n_channels * dtype_size).
// `out` receives the encoded bitstream. `out_cap` should be at least
// c2d_image_encode_max_size(...).
size_t c2d_image_encode(const void *in, uint32_t width, uint32_t height,
                        c2d_dtype dt, uint32_t n_channels,
                        float target_ratio, uint32_t flags,
                        uint8_t *out, size_t out_cap);

// Decodes a full image. `out` must hold w*h*n_channels * dtype_size bytes.
void c2d_image_decode(const uint8_t *in, size_t in_len, void *out);

// ---- Panic hook ------------------------------------------------------------

typedef void (*c2d_panic_fn)(const char *file, int line, const char *msg);
void c2d_set_panic_hook(c2d_panic_fn hook);
void c2d_panic(const char *file, int line, const char *msg);

#define c2d_assert(cond)    do { if (!(cond)) c2d_panic(__FILE__, __LINE__, #cond); } while (0)
#define c2d_invariant(cond) c2d_assert(cond)

#if defined(__GNUC__) || defined(__clang__)
#define c2d_likely(x)   __builtin_expect(!!(x), 1)
#define c2d_unlikely(x) __builtin_expect(!!(x), 0)
#else
#define c2d_likely(x)   (x)
#define c2d_unlikely(x) (x)
#endif

#ifdef __cplusplus
}
#endif

#endif // C2D_H
