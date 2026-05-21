// c2d_bench: load PNG -> 256x256 tile -> encode/decode with c2d at a sweep of
// quality settings -> emit TSV with (codec, image, setting, bpp, psnr_db).
// Also provides a `psnr` subcommand so a shell driver can compute PSNR between
// the original and a decoded competitor (JPEG/JPEG2000).

#include "c2d.h"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Load a PNG as 8-bit, choose channel count from the file (3 for typical Kodak).
static uint8_t *load_png(const char *path, int *w, int *h, int *nch) {
    int n;
    uint8_t *p = stbi_load(path, w, h, &n, 0);
    if (!p) { fprintf(stderr, "stbi_load failed: %s\n", path); return NULL; }
    *nch = n;
    return p;
}

static double psnr_u8(const uint8_t *a, const uint8_t *b, size_t n) {
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= (double)n;
    if (mse < 1e-30) return 200.0;
    return 10.0 * log10((255.0 * 255.0) / mse);
}

// Simple 8x8-block-mean SSIM (no Gaussian window — uses non-overlapping
// 8x8 means). Returns mean SSIM over all blocks of one channel.
// Standard SSIM constants for 8-bit data.
static double ssim_channel(const uint8_t *a, const uint8_t *b,
                           int w, int h, int nch, int chan) {
    const double C1 = (0.01 * 255.0) * (0.01 * 255.0);
    const double C2 = (0.03 * 255.0) * (0.03 * 255.0);
    int nbx = w / 8, nby = h / 8;
    if (nbx == 0 || nby == 0) return 0.0;
    double total = 0.0;
    int blocks = 0;
    for (int by = 0; by < nby; by++) {
        for (int bx = 0; bx < nbx; bx++) {
            double sa = 0, sb = 0;
            for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) {
                size_t i = (size_t)((by * 8 + y) * w + (bx * 8 + x)) * nch + chan;
                sa += a[i]; sb += b[i];
            }
            double ma = sa / 64.0, mb = sb / 64.0;
            double va = 0, vb = 0, cov = 0;
            for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) {
                size_t i = (size_t)((by * 8 + y) * w + (bx * 8 + x)) * nch + chan;
                double da = (double)a[i] - ma, db = (double)b[i] - mb;
                va += da * da; vb += db * db; cov += da * db;
            }
            va /= 63.0; vb /= 63.0; cov /= 63.0;
            double num = (2 * ma * mb + C1) * (2 * cov + C2);
            double den = (ma * ma + mb * mb + C1) * (va + vb + C2);
            total += num / den;
            blocks++;
        }
    }
    return total / blocks;
}

static double ssim_mean(const uint8_t *a, const uint8_t *b, int w, int h, int nch) {
    double sum = 0;
    for (int c = 0; c < nch; c++) sum += ssim_channel(a, b, w, h, nch, c);
    return sum / nch;
}

// Encode entire image via the image container; report total bytes.
static size_t roundtrip_c2d_image(const uint8_t *src, int w, int h, int nch,
                                  float ratio, uint32_t flags,
                                  uint8_t *recon) {
    size_t cap = c2d_image_encode_max_size((uint32_t)w, (uint32_t)h, C2D_DTYPE_U8, (uint32_t)nch);
    uint8_t *bs = (uint8_t *)malloc(cap);
    size_t n = c2d_image_encode(src, (uint32_t)w, (uint32_t)h, C2D_DTYPE_U8,
                                (uint32_t)nch, ratio, flags, bs, cap);
    if (n == 0) { fprintf(stderr, "image encode failed\n"); free(bs); return 0; }
    c2d_image_decode(bs, n, recon);
    free(bs);
    return n;
}

// Tile a (w*h*nch) image into 256x256 blocks, encode each, decode, recompose.
// Returns total compressed bytes; writes reconstruction into `recon`.
// Edge tiles are zero-padded for encode and cropped on output.
static size_t roundtrip_c2d_tiled(const uint8_t *src, int w, int h, int nch,
                                  float ratio, uint32_t flags,
                                  uint8_t *recon) {
    size_t cap = c2d_tile_encode_max_size(C2D_DTYPE_U8, (uint32_t)nch);
    uint8_t *bs = (uint8_t *)malloc(cap);
    uint8_t *tile_in  = (uint8_t *)malloc(C2D_PIXELS_PER_TILE * nch);
    uint8_t *tile_out = (uint8_t *)malloc(C2D_PIXELS_PER_TILE * nch);

    size_t total_bytes = 0;
    for (int ty = 0; ty < h; ty += C2D_TILE_SIDE) {
        for (int tx = 0; tx < w; tx += C2D_TILE_SIDE) {
            memset(tile_in, 0, C2D_PIXELS_PER_TILE * nch);
            int tw = (tx + (int)C2D_TILE_SIDE > w) ? (w - tx) : (int)C2D_TILE_SIDE;
            int th = (ty + (int)C2D_TILE_SIDE > h) ? (h - ty) : (int)C2D_TILE_SIDE;
            // Replicate edge into pad region to avoid spurious high-freq energy.
            for (int y = 0; y < (int)C2D_TILE_SIDE; y++) {
                int sy = y < th ? y : th - 1;
                for (int x = 0; x < (int)C2D_TILE_SIDE; x++) {
                    int sx = x < tw ? x : tw - 1;
                    memcpy(tile_in + (y * (int)C2D_TILE_SIDE + x) * nch,
                           src + ((ty + sy) * w + (tx + sx)) * nch,
                           (size_t)nch);
                }
            }
            size_t n = c2d_tile_encode(tile_in, C2D_DTYPE_U8, (uint32_t)nch,
                                       ratio, flags, bs, cap);
            if (n == 0) { fprintf(stderr, "c2d encode failed\n"); exit(1); }
            total_bytes += n;
            c2d_tile_decode(bs, n, tile_out);
            // Copy back valid region only.
            for (int y = 0; y < th; y++) {
                memcpy(recon + ((ty + y) * w + tx) * nch,
                       tile_out + (y * (int)C2D_TILE_SIDE) * nch,
                       (size_t)tw * nch);
            }
        }
    }
    free(bs); free(tile_in); free(tile_out);
    return total_bytes;
}

static int cmd_c2d_sweep(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: c2d-sweep <png>\n"); return 1; }
    const char *path = argv[1];
    int w, h, nch;
    uint8_t *src = load_png(path, &w, &h, &nch);
    if (!src) return 1;
    size_t npix = (size_t)w * h * nch;
    uint8_t *recon = (uint8_t *)malloc(npix);
    const char *name = strrchr(path, '/'); name = name ? name + 1 : path;

    const float ratios[] = { 2, 3, 5, 8, 12, 20, 30, 40, 50, 60, 80, 120 };
    struct { uint32_t flags; const char *suffix; } img_modes[] = {
        { C2D_FLAG_COLOR_YCOCG,                                            "-ycocg" },
        { C2D_FLAG_COLOR_YCOCG | C2D_FLAG_BITPLANE,                        "-ycocg+bp" },
        { C2D_FLAG_COLOR_YCOCG | C2D_FLAG_BITPLANE | C2D_FLAG_LOW_BPP,     "-ycocg+bp+lo" },
    };
    // tile codec (unified only — tile API doesn't support ctx-split)
    const uint32_t tile_flagsets[] = { 0, C2D_FLAG_COLOR_YCOCG };
    for (size_t fi = 0; fi < sizeof(tile_flagsets)/sizeof(tile_flagsets[0]); fi++) {
        uint32_t flags = tile_flagsets[fi];
        if ((flags & C2D_FLAG_COLOR_YCOCG) && nch != 3) continue;
        const char *codec_tile = (flags & C2D_FLAG_COLOR_YCOCG) ? "c2d-ycocg" : "c2d";
        for (size_t ri = 0; ri < sizeof(ratios)/sizeof(ratios[0]); ri++) {
            size_t cb = roundtrip_c2d_tiled(src, w, h, nch, ratios[ri], flags, recon);
            double p  = psnr_u8(src, recon, npix);
            double s  = ssim_mean(src, recon, w, h, nch);
            double bpp = (double)cb * 8.0 / ((double)w * (double)h);
            printf("%s\t%s\tratio=%.0f\t%zu\t%.4f\t%.3f\t%.5f\n",
                   codec_tile, name, (double)ratios[ri], cb, bpp, p, s);
        }
    }
    // image codec — all flag combos that apply.
    for (size_t mi = 0; mi < sizeof(img_modes)/sizeof(img_modes[0]); mi++) {
        uint32_t flags = img_modes[mi].flags;
        if ((flags & C2D_FLAG_COLOR_YCOCG) && nch != 3) continue;
        char codec[64];
        snprintf(codec, sizeof(codec), "c2di%s", img_modes[mi].suffix);
        for (size_t ri = 0; ri < sizeof(ratios)/sizeof(ratios[0]); ri++) {
            size_t cb = roundtrip_c2d_image(src, w, h, nch, ratios[ri], flags, recon);
            if (cb == 0) continue;
            double p  = psnr_u8(src, recon, npix);
            double s  = ssim_mean(src, recon, w, h, nch);
            double bpp = (double)cb * 8.0 / ((double)w * (double)h);
            printf("%s\t%s\tratio=%.0f\t%zu\t%.4f\t%.3f\t%.5f\n",
                   codec, name, (double)ratios[ri], cb, bpp, p, s);
        }
    }
    free(src); free(recon);
    return 0;
}

static int cmd_psnr(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: psnr <orig> <recon>\n"); return 1; }
    int w1, h1, c1, w2, h2, c2;
    uint8_t *a = load_png(argv[1], &w1, &h1, &c1);
    uint8_t *b = load_png(argv[2], &w2, &h2, &c2);
    if (!a || !b) return 1;
    if (w1 != w2 || h1 != h2 || c1 != c2) {
        fprintf(stderr, "psnr: shape mismatch %dx%dx%d vs %dx%dx%d\n",
                w1,h1,c1, w2,h2,c2);
        return 1;
    }
    double p = psnr_u8(a, b, (size_t)w1 * h1 * c1);
    printf("%.3f\n", p);
    free(a); free(b);
    return 0;
}

static int cmd_ssim(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ssim <orig> <recon>\n"); return 1; }
    int w1, h1, c1, w2, h2, c2;
    uint8_t *a = load_png(argv[1], &w1, &h1, &c1);
    uint8_t *b = load_png(argv[2], &w2, &h2, &c2);
    if (!a || !b) return 1;
    if (w1 != w2 || h1 != h2 || c1 != c2) {
        fprintf(stderr, "ssim: shape mismatch\n"); return 1;
    }
    double s = ssim_mean(a, b, w1, h1, c1);
    printf("%.5f\n", s);
    free(a); free(b);
    return 0;
}

// Convert any stb-readable image to a binary PPM (P6, 8-bit, RGB).
static int cmd_to_ppm(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: to-ppm <in> <out.ppm>\n"); return 1; }
    int w, h, n;
    uint8_t *p = stbi_load(argv[1], &w, &h, &n, 3);  // force RGB
    if (!p) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    FILE *f = fopen(argv[2], "wb");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(p, 1, (size_t)w * h * 3, f);
    fclose(f);
    free(p);
    return 0;
}

// Emit "<width> <height> <channels>" for an image (any stb-readable format).
static int cmd_info(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: info <img>\n"); return 1; }
    int w, h, n;
    if (!stbi_info(argv[1], &w, &h, &n)) { fprintf(stderr, "info failed\n"); return 1; }
    printf("%d %d %d\n", w, h, n);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage:\n"
                        "  c2d_bench c2d-sweep <png>     # sweep c2d ratios on one image, TSV\n"
                        "  c2d_bench psnr <a> <b>        # psnr between two stb-readable images\n"
                        "  c2d_bench to-ppm <in> <out>   # convert any stb-readable image to PPM\n"
                        "  c2d_bench info <img>          # print 'w h nch'\n");
        return 1;
    }
    const char *cmd = argv[1];
    if (!strcmp(cmd, "c2d-sweep")) return cmd_c2d_sweep(argc - 1, argv + 1);
    if (!strcmp(cmd, "psnr"))      return cmd_psnr(argc - 1, argv + 1);
    if (!strcmp(cmd, "ssim"))      return cmd_ssim(argc - 1, argv + 1);
    if (!strcmp(cmd, "to-ppm"))    return cmd_to_ppm(argc - 1, argv + 1);
    if (!strcmp(cmd, "info"))      return cmd_info(argc - 1, argv + 1);
    fprintf(stderr, "unknown command: %s\n", cmd);
    return 1;
}
