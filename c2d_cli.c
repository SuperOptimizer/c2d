// c2d - command-line encoder/decoder for the c2d image codec.
//
// Subcommands:
//   c2d encode <input.png> <output.c2d> [--ratio N] [--low-bpp]
//   c2d decode <input.c2d> <output.png>
//   c2d info <input.c2d>
//   c2d version
//
// Supported input/output: PNG, JPEG (read only), BMP, TGA via stb_image.

#include "c2d.h"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static int usage(const char *prog) {
    fprintf(stderr,
        "c2d %s (format v%u)\n"
        "\n"
        "usage:\n"
        "  %s encode <in.png> <out.c2d> [--ratio N] [--low-bpp]\n"
        "  %s decode <in.c2d> <out.png>\n"
        "  %s info <in.c2d>\n"
        "  %s version\n"
        "\n"
        "options:\n"
        "  --ratio N    target compression ratio (uncompressed/compressed).\n"
        "               default 20. typical 5-100. higher = smaller + lossier.\n"
        "  --low-bpp    enable perceptual mode (recommended for ratio >= 20).\n"
        "               trades a small PSNR loss for better SSIM at low bpp.\n",
        C2D_VERSION_STRING, c2d_format_version(),
        prog, prog, prog, prog);
    return 2;
}

static uint8_t *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); fprintf(stderr, "%s: empty/unreadable\n", path); return NULL; }
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (!buf) { fclose(f); fprintf(stderr, "out of memory\n"); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f); free(buf);
        fprintf(stderr, "read %s: short read\n", path);
        return NULL;
    }
    fclose(f);
    *out_len = (size_t)n;
    return buf;
}

static int write_file(const char *path, const uint8_t *buf, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return -1; }
    if (fwrite(buf, 1, len, f) != len) {
        fclose(f);
        fprintf(stderr, "write %s: short write\n", path);
        return -1;
    }
    fclose(f);
    return 0;
}

static int cmd_encode(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: c2d encode <in.png> <out.c2d> [--ratio N] [--low-bpp]\n");
        return 2;
    }
    const char *in_path = argv[0];
    const char *out_path = argv[1];
    float ratio = 20.0f;
    uint32_t flags = C2D_FLAG_COLOR_YCOCG | C2D_FLAG_BITPLANE;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ratio") && i + 1 < argc) {
            ratio = (float)atof(argv[++i]);
            if (ratio < 1.0f) { fprintf(stderr, "ratio must be >= 1\n"); return 2; }
        } else if (!strcmp(argv[i], "--low-bpp")) {
            flags |= C2D_FLAG_LOW_BPP;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }

    int w, h, nch;
    uint8_t *src = stbi_load(in_path, &w, &h, &nch, 0);
    if (!src) { fprintf(stderr, "load %s: %s\n", in_path, stbi_failure_reason()); return 1; }
    if (nch < 1 || nch > 4) {
        fprintf(stderr, "unsupported channel count: %d\n", nch);
        stbi_image_free(src); return 1;
    }
    if (nch != 3) flags &= ~C2D_FLAG_COLOR_YCOCG;  // YCoCg only for RGB

    size_t cap = c2d_image_encode_max_size((uint32_t)w, (uint32_t)h, C2D_DTYPE_U8, (uint32_t)nch);
    uint8_t *enc = (uint8_t *)malloc(cap);
    if (!enc) { fprintf(stderr, "out of memory\n"); stbi_image_free(src); return 1; }

    size_t enc_size = c2d_image_encode(src, (uint32_t)w, (uint32_t)h, C2D_DTYPE_U8,
                                       (uint32_t)nch, ratio, flags, enc, cap);
    stbi_image_free(src);
    if (enc_size == 0) {
        fprintf(stderr, "encode failed (out of capacity?)\n");
        free(enc); return 1;
    }
    if (write_file(out_path, enc, enc_size) < 0) { free(enc); return 1; }

    double raw_bytes = (double)w * (double)h * (double)nch;
    double bpp = (double)enc_size * 8.0 / ((double)w * (double)h);
    fprintf(stderr, "%dx%dx%d -> %zu bytes (ratio %.2f, %.3f bpp)\n",
            w, h, nch, enc_size, raw_bytes / (double)enc_size, bpp);
    free(enc);
    return 0;
}

static int cmd_decode(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: c2d decode <in.c2d> <out.png>\n");
        return 2;
    }
    const char *in_path = argv[0];
    const char *out_path = argv[1];

    size_t in_len;
    uint8_t *in = read_file(in_path, &in_len);
    if (!in) return 1;
    if (!c2d_is_image(in, in_len)) {
        fprintf(stderr, "%s: not a c2d image\n", in_path);
        free(in); return 1;
    }
    if (!c2d_image_validate(in, in_len)) {
        fprintf(stderr, "%s: invalid c2d image\n", in_path);
        free(in); return 1;
    }

    c2d_image_info info;
    c2d_image_inspect(in, in_len, &info);
    if (info.dtype != C2D_DTYPE_U8) {
        fprintf(stderr, "%s: dtype %d not yet supported by CLI (only u8)\n",
                in_path, (int)info.dtype);
        free(in); return 1;
    }

    size_t out_bytes = (size_t)info.width * (size_t)info.height * (size_t)info.n_channels;
    uint8_t *out = (uint8_t *)malloc(out_bytes);
    if (!out) { fprintf(stderr, "out of memory\n"); free(in); return 1; }
    c2d_image_decode(in, in_len, out);
    free(in);

    // Pick writer based on output extension.
    const char *ext = strrchr(out_path, '.');
    int wrote = 0;
    if (ext && (!strcmp(ext, ".png") || !strcmp(ext, ".PNG"))) {
        wrote = stbi_write_png(out_path, (int)info.width, (int)info.height,
                               (int)info.n_channels, out,
                               (int)info.width * (int)info.n_channels);
    } else if (ext && (!strcmp(ext, ".bmp") || !strcmp(ext, ".BMP"))) {
        wrote = stbi_write_bmp(out_path, (int)info.width, (int)info.height,
                               (int)info.n_channels, out);
    } else if (ext && (!strcmp(ext, ".tga") || !strcmp(ext, ".TGA"))) {
        wrote = stbi_write_tga(out_path, (int)info.width, (int)info.height,
                               (int)info.n_channels, out);
    } else {
        fprintf(stderr, "unsupported output extension (use .png/.bmp/.tga)\n");
        free(out); return 1;
    }
    free(out);
    if (!wrote) { fprintf(stderr, "write %s failed\n", out_path); return 1; }
    return 0;
}

static int cmd_info(int argc, char **argv) {
    if (argc < 1) { fprintf(stderr, "usage: c2d info <in.c2d>\n"); return 2; }
    size_t in_len;
    uint8_t *in = read_file(argv[0], &in_len);
    if (!in) return 1;
    if (!c2d_is_image(in, in_len)) {
        fprintf(stderr, "%s: not a c2d image\n", argv[0]);
        free(in); return 1;
    }
    if (!c2d_image_validate(in, in_len)) {
        fprintf(stderr, "%s: invalid c2d image\n", argv[0]);
        free(in); return 1;
    }
    c2d_image_info info;
    c2d_image_inspect(in, in_len, &info);
    static const char *dtype_names[] = {
        "u8", "u16", "u32", "s8", "s16", "s32", "f32"
    };
    const char *dt = (info.dtype < 7) ? dtype_names[info.dtype] : "?";
    printf("file:        %s\n", argv[0]);
    printf("format:      c2d image v%u\n", c2d_format_version());
    printf("dimensions:  %u x %u\n", info.width, info.height);
    printf("channels:    %u\n", info.n_channels);
    printf("dtype:       %s\n", dt);
    printf("bytes:       %zu\n", in_len);
    printf("bpp:         %.3f\n",
           (double)in_len * 8.0 / ((double)info.width * (double)info.height));
    printf("flags:       0x%04x", info.flags);
    if (info.flags & C2D_FLAG_COLOR_YCOCG) printf(" COLOR_YCOCG");
    if (info.flags & C2D_FLAG_CTX_SPLIT)   printf(" CTX_SPLIT");
    if (info.flags & C2D_FLAG_CTX_NEIGH)   printf(" CTX_NEIGH");
    if (info.flags & C2D_FLAG_CTX_MAG)     printf(" CTX_MAG");
    if (info.flags & C2D_FLAG_BITPLANE)    printf(" BITPLANE");
    if (info.flags & C2D_FLAG_LOW_BPP)     printf(" LOW_BPP");
    printf("\n");
    free(in);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return usage(argv[0]);
    const char *sub = argv[1];
    if (!strcmp(sub, "encode")) return cmd_encode(argc - 2, argv + 2);
    if (!strcmp(sub, "decode")) return cmd_decode(argc - 2, argv + 2);
    if (!strcmp(sub, "info"))   return cmd_info(argc - 2, argv + 2);
    if (!strcmp(sub, "version") || !strcmp(sub, "--version") || !strcmp(sub, "-V")) {
        printf("c2d %s (format v%u)\n", C2D_VERSION_STRING, c2d_format_version());
        return 0;
    }
    if (!strcmp(sub, "help") || !strcmp(sub, "--help") || !strcmp(sub, "-h")) {
        return usage(argv[0]);
    }
    fprintf(stderr, "unknown subcommand: %s\n", sub);
    return usage(argv[0]);
}
