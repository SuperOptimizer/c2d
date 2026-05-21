// Quick throughput benchmark for c2d.

#include "c2d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    const struct { const char *label; uint32_t nch; uint32_t flags; } cases[] = {
        { "u8 nch=1",       1, 0 },
        { "u8 nch=3",       3, 0 },
        { "u8 nch=3 ycocg", 3, C2D_FLAG_COLOR_YCOCG },
        { "u8 nch=4",       4, 0 },
    };
    int iters = 100;
    for (size_t k = 0; k < sizeof(cases)/sizeof(cases[0]); k++) {
        uint32_t nch = cases[k].nch;
        size_t bytes = c2d_dtype_tile_bytes_n(C2D_DTYPE_U8, nch);
        uint8_t *src = (uint8_t *)malloc(bytes);
        uint8_t *dst = (uint8_t *)malloc(bytes);
        for (size_t i = 0; i < bytes; i++) src[i] = (uint8_t)((i * 7919) ^ (i >> 3));

        size_t cap = c2d_tile_encode_max_size(C2D_DTYPE_U8, nch);
        uint8_t *bs = (uint8_t *)malloc(cap);

        c2d_encoder *enc = c2d_encoder_new();
        c2d_decoder *dec = c2d_decoder_new();
        double t0 = now_sec(); size_t n = 0;
        for (int i = 0; i < iters; i++)
            n = c2d_encoder_tile_encode(enc, src, C2D_DTYPE_U8, nch, 10.0f, cases[k].flags, bs, cap);
        double t1 = now_sec();
        for (int i = 0; i < iters; i++) c2d_decoder_tile_decode(dec, bs, n, dst);
        double t2 = now_sec();
        c2d_encoder_free(enc); c2d_decoder_free(dec);

        double enc_mb = (double)bytes * iters / (t1 - t0) / (1024.0 * 1024.0);
        double dec_mb = (double)bytes * iters / (t2 - t1) / (1024.0 * 1024.0);
        printf("%-18s: encode %.1f MB/s, decode %.1f MB/s, ratio %.2f\n",
               cases[k].label, enc_mb, dec_mb, (double)bytes / (double)n);
        free(src); free(dst); free(bs);
    }

    return 0;
}
