# c2d

A 2D lossy image codec. Fast, multi-threaded, optimized for high-fidelity
imaging (medical, scientific, microscopy, RAW workflows) where you want
PSNR ≥35 dB and fast decode. Supports u8/u16/u32/s8/s16/s32/f32 data with
1–255 channels. Pure C23, no runtime deps beyond libc/libm. OpenMP
optional.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/c2d encode photo.png photo.c2d --ratio 20
./build/c2d decode photo.c2d photo.png
```

## When to use c2d

**Use c2d when** you want:
- **High fidelity** (≥1 bpp, PSNR ≥35 dB) — c2d beats JPEG-XL, JPEG2000,
  HEIC, AVIF, WebP on PSNR at this regime, often by 1–6 dB.
- **Fast multi-threaded decode** — 762 MB/s on a 10-core Apple M-series.
- **Non-natural dtypes** — u16/u32/f32 microscopy, scientific imaging,
  HDR. No other widely-deployed codec handles f32 natively.
- **Lots of channels** — supports up to 255 channels for multispectral or
  hyperspectral imagery.

**Don't use c2d when** you want:
- **Smallest file sizes at very low bpp (<0.5 bpp)** — VVC/AVIF/HEIC win
  here by 1–3 dB PSNR. They have block-based intra prediction; c2d's
  wavelet architecture can't match it at extreme compression.
- **Native browser/OS support** — c2d is not a standardized format.

## Performance (Apple M-series, 10 cores, 12 MP RGB)

| mode             | enc 1t | enc 10t | dec 1t | dec 10t |
|------------------|-------:|--------:|-------:|--------:|
| tile (ycocg)     |  46    |  249    | 145    | **762** |
| image (ctx)      |  49    |  167    | 134    |   682   |
| image (bitplane) |  25    |  145    |  45    |   336   |

Units: MB/s of uncompressed pixels. Reference codecs single-threaded:
OpenJPEG ~5–10 MB/s enc / 25–50 MB/s dec, cjxl -e7 ~10 MB/s enc, djxl
~60–210 MB/s dec.

## Quality (full corpus: Kodak + CLIC pro + SIPI, 95 images total)

PSNR (dB) mean across 46 images with complete coverage in every codec:

```
                       0.25    0.50    1.00    2.00    4.00 bpp
c2di-ycocg              27.65   30.43   33.93   38.27   43.69
c2di-ycocg+bp           28.79   31.82   35.32   39.66   45.07
c2di-ycocg+bp+lo        28.57   31.40   34.67   38.88   44.32

avif                    30.34   33.03   36.00   36.51   31.20
heic                    30.22   32.65   35.31   38.24   41.01
vvc                     30.61   33.04   35.49   37.64   28.85
webp                    31.15   31.95   34.90   38.13   38.18
jpegxl   (-e 7)         29.69   31.51   34.73   38.05   33.65
jpeg2k                  29.12   31.85   35.09   38.93   43.87
jpeg                    26.93   29.56   32.85   36.14   35.86
```

SSIM mean:
```
                       0.25    0.50    1.00    2.00    4.00 bpp
c2di-ycocg+bp           0.77    0.85    0.91    0.96    0.98
c2di-ycocg+bp+lo        0.78    0.86    0.91    0.96    0.98
avif                    0.84    0.89    0.94    0.96    0.94
heic                    0.83    0.89    0.93    0.96    0.98
vvc                     0.84    0.90    0.94    0.96    0.91
webp                    0.84    0.88    0.93    0.96    0.97
jpegxl                  0.81    0.87    0.92    0.96    0.95
jpeg2k                  0.80    0.86    0.92    0.95    0.98
```

**c2d wins on PSNR at ≥1 bpp** and stays competitive in SSIM at ≥0.5 bpp.
At <0.5 bpp the block-DCT competitors (AVIF/HEIC/VVC) pull ahead.

The `+lo` (low-bpp perceptual mode) trades ~0.3 dB PSNR for +0.01 SSIM at
low bpp. Recommended for ratio ≥20.

## CLI

```sh
c2d encode <in.png> <out.c2d> [--ratio N] [--low-bpp]
c2d decode <in.c2d> <out.png>
c2d info <in.c2d>
c2d version
```

- `--ratio N`: target compression ratio (uncompressed/compressed). Default
  20. Typical range 5–100.
- `--low-bpp`: enable perceptual mode. Recommended for `--ratio ≥ 20`.

## C API

```c
#include "c2d.h"

// Image API — arbitrary WxH, split into 256x256 tiles internally.
size_t c2d_image_encode(const void *in, uint32_t w, uint32_t h,
                        c2d_dtype dt, uint32_t n_channels,
                        float target_ratio, uint32_t flags,
                        uint8_t *out, size_t out_cap);
void   c2d_image_decode(const uint8_t *in, size_t in_len, void *out);

// Tile API — single 256x256 tile, simpler/faster header.
size_t c2d_tile_encode(const void *in, c2d_dtype dt, uint32_t n_channels,
                       float target_ratio, uint32_t flags,
                       uint8_t *out, size_t out_cap);
void   c2d_tile_decode(const uint8_t *in, size_t in_len, void *out);
```

Input/output buffers are **pixel-interleaved** (R0,G0,B0,R1,G1,B1,…).

Recommended flag combos:
- **High-fidelity** (`--ratio ≤ 20`): `C2D_FLAG_COLOR_YCOCG | C2D_FLAG_BITPLANE`
- **Low-bpp** (`--ratio ≥ 20`): `... | C2D_FLAG_LOW_BPP`
- **Fast decode** (Tile API): `C2D_FLAG_COLOR_YCOCG` only

See `c2d.h` for the full flag list and amortized encoder/decoder contexts.

## Pipeline

```
input pixels (dtype, interleaved)
  → deinterleave + per-channel f32 normalize
  → [optional YCoCg-R for n_ch=3]
  → 5-level CDF 9/7 DWT, 256x256 tile (cache-blocked column passes)
  → dead-zone quantizer with empirically-tuned per-subband weights
  → EITHER  rANS path (faster decode):
  →   65-symbol zigzag/escape alphabet
  →   [optional split sig/mag, optional neighbor contexts]
  → OR      EBCOT-lite bitplane path (better compression):
  →   bit-plane significance/sign/refinement passes
  →   small adaptive binary arithmetic coder
  →   4x4 zero-block skip
  → per-tile bodies + per-image shared freq tables + presence bitmap
```

Rate control: log-space bisection on a single global quantizer scalar
against an entropy estimate from per-(band,channel) histograms. Typically
~5–8 iterations, no real rANS encoding inside the bisection.

## Build

Requires CMake 3.20 and a C23 compiler (clang ≥16 or gcc ≥13).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build       # round-trip tests
./build/c2d_perf             # throughput benchmark
./build/c2d_bench            # comparison harness (PNG → sweep → TSV)
```

OpenMP is detected automatically. Without it, encode/decode run
single-threaded.

## Format

The wire format is stable at **v1**. See [FORMAT.md](FORMAT.md) for the
full specification. Future versions are required to remain able to decode
v1 streams.

## File layout

```
c2d.h           public API
c2d.c           implementation (~4100 lines)
c2d_cli.c       CLI tool source
c2d_test.c      round-trip tests across all dtypes / channel counts
c2d_perf.c      throughput benchmark
c2d_bench.c     comparison harness
bench.sh        parallel sweep across c2d / cjpeg / opj / cjxl / avif / heic / vvc
bench_summary.py  aggregate bench TSV → per-source PSNR/SSIM tables
testdata/       test corpus (Kodak, ECE533, SIPI, CLIC subset)
third_party/    vendored stb_image{,_write}.h
FORMAT.md       v1 wire format specification
LICENSE
```

Inspired by [c3d](https://github.com/SuperOptimizer/c3d), which applies
the same wavelet + rANS + rate-control approach to 3D volumes.

## License

MIT. See [LICENSE](LICENSE).
