# c2d wire format v1

This document specifies the c2d bitstream format. Version 1 is stable;
future versions must remain able to decode v1 streams.

All multi-byte integers are **little-endian**. Floats are IEEE 754
binary32, also little-endian.

Two container shapes are defined:
- **Tile** (magic `C2DT`) — a single 256×256 image, self-contained header.
- **Image** (magic `C2DI`) — arbitrary WxH image, split into 256×256 tiles
  with shared per-(band,channel) frequency tables.

The Image container is the primary publishing format; Tile is mainly for
streaming/embedding individual tiles inside larger formats.

## Common building blocks

### LEB128 varint

Unsigned 32-bit LEB128. Low 7 bits of each byte hold value, MSB is
continuation. Up to 5 bytes for u32.

### Fixed geometry

- `tile_side` = 256 (fixed in v1)
- `n_lods` = 6 (LOD 0 = full 256×256, LOD k = 256/2^k × 256/2^k)
- `n_dwt_levels` = 5
- `n_subbands` = 16 (1 LL at level 5 + 3 detail bands at each of levels 5..1)

Subband enumeration order (used everywhere indices appear):
```
b=0:  LL  at level 5 (8x8)
b=1:  HL  at level 5      b=2:  LH at level 5      b=3:  HH at level 5
b=4:  HL  at level 4      b=5:  LH at level 4      b=6:  HH at level 4
b=7:  HL  at level 3      b=8:  LH at level 3      b=9:  HH at level 3
b=10: HL  at level 2      b=11: LH at level 2      b=12: HH at level 2
b=13: HL  at level 1      b=14: LH at level 1      b=15: HH at level 1
```
where level k subbands are (256/2^k) × (256/2^k).

### Wavelet

The forward DWT is the 9/7 irreversible Cohen-Daubechies-Feauveau
biorthogonal wavelet (JPEG2000 Annex F.4.8.1 lifting form), applied
separably 5 times with whole-sample symmetric boundary extension. The
inverse DWT is the exact inverse and operates partial-depth for LOD decode.

### Quantizer

Uniform dead-zone scalar quantizer per JPEG2000:
```
forward:  q = sign(x) * floor(|x| / step)        (dead-zone width = step)
inverse:  x_hat = sign(q) * (|q| + 0.375) * step for q != 0
```
The +0.375 reconstruction bias is the Laplacian-optimal centroid offset.
The per-subband step is `q_global * subband_weight(level, kind, flags)`.

For decoders: `subband_weight` is the deterministic function defined in
`subband_baseline_flags()` in c2d.c. The same function must be implemented
identically on both sides. The PSNR-tuned weights are:

```
LL: 0.12
detail band (level L, kind in {HL, LH, HH}):
   w = 2^(5-L)
   b = w^0.9
   if HH: b *= 2.3
```

With `C2D_FLAG_LOW_BPP` (perceptual mode), the weights change to:
```
LL: 0.10
detail band:
   w = 2^(5-L)
   b = w^0.92
   if level == 1:                b *= 2.0
   else if level in {3, 4}:      b *= 0.80
   if HH:                        b *= 2.5
```

For YCoCg images, the chroma steps are scaled by an extra factor of 2 (the
Co and Cg planes are coded coarser than Y). The decoder must apply the
same scaling.

### Coefficient coding

Two coding paths are defined; the bitstream flag `BITPLANE` selects.

**Path A — rANS** (no `BITPLANE` flag): see `c2d.c` for the precise
interleaved-rANS implementation. Coefficients are mapped through
zigzag+escape into a 65-symbol alphabet, optionally split into
significance / magnitude streams and conditioned on neighbor contexts
(flags `CTX_SPLIT`, `CTX_NEIGH`, `CTX_MAG`). Frequency tables are coded in
the header (Image container) or per-subband (Tile container).

**Path B — EBCOT-lite bitplane AC** (`BITPLANE` flag set): bit-plane
significance + sign + refinement passes coded with a small adaptive binary
arithmetic coder. Recommended over Path A; +0.5-1.5 dB at low bpp with no
explicit frequency tables. See `bp_encode_band` / `bp_decode_band` for the
precise pass order, context-model layout, and 4×4 zero-skip predicate.

The format reserves the low bit of each EBCOT band's leading K-varint for
a future truncation flag; v1 encoders write 0, v1 decoders must skip the
extra varint if the bit is 1 (treating it as truncation).

## Image container (`C2DI`)

```
offset  size  field
------  ----  -----
 0      4     magic        = "C2DI"  (0x43 0x32 0x44 0x49)
 4      1     version      = 1
 5      1     dtype        (see "Dtypes" below)
 6      1     n_channels   1..255
 7      2     flags        u16 little-endian (see "Flags" below)
 9      1     reserved     = 0
10      4     width        u32 pixels
14      4     height       u32 pixels
18      4     tile_side    u32 = 256 in v1
22      4     q            f32 — global quantization step
26      8*nch per-channel  (dc f32, scale f32) * n_channels
                           (decoded pixel = plane * (1/scale) + dc)
...     var   presence     ceil(n_subbands * n_channels / 8) bytes
                           bit (b*nch + c) set iff that (band, channel)
                           contributes at least one non-zero coefficient.

...     var   freq tables  for each (b, c) marked present in `presence`,
                           the shared frequency table(s). Layout depends
                           on flags:
                             default rANS:     1 mag table
                             +CTX_SPLIT:       1 sig pair (u16) + 1 mag table
                             +CTX_NEIGH:       15 sig pairs + 1 mag table
                             +CTX_MAG:         15 sig pairs + 4 mag tables
                           Each mag table: u8 n_entries + (u8 sym, u16 freq) * n_entries.
                           BITPLANE mode: no freq tables (skipped entirely).

...     4     n_tiles_x    u32
...     4     n_tiles_y    u32
                           Tile (tx, ty) covers pixels [tx*256, (tx+1)*256)
                           × [ty*256, (ty+1)*256). Right/bottom edges
                           replicate boundary pixels.

...     ntx*nty  qmul      (only if LOW_BPP flag set)
                           Per-tile q multiplier index, u8 each.
                           multiplier = 2^((idx-128)/64), range ~[0.25, 4.0].

...     ntx*nty  filter    (only if LOW_BPP flag set)
                           Per-tile decoder filter strength, u8 each.
                           0 = off, 1..15 = sigma filter strength.
                           Sigma in pixel units: 1 + 2*(strength-1).

...     ntx*nty*8  dir     tile directory: (offset u32, length u32) per tile.
                           Offsets are absolute file offsets to tile body.

...     var   tile bodies  see "Tile body" below.
```

### Tile body

Tiles are encoded in row-major order: `tile_id = ty * n_tiles_x + tx`.

```
field
-----
length-prefix array:
  for i in 0..n_present (where n_present = popcount of presence bitmap):
    length varint (bytes in the i-th present subband body)

subband bodies, concatenated, in the order of (b, c) with bit set in
`presence`. Body content depends on coding path:

  BITPLANE (Path B):
    K_tag varint  (low bit reserved; high bits = K = max bit-plane)
    AC bitstream (raw — no internal length prefix)

  default rANS (Path A):
    flags u8       (bit 0 = has_escapes, bit 1 = all_zero)
    if !all_zero:
      rans_size varint
      final_state RANS_STATE_BYTES (=32) bytes
      renorm stream [rans_size]
      if has_escapes: esc_size varint + esc_stream

  CTX_SPLIT variants:
    similar to rANS but with split sig/mag streams; see source.
```

## Tile container (`C2DT`)

Self-contained single 256×256 tile.

```
offset  size  field
------  ----  -----
 0      4     magic        = "C2DT"
 4      1     version      = 1
 5      1     dtype
 6      1     n_channels
 7      1     flags        u8 (note: only 1 byte; tile container does not
                            support LOW_BPP / VAR_Q / DEC_FILTER which
                            require image-level state).
 8      4     q            f32
12      4*6   lod_offsets  u32 each — byte length to decode LOD k
36      8*nch per-channel  (dc f32, scale f32) * n_channels

...     var   per-(band,channel) directory
                            16 * n_channels * (u32 offset, u32 length)

...     var   per-(band,channel) frequency table (always inline for Tile)

...     var   subband bodies
```

The Tile container does **not** support LOW_BPP — that flag requires
per-tile state shared across an image and only makes sense in the Image
container.

## Dtypes

```
0  U8
1  U16
2  U32
3  S8
4  S16
5  S32
6  F32
```

All dtypes are supported by encode/decode; for natural images use U8.
The pipeline normalizes to f32 internally regardless of dtype.

## Flags

Bitmask of `C2D_FLAG_*` constants (see c2d.h):

```
bit 0  COLOR_YCOCG    Reversible RGB->YCoCg-R color transform. Only valid
                      when n_channels == 3.
bit 1  CTX_SPLIT      Split sig/mag entropy streams. (Image only.)
bit 2  CTX_NEIGH      Neighbor context for sig stream. Requires CTX_SPLIT.
bit 3  CTX_MAG        Magnitude contexts. Requires CTX_NEIGH.
bit 4  BITPLANE       Use bit-plane EBCOT-lite coder instead of rANS.
                      Replaces all CTX_* flags. Recommended.
bit 5  LOW_BPP        Perceptual low-bpp mode. SSIM-tuned quant matrix,
                      per-tile adaptive q, decoder smoothing filter.
                      Adds 2 bytes per tile. (Image only.)
bits 6-15  reserved   Must be 0 in v1.
```

`CTX_*` flags and `BITPLANE` are mutually exclusive (BITPLANE wins).

## Backward compatibility

Future v2+ formats must remain able to decode v1. Specifically:
- The magic bytes (`C2DI` / `C2DT`) are reserved for c2d images at all
  versions.
- Byte 4 = format version is the version dispatch byte.
- A v1 decoder MUST refuse to decode anything with version != 1.
- A v2+ decoder MUST handle version == 1 correctly per this spec.

Bits 6-15 of the flags word and the reserved byte at offset 9 are
provisional extension points; they MUST be 0 in v1 streams. A v1 decoder
MUST refuse v1 streams that have non-zero reserved bits.

## Reference implementation

The reference implementation is the C source in this repository (`c2d.c` /
`c2d.h`). Anywhere this spec is ambiguous, the C source is authoritative
for v1.
