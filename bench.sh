#!/usr/bin/env bash
# bench.sh - parallel rate-distortion sweep across c2d / cjpeg / opj_compress
# on a multi-source corpus.
# Outputs TSV: codec  image  setting  bytes  bpp  psnr_db
#
# A second column-prefix encodes the corpus source as "<source>:<image>"
# unless overridden via positional args (which still emit raw "<image>").
set -euo pipefail
cd "$(dirname "$0")"

BENCH=./build/c2d_bench
TMP="$(mktemp -d)"
trap "rm -rf '$TMP'" EXIT

JPEG_Q=(95 85 75 60 45 30 18 10 5)
J2K_RATES=(2 4 8 16 32 64 128)
# JPEG-XL: distance values. 0.5=very high quality .. 14=heavy compression.
JXL_DIST=(0.5 1.0 1.5 2.5 4.0 6.0 9.0 14.0)
# WebP: quality 0..100 (higher = better quality).
WEBP_Q=(98 90 80 65 50 35 25 15 8 3)
# AVIF: quality 0..100 (higher = better quality).
AVIF_Q=(85 70 55 40 30 22 15 10 5)
# HEIC: quality 0..100 (higher = better quality).
HEIC_Q=(85 70 55 40 30 22 15 10 5)
# VVC: QP 0..63 (lower = better quality).
VVC_QP=(18 24 30 36 42 48 54 60)
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 8)}"

# Process a single image; write its rows to its own TSV in $TMP/parts.
process_one() {
  local img="$1"
  local name part ppm w h nch pixels
  name="$(basename "$img")"
  part="$TMP/parts/${name}.tsv"
  ppm="$TMP/${name%.*}.ppm"

  read w h nch < <($BENCH info "$img")
  pixels=$((w * h))
  $BENCH to-ppm "$img" "$ppm"

  {
    # c2d (and c2d-ycocg for 3ch).
    $BENCH c2d-sweep "$img" | awk -v n="$name" -F'\t' '{ printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",$1,n,$3,$4,$5,$6,$7 }'

    # cjpeg.
    for q in "${JPEG_Q[@]}"; do
      local out="$TMP/${name}.q${q}.jpg"
      local rec="$TMP/${name}.q${q}.ppm"
      cjpeg -quality "$q" -outfile "$out" "$ppm" 2>/dev/null
      djpeg -outfile "$rec" "$out" 2>/dev/null
      local psnr bytes bpp
      psnr=$($BENCH psnr "$ppm" "$rec")
      local ssim=$($BENCH ssim "$ppm" "$rec")
      bytes=$(wc -c < "$out" | tr -d ' ')
      bpp=$(awk -v b="$bytes" -v p="$pixels" 'BEGIN{printf "%.4f", b*8/p}')
      printf 'jpeg\t%s\tq=%s\t%s\t%s\t%s\t%s\n' "$name" "$q" "$bytes" "$bpp" "$psnr" "$ssim"
    done

    # opj_compress.
    for r in "${J2K_RATES[@]}"; do
      local out="$TMP/${name}.r${r}.j2k"
      local rec="$TMP/${name}.r${r}.ppm"
      if ! opj_compress -i "$ppm" -o "$out" -r "$r" >/dev/null 2>&1; then continue; fi
      if ! opj_decompress -i "$out" -o "$rec" >/dev/null 2>&1; then continue; fi
      local psnr bytes bpp
      psnr=$($BENCH psnr "$ppm" "$rec")
      local ssim=$($BENCH ssim "$ppm" "$rec")
      bytes=$(wc -c < "$out" | tr -d ' ')
      bpp=$(awk -v b="$bytes" -v p="$pixels" 'BEGIN{printf "%.4f", b*8/p}')
      printf 'jpeg2k\t%s\tr=%s\t%s\t%s\t%s\t%s\n' "$name" "$r" "$bytes" "$bpp" "$psnr" "$ssim"
    done

    # JPEG-XL.
    for d in "${JXL_DIST[@]}"; do
      local out="$TMP/${name}.d${d}.jxl"
      local rec="$TMP/${name}.d${d}.ppm"
      if ! cjxl --quiet -d "$d" -e 7 "$ppm" "$out" >/dev/null 2>&1; then continue; fi
      if ! djxl --quiet "$out" "$rec" >/dev/null 2>&1; then continue; fi
      local psnr bytes bpp
      psnr=$($BENCH psnr "$ppm" "$rec")
      local ssim=$($BENCH ssim "$ppm" "$rec")
      bytes=$(wc -c < "$out" | tr -d ' ')
      bpp=$(awk -v b="$bytes" -v p="$pixels" 'BEGIN{printf "%.4f", b*8/p}')
      printf 'jpegxl\t%s\td=%s\t%s\t%s\t%s\t%s\n' "$name" "$d" "$bytes" "$bpp" "$psnr" "$ssim"
    done

    # WebP (lossy VP8). Takes PNG directly.
    for q in "${WEBP_Q[@]}"; do
      local out="$TMP/${name}.wq${q}.webp"
      local rec="$TMP/${name}.wq${q}.png"
      if ! cwebp -quiet -q "$q" "$img" -o "$out" 2>/dev/null; then continue; fi
      if ! dwebp -quiet "$out" -o "$rec" 2>/dev/null; then continue; fi
      local psnr bytes bpp ssim
      psnr=$($BENCH psnr "$img" "$rec")
      ssim=$($BENCH ssim "$img" "$rec")
      bytes=$(wc -c < "$out" | tr -d ' ')
      bpp=$(awk -v b="$bytes" -v p="$pixels" 'BEGIN{printf "%.4f", b*8/p}')
      printf 'webp\t%s\tq=%s\t%s\t%s\t%s\t%s\n' "$name" "$q" "$bytes" "$bpp" "$psnr" "$ssim"
    done

    # AVIF (AV1 intra). Takes PNG directly.
    for q in "${AVIF_Q[@]}"; do
      local out="$TMP/${name}.aq${q}.avif"
      local rec="$TMP/${name}.aq${q}.png"
      if ! avifenc -q "$q" -s 6 "$img" "$out" >/dev/null 2>&1; then continue; fi
      if ! avifdec "$out" "$rec" >/dev/null 2>&1; then continue; fi
      local psnr bytes bpp ssim
      psnr=$($BENCH psnr "$img" "$rec")
      ssim=$($BENCH ssim "$img" "$rec")
      bytes=$(wc -c < "$out" | tr -d ' ')
      bpp=$(awk -v b="$bytes" -v p="$pixels" 'BEGIN{printf "%.4f", b*8/p}')
      printf 'avif\t%s\tq=%s\t%s\t%s\t%s\t%s\n' "$name" "$q" "$bytes" "$bpp" "$psnr" "$ssim"
    done

    # HEIC (HEVC intra via x265). Takes PNG directly.
    for q in "${HEIC_Q[@]}"; do
      local out="$TMP/${name}.hq${q}.heic"
      local rec="$TMP/${name}.hq${q}.png"
      if ! heif-enc -q "$q" -o "$out" "$img" >/dev/null 2>&1; then continue; fi
      if ! heif-dec "$out" "$rec" >/dev/null 2>&1; then continue; fi
      local psnr bytes bpp ssim
      psnr=$($BENCH psnr "$img" "$rec")
      ssim=$($BENCH ssim "$img" "$rec")
      bytes=$(wc -c < "$out" | tr -d ' ')
      bpp=$(awk -v b="$bytes" -v p="$pixels" 'BEGIN{printf "%.4f", b*8/p}')
      printf 'heic\t%s\tq=%s\t%s\t%s\t%s\t%s\n' "$name" "$q" "$bytes" "$bpp" "$psnr" "$ssim"
    done

    # VVC intra (H.266). Needs YUV input via ffmpeg.
    local yuv8="$TMP/${name}.yuv"
    if ffmpeg -y -loglevel error -i "$img" -pix_fmt yuv420p "$yuv8" 2>/dev/null; then
      for qp in "${VVC_QP[@]}"; do
        local out="$TMP/${name}.vq${qp}.vvc"
        local yuv_out="$TMP/${name}.vq${qp}.yuv"
        local rec="$TMP/${name}.vq${qp}.png"
        if ! vvencapp -i "$yuv8" -s "${w}x${h}" -c yuv420 --frames 1 -q "$qp" --preset fast -o "$out" >/dev/null 2>&1; then continue; fi
        if ! vvdecapp -b "$out" -o "$yuv_out" >/dev/null 2>&1; then continue; fi
        if ! ffmpeg -y -loglevel error -f rawvideo -pix_fmt yuv420p10le -s "${w}x${h}" -i "$yuv_out" -pix_fmt rgb24 -frames:v 1 "$rec" 2>/dev/null; then continue; fi
        local psnr bytes bpp ssim
        psnr=$($BENCH psnr "$img" "$rec")
        ssim=$($BENCH ssim "$img" "$rec")
        bytes=$(wc -c < "$out" | tr -d ' ')
        bpp=$(awk -v b="$bytes" -v p="$pixels" 'BEGIN{printf "%.4f", b*8/p}')
        printf 'vvc\t%s\tqp=%s\t%s\t%s\t%s\t%s\n' "$name" "$qp" "$bytes" "$bpp" "$psnr" "$ssim"
      done
    fi
  } > "$part"
  echo "  done $name" >&2
}

export -f process_one
export BENCH TMP
export JPEG_Q_STR="${JPEG_Q[*]}"
export J2K_RATES_STR="${J2K_RATES[*]}"
export JXL_DIST_STR="${JXL_DIST[*]}"
export WEBP_Q_STR="${WEBP_Q[*]}"
export AVIF_Q_STR="${AVIF_Q[*]}"
export HEIC_Q_STR="${HEIC_Q[*]}"
export VVC_QP_STR="${VVC_QP[*]}"
worker() {
  local img="$1"
  read -ra JPEG_Q <<< "$JPEG_Q_STR"
  read -ra J2K_RATES <<< "$J2K_RATES_STR"
  read -ra JXL_DIST <<< "$JXL_DIST_STR"
  read -ra WEBP_Q <<< "$WEBP_Q_STR"
  read -ra AVIF_Q <<< "$AVIF_Q_STR"
  read -ra HEIC_Q <<< "$HEIC_Q_STR"
  read -ra VVC_QP <<< "$VVC_QP_STR"
  process_one "$img"
}
export -f worker

mkdir -p "$TMP/parts"
echo "Running with JOBS=$JOBS" >&2

# Default corpus: ALL lossless PNGs across categories.
if [ $# -eq 0 ]; then
  IMAGES=( \
    testdata/kodak/kodim*.png \
    testdata/ece533/*.png \
    testdata/sipi/*.png \
    testdata/clic_pro/*.png \
  )
else
  IMAGES=("$@")
fi

# Filter to only existing files; warn on missing.
EXISTING=()
for img in "${IMAGES[@]}"; do
  if [ -f "$img" ]; then
    EXISTING+=("$img")
  fi
done
echo "Total images: ${#EXISTING[@]}" >&2

printf '%s\n' "${EXISTING[@]}" | xargs -n1 -P"$JOBS" -I{} bash -c 'worker "$@"' _ {}

# Header + concatenate in original image order.
printf 'codec\timage\tsetting\tbytes\tbpp\tpsnr_db\tssim\n'
for img in "${EXISTING[@]}"; do
  name="$(basename "$img")"
  cat "$TMP/parts/${name}.tsv" 2>/dev/null
done
