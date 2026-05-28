#!/bin/bash
#SBATCH --job-name=czarr_decode_profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/decode_profile_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/decode_profile_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01
module load nsight/2025.3.1

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
nsys --version | head -1
echo

OUT="$REPO/bench/logs/profile_${SLURM_JOB_ID}"
mkdir -p "$OUT"
STORE=$(mktemp -p "${TMPDIR:-/tmp}" -d)/decode.zarr
trap 'rm -rf "$(dirname "$STORE")"' EXIT

for COMP in zstd lz4 ans; do
  echo "================ compressor=$COMP ================"
  REP="$OUT/decode_$COMP"
  # --cuda-memory-usage gives the device-memory timeline; trace cuda+nvtx
  # so the codec NVTX ranges segment the decode.  Skip --gpu-metrics
  # (needs elevated HW-counter perms on shared nodes).
  nsys profile \
      --trace=cuda,nvtx,osrt \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      --output="$REP" \
      uv run --extra cu12 python -u bench/profile/decode_profile.py \
          --path "${STORE}.${COMP}" --compressor "$COMP" --rmm-gb 4 --reps 12 --warmup 3 2>&1
  echo

  echo "---- nsys stats: kernel + memcpy + NVTX (compressor=$COMP) ----"
  nsys stats \
      --report cuda_gpu_kern_sum \
      --report cuda_gpu_mem_size_sum \
      --report nvtx_pushpop_sum \
      "$REP.nsys-rep" 2>&1 | sed -n '1,60p'
  echo

  echo "---- NVTX region breakdown via nsys_analyze (compressor=$COMP) ----"
  nsys export --type=sqlite --force-overwrite=true --output="$REP.sqlite" "$REP.nsys-rep" >/dev/null 2>&1 \
      && uv run --extra cu12 python bench/profile/nsys_analyze.py "$REP.sqlite" --top 15 2>&1 | sed -n '1,80p' \
      || echo "(sqlite export / analyze skipped)"
  echo
done

echo "reports in: $OUT"
echo "=== exit ==="
