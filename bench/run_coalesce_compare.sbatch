#!/bin/bash
#SBATCH --job-name=czarr_coalesce_compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/coalesce_compare_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/coalesce_compare_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
echo

# /local/scratch is the node-local NVMe — best surface for cuFile.
SHARD=$(mktemp -p /local/scratch -d)/shard.bin
trap 'rm -rf "$(dirname "$SHARD")"' EXIT

UV_ARGS="--extra cu12"

# Sweep over chunk sizes — small chunks are where coalesce should win
# most (per-call overhead dominates the read time).
for CHUNK in 65536 262144 1048576; do
  echo "=== N=32 chunk=$CHUNK ==="
  uv run $UV_ARGS python -u -X faulthandler \
      bench/storage/coalesce_compare.py \
      --shard "$SHARD" \
      --n-chunks 32 \
      --chunk-bytes $CHUNK \
      --max-fused-bytes $((64 << 20)) \
      --reps 20 --warmup 5 2>&1
  echo
done

# Large-chunk sanity: when per-call overhead is small relative to read
# time, coalesce should be approximately a tie (no regression).
echo "=== N=8 chunk=4MiB (large-chunk regime) ==="
uv run $UV_ARGS python -u -X faulthandler \
    bench/storage/coalesce_compare.py \
    --shard "$SHARD" \
    --n-chunks 8 \
    --chunk-bytes $((4 << 20)) \
    --max-fused-bytes $((64 << 20)) \
    --reps 20 --warmup 5 2>&1

echo "=== exit ==="
