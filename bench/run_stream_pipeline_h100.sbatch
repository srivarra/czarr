#!/bin/bash
#SBATCH --job-name=czarr_stream_pipe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/stream_pipe_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/stream_pipe_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "GDS active: $( [[ -e /proc/driver/nvidia-fs ]] && echo yes || echo no )"
echo

# Node-local NVMe so the read half is realistic GPU-direct, not NFS-bound.
BASE="${TMPDIR:-/local/scratch}"
[ -w "$BASE" ] || BASE="$REPO/bench/logs"
STORE="$BASE/stream.zarr"

# Async consumer (no per-iter sync) — the realistic training pattern, where
# CUDA's async queue already overlaps read with compute.
echo "================ async consumer / prefetch=2 / store=gpu ================"
uv run --extra cu12 python -m bench.pipeline.stream_pipeline_h100 \
    --path "$STORE" --store gpu --prefetch 2 --rewrite --reps 5 --warmup 2 2>&1
echo

# Sync-per-iter consumer — drains the async queue every step; this is the only
# regime where an explicit prefetch thread can recover the overlap.
echo "================ sync-per-iter consumer / prefetch=2 / store=gpu ================"
uv run --extra cu12 python -m bench.pipeline.stream_pipeline_h100 \
    --path "$STORE" --store gpu --prefetch 2 --rewrite --sync-consumer --reps 5 --warmup 2 2>&1

echo "=== exit ==="
