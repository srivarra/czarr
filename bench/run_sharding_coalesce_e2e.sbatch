#!/bin/bash
#SBATCH --job-name=czarr_sharding_e2e
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/sharding_e2e_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/sharding_e2e_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
echo

# Build sharded zarr on node-local scratch when available so we
# measure cuFile + NVMe; fall back to VAST NFS (slower per-read but
# still exercises the coalesce path).
SCRATCH_BASE=/local/scratch/users/$USER
mkdir -p "$SCRATCH_BASE" 2>/dev/null || true
if [ ! -w "$SCRATCH_BASE" ]; then
  SCRATCH_BASE="$REPO/bench/logs"
fi
ROOT_DIR=$(mktemp -p "$SCRATCH_BASE" -d) || { echo "mktemp failed in $SCRATCH_BASE" >&2; exit 1; }
ZARR_ROOT="$ROOT_DIR/sharded.zarr"
echo "scratch base: $SCRATCH_BASE"
echo "zarr root:    $ZARR_ROOT"
trap 'rm -rf "$ROOT_DIR"' EXIT

uv run --extra cu12 python -u -X faulthandler \
    bench/storage/sharding_coalesce_e2e.py \
    --root "$ZARR_ROOT" \
    --reps 8 --warmup 3 2>&1

echo "=== exit ==="
