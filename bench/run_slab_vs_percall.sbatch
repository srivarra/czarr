#!/bin/bash
#SBATCH --job-name=czarr_slab_vs_percall
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/slab_vs_percall_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/slab_vs_percall_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
ls /proc/driver/nvidia-fs >/dev/null 2>&1 && echo "real GDS: yes (nvidia_fs loaded)" || echo "real GDS: no (compat mode)"
echo

# /local/scratch is the confirmed node-local NVMe xfs mount (/dev/md0),
# the only surface where real GDS DMA — and therefore register-once —
# can pay off.  $TMPDIR (often the harness's tmpfs) and VAST NFS both
# force cuFile through a host bounce regardless of buffer registration,
# masking the slab's DMA benefit.  Pin to /local/scratch explicitly and
# report the backing mount so the result is unambiguous.
SCRATCH_BASE=/local/scratch
if [ ! -w "$SCRATCH_BASE" ]; then
  echo "WARNING: $SCRATCH_BASE not writable; falling back (results will NOT reflect GDS DMA)" >&2
  SCRATCH_BASE="${TMPDIR:-$REPO/bench/logs}"
fi
SHARD_DIR=$(mktemp -p "$SCRATCH_BASE" -d) || { echo "mktemp failed in $SCRATCH_BASE" >&2; exit 1; }
SHARD="$SHARD_DIR/shard.bin"
echo "scratch base: $SCRATCH_BASE"
echo "shard path:   $SHARD"
echo "backing mount:"
df -hT "$SHARD_DIR" | tail -1
trap 'rm -rf "$SHARD_DIR"' EXIT

uv run --extra cu12 python -u -X faulthandler \
    bench/buffer/slab_vs_percall.py \
    --shard "$SHARD" \
    --reps 15 --warmup 4 2>&1

echo "=== exit ==="
