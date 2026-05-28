#!/bin/bash
#SBATCH --job-name=czarr_overlap_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/overlap_sweep_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/overlap_sweep_%j.err

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

# Store on node-local NVMe so the read path isn't NFS-bound.  GPULocalStore
# (cuFile) is the representative GPU-direct path; falls back gracefully.
BASE="${TMPDIR:-/local/scratch}"
[ -w "$BASE" ] || BASE="$REPO/bench/logs"
STORE="$BASE/overlap.zarr"

# ANS — cheap decode (no per-batch table rebuild): overlap predicted to WIN.
# zstd — per-batch FSE/Huffman rebuild: overlap predicted to LOSE (control).
for COMP in ans zstd; do
  echo "================ compressor=$COMP / store=gpu ================"
  uv run --extra cu12 python -m bench.overlap.sweep_h200 \
      --path "$STORE" --compressor "$COMP" --store gpu --rewrite --reps 8 --warmup 3 2>&1
  echo
done

echo "=== exit ==="
