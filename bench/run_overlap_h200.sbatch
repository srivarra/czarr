#!/bin/bash
#SBATCH --job-name=czarr_overlap_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-overlap/bench/logs/overlap_h200_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-overlap/bench/logs/overlap_h200_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-overlap
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "GDS active: $( [[ -e /proc/driver/nvidia-fs ]] && echo yes || echo no )"
echo

echo "=== batch_size sweep ==="
uv run --extra cu12 python -m bench.overlap.sweep_h200 --rewrite 2>&1
echo "=== exit: $? ==="
