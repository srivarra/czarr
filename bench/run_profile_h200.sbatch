#!/bin/bash
#SBATCH --job-name=czarr_prof_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-gpu-buffer/bench/logs/prof_h200_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-gpu-buffer/bench/logs/prof_h200_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-gpu-buffer
mkdir -p "$REPO/bench/logs" "$REPO/bench/profiles"
cd "$REPO"

module load cuda/13.1.0_590.44.01
module load nsight/2025.3.1 || true
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "GDS active: $( [[ -e /proc/driver/nvidia-fs ]] && echo yes || echo no )"
which nsys
echo

OUT="$REPO/bench/profiles/h200_${SLURM_JOB_ID}"

echo "=== nsys profile ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --stats=true \
    --force-overwrite=true \
    --output="$OUT" \
    uv run --extra cu12 python -m bench.buffer.profile_h200 --reps 3 --warmup 1 2>&1

echo "=== exit: $? ==="
echo "profile saved to ${OUT}.nsys-rep"
