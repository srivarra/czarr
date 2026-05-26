#!/bin/bash
#SBATCH --job-name=czarr_lz4_e2e
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-v0.1/bench/logs/lz4_e2e_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-v0.1/bench/logs/lz4_e2e_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr.czarr-v0.1
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
echo

uv run --with numcodecs --with lz4 --extra cu12 python -u -X faulthandler -m bench.cuda_array.lz4_native_e2e --reps 6 --warmup 2 2>&1
echo "=== exit: $? ==="
