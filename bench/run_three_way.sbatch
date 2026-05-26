#!/bin/bash
#SBATCH --job-name=czarr_3way
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/three_way_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/three_way_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
echo

# `--with torch` brings the cu12 wheel (compatible with our cu12 install).
uv run --with torch --with numcodecs --with lz4 --extra cu12 \
    python -u -X faulthandler -m bench.cuda_array.three_way --rewrite 2>&1
echo "=== exit: $? ==="
