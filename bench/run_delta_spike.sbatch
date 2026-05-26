#!/bin/bash
#SBATCH --job-name=czarr_delta_spike
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/delta_spike_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/delta_spike_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
echo

uv run --with cuda-cccl --with numba-cuda --with numcodecs --extra cu12 \
    python -u -X faulthandler \
    .planning/research/cuda-array/spikes/delta_cuda_compute.py 2>&1
echo "=== exit: $? ==="
