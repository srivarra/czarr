#!/bin/bash
#SBATCH --job-name=czarr_cccl_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cccl_probe_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cccl_probe_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

# cuda/13.1.0 ships libnvrtc.so.13 + libnvJitLink.so.13 — the cu13
# bindings cuda-cccl picks first on a CUDA-13 driver.  No extra paths.
module load cuda/13.1.0_590.44.01

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
echo

uv run --with cuda-cccl --with numba-cuda --extra cu12 \
    python -u -X faulthandler \
    .planning/research/cuda-array/spikes/cuda_compute_probe.py 2>&1
echo "=== exit: $? ==="
