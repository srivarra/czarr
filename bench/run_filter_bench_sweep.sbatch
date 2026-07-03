#!/bin/bash
#SBATCH --job-name=czarr_phase2_filter_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/filter_bench_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/filter_bench_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
echo

UV_ARGS="--with cuda-cccl --with numba-cuda --with numcodecs --extra cu12"

echo "=== Delta (cupy.cumsum vs cuda.compute scan) ==="
uv run $UV_ARGS python -u -X faulthandler \
    .planning/research/cuda-array/spikes/delta_cuda_compute.py 2>&1
echo

echo "=== FixedScaleOffset (cupy elementwise vs cuda.compute make_unary_transform) ==="
uv run $UV_ARGS python -u -X faulthandler \
    .planning/research/cuda-array/spikes/fso_cuda_compute.py 2>&1
echo

echo "=== BitRound (round-to-even vs legacy add-half) ==="
uv run $UV_ARGS python -u -X faulthandler \
    .planning/research/cuda-array/spikes/bitround_perf.py 2>&1
echo

echo "=== exit ==="
