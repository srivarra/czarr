#!/bin/bash
#SBATCH --job-name=czarr_cpu_zstd_gpu_decode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cpu_zstd_gpu_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cpu_zstd_gpu_%j.err

set -uo pipefail
cd /hpc/mydata/sricharan.varra/Dev/czarr
module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
hostname
nvidia-smi --query-gpu=name --format=csv,noheader
uv run python bench/storage/probe_cpu_zstd_gpu_decode.py
