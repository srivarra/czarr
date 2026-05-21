#!/bin/bash
#SBATCH --job-name=czarr_cufile_async_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cufile_async_probe_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cufile_async_probe_%j.err

set -euo pipefail

mkdir -p /hpc/mydata/sricharan.varra/Dev/czarr/bench/logs
cd /hpc/mydata/sricharan.varra/Dev/czarr

module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== node ==="
hostname
echo "=== gpu ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "=== nvidia_fs ==="
lsmod | grep -i nvidia_fs || echo "(not loaded)"
ls /proc/driver/nvidia-fs 2>/dev/null && echo "/proc/driver/nvidia-fs present" || echo "no /proc/driver/nvidia-fs"
echo "=== run ==="
uv run python bench/storage/probe_cufile_async.py
