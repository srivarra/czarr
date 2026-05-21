#!/bin/bash
#SBATCH --job-name=czarr_batch_io_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/batch_io_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/batch_io_%j.err

set -uo pipefail

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
ls /proc/driver/nvidia-fs 2>/dev/null && echo "/proc/driver/nvidia-fs present" || echo "(missing)"
echo "=== run ==="
uv run python bench/storage/probe_batch_io.py
