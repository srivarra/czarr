#!/bin/bash
#SBATCH --job-name=czarr_tests_h100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/tests_h100_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/tests_h100_%j.err

set -uo pipefail  # don't exit on first non-zero; we want full output

mkdir -p /hpc/mydata/sricharan.varra/Dev/czarr/bench/logs
cd /hpc/mydata/sricharan.varra/Dev/czarr

module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== node ==="
hostname
echo "=== gpu ==="
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== run full test suite ==="
uv run pytest tests/ -v 2>&1
echo "=== exit code: $? ==="
