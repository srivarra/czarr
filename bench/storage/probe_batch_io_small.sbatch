#!/bin/bash
#SBATCH --job-name=czarr_batch_io_small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/batch_io_small_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/batch_io_small_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== node ==="; hostname
echo "=== gpu ==="; nvidia-smi --query-gpu=name --format=csv,noheader

uv run --extra cu12 --group test python -m bench.storage.probe_batch_io_smallchunk 2>&1
echo "=== exit: $? ==="
