#!/bin/bash
#SBATCH --job-name=czarr_sweep_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/sweep_h200_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/sweep_h200_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

LOCAL_SCRATCH="/local/scratch/${USER}"
mkdir -p "$LOCAL_SCRATCH"

echo "=== node info ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo
echo "=== cuFile / GDS status ==="
if [[ -e /proc/driver/nvidia-fs ]]; then
    echo "nvidia-fs kernel module: LOADED — real GDS available"
else
    echo "nvidia-fs kernel module: NOT LOADED — cuFile in compat mode"
fi
echo
echo "=== run pipeline sweep ==="
CSV="$REPO/bench/logs/sweep_h200_${SLURM_JOB_ID}.csv"
uv run --extra cu12 --group test python -m bench.zarr.pipeline_sweep --output "$CSV" 2>&1
echo
wc -l "$CSV"
echo "=== exit code: $? ==="
