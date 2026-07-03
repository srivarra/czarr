#!/bin/bash
#SBATCH --job-name=czarr_blosc_e2e
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/blosc_e2e_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/blosc_e2e_%j.err

set -uo pipefail
REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"
cd "$REPO"

module load cuda/13.1.0_590.44.01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== node ==="; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "GDS active: $( [[ -e /proc/driver/nvidia-fs ]] && echo yes || echo no )"
echo

# Store on node-local NVMe so cuFile/GDS reads compressed bytes straight to GPU.
export TMPDIR="${TMPDIR:-/local/scratch}"
[ -w "$TMPDIR" ] || export TMPDIR="$REPO/bench/logs"
echo "TMPDIR=$TMPDIR"

uv run --extra cu12 python -m bench.blosc.e2e_h100 2>&1
echo "=== exit ==="
