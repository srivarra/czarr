#!/bin/bash
# Single sbatch for the whole czarr-bench CLI. GPU type is chosen at submit:
#   sbatch --gres=gpu:h100:1 bench/run_bench.sbatch run blosc-e2e --param chunk_mib=128
#   sbatch --gres=gpu:h200:1 bench/run_bench.sbatch sweep blosc-e2e
#   NSYS=1 sbatch --gres=gpu:h100:1 bench/run_bench.sbatch run blosc-e2e --nsys
# Everything after the script name is passed verbatim to `czarr-bench`.
#SBATCH --job-name=czarr_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/bench_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/bench_%j.err

set -uo pipefail
REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs" "$REPO/bench/results"; cd "$REPO"

module load cuda/13.1.0_590.44.01
[[ "${NSYS:-0}" == "1" ]] && module load nsight/2025.3.1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== node ==="; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "GDS active: $( [[ -e /proc/driver/nvidia-fs ]] && echo yes || echo no )"

# GDS cannot DMA from tmpfs — never trust an inherited TMPDIR (may be /tmp).
# Force a real block FS (NVMe or VAST). See memory project_gds_read_after_write.
unset TMPDIR
# Prefer node-local NVMe; fall back to VAST under $MYDATA but OUTSIDE the repo
# tree (fixtures are hundreds of MB of regenerable .zarr — never in-tree).
for CAND in "/local/scratch/$USER" /local/scratch "/hpc/mydata/$USER/.czarr-bench-fixtures"; do
  if mkdir -p "$CAND/czarrbench" 2>/dev/null && [ -w "$CAND/czarrbench" ]; then export TMPDIR="$CAND/czarrbench"; break; fi
done
echo "TMPDIR=$TMPDIR"; df -T "$TMPDIR" | sed 's/^/  /'; echo

RUN=(uv run --extra cu12 --extra bench)
if [[ "${NSYS:-0}" == "1" ]]; then
  # Capture only the cudaProfilerStart/Stop window the harness brackets.
  PREFIX=(nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
          --trace=cuda,nvtx -f true -o "$REPO/bench/results/nsys_${SLURM_JOB_ID}")
  echo "=== profiling under nsys ==="
  "${PREFIX[@]}" "${RUN[@]}" czarr-bench "$@"
else
  "${RUN[@]}" czarr-bench "$@"
fi
echo "=== exit $? ==="
