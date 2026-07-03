#!/bin/bash
# cuFile knob sweep over the read-path gds arm (phase-4 step 6).
# libcufile reads its config once at driver_open, so each knob combo runs in
# a fresh czarr-bench process via CZARR_CUFILE_KNOBS (see benches/read_path.py).
#   sbatch bench/run_cufile_sweep.sbatch                 # H100 (default)
#   sbatch --gres=gpu:h200:1 bench/run_cufile_sweep.sbatch
#SBATCH --job-name=czarr_cufile_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cufile_sweep_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cufile_sweep_%j.err

set -uo pipefail
REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs" "$REPO/bench/results"; cd "$REPO"

module load cuda/13.1.0_590.44.01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== node ==="; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "GDS active: $( [[ -e /proc/driver/nvidia-fs ]] && echo yes || echo no )"

# Same real-FS TMPDIR dance as run_bench.sbatch — GDS cannot DMA from tmpfs.
unset TMPDIR
for CAND in "/local/scratch/$USER" /local/scratch "/hpc/mydata/$USER/.czarr-bench-fixtures"; do
  if mkdir -p "$CAND/czarrbench" 2>/dev/null && [ -w "$CAND/czarrbench" ]; then export TMPDIR="$CAND/czarrbench"; break; fi
done
echo "TMPDIR=$TMPDIR"; df -T "$TMPDIR" | sed 's/^/  /'; echo

RUN=(uv run --extra cu12 --extra bench czarr-bench run read-path
     --param impl=gds --param chunk_mib=64 --param n_chunks=16)

# Baseline: library defaults (no knobs).
echo "=== knobs: defaults ==="
"${RUN[@]}"

# Axis 1: internal thread pool x request parallelism (the execution.* knobs
# flagged in the cufile-async postmortem as the ones to tune on the sync path).
for THREADS in 4 8 16; do
  for PAR in 4 8 16; do
    echo "=== knobs: max_io_threads=$THREADS max_request_parallelism=$PAR ==="
    CZARR_CUFILE_KNOBS="{\"max_io_threads\": $THREADS, \"max_request_parallelism\": $PAR}" "${RUN[@]}"
  done
done

# Axis 2: large-read split threshold x per-request direct-IO size, at the
# best-guess parallelism from axis 1's defaults.
for THRESH_KB in 1024 8192; do
  for DIO_KB in 4096 16384; do
    echo "=== knobs: min_io_threshold_size_kb=$THRESH_KB max_direct_io_size_kb=$DIO_KB ==="
    CZARR_CUFILE_KNOBS="{\"min_io_threshold_size_kb\": $THRESH_KB, \"max_direct_io_size_kb\": $DIO_KB}" "${RUN[@]}"
  done
done

echo "=== sweep done; rows in bench/results/read-path.jsonl (regime.cufile_knobs) ==="
