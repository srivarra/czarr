#!/bin/bash
#SBATCH --job-name=cufile_debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cufile_debug_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/cufile_debug_%j.err

set -uo pipefail
REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"; cd "$REPO"
module load cuda/13.1.0_590.44.01

echo "=== node ==="; hostname
nvidia-smi --query-gpu=name --format=csv,noheader
echo "GDS active: $( [[ -e /proc/driver/nvidia-fs ]] && echo yes || echo no )"

# Force node-local NVMe (GDS-capable). The inherited TMPDIR (/tmp/...) may be
# tmpfs/RAM-backed, which GDS cannot DMA from (reads return zeros).
unset TMPDIR
for CAND in "/local/scratch/$USER" /local/scratch /scratch/local "$REPO/bench/logs"; do
  if mkdir -p "$CAND/cufiledbg" 2>/dev/null && [ -w "$CAND/cufiledbg" ]; then export TMPDIR="$CAND/cufiledbg"; break; fi
done
echo "TMPDIR=$TMPDIR"
echo "FS type:"; df -T "$TMPDIR" | sed 's/^/  /'
echo

# CONTROL: the real on-disk /hpc/projects chunk (known-good in job 33631508)
echo "##### CONTROL: real /hpc/projects chunk (strategy A) #####"
uv run --extra cu12 python -m bench.cufile_debug --strategy A 2>&1
echo "  exit=$?"; echo

# SIZE sweep on NVMe (strategy A baseline)
echo "##### SIZE SWEEP on NVMe (strategy A) #####"
for MIB in 32 128 256; do
  echo "---- size=${MIB}MiB ----"
  uv run --extra cu12 python -m bench.cufile_debug --strategy A --gen-mib "$MIB" 2>&1
  echo "  exit=$?"
done
echo

# STRATEGY matrix @192MiB on NVMe
echo "##### STRATEGY MATRIX @192MiB on NVMe #####"
for S in A B C D; do
  echo "---- strategy $S @192MiB ----"
  uv run --extra cu12 python -m bench.cufile_debug --strategy "$S" --gen-mib 192 2>&1
  echo "  exit=$?"
done
echo "=== exit ==="
