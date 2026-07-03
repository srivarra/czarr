#!/bin/bash
#SBATCH --job-name=czarr_blosc_wave
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/blosc_wave_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/blosc_wave_%j.err

set -uo pipefail
REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"; cd "$REPO"

module load cuda/13.1.0_590.44.01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== node ==="; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "GDS active: $( [[ -e /proc/driver/nvidia-fs ]] && echo yes || echo no )"; echo

# GDS cannot DMA from tmpfs — never trust an inherited TMPDIR (may be /tmp).
# Force a real block FS (NVMe or VAST). See memory project_gds_read_after_write.
unset TMPDIR
for CAND in "/local/scratch/$USER" /local/scratch "$REPO/bench/logs"; do
  if mkdir -p "$CAND/wavebench" 2>/dev/null && [ -w "$CAND/wavebench" ]; then export TMPDIR="$CAND/wavebench"; break; fi
done
echo "TMPDIR=$TMPDIR"; df -T "$TMPDIR" | sed 's/^/  /'; echo

# realistic 256 MiB chunks (matches waveorder layout) — the number that matters
echo "================ chunk-z=128 (256 MiB chunks) ================"
uv run --extra cu12 python -m bench.blosc.wave_h100 --chunk-z 128 --nz 512 2>&1
echo
# pessimistic 8 MiB chunks (many reads) — where per-chunk overhead shows
echo "================ chunk-z=4 (8 MiB chunks) ================"
uv run --extra cu12 python -m bench.blosc.wave_h100 --chunk-z 4 --nz 512 2>&1
echo "=== exit ==="
