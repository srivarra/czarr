#!/bin/bash
# Probe GDS (GPUDirect Storage) availability on a Bruno GPU node.
# Submit per GPU type:  sbatch --gres=gpu:h200:1 run_gds_probe.sbatch  (etc.)
#SBATCH --job-name=gds_probe
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/gds_probe_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/gds_probe_%j.err

set -uo pipefail
REPO=/hpc/mydata/sricharan.varra/Dev/czarr
mkdir -p "$REPO/bench/logs"; cd "$REPO"
module load cuda/13.1.0_590.44.01

echo "=== node ==="; hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo
echo "nvidia-fs module present (real GDS):"
[[ -e /proc/driver/nvidia-fs ]] && echo "  YES  /proc/driver/nvidia-fs exists" || echo "  NO   (compat mode only)"
[[ -e /proc/driver/nvidia-fs/version ]] && { echo "  version:"; sed 's/^/    /' /proc/driver/nvidia-fs/version; }
echo
echo "gds_stats / cuFile tools:"
command -v gds_stats >/dev/null 2>&1 && echo "  gds_stats on PATH" || echo "  gds_stats not on PATH"
ls /usr/local/cuda*/gds/tools 2>/dev/null | sed 's/^/    /' || true
echo
echo "cuFile driver properties (compat vs GDS, feature flags):"
uv run --extra cu12 python - <<'PY' 2>&1 | sed 's/^/  /'
try:
    from cuda.bindings import cufile
    cufile.driver_open()
    for attr in ("driver_get_properties", "get_version"):
        fn = getattr(cufile, attr, None)
        if fn:
            try:
                print(f"{attr}: {fn()}")
            except Exception as e:
                print(f"{attr}: <err {type(e).__name__}: {e}>")
    cufile.driver_close()
except Exception as e:
    print(f"cufile probe failed: {type(e).__name__}: {e}")
PY
echo "=== exit ==="
