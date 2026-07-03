#!/bin/bash
#SBATCH --job-name=czarr_jit_cold_warm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/jit_cold_warm_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/jit_cold_warm_%j.err

set -uo pipefail

REPO=/hpc/mydata/sricharan.varra/Dev/czarr
cd "$REPO"
mkdir -p "$REPO/bench/logs"

module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# The probe script we want to time twice — one full configure_gpu +
# arr[:] round-trip through the Shuffle filter (cupy transpose) +
# nvCOMP zstd, comparing cold vs warm kernel caches.
PROBE=$(cat <<'PY'
import os, sys, time, tempfile
os.environ.setdefault("CUFILE_FORCE_COMPAT_MODE", "true")
t0 = time.perf_counter()
import cupy as cp, numpy as np, zarr, czarr
import_t = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
czarr.configure_gpu()
cfg_t = (time.perf_counter() - t1) * 1000

with tempfile.TemporaryDirectory(dir="/hpc/mydata/sricharan.varra/Dev/czarr",
                                 prefix=".czarr_jit_") as td:
    store = czarr.GPULocalStore(f"{td}/s.zarr")
    arr = zarr.create_array(store=store, shape=(32, 256, 256), chunks=(8, 64, 64),
                            dtype="float32",
                            compressors=[czarr.Shuffle(elementsize=4), czarr.Zstd()],
                            overwrite=True)
    arr[:] = cp.asarray(np.random.default_rng(0).standard_normal((32, 256, 256)).astype("f4"))
    cp.cuda.Stream.null.synchronize()

    t2 = time.perf_counter()
    arr_r = zarr.open(czarr.GPULocalStore(f"{td}/s.zarr", read_only=True), mode="r")
    out = arr_r[:]
    cp.cuda.Stream.null.synchronize()
    read_t = (time.perf_counter() - t2) * 1000

print(f"import={import_t:.0f} ms  configure={cfg_t:.0f} ms  first_read={read_t:.0f} ms  "
      f"total={import_t + cfg_t + read_t:.0f} ms")
PY
)

echo "=== node ==="; hostname
echo "=== gpu ==="; nvidia-smi --query-gpu=name --format=csv,noheader
echo

echo "=== COLD (caches purged) ==="
rm -rf ~/.cupy/kernel_cache ~/.nv/ComputeCache
uv run --extra cu12 --group test python -c "$PROBE"
echo

echo "=== WARM 1 (caches primed by cold run) ==="
uv run --extra cu12 --group test python -c "$PROBE"
echo

echo "=== WARM 2 (definitely warm) ==="
uv run --extra cu12 --group test python -c "$PROBE"
echo

echo "=== cache sizes after run ==="
du -sh ~/.cupy/kernel_cache ~/.nv/ComputeCache 2>/dev/null || true
