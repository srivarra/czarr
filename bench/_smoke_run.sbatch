#!/bin/bash
#SBATCH --job-name=czarr_bench_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/smoke_%j.log
#SBATCH --error=/hpc/mydata/sricharan.varra/Dev/czarr/bench/logs/smoke_%j.err

set -uo pipefail
cd /hpc/mydata/sricharan.varra/Dev/czarr
module load cuda/13.1.0_590.44.01
export LD_LIBRARY_PATH="/hpc/apps/x86_64/cuda/13.1.0_590.44.01/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# Quick "do all moved scripts still import" smoke — just import each module.
uv run python -c "
import importlib
mods = [
    'bench.baseline',
    'bench.codec.exp_batch', 'bench.codec.exp_chunk_size',
    'bench.codec.exp_streams', 'bench.codec.exp_hlif_out',
    'bench.zarr.exp_zarr_e2e', 'bench.zarr.exp_zarr_chunksize',
    'bench.zarr.exp_large_arrays', 'bench.zarr.exp_ome_zarr',
    'bench.zarr.exp_sharding', 'bench.zarr.exp_sharding_diag',
    'bench.storage.exp_gpu_local_store', 'bench.storage.probe_cufile_async',
]
fails = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f'OK   {m}')
    except Exception as e:
        fails.append((m, e))
        print(f'FAIL {m}: {type(e).__name__}: {e}')
print()
print('summary:', len(mods) - len(fails), 'pass /', len(fails), 'fail')
import sys
sys.exit(0 if not fails else 1)
"
