# czarr benchmarks

Measurements that informed czarr's design choices.  Each script's docstring
has the `Run:` invocation.  All scripts are pure benchmarks — no asserts, no
test discovery — and dump tables to stdout.

## Layout

| Dir | What it measures |
|---|---|
| `baseline.py` | Headline GPU-vs-CPU table for all 10 codecs.  Run this first when comparing hardware. |
| `codec/` | nvCOMP codec internals only (no Zarr).  Batch-decode parallelism, chunk-size sweep, stream count, HLIF `out=` reuse. |
| `zarr/` | Full Zarr pipeline read/write at varying scale — chunk size, sharding, OME-Zarr, large arrays. |
| `storage/` | `GPULocalStore` (cuFile) under disk load, plus the cuFile async API recon probe. |

## Run convention

All scripts run from repo root as Python modules:

```bash
uv run --extra cu12 --group test python -m bench.codec.exp_batch
uv run --extra cu13 --group test python -m bench.zarr.exp_zarr_e2e   # CUDA 13
```

GPU jobs go through SLURM on Bruno — see `bench/run_tests_h100.sbatch` and
`bench/storage/probe_cufile_async.sbatch` for templates.

## Logs

`bench/logs/` is gitignored.  SLURM stdout/stderr land there as
`<jobname>_<jobid>.log`/`.err`.
