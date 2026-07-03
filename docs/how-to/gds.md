# GPUDirect Storage setup

**Goal:** get real disk → GPU DMA working, know when you're in compatibility mode instead, and avoid the classic traps.

## The two cuFile modes

| Mode | Requirements | What happens |
|---|---|---|
| **Real GDS** | `nvidia_fs` kernel module + libcufile + supported FS (ext4/xfs on NVMe, Lustre, some NFS) | Reads DMA directly into GPU memory, no host bounce |
| **Compat mode** | libcufile only | cuFile stages through a pinned host bounce internally — still faster than naive read+upload |

czarr uses cuFile in both modes automatically. Check what you have:

```python
from czarr import GPULocalStore, cufile

store = GPULocalStore("/data/store.zarr")
print(store.gds_available)    # libcufile loaded and driver opened
```

```bash
ls /proc/driver/nvidia-fs     # exists => nvidia_fs module loaded (real GDS possible)
/usr/local/cuda/gds/tools/gdscheck -p    # full platform report, if installed
```

## Traps

!!! danger "GDS cannot read from tmpfs"

    cuFile reads from `tmpfs` (`/tmp` on many systems) return **zeros or errors** — there is no block device to DMA from. Keep stores on a real filesystem (NVMe, Lustre, NFS). If a benchmark suddenly reads all-zero data, check where `$TMPDIR` points before anything else.

!!! warning "Compat mode can hide misconfiguration"

    cuFile silently falls back to compat mode when `nvidia_fs` is missing. To fail loudly instead:

    ```python
    from czarr import cufile
    cufile.configure(allow_compat_mode=False)   # before the first read
    ```

## Tuning knobs

All of `cufile.json`'s knobs are scriptable, process-wide, and must be set **before the first cuFile call** (libcufile reads its config once at driver open):

```python
from czarr import cufile

cufile.configure(
    max_io_threads=8,               # internal pool (default 4)
    max_request_parallelism=8,      # hard max 8 in libcufile
    min_io_threshold_size_kb=8192,  # large-read split threshold
    max_direct_io_size_kb=16384,    # per-request IO chunk
)
```

Measured on H100 + VAST (see `bench/results/read-path.jsonl`): **the defaults are already right** — the thread/parallelism axes are flat and `max_request_parallelism` clamps at 8. Treat `configure()` as an escape hatch for unusual storage, not a required step.

## Per-read parallelism

The knob that does matter is the czarr-side read threadpool — cuFile calls are blocking, and czarr issues them in parallel:

```python
arr.retrieve_array_subset(selection, max_workers=16)   # tier 2, per call
czarr.configure_gpu(async_concurrency=32)              # tier 1, pipeline-wide
```

## Reference numbers (H100, real GDS)

From `czarr-bench sweep read-path` — raw storage → GPU transfer, decode excluded:

| Transfer | 8 MiB chunks | 64 MiB | 256 MiB |
|---|---|---|---|
| cuFile GDS | 6.6 GiB/s | 19.5 | **24.6** |
| pinned bounce + H2D | **11.4** | 15.1 | 18.1 |
| pageable read + H2D | 5.4 | 5.2 | 5.0 |

GDS wins at large reads; the pinned bounce still wins below ~16 MiB. czarr's compat fallback is the pinned path, so small-chunk workloads lose little without GDS.
