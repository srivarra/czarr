# GPUDirect Storage setup

cuFile runs in one of two modes. czarr uses whichever the system provides.

| Mode | Requirements | Behavior |
|---|---|---|
| GDS | `nvidia_fs` kernel module, libcufile, supported filesystem (ext4/xfs on NVMe, Lustre, some NFS) | Reads DMA directly into GPU memory |
| Compatibility | libcufile only | cuFile stages through an internal pinned host bounce |

To check the current state:

```python
from czarr import GPULocalStore

store = GPULocalStore("/data/store.zarr")
print(store.gds_available)    # libcufile loaded and driver opened
```

```bash
ls /proc/driver/nvidia-fs                # exists when the kernel module is loaded
/usr/local/cuda/gds/tools/gdscheck -p    # full platform report, if installed
```

GDS is faster for large reads. Below roughly 16 MiB per read the pinned path is competitive, and the compatibility fallback is the pinned path, so small-chunk workloads lose little without GDS.

## Failure modes

cuFile cannot read tmpfs. Reads from `/tmp` on tmpfs-backed systems return zeros or fail; there is no block device to DMA from. Keep stores on a real filesystem, and check `$TMPDIR` first when reads return all zeros.

Compat mode is silent. When `nvidia_fs` is missing, cuFile falls back without an error. To fail instead:

```python
from czarr import cufile
cufile.configure(allow_compat_mode=False)   # before the first read
```

## Configuration knobs

The `cufile.json` knobs are settable in-process. libcufile reads its configuration once at driver open, so `configure()` must run before the first cuFile call:

```python
from czarr import cufile

cufile.configure(
    max_io_threads=8,               # internal pool, default 4
    max_request_parallelism=8,      # libcufile maximum is 8
    min_io_threshold_size_kb=8192,
    max_direct_io_size_kb=16384,
)
```

The defaults are appropriate on the systems we have tested; treat `configure()` as an escape hatch for unusual storage.

The knob that does matter is czarr's own read threadpool; cuFile calls block, and czarr issues them in parallel:

```python
arr.retrieve_array_subset(selection, max_workers=16)   # explicit API, per call
czarr.configure_gpu(async_concurrency=32)              # zarr pipeline
```

czarr caches registered cuFile file handles process-wide, revalidating each file's inode/mtime/size per read. `czarr.cufile.clear_handle_cache()` drops the cache.
