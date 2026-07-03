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

A sweep of these on H100 against VAST (`bench/results/read-path.jsonl`) showed no improvement over the defaults.

The knob that does matter is czarr's own read threadpool; cuFile calls block, and czarr issues them in parallel:

```python
arr.retrieve_array_subset(selection, max_workers=16)   # explicit API, per call
czarr.configure_gpu(async_concurrency=32)              # zarr pipeline
```

czarr also caches registered cuFile file handles process-wide (open plus register costs about 1 ms per file), revalidating each file's inode/mtime/size per read. `czarr.cufile.clear_handle_cache()` drops the cache.

## Transfer measurements

`czarr-bench sweep read-path`, H100, GDS, storage-to-GPU transfer only:

| Transfer | 8 MiB chunks | 64 MiB | 256 MiB |
|---|---|---|---|
| cuFile GDS | 6.6 GiB/s | 19.5 | 24.6 |
| pinned bounce + H2D | 11.4 | 15.1 | 18.1 |
| pageable read + H2D | 5.4 | 5.2 | 5.0 |

GDS wins at large reads; the pinned bounce wins below roughly 16 MiB. The compat fallback is the pinned path, so small-chunk workloads lose little without GDS.
