# czarr

GPU-accelerated NVIDIA nvCOMP codecs for Zarr 3.x — read and write Zarr arrays on the GPU at 30-238 GB/s, 10-30× faster than CPU codecs at multi-megabyte chunks.

czarr plugs into zarr's official extension points:

- **Codecs** — 10 `BytesBytesCodec` subclasses registered with `zarr.registry.register_codec`
- **Codec pipeline** — `CzarrCodecPipeline` registered with `register_pipeline` (batches all decode calls)
- **Stores** — `GPULocalStore` extends `zarr.storage.LocalStore` for direct disk-to-GPU reads via cuFile

## What's in the box

- **10 GPU codecs**:
  - Native (nvCOMP-only, max perf): `ANS`, `Bitcomp`, `Cascaded`, `Deflate`, `GDeflate`, `Snappy`
  - Compat (read existing CPU-written stores on GPU transparently): `Zstd`, `LZ4`, `Gzip`, `Zlib`
- **`configure_gpu()`** — one-call setup: batched codec pipeline + GPU buffer prototypes + (optional) RMM pool + transparent compat-codec selection
- **`GPULocalStore`** — cuFile-backed store for direct disk-to-GPU reads on systems with NVIDIA GPUDirect Storage
- **`use_rmm_pool`** — one-call RMM pool wiring; nvCOMP scratch + cupy allocations all share one pool

## Installation

czarr ships separate optional builds for CUDA 12 and CUDA 13 because the underlying NVIDIA wheels are CUDA-version-specific. Pick one:

```bash
pip install "czarr[cu12]"     # CUDA 12 systems
pip install "czarr[cu13]"     # CUDA 13 systems
```

(The base `pip install czarr` will fail at import — you must pick a CUDA build.)

## Quickstart

```python
import czarr
import zarr
import cupy as cp

# 1. One-call setup: batched pipeline + GPU buffers + (optional) RMM pool.
czarr.configure_gpu(rmm_pool_gb=1.0)         # rmm_pool_gb is optional

# 2. Create an array with a GPU codec
arr = zarr.create_array(
    store="path/to/data.zarr",
    shape=(64, 3, 32, 512, 512),             # TCZYX
    chunks=(1, 1, 32, 512, 512),             # one TCZ-volume per chunk
    dtype="float32",
    compressors=[czarr.ANS()],               # ANS is the safe default
)

# 3. Write — accepts cupy or numpy input
arr[:] = cp.asarray(my_data)

# 4. Read — output is cupy.ndarray on GPU; one nvCOMP batched decode for the whole selection
data = arr[:]                                # 30-238 GB/s depending on hardware
print(type(data))                            # cupy.ndarray
```

### Transparent CPU → GPU decode of existing zarr stores

After `configure_gpu()`, czarr's compat codecs (`Zstd`, `LZ4`, `Gzip`, `Zlib`) shadow the stdlib CPU codecs in zarr's registry — any existing zarr store written with `numcodecs.Zstd`, `numcodecs.LZ4`, `numcodecs.GZip`, `numcodecs.Zlib`, or `zarr.codecs.ZstdCodec` / `GzipCodec` decodes on the GPU with no migration:

```python
import czarr, zarr
czarr.configure_gpu()
arr = zarr.open_array(store="legacy_cpu_written.zarr")
out = arr[:]                                  # cupy.ndarray, GPU decoded
```

The bytestream is bit-identical with libzstd / libdeflate / libz / liblz4 — no re-encode needed.

## Disk-backed (cuFile)

```python
from czarr import GPULocalStore
import czarr, zarr

czarr.configure_gpu()
store = GPULocalStore("/path/to/store.zarr")
print(store.gds_available)           # True if libcufile loaded

arr = zarr.create_array(
    store=store, shape=..., chunks=..., dtype="float32",
    compressors=[czarr.ANS()],
)
arr[:] = data_dev                    # cuFile write path
out = arr[:]                         # cuFile read path
```

`GPULocalStore` falls back to plain `LocalStore` POSIX I/O when the requested buffer prototype is host or when libcufile is unavailable. On systems where NVIDIA GPUDirect Storage is properly configured (libcufile + `nvidia_fs` kernel module), reads land directly in GPU memory; on systems without GDS, cuFile silently runs in compatibility mode (pinned-host bounce + async memcpy — still better than naive read+upload).

## Composing with RAPIDS / cuDF / cuML

If your pipeline uses other RAPIDS libraries, route everything through one RMM pool to share device memory:

```python
import czarr

czarr.use_rmm_pool(initial_size=2**30)   # 1 GiB up front, grows as needed
# now nvCOMP scratch + cupy + cuDF + kvikIO all draw from one pool
```

Without this call, czarr uses cupy's default `MemoryPool` — also pool-backed but isolated from RAPIDS.

## Practical recipes

| If you... | Then... |
|---|---|
| Don't know which codec to pick | **`ANS`** — flat 30 GB/s, scales to 4 GiB+, lowest scratch |
| Want max throughput on small arrays | `Bitcomp` (35 GB/s, OOMs at 4 GiB on A40) |
| Need a libzstd-compatible byte stream | `Zstd` (compat codec — `czarr.Zstd()`) |
| Reading an existing CPU-written zarr (zstd/lz4/gzip/zlib) | `czarr.configure_gpu()` then `zarr.open_array(...)` — automatic GPU decode |
| Have chunks <256 KiB | Don't bother — CPU multi-thread Blosc wins |
| Have chunks 1-4 MiB | GPU pulls ahead, 3-15× CPU |
| Have chunks ≥16 MiB | GPU dominates, 14-60× CPU |
| Want to use Zarr's sharding codec | **Avoid** — it dispatches inner chunks one at a time, killing our batch API. Use big chunks without sharding instead. |
| Reading from disk | `czarr.GPULocalStore(path)` instead of `zarr.storage.LocalStore` |
| Composing with cuDF/cuML | `czarr.use_rmm_pool()` once at startup |
| Using `cuda.core.Stream` / `rmm.pylibrmm.stream.Stream` for codec | Pass via `cuda_stream=...` — any `__cuda_stream__`-compliant object works |

## Chunk shape guide for TCZYX microscopy

| Access pattern | Recommended chunks | Why |
|---|---|---|
| Read whole timepoints (`arr[t]`) | `(1, 1, Z, Y, X)` — one volume per chunk | Each crop = one chunk decode |
| Read 2D Z-planes (`arr[t, c, z]`) | `(1, 1, 1, Y, X)` — one plane per chunk | Each plane is its own chunk |
| Read XY tiles (`arr[t, c, z, y_slice, x_slice]`) | `(1, 1, 1, tile_y, tile_x)` | Tile-aligned access |
| Read full timeseries (`arr[:, c, z]`) | `(T, 1, 1, Y, X)` — one timeseries per chunk | Sequential T traversal |

The right answer depends on access pattern. Volume-per-chunk works best for ML training where each sample is one timepoint; plane-per-chunk works for visualization / ROI analysis where you read individual Z-planes.

## API reference

| Symbol | Purpose |
|---|---|
| `configure_gpu(*, batch_size=None, async_concurrency=32, rmm_pool_gb=None)` | One-call setup: pipeline + buffers + (optional) RMM pool + compat codec selection |
| `Codec` | Base class for all GPU codecs |
| `ANS, Bitcomp, Cascaded, Deflate, GDeflate, Snappy` | Native — nvCOMP-only bitstream, max throughput |
| `Zstd, LZ4, Gzip, Zlib` | Compat — bit-identical with libzstd / liblz4 / libdeflate / libz; shadow CPU codecs in registry after `configure_gpu()` |
| `Checksum` | Enum for codec config |
| `CzarrCodecPipeline` | The codec pipeline registered with zarr (batches every selection's decode into one nvCOMP call) |
| `GPULocalStore(root, *, read_only=False, force_gpu=False)` | cuFile-backed local store |
| `register_nvcomp_allocator(allocator=None)` | Hook nvCOMP into cupy's allocator (auto-called) |
| `use_rmm_pool(initial_size=1<<30, maximum_size=None)` | Switch the whole stack to an RMM pool |

Each codec accepts `chunk_size` (nvCOMP internal chunk, default 64 KiB), `checksum_policy` (`Checksum.NO_COMPUTE_NO_VERIFY` by default), `device_id`, and `cuda_stream`.  Compat codecs additionally accept `level` / `checksum` for metadata round-trip with the CPU equivalents (nvCOMP picks its own internal level).

## Performance numbers

Measured on **NVIDIA A40, RMM pool, GPU buffer prototype** (see `bench/` for the scripts):

| Workload | Throughput |
|---|---|
| Codec only, mono buffer, Bitcomp | **234 GB/s** decode |
| Zarr `arr[:]` 64 MiB chunks, ANS | **69 GB/s** read |
| CPU baseline (Blosc-lz4 16 threads) | ~3 GB/s read |
| **Net win over CPU** | **17-30×** |

## Limitations and known issues

- **Sharding codec interaction is broken.** Zarr's sharding codec dispatches inner chunks to our codec one at a time, defeating our batch decode (we measured 0.5 GB/s with sharding vs 28 GB/s without). Workaround: don't use the sharding codec; pick a chunk shape that matches your access pattern instead.
- **Bitcomp OOMs at 4 GiB+ on A40.** Higher scratch overhead than ANS. Use ANS for big workloads.
- **GPU advantage requires chunks ≥1 MiB.** Below that, CPU multi-threaded Blosc wins.
- **Real GPUDirect Storage requires the `nvidia_fs` kernel module.** On systems without it, `GPULocalStore` falls back to cuFile compatibility mode (still better than naive but no real DMA). Bruno HPC: H100 nodes have it; A40 nodes don't.
- **`cupy` is a hard dep** because Zarr's GPU buffer prototype is hard-coded to `cupy.ndarray`. We can't easily swap it for `rmm.DeviceBuffer` or `cuda.core.Buffer` without re-implementing a CUDA ndarray library.

## Contributing

For bug reports and feature requests, please use the [issue tracker][].
For questions and discussion, the [scverse discourse][] is a good place to start.

## License

MIT.

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/srivarra/czarr/issues
