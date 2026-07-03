# czarr

GPU-native reading and writing of Zarr v3 arrays — nvCOMP codecs, GPUDirect Storage via cuFile, and an explicit low-level read path. 10-30× faster than CPU codecs at multi-megabyte chunks; ~20 GiB/s end-to-end reads on H100 + GDS.

czarr gives you two tiers on one substrate:

- **Tier 1 — drop-in zarr.** `configure_gpu()` plugs GPU codecs, a batched pipeline, and GPU buffers into zarr's official extension points. Existing code and existing CPU-written stores work unchanged. `CudaZarrArray` adds a fast path that routes basic-indexing reads through tier 2 automatically.
- **Tier 2 — explicit.** `czarr.core.Array` and `czarr.lowlevel` expose the staged read path (plan → coalesced cuFile reads → one batched nvCOMP decode) as plain functions with no global state. Parse once, read many, tune per call.

## What's in the box

- **GPU compressors** (nvCOMP):
  - Native bitstream, max perf: `ANS`, `Bitcomp`, `Cascaded`, `Deflate`, `GDeflate`, `Snappy`
  - Compat, bit-identical with libzstd/liblz4/libdeflate/libz: `Zstd`, `LZ4`, `Gzip`, `Zlib`
  - `Blosc` — GPU decode of blosc `[bitshuffle|shuffle, zstd]` containers via nvCOMP's native batched API
- **GPU filters**: `Shuffle`, `Delta`, `FixedScaleOffset`, `BitRound`
- **Coalescing sharding codec** — transparently replaces zarr's `sharding_indexed` on read; partial-shard reads fuse adjacent chunks into single cuFile calls (22× on the H100 microbench)
- **`GPULocalStore`** — cuFile-backed store: disk → GPU memory directly on GDS systems, compat-mode bounce elsewhere
- **`czarr.core.Array` / `AsyncArray`** — explicit read API returning `cupy.ndarray`
- **`czarr.lowlevel`** — the staged plumbing itself: `DecodePlan`, `read`, `decode`, `read_array`
- **`czarr-bench`** — the benchmark CLI used for every number in this README (`pip install "czarr[bench]"`)

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
import cupy as cp

czarr.configure_gpu()                        # one-call setup (optional rmm_pool_gb=...)

# Create + write through zarr's machinery
arr = czarr.create_cuda_array(
    store="path/to/data.zarr",
    shape=(64, 3, 32, 512, 512),             # TCZYX
    chunks=(1, 1, 32, 512, 512),             # one TCZ-volume per chunk
    dtype="float32",
    compressors=[czarr.ANS()],               # ANS is the safe default
)
arr[:] = cp.asarray(my_data)                 # accepts cupy or numpy

# Read — cupy.ndarray on device; basic selections take the lowlevel fast path
data = arr[:]
plane = arr[3, 0, 16]                        # coalesced cuFile reads, one batched decode
```

### Explicit reads (tier 2)

No global state, no zarr config — parse the metadata once, read many times:

```python
from czarr.core import Array

arr = Array.open("path/to/data.zarr")        # one zarr.json read, nothing else
out = arr[0:4, :, 8:24]                      # cupy.ndarray; ndim-preserving selections
chunk = arr.retrieve_chunk((0, 0, 0))        # one decode unit
raw = arr.retrieve_encoded_chunk((0, 0, 0))  # pre-decode device bytes (escape hatch)

out = arr.retrieve_array_subset(             # per-call knobs, zero globals
    (slice(0, 4), ..., slice(8, 24)),
    max_workers=16, max_fused_bytes=64 << 20,
)
```

`AsyncArray` is the awaitable dual (`await arr.retrieve_array_subset(...)`); each call runs the sync path in a worker thread, and cuFile reads release the GIL so concurrent awaits genuinely overlap.

### Transparent CPU → GPU decode of existing zarr stores

After `configure_gpu()`, czarr's compat codecs (`Zstd`, `LZ4`, `Gzip`, `Zlib`) shadow the CPU codecs in zarr's registry — any existing store written with `numcodecs` or `zarr.codecs` equivalents decodes on the GPU with no migration:

```python
import czarr, zarr
czarr.configure_gpu()
arr = zarr.open_array(store="legacy_cpu_written.zarr")
out = arr[:]                                  # cupy.ndarray, GPU decoded
```

The bytestream is bit-identical with libzstd / libdeflate / libz / liblz4 — no re-encode needed. Blosc-compressed stores (`[bitshuffle, zstd]`, the common bioimaging layout) decode on GPU too, via `czarr.Blosc` (decode-only).

## Disk-backed (cuFile / GPUDirect Storage)

```python
from czarr import GPULocalStore
import czarr, zarr

czarr.configure_gpu()
store = GPULocalStore("/path/to/store.zarr")
print(store.gds_available)           # True if libcufile loaded

arr = zarr.open_array(store=store, mode="r")
out = arr[:]                         # disk -> GPU, no host round-trip
```

`GPULocalStore` falls back to plain `LocalStore` POSIX I/O when the requested buffer prototype is host-side or libcufile is unavailable. With real GDS (`nvidia_fs` kernel module), reads DMA directly into GPU memory; without it, cuFile runs in compatibility mode (pinned-host bounce — still better than naive read+upload).

cuFile's process-wide knobs are scriptable before the first read:

```python
from czarr import cufile
cufile.configure(max_io_threads=8, allow_compat_mode=False)  # cufile.json equivalents
```

(Measured on H100: the defaults are already right — see `bench/results/read-path.jsonl`.)

## Composing with RAPIDS / cuDF / cuML

Route everything through one RMM pool to share device memory:

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
| Reading an existing CPU-written zarr (zstd/lz4/gzip/zlib/blosc) | `czarr.configure_gpu()` then `zarr.open_array(...)` — automatic GPU decode |
| Have chunks <256 KiB | Don't bother — CPU multi-thread Blosc wins |
| Have chunks 1-4 MiB | GPU pulls ahead, 3-15× CPU |
| Have chunks ≥16 MiB | GPU dominates, 14-60× CPU |
| Using zarr's sharding codec | Just `configure_gpu()` — czarr's coalescing override fuses partial-shard reads automatically |
| Reading from disk | `czarr.GPULocalStore(path)` instead of `zarr.storage.LocalStore` |
| Want reads without touching global zarr config | `czarr.core.Array.open(path)` |
| Composing with cuDF/cuML | `czarr.use_rmm_pool()` once at startup |
| Using `cuda.core.Stream` / `rmm` streams with a codec | Pass via `cuda_stream=...` — any `__cuda_stream__`-compliant object works |

## Chunk shape guide for TCZYX microscopy

| Access pattern | Recommended chunks | Why |
|---|---|---|
| Read whole timepoints (`arr[t]`) | `(1, 1, Z, Y, X)` — one volume per chunk | Each crop = one chunk decode |
| Read 2D Z-planes (`arr[t, c, z]`) | `(1, 1, 1, Y, X)` — one plane per chunk | Each plane is its own chunk |
| Read XY tiles (`arr[t, c, z, y_slice, x_slice]`) | `(1, 1, 1, tile_y, tile_x)` | Tile-aligned access |
| Read full timeseries (`arr[:, c, z]`) | `(T, 1, 1, Y, X)` — one timeseries per chunk | Sequential T traversal |

The right answer depends on access pattern. Volume-per-chunk works best for ML training where each sample is one timepoint; plane-per-chunk works for visualization / ROI analysis where you read individual Z-planes.

## Performance numbers

End-to-end reads, H100 + GPUDirect Storage, blosc `[bitshuffle, zstd]` fixture, 128-256 MiB chunks (`czarr-bench sweep zarr-read`, see `bench/results/`):

| Read stack | Throughput |
|---|---|
| czarr — lowlevel / tier-1 fast path / zarr pipeline | **~20 GiB/s** (parity across all three) |
| kvikio `GDSStore` + GPU codecs (external baseline) | 1.6-1.9 GiB/s |
| CPU decode + H2D (`LocalStore` + numcodecs) | ~1 GiB/s |

Codec-only decode on A40: Bitcomp **234 GB/s**, zarr `arr[:]` with ANS at 64 MiB chunks **69 GB/s** vs ~3 GB/s CPU Blosc-lz4 (16 threads).

## Limitations and known issues

- **Decode-oriented.** The lowlevel fast path covers zstd, blosc, and shuffle for reads; other codec chains automatically fall back to the zarr pipeline (still GPU-decoded where a czarr codec exists). `Blosc` is decode-only — write with `[Shuffle, Zstd]` for GPU-decodable output.
- **Bitcomp OOMs at 4 GiB+ on A40.** Higher scratch overhead than ANS. Use ANS for big workloads.
- **GPU advantage requires chunks ≥1 MiB.** Below that, CPU multi-threaded Blosc wins.
- **Real GPUDirect Storage requires the `nvidia_fs` kernel module.** Without it, `GPULocalStore` falls back to cuFile compatibility mode (still better than naive, but no true DMA). Also: GDS cannot read from `tmpfs` — keep stores on a real filesystem.
- **`cupy` is a hard dep** because Zarr's GPU buffer prototype is hard-coded to `cupy.ndarray`.
- **Linux only.**

## Contributing

For bug reports and feature requests, please use the [issue tracker][].

## License

MIT.

[issue tracker]: https://github.com/srivarra/czarr/issues
