# czarr

czarr reads and writes Zarr v3 arrays on NVIDIA GPUs. Compression runs through nvCOMP, file I/O runs through cuFile (GPUDirect Storage where the system supports it), and reads return `cupy.ndarray`.

Documentation: <https://czarr.readthedocs.io/>

## Installation

The NVIDIA wheels are CUDA-version-specific, so the CUDA build is an explicit choice:

```bash
pip install "czarr[cu12]"     # CUDA 12
pip install "czarr[cu13]"     # CUDA 13
```

`pip install czarr` without an extra installs no CUDA wheels and fails at import. Linux only.

## Usage

Through zarr, after one setup call:

```python
import czarr, zarr

czarr.configure_gpu()
arr = zarr.open_array("data.zarr")     # any zstd/lz4/gzip/zlib/blosc store,
out = arr[:]                           # including CPU-written ones; decodes on GPU
```

Or through the explicit API, which parses metadata once and holds no global state:

```python
from czarr.core import Array

arr = Array.open("data.zarr")
out = arr[0:4, :, 8:24]
```

Both return `cupy.ndarray`. Writing goes through zarr (`czarr.create_cuda_array` or `zarr.create_array` with czarr codecs). See the documentation for the how-to guides and API reference.

## Codec selection

| Situation | Codec |
|---|---|
| Default | `ANS` (30 GB/s decode, lowest scratch, scales past 4 GiB) |
| Small arrays, max throughput | `Bitcomp` (35 GB/s; OOMs at 4 GiB on A40) |
| Interoperable bitstream | `Zstd`, `LZ4`, `Gzip`, `Zlib` (bit-identical with the CPU libraries) |
| Existing blosc stores | decode-only via `czarr.Blosc`; write new data as `[Shuffle, Zstd]` |

Chunks below 256 KiB decode faster on CPU Blosc; the GPU advantage starts near 1 MiB and reaches 14-60x at 16 MiB and above.

## Chunk shapes for TCZYX microscopy

| Access pattern | Chunks |
|---|---|
| Whole timepoints, `arr[t]` | `(1, 1, Z, Y, X)` |
| Z-planes, `arr[t, c, z]` | `(1, 1, 1, Y, X)` |
| XY tiles | `(1, 1, 1, tile_y, tile_x)` |
| Full timeseries, `arr[:, c, z]` | `(T, 1, 1, Y, X)` |

Match the chunk to the access pattern so each read decodes only the chunks it touches. Sharded stores are handled: czarr replaces zarr's `sharding_indexed` codec with a variant that coalesces partial-shard reads.

## Measurements

H100, GPUDirect Storage, blosc `[bitshuffle, zstd]` store, 128-256 MiB chunks (`czarr-bench sweep zarr-read`; raw rows in `bench/results/`):

| Read stack | GiB/s |
|---|---|
| czarr (lowlevel, tier-1 fast path, and zarr pipeline are equivalent) | 20 |
| kvikio `GDSStore` + GPU codecs | 1.6-1.9 |
| `LocalStore` + numcodecs, CPU decode + H2D | 1.0 |

A40, codec only: Bitcomp decodes at 234 GB/s; zarr `arr[:]` with ANS at 64 MiB chunks reads at 69 GB/s against 3 GB/s for 16-thread CPU Blosc-lz4.

## Limitations

- The lowlevel read path covers zstd, blosc, and shuffle. Other codec chains fall back to the zarr pipeline, GPU-decoded where a czarr codec exists.
- `Blosc` is decode-only.
- Real GPUDirect Storage requires the `nvidia_fs` kernel module; without it cuFile stages through a pinned host bounce. cuFile cannot read tmpfs.
- `cupy` is a hard dependency; zarr's GPU buffer prototype is hard-coded to `cupy.ndarray`.

## License

MIT.
