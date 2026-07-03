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
| Default | `ANS` (lowest scratch, scales past 4 GiB) |
| Small arrays | `Bitcomp` (fastest below 4 GiB; OOMs above on A40) |
| Interoperable bitstream | `Zstd`, `LZ4`, `Gzip`, `Zlib` (bit-identical with the CPU libraries) |
| Existing blosc stores | decode-only via `czarr.Blosc`; write new data as `[Shuffle, Zstd]` |

GPU decode pays off from roughly 1 MiB chunks upward; below 256 KiB, multi-threaded CPU Blosc is faster.

## Chunk shapes for TCZYX microscopy

| Access pattern | Chunks |
|---|---|
| Whole timepoints, `arr[t]` | `(1, 1, Z, Y, X)` |
| Z-planes, `arr[t, c, z]` | `(1, 1, 1, Y, X)` |
| XY tiles | `(1, 1, 1, tile_y, tile_x)` |
| Full timeseries, `arr[:, c, z]` | `(T, 1, 1, Y, X)` |

Match the chunk to the access pattern so each read decodes only the chunks it touches. Sharded stores are handled: czarr replaces zarr's `sharding_indexed` codec with a variant that coalesces partial-shard reads.

## Limitations

- The lowlevel read path covers zstd, blosc, and shuffle. Other codec chains fall back to the zarr pipeline, GPU-decoded where a czarr codec exists.
- `Blosc` is decode-only.
- Real GPUDirect Storage requires the `nvidia_fs` kernel module; without it cuFile stages through a pinned host bounce. cuFile cannot read tmpfs.
- `cupy` is a hard dependency; zarr's GPU buffer prototype is hard-coded to `cupy.ndarray`.

## License

MIT.
