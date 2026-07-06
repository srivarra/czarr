# czarr

<div align="center">

|             |                                                                                                                                                                                            |
| :---------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Status**  | [![Build][badge-build]][link-build] [![Tests][badge-test]][link-test] [![Documentation][badge-docs]][link-docs] [![codecov][badge-codecov]][link-codecov] [![prek][badge-prek]][link-prek] [![zizmor][badge-zizmor]][link-zizmor] |
|  **Meta**   | [![Hatch project][badge-hatch]][link-hatch] [![Ruff][badge-ruff]][link-ruff] [![ty][badge-ty]][link-ty] [![uv][badge-uv]][link-uv] [![Conventional Commits][badge-cc]][link-cc] [![Commitizen friendly][badge-cz]][link-cz] [![License][badge-license]][link-license]               |
| **Package** | [![PyPI][badge-pypi]][link-pypi] [![PyPI][badge-python-versions]][link-pypi]                                                                                                               |
|             |                                                                                                                                                                                            |

</div>

[badge-build]: https://github.com/srivarra/czarr/actions/workflows/build.yaml/badge.svg
[badge-test]: https://github.com/srivarra/czarr/actions/workflows/test.yaml/badge.svg
[badge-docs]: https://img.shields.io/readthedocs/czarr?logo=readthedocs
[badge-codecov]: https://codecov.io/gh/srivarra/czarr/graph/badge.svg
[badge-prek]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json
[badge-zizmor]: https://img.shields.io/badge/%F0%9F%8C%88-zizmor-white?labelColor=white
[badge-hatch]: https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg
[badge-ruff]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[badge-ty]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json
[badge-uv]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[badge-cc]: https://img.shields.io/badge/Conventional%20Commits-1.0.0-fe5196?logo=conventionalcommits&logoColor=white
[badge-cz]: https://img.shields.io/badge/commitizen-friendly-brightgreen.svg
[badge-license]: https://img.shields.io/badge/License-MIT-yellow.svg
[badge-pypi]: https://img.shields.io/pypi/v/czarr.svg?logo=pypi&label=PyPI&logoColor=gold
[badge-python-versions]: https://img.shields.io/pypi/pyversions/czarr.svg?logo=python&label=Python&logoColor=gold
[link-build]: https://github.com/srivarra/czarr/actions/workflows/build.yaml
[link-test]: https://github.com/srivarra/czarr/actions/workflows/test.yaml
[link-docs]: https://czarr.readthedocs.io/
[link-codecov]: https://codecov.io/gh/srivarra/czarr
[link-prek]: https://github.com/j178/prek
[link-zizmor]: https://github.com/zizmorcore/zizmor
[link-hatch]: https://github.com/pypa/hatch
[link-ruff]: https://github.com/astral-sh/ruff
[link-ty]: https://github.com/astral-sh/ty
[link-uv]: https://github.com/astral-sh/uv
[link-cc]: https://www.conventionalcommits.org
[link-cz]: https://commitizen-tools.github.io/commitizen/
[link-license]: https://opensource.org/licenses/MIT
[link-pypi]: https://pypi.org/project/czarr/

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

## An experiment in AI-assisted development

czarr is an experiment in AI-assisted development ("vibe coding"): most implementation was written by Claude Code, with a human directing design, reviewing changes, and interpreting benchmarks. The bet under test is that correctness can rest on hard gates rather than line-by-line authorship: a test suite CI runs on seven GPU architectures across CUDA 12 and 13, and benchmarks against real hardware. Read the code with the skepticism you would apply to any unfamiliar codebase.

## License

MIT.
