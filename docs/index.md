# czarr

czarr reads and writes Zarr v3 arrays on NVIDIA GPUs. Compression runs through nvCOMP, file I/O runs through cuFile (GPUDirect Storage where the system supports it), and reads return `cupy.ndarray`.

There are two ways to use it. `configure_gpu()` plugs GPU codecs, a batched decode pipeline, and GPU buffers into zarr's extension points, so existing zarr code and existing CPU-written stores work unchanged:

```python
import czarr, zarr

czarr.configure_gpu()
arr = zarr.open_array("existing_store.zarr")
out = arr[:]
```

`czarr.core.Array` is the explicit alternative. It parses metadata once, holds no global state, and takes per-call options:

```python
from czarr.core import Array

arr = Array.open("existing_store.zarr")
out = arr[0:4, :, 8:24]
```

`CudaZarrArray`, returned by the czarr factories, connects the two: basic-indexing reads go through a cached `core.Array`, everything else through zarr.

New users should start with [Getting started](getting-started.md). The design rationale is in [Architecture](explanation/architecture.md).

## Installation

```bash
pip install "czarr[cu12]"     # CUDA 12
pip install "czarr[cu13]"     # CUDA 13
```

The NVIDIA wheels are CUDA-version-specific; `pip install czarr` without an extra fails at import. Linux only.
