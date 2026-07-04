---
icon: lucide/house
description: GPU-native reading and writing of Zarr v3 arrays — nvCOMP codecs, GPUDirect Storage, and an explicit low-level read path.
---

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

<div class="grid cards" markdown>

- **Getting started**

    ---

    Create, write, and read a GPU-compressed array through both tiers.

    [Getting started](getting-started.md)

- **How-to guides**

    ---

    CPU-store decode, explicit reads, GPUDirect Storage setup, benchmarking.

    [GPU-decode existing stores](how-to/read-cpu-stores.md)

- **Architecture**

    ---

    Why two tiers, how a read flows, and which designs were measured out.

    [The two-tier architecture](explanation/architecture.md)

- **API reference**

    ---

    Signatures and behavior, rendered from the docstrings.

    [czarr reference](reference/czarr.md)

</div>

## Installation

```bash
pip install "czarr[cu12]"     # CUDA 12
pip install "czarr[cu13]"     # CUDA 13
```

The NVIDIA wheels are CUDA-version-specific; `pip install czarr` without an extra fails at import. Linux only.
