# Getting started

This tutorial takes you from a fresh environment to reading and writing a GPU-compressed Zarr array. You need a Linux machine with an NVIDIA GPU and either CUDA 12 or CUDA 13.

## 1. Install

```bash
pip install "czarr[cu12]"     # or czarr[cu13] to match your CUDA version
```

Verify the install:

```python
import czarr
print(czarr.__version__)
```

## 2. Configure the GPU path

One call wires zarr up for GPU work — batched codec pipeline, GPU buffer prototypes, and codec registrations:

```python
import czarr

czarr.configure_gpu()
```

Everything after this point is ordinary zarr. `configure_gpu` also works as a context manager (`with czarr.configure_gpu(): ...`) when you want the prior zarr configuration restored on exit.

## 3. Create and write an array

```python
import cupy as cp
import numpy as np

arr = czarr.create_cuda_array(
    store="quickstart.zarr",
    shape=(16, 1024, 1024),
    chunks=(4, 1024, 1024),          # ~16 MiB chunks — GPU-friendly territory
    dtype="float32",
    compressors=[czarr.ANS()],       # nvCOMP-native, the safe default
)

data = np.random.default_rng(0).standard_normal((16, 1024, 1024), dtype="float32")
arr[:] = data                        # accepts numpy or cupy
```

`create_cuda_array` wraps a string/`Path` store in [`GPULocalStore`][czarr.GPULocalStore] automatically, so reads go disk → GPU via cuFile where the system supports it.

## 4. Read it back

```python
out = arr[:]                         # whole array
print(type(out))                     # <class 'cupy.ndarray'> — it never touched the host

plane = arr[3]                       # basic selections use the lowlevel fast path:
tile = arr[2:6, 256:512, 256:512]    # coalesced reads, one batched nvCOMP decode
```

Verify the round trip:

```python
np.testing.assert_array_equal(cp.asnumpy(out), data)
```

## 5. Same array, explicit tier

The identical store reads through [`czarr.core.Array`][czarr.core.Array] with no global configuration at all — useful in libraries and long-lived services that must not mutate zarr's process-wide state:

```python
from czarr.core import Array

a = Array.open("quickstart.zarr")    # one zarr.json read
print(a.shape, a.dtype, a.chunk_shape)

out = a[2:6]                         # cupy.ndarray
```

## Where next

- Have existing CPU-written stores? → [GPU-decode existing stores](how-to/read-cpu-stores.md)
- Want per-call tuning, async reads, or raw encoded bytes? → [Explicit reads](how-to/explicit-reads.md)
- Setting up a GDS machine? → [GPUDirect Storage setup](how-to/gds.md)
