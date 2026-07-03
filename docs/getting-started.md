# Getting started

This tutorial creates a GPU-compressed Zarr array, writes to it, and reads it back both through zarr and through the explicit API. It assumes a Linux machine with an NVIDIA GPU and CUDA 12 or 13.

## Install

```bash
pip install "czarr[cu12]"     # or czarr[cu13]
```

```python
import czarr
print(czarr.__version__)
```

## Configure zarr for the GPU

```python
import czarr

czarr.configure_gpu()
```

This registers the GPU codecs, sets the batched decode pipeline, and switches zarr's buffer prototypes to device memory. It also works as a context manager, restoring the prior zarr configuration on exit.

## Create and write

```python
import cupy as cp
import numpy as np

arr = czarr.create_cuda_array(
    store="quickstart.zarr",
    shape=(16, 1024, 1024),
    chunks=(4, 1024, 1024),
    dtype="float32",
    compressors=[czarr.ANS()],
)

data = np.random.default_rng(0).standard_normal((16, 1024, 1024), dtype="float32")
arr[:] = data
```

`create_cuda_array` wraps a string or `Path` store in [`GPULocalStore`][czarr.GPULocalStore], which reads disk to GPU through cuFile where the system supports it. Writes accept numpy or cupy.

## Read

```python
out = arr[:]
print(type(out))                     # <class 'cupy.ndarray'>

plane = arr[3]
tile = arr[2:6, 256:512, 256:512]
```

Basic selections (integers, step-1 slices, `Ellipsis`) go through the lowlevel path: coalesced cuFile reads followed by one batched nvCOMP decode. Other selections go through zarr.

```python
np.testing.assert_array_equal(cp.asnumpy(out), data)
```

## Read through the explicit API

The same store, without `configure_gpu` or any other global state:

```python
from czarr.core import Array

a = Array.open("quickstart.zarr")    # one zarr.json read
print(a.shape, a.dtype, a.chunk_shape)

out = a[2:6]
```

[Explicit reads](how-to/explicit-reads.md) covers per-call options, the async variant, and the individual pipeline stages. [GPUDirect Storage setup](how-to/gds.md) covers cuFile modes and diagnostics.
