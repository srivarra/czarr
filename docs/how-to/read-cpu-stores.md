---
icon: lucide/refresh-cw
description: Decode existing CPU-written zarr stores on the GPU without migration.
tags:
  - Codecs
---

# GPU-decode existing CPU stores

A zarr store written with CPU codecs (numcodecs or `zarr.codecs`) can be decoded on the GPU without migration or re-encoding.

## Zstd, LZ4, Gzip, Zlib

czarr's compat codecs consume and produce byte streams bit-identical with libzstd, liblz4, libdeflate, and libz. `configure_gpu()` registers them over the CPU codecs in zarr's registry, so a store's metadata resolves to the GPU implementations:

```python
import czarr, zarr

czarr.configure_gpu()
arr = zarr.open_array("legacy_cpu_written.zarr")
out = arr[:]                        # cupy.ndarray
```

To scope the registry changes, use the context-manager form:

```python
with czarr.configure_gpu():
    gpu_out = arr[:]
cpu_out = arr[:]                    # prior zarr configuration restored
```

## Blosc

Blosc containers with bitshuffle or byte shuffle, the common bioimaging layout, decode through nvCOMP's batched API plus czarr's shuffle kernels. The same `configure_gpu()` call covers them.

czarr does not encode blosc containers. Write new GPU-decodable data with `compressors=[czarr.Shuffle(...), czarr.Zstd()]`.

## Sharded stores

`configure_gpu()` registers a coalescing replacement for zarr's `sharding_indexed` codec. Partial-shard reads fuse adjacent inner chunks into single cuFile calls instead of zarr's one call per chunk. Nothing needs to be configured per store.

## Codec backends

The `Shuffle`, `Delta`, and `FixedScaleOffset` filters have more than one GPU implementation producing the same bitstream. To pin one:

```python
czarr.configure_gpu(codec_backend_overrides={"shuffle": "cupy"})
```

## Fallback behavior

Codec chains without a czarr GPU implementation decode through zarr's normal CPU path. `zarr.open_array(...).metadata.codecs` shows what a store uses.
