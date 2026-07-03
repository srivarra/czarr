# GPU-decode existing CPU-written stores

**Goal:** read a zarr store written with CPU codecs (numcodecs / `zarr.codecs`) on the GPU, without migrating or re-encoding anything.

## Zstd, LZ4, Gzip, Zlib

czarr's compat codecs produce and consume byte streams bit-identical with libzstd / liblz4 / libdeflate / libz. After `configure_gpu()` they shadow the CPU codecs in zarr's registry, so the store's metadata resolves to the GPU implementations transparently:

```python
import czarr, zarr

czarr.configure_gpu()
arr = zarr.open_array("legacy_cpu_written.zarr")   # written years ago with numcodecs.Zstd
out = arr[:]                                       # cupy.ndarray, GPU decoded
```

No store changes, no flags. Reverting is equally clean — use the context-manager form to scope it:

```python
with czarr.configure_gpu():
    gpu_out = arr[:]        # GPU decode inside the block
cpu_out = arr[:]            # stock zarr behavior restored
```

## Blosc (bitshuffle/shuffle + zstd)

The common bioimaging layout — blosc containers with bitshuffle — decodes on the GPU through nvCOMP's native batched API plus czarr's shuffle kernels:

```python
czarr.configure_gpu()
arr = zarr.open_array("waveorder_style_store.zarr")   # blosc [bitshuffle, zstd]
out = arr[:]
```

!!! note "Blosc is decode-only"

    czarr never encodes blosc containers. For new GPU-decodable data, write with `compressors=[czarr.Shuffle(...), czarr.Zstd()]` instead.

## Sharded stores

Nothing extra to do. `configure_gpu()` registers czarr's coalescing override for `sharding_indexed`, so partial-shard reads fuse adjacent inner chunks into single cuFile calls instead of zarr's one-`get`-per-chunk loop (22× on the H100 microbench for 32×64 KiB inner chunks).

## Pinning a codec backend

Filter codecs (`Shuffle`, `Delta`, `FixedScaleOffset`) have interchangeable GPU implementations. The bitstream is identical either way; pin one per codec if profiling says so:

```python
czarr.configure_gpu(codec_backend_overrides={"shuffle": "cupy"})
```

## What falls back

Codec chains czarr has no GPU implementation for decode through zarr's normal CPU path — the read still works, it's just not GPU-accelerated. Check what a store uses with `zarr.open_array(...).metadata.codecs`.
