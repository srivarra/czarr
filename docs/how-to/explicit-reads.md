# Explicit reads

`czarr.core.Array` and `czarr.lowlevel` read zarr arrays without touching zarr's global configuration. Use them in library code that must not mutate process-wide state, in hot loops that read one array many times, and when a single pipeline stage needs to be run or measured in isolation.

## czarr.core.Array

```python
from czarr.core import Array

arr = Array.open("data.zarr")          # one zarr.json read, cached process-wide
arr.shape, arr.dtype, arr.chunk_shape  # properties, no I/O
```

Reads return `cupy.ndarray`. Selections take integers, step-1 slices, `Ellipsis`, and tuples of those. Selections are ndim-preserving: an integer keeps a length-1 axis.

```python
out = arr[0:4, :, 8:24]

out = arr.retrieve_array_subset(
    (slice(0, 4), ..., slice(8, 24)),
    max_workers=16,                    # read threadpool size
    max_fused_bytes=64 << 20,          # coalescing cap per fused read
    out=preallocated,                  # cupy array, exact shape and dtype
)

chunk = arr.retrieve_chunk((0, 0, 0))          # one decode unit; fill value if missing
raw = arr.retrieve_encoded_chunk((0, 0, 0))    # encoded device bytes, or None
```

Fancy indexing, `step != 1`, and newaxis raise `TypeError` or `NotImplementedError`, as do codec chains outside the lowlevel decode scope (zstd, blosc, shuffle). Handle those through zarr.

`Array.open` shares plans process-wide, revalidated against `zarr.json`'s stat signature, so reopening an array costs one `stat` and reuses the parsed shard indexes. Pass `cached=False` for a private plan when the store mutates underneath you.

## AsyncArray

```python
from czarr.core import AsyncArray

arr = AsyncArray.open("data.zarr")
a, b = await asyncio.gather(
    arr.retrieve_chunk((0, 0, 0)),
    arr.retrieve_chunk((1, 0, 0)),
)
```

Each call runs the sync path in a worker thread. cuFile reads release the GIL, so concurrent awaits overlap at the I/O level.

## czarr.lowlevel

The stages compose the same way `Array` uses them:

```python
from czarr import lowlevel

plan = lowlevel.open_plan("data.zarr")            # metadata -> DecodePlan
requests = plan.ranges(np.s_[0:4])                # coalesced byte ranges
buffers = lowlevel.read(requests)                 # threaded cuFile reads
out = lowlevel.decode(plan, requests, buffers, np.s_[0:4])
```

```python
out = lowlevel.read_array("data.zarr", np.s_[0:4])
out = lowlevel.read_array(None, np.s_[0:4], plan=plan)
```

`DecodePlan` does not import cupy; plans build on hosts without a GPU.

## Interaction with CudaZarrArray

`CudaZarrArray` routes basic-indexing reads through a cached `core.Array` and falls back to `zarr.Array.__getitem__` for everything else. `arr._fast_array()` returns the cached instance, or `None` when the store or codec chain is out of scope.
