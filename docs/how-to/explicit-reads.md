# Explicit reads with core.Array and lowlevel

**Goal:** read zarr arrays on the GPU without touching zarr's global configuration — with per-call tuning, async duals, and access to each pipeline stage.

## When to reach for this tier

- Library code that must not mutate process-wide zarr state
- Hot loops where you want to parse metadata once and read many times
- Per-call control over threading, coalescing, output buffers, and streams
- Benchmarking or debugging a single stage in isolation

## The object API: `czarr.core.Array`

```python
from czarr.core import Array

arr = Array.open("data.zarr")          # exactly one zarr.json read
arr.shape, arr.dtype, arr.chunk_shape  # plain properties, no I/O
```

Reads return `cupy.ndarray` and accept basic indexing — integers, step-1 slices, `Ellipsis`, and tuples thereof. Selections are **ndim-preserving**: an integer keeps a length-1 axis (tier 1 squeezes on top for numpy semantics).

```python
out = arr[0:4, :, 8:24]

out = arr.retrieve_array_subset(       # same read, with per-call knobs
    (slice(0, 4), ..., slice(8, 24)),
    max_workers=16,                    # read threadpool size
    max_fused_bytes=64 << 20,          # coalescing cap per fused read
    out=preallocated,                  # exact shape+dtype cupy array
)

chunk = arr.retrieve_chunk((0, 0, 0))          # one decode unit, fill-value for missing
raw = arr.retrieve_encoded_chunk((0, 0, 0))    # pre-decode device bytes, or None
```

Unsupported selections (fancy indexing, `step != 1`, newaxis) and codec chains outside the lowlevel decode scope raise `NotImplementedError` / `TypeError` — fall back to tier 1 for those.

## Async

`AsyncArray` is the awaitable dual. Each call runs the sync path in a worker thread; cuFile reads release the GIL, so concurrent awaits genuinely overlap I/O:

```python
from czarr.core import AsyncArray

arr = AsyncArray.open("data.zarr")
a, b = await asyncio.gather(
    arr.retrieve_chunk((0, 0, 0)),
    arr.retrieve_chunk((1, 0, 0)),
)
```

## The staged plumbing: `czarr.lowlevel`

Every stage is callable on its own:

```python
from czarr import lowlevel

plan = lowlevel.open_plan("data.zarr")            # metadata -> DecodePlan (host-only, no cupy)
requests = plan.ranges(np.s_[0:4])                # coalesced byte ranges + chunk mapping
buffers = lowlevel.read(requests)                 # threaded cuFile reads -> device buffers
out = lowlevel.decode(plan, requests, buffers, np.s_[0:4])   # batched nvCOMP + scatter
```

Or the one-liner that composes them:

```python
out = lowlevel.read_array("data.zarr", np.s_[0:4])
out = lowlevel.read_array(None, np.s_[0:4], plan=plan)   # reuse a plan (and its shard-index cache)
```

`DecodePlan` is deliberately cupy-free — plans build on login nodes and in host-only tests. Shard indexes are parsed once and cached on the plan; drop the plan to drop the cache.

## Mixing tiers

`CudaZarrArray` (tier 1) already routes basic-indexing reads through a cached `core.Array`. Grab the underlying plan when you need to go lower:

```python
arr = czarr.open_cuda_array("data.zarr")   # zarr-compatible: writes, fancy indexing, groups
fast = arr._fast_array()                    # the cached core.Array (None if out of scope)
```
