# Agent 2: CCCL `cuda.compute` & `cuda.coop._experimental` for czarr Codec Implementation

**Scope:** Map every algorithm, iterator, and operator surface in NVIDIA's CCCL Python bindings — `cuda.compute` (device-wide algorithms) and `cuda.coop._experimental` (block- and warp-level cooperative primitives) — and judge feasibility of composing czarr's nvCOMP-replacement codecs from them.

**Sources (all fetched 2026-05-23):**
- Landing: https://nvidia.github.io/cccl/unstable/python/compute/index.html
- API reference: https://nvidia.github.io/cccl/unstable/python/compute_api.html
- Coop API: https://nvidia.github.io/cccl/unstable/python/coop_api.html
- Coop overview: https://nvidia.github.io/cccl/unstable/python/coop.html
- Developer overview: https://nvidia.github.io/cccl/unstable/python/compute/developer_overview.html
- Setup: https://nvidia.github.io/cccl/unstable/python/setup.html
- Source: https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl
- Examples: https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/compute/examples

---

## Executive Summary

CCCL Python is split into **two layers** worth caring about:

1. **`cuda.compute`** — device-wide algorithms (CUB/Thrust-equivalent). Public beta. Composable primitives (`reduce_into`, `inclusive_scan`, `radix_sort`, `merge_sort`, `histogram_even`, `unary_transform`, `select`, `unique_by_key`, `segmented_*`, `three_way_partition`, `lower_bound`/`upper_bound`) plus rich iterators (`Counting`, `Constant`, `Transform[Output]`, `Zip`, `Permutation`, `Reverse`, `Shuffle`, `CacheModified`, `Discard`). Custom operators are JIT-compiled from Python lambdas via numba-cuda to LTO-IR, then nvJitLink'd into the generated kernel — so operators are **inlined**, not called through function pointers.
2. **`cuda.coop._experimental`** — block-/warp-level collective primitives (CUB BlockReduce, BlockScan, BlockRadixSort, BlockMergeSort, BlockLoad, BlockStore, BlockExchange; WarpReduce, WarpScan, WarpMergeSort). These are **callable from inside a `@numba.cuda.jit` kernel** linked with the generated LTO-IR files. The full namespace and API are explicitly slated to change.

**Key positive findings for czarr:**

- **Iterators are first-class and fuse** — `TransformIterator` (input), `TransformOutputIterator`, `ZipIterator`, `PermutationIterator`, `ReverseIterator` all compose with any device-wide algorithm. The `running_average.py` example chains `ZipIterator(d_in, ConstantIterator(1)) → inclusive_scan → TransformOutputIterator(d_out, divide)` in a single GPU pass with one kernel launch. This is the kind of fusion that makes shuffle/delta/bitround codecs cheap.
- **`RawOp`** lets you ship pre-compiled C++ LTO-IR as an operator, with optional state bytes (used for atomic counters, scratch pointers, etc.). This means C++-written hot loops can still be composed in Python with device-wide algorithms — useful escape hatch when Numba's Python subset can't express something.
- **Block-level primitives exist as Python-callable Numba kernel components** — `coop.block.make_radix_sort_keys`, `make_merge_sort_keys`, `make_load`/`make_store` with all the CUB load/store algorithms (`direct`, `striped`, `vectorize`, `transpose`, `warp_transpose`, `warp_transpose_timesliced`), `make_exchange` for striped-to-blocked rearrangement. Means a custom tile kernel for shuffle/byteshuffle is feasible without writing C++.
- All algorithms accept a `stream=` kwarg, all device arrays are passed via `__cuda_array_interface__`, and there are no implicit syncs after launch. Async chaining works.

**Key negative findings:**

- **No compression primitives.** No Huffman, ANS, rANS, range coder, LZ77/LZ4 matcher, dictionary builder. No bit-packing helpers, no varint encoders. CCCL is *parallel algorithms*, not *compression*. Any entropy-coded codec (Zstd/LZ4/Deflate/Snappy/Bitcomp/ANS/Cascaded) needs to be written from scratch on top of these primitives — and the parallel-Huffman / parallel-rANS literature is what we'd have to implement, not pull off a shelf.
- **No fixed-block-size LZ matcher, no suffix array, no hash chain** — would need to be hand-rolled with custom Numba/C++ device code, then exposed as either a `RawOp` or a raw `@cuda.jit` kernel.
- **Stable URL does not exist** (`https://nvidia.github.io/cccl/python/...` returns 404; only `/unstable/` resolves). The docs explicitly warn: *"cuda.compute is in public beta. The API is subject to change without notice"* and *"cuda.coop._experimental is in public beta. The API is subject to change without notice."* The coop overview adds: *"the Python package namespace and API details will change in a subsequent release."*
- **Numba-CUDA is the compilation pipeline.** All Python operator support inherits Numba's Python subset — no full Python, no exceptions, no allocations, no recursion. Stateful operators need either closure-captured device arrays or `RawOp` state bytes packed manually with `struct.pack`. This is workable but it is *not* "write any Python, GPU goes brrr."
- **Caches grow unbounded.** Compiled cubins live in process memory keyed on dtype + op bytecode + closure contents + compute capability. There's a `cuda.compute.clear_all_caches()` escape hatch but no automatic eviction. For long-lived workers, this is a leak.
- **No explicit allocator/memory-resource integration.** Algorithms accept temp-storage buffers exposing `__cuda_array_interface__` (so RMM/CuPy pools work), but there's no documented hook into cudaMemPool or graph-managed allocations. cuda.core's resource API isn't bridged in the algorithm signatures.
- **No CUDA Graph capture documented.** The developer overview is silent on graph capture/replay; stream kwargs are there but there's no `make_*` flag that says "this is graph-safe."

**Verdict for czarr:**

- **Filters** (Shuffle, BitRound, FixedScaleOffset, Delta) — *cleanly composable* in `cuda.compute` with iterators + transforms. This is where CCCL shines for us.
- **Compressors** (Zstd, LZ4, Snappy, Deflate, GDeflate, Bitcomp, ANS, Cascaded) — *not directly composable*. You'd be implementing parallel Huffman/rANS/LZ on top of `radix_sort` + `inclusive_scan` + `segmented_reduce` + custom `RawOp`/block kernels. That's a multi-month research project per codec. The pragmatic path is: write the algorithm in C++ (LTO-IR), expose via `RawOp` or a raw `@cuda.jit` kernel, and let `cuda.compute` handle the surrounding glue.
- **Hopper-only path:** none of the CCCL primitives require sm_90, none use TMA/cuTile (cuTile is a separate library). So we *escape* the current cuTile/Hopper Shuffle bug just by using `cuda.compute` for Shuffle.

---

## Stability and Maturity Surface

| Module | Status | Notes |
|---|---|---|
| `cuda.compute` | "public beta" | Docs only at `/unstable/` path; stable URL 404s. API subject to change without notice. |
| `cuda.coop._experimental` | "public beta" + `_experimental` namespace | Docs say *"namespace and API details will change in a subsequent release."* |
| `cuda.cccl` | exists as a sibling package alongside `compute`/`coop` in source tree (`python/cuda_cccl/cuda/cccl/`) but not documented on the landing page | Inferred to be the umbrella/installation package. |
| `cuda.cooperative` (legacy) | **not mentioned** in current docs | The new `cuda.coop._experimental` appears to supersede the older `cuda.cooperative` mentioned in older Conda packages, though no migration page is published. |
| `cuda.parallel` (legacy) | **not mentioned** | `cuda.compute` appears to supersede; again no migration page. |

**Install (CUDA 13 / Python ≥3.9 / SM ≥6.0 / Linux Ubuntu 20.04+):**

```bash
pip install cuda-cccl[cu13]              # full install with Numba
pip install cuda-cccl[minimal-cu13]      # without Numba (RawOp only, no Python operators)
pip install cuda-cccl[sysctk13]          # system CUDA toolkit instead of pip packages
```

For czarr's stack (Python 3.13, CUDA 13.1, Bruno NFS) — `pip install cuda-cccl[cu13]` should be the right move. The minimal variant skips Numba entirely and forces all custom ops through `RawOp` (C++ LTO-IR) — could be a strategy for production deployment to avoid the Numba JIT memory footprint.

---

## Architecture: How the Compilation Pipeline Works

From the developer overview, the pipeline is:

1. **Python operator** (lambda / `def`) handed to a `cuda.compute` algorithm.
2. **numba-cuda compiles it to LTO-IR** (link-time intermediate representation, not PTX). Closure-captured device arrays become pointer constants; host scalars become immediate constants.
3. The CCCL C++ side **generates the kernel source as a string** templated on the input/output types and iterator descriptors, then runs it through **NVRTC** to produce another LTO-IR.
4. **`nvJitLink` links the operator LTO-IR with the generated kernel LTO-IR** into a single cubin. Because LTO-IR is post-frontend but pre-codegen, **the operator gets inlined into the kernel** — no function-pointer indirection, no runtime dispatch. This is why CCCL claims near-CUB performance from Python.
5. The cubin is cached in-process keyed on `(dtype, iterator kind, op bytecode + closure hash, compute_capability, algorithm params)`. Repeated calls hit the cache. `cuda.compute.clear_all_caches()` flushes.

**Implication for czarr:** Cold-launch cost = NVRTC + nvJitLink (hundreds of ms first time, depending on op complexity). Warm-launch = pure kernel dispatch. Compatible with our existing "first call is slow" benchmark pattern. The cache key includes closure-captured device arrays *by pointer/shape/dtype*, not contents — so swapping buffers does not invalidate the cache, which is exactly what we want for a streaming Zarr read pipeline.

---

# Per-Primitive Catalog (`cuda.compute.algorithms`)

All algorithms come in two flavors:

- **Immediate** (`reduce_into(...)`) — allocates temp storage internally, executes, returns. Convenient but allocates on every call.
- **Object-based** (`make_reduce_into(...)`) — two-phase: (1) construct callable, (2) call with `temp_storage=None` to query size, (3) allocate user buffer, (4) call with `temp_storage=<buf>` to execute. Reusable across calls with same shapes.

All signatures are **keyword-only**. Naming convention: `d_*` = device array (CuPy/PyTorch/Numba — anything with `__cuda_array_interface__`), `h_*` = host array (NumPy).

## Reductions

### `reduce_into` / `make_reduce_into`
```python
cuda.compute.reduce_into(*, d_in, d_out, num_items, op, h_init, stream=None)
```

- **Op signature:** `(T, T) -> T` where `T` matches `h_init.dtype`.
- **Built-in ops via `OpKind`:** `PLUS`, `MINUS`, `MULTIPLIES`, `MAXIMUM`, `MINIMUM`, `BIT_AND`, `BIT_OR`, `BIT_XOR`, etc. (full list below).
- **Performance hint:** Throughput-bound. Latency is single kernel launch.
- **User buffers:** `d_out` is user-provided (size-1 array). Temp storage is allocated internally in immediate form; user-controlled in object form.
- **Maturity:** Documented, has multiple tested examples, stable example surface.

**Codec relevance:** Useful as a tail step for any codec that needs a sum/max/min — e.g. computing a global min for FixedScaleOffset, or a population count for bit-packing decisions.

### `segmented_reduce` / `make_segmented_reduce`
```python
cuda.compute.segmented_reduce(*, d_in, d_out, num_segments,
    start_offsets_in, end_offsets_in, op, h_init,
    max_segment_size=None, stream=None)
```

- Independently reduces each segment defined by offset arrays.
- `max_segment_size` is an *optional optimization hint* — provided, dispatches to a specialized kernel.
- **Codec relevance:** Critical for per-block summary statistics over Zarr chunks. E.g. per-shard max/min for delta encoding, per-shard histograms for entropy modeling.

## Scans

### `inclusive_scan` / `exclusive_scan` (and `make_*`)
```python
cuda.compute.inclusive_scan(*, d_in, d_out, op, init_value, num_items, stream=None)
cuda.compute.exclusive_scan(*, d_in, d_out, op, init_value, num_items, stream=None)
```

- `init_value` can be `None` for `inclusive_scan` (skip the initial offset).
- **Op signature:** `(T, T) -> T`.
- **Performance hint:** ~bandwidth-bound for sum scans; small overhead per launch.
- **Codec relevance:** *Heavily relevant.* Prefix-sum on compressed chunk lengths to compute output offsets, prefix-sum on selected-bit-counts for stream packing, cumulative byte offsets for variable-length encoding. Every parallel-compression algorithm I know uses prefix-sums for output positioning.

### Composable with `TransformOutputIterator`
The `running_average.py` example fuses *zip → inclusive_scan → divide → write* into a single kernel:

```python
@gpu_struct
class SumAndCount:
    sum: np.float32
    count: np.int32

def add_op(x1: SumAndCount, x2: SumAndCount) -> SumAndCount:
    return SumAndCount(x1.sum + x2.sum, x1.count + x2.count)

def write_op(x: SumAndCount) -> np.float32:
    return x.sum / x.count

it_input = ZipIterator(d_input, ConstantIterator(np.int32(1)))
it_output = TransformOutputIterator(d_output, write_op)

cuda.compute.inclusive_scan(d_in=it_input, d_out=it_output,
    op=add_op, init_value=SumAndCount(0.0, 0), num_items=len(d_input))
```

This pattern — fused scan + transform — is exactly how you'd implement delta-decode-then-prefix-sum (RLV decode) in one pass.

## Transforms

### `unary_transform` / `binary_transform` (and `make_*`)
```python
cuda.compute.unary_transform(*, d_in, d_out, op, num_items, stream=None)
cuda.compute.binary_transform(*, d_in1, d_in2, d_out, op, num_items, stream=None)
```

- **Critical note from docs:** *"The `op` function can reference device arrays as globals or closures - they will be automatically captured as state arrays, enabling stateful operations like counting."*
- **Op signature:** `(T) -> U` for unary; `(T1, T2) -> U` for binary.
- **Struct support:** With `@gpu_struct`-decorated types, ops must have explicit type annotations (`(p1: Point2D, p2: Point2D) -> Point2D`) for Numba to infer correctly.
- **User buffers:** Both d_in and d_out are user-provided. No temp storage needed for transforms.

**Codec relevance:** This is the *workhorse* for filters. ByteShuffle = `unary_transform` with a permutation op. BitRound = `unary_transform` with a mask op. FixedScaleOffset = `binary_transform` with the scale/offset constants. Delta = `binary_transform` paired with a shifted iterator.

## Sorts

### `radix_sort` / `make_radix_sort`
```python
cuda.compute.radix_sort(*, d_in_keys, d_out_keys, d_in_values=None, d_out_values=None,
    num_items, order: SortOrder, begin_bit=None, end_bit=None, stream=None)
```

- **`SortOrder.ASCENDING` / `DESCENDING`.**
- **Optional `begin_bit`/`end_bit`** — sort only on a subset of bits. Performance speedup proportional to bits.
- **`DoubleBuffer` support** — pass `DoubleBuffer(buf_a, buf_b)` instead of in/out arrays to halve memory usage; result is in `db.current()`.
- **Performance hint:** Throughput-bound. CUB radix sort is roughly 30 GB/s on H100 for 32-bit keys.
- **Codec relevance:** Indirect — could be used as a building block for parallel Huffman tree construction (sort symbols by frequency) or for column-major reordering of high-cardinality data before entropy coding. Not a primary win.

### `merge_sort` / `make_merge_sort`
```python
cuda.compute.merge_sort(*, d_in_keys, d_in_values=None,
    d_out_keys, d_out_values=None, num_items, op, stream=None)
```

- **Op signature:** `(T, T) -> int8` (strict weak ordering — docs warn that `>=` semantics will silently corrupt memory).
- Supports arbitrary comparators (radix_sort doesn't).
- **Codec relevance:** Same as radix — useful for parallel-Huffman, otherwise indirect.

### `segmented_sort` / `make_segmented_sort`
```python
cuda.compute.segmented_sort(*, d_in_keys, d_out_keys=None, d_in_values=None, d_out_values=None,
    num_items, num_segments, start_offsets_in, end_offsets_in, order, stream=None)
```

- Per-segment sort using offsets. Both array-of-keys and `DoubleBuffer` inputs supported.
- **Codec relevance:** Per-block histogram sort for entropy modeling.

## Selection / Partitioning

### `select` / `make_select`
```python
cuda.compute.select(*, d_in, d_out, d_num_selected_out, cond, num_items, stream=None)
```

- **`cond` signature:** `(T) -> uint8` (1 = keep, 0 = drop).
- Stateful predicates: closure captures device arrays automatically (e.g. atomic counter).
- Output is compacted; count written to `d_num_selected_out[0]`.

**Codec relevance:** Run-length decoding (select non-repeat positions), sparse encoding (select non-zero).

### `unique_by_key` / `make_unique_by_key`
```python
cuda.compute.unique_by_key(*, d_in_keys, d_in_items, d_out_keys, d_out_items,
    d_out_num_selected, op, num_items, stream=None)
```

- Keeps the first key + value from each run of consecutive equal keys.
- **Codec relevance:** Run-length encoding (RLE), which Cascaded uses as a stage. Plus useful for detecting palettable input chunks.

### `three_way_partition` / `make_three_way_partition`
```python
cuda.compute.three_way_partition(*, d_in,
    d_first_part_out, d_second_part_out, d_unselected_out, d_num_selected_out,
    select_first_part_op, select_second_part_op, num_items, stream=None)
```

- Partitions into three buckets with two unary predicates.
- **Codec relevance:** Slight — can be used to separate literals from match references in an LZ scheme without a second pass, but rarely the dominant cost.

## Histograms

### `histogram_even` / `make_histogram_even`
```python
cuda.compute.histogram_even(*, d_samples, d_histogram, num_output_levels,
    lower_level, upper_level, num_samples, stream=None)
```

- Evenly-spaced bins between `[lower_level, upper_level)`.
- Note: no `histogram_range` (custom bin edges). Only even bins.
- **Codec relevance:** Build symbol frequency tables for entropy coding. The 256-bin uint8 histogram is the foundational step for Huffman/rANS code-table generation. Big primitive for us.

## Binary search

### `lower_bound` / `upper_bound` (and `make_*`)
```python
cuda.compute.lower_bound(*, d_data, num_items, d_values, num_values, d_out,
    comp=None, stream=None)
```

- Parallel binary search of sorted `d_data` for each of `d_values`. Default `comp = OpKind.LESS`.
- Result is `uintp` (size_t) indices.
- **Codec relevance:** Symbol lookup against a sorted code table in custom decoders.

## Utility Types

### `DoubleBuffer`
```python
class DoubleBuffer:
    def __init__(self, d_current, d_alternate)
    def current() -> DeviceArrayLike
    def alternate() -> DeviceArrayLike
```

Used by `radix_sort` and `segmented_sort` to halve memory by ping-ponging.

### `SortOrder` enum
- `ASCENDING = 0`, `DESCENDING = 1`.

---

# Iterators (`cuda.compute.iterators`)

All iterators inherit from `IteratorBase` and implement `to_cccl_iter()`. They compose freely — any iterator can wrap any other (subject to dtype rules). Key property: **iterators are lazy** and **fuse** with algorithms into a single kernel.

## `CountingIterator`
```python
CountingIterator(start: np.number)
```
Emits `start, start+1, start+2, ...`. Like `thrust::counting_iterator`. Useful for generating indices on-the-fly.

## `ConstantIterator`
```python
ConstantIterator(value: np.number)
```
Emits the same value forever. Useful for "join an array with a 1" before scan (see `running_average` example).

## `TransformIterator`
```python
TransformIterator(underlying, transform_op, value_type=None, is_input=True)
```
- Applies `transform_op: (T) -> U` lazily on read.
- **Operator can be a Python lambda, OpKind, or RawOp.**
- **Composes:** `TransformIterator(d_in, lambda x: x**2)` then `reduce_into(d_in=that, op=PLUS)` = sum-of-squares in one kernel.

## `TransformOutputIterator`
```python
TransformOutputIterator(underlying, transform_op, output_value_type=None)
```
- Applies `transform_op` lazily on *write* — receives the algorithm's result, transforms it, writes to `underlying`.
- **Requires type annotations** on `transform_op` because Numba can't always infer.

## `ZipIterator`
```python
ZipIterator(*iters)   # or ZipIterator([it1, it2, it3])
```
- Emits tuples of values from N underlying iterators/arrays.
- Used in struct-key sorts: `ZipIterator(values, indices) → sort → output`.
- The reduction with `(index, value)` pair to find argmax is one of the documented examples.

## `PermutationIterator`
```python
PermutationIterator(values, indices)
```
- At position `i`, yields `values[indices[i]]`. Like `thrust::permutation_iterator`.
- **Codec relevance:** Byteshuffle = `PermutationIterator(d_in, shuffle_index_table)` then write through `unary_transform` with identity op. **This is the cleanest in-CCCL way to implement byteshuffle.**

## `ReverseIterator`
```python
ReverseIterator(underlying)
```
Reverses iteration direction.

## `ShuffleIterator`
```python
ShuffleIterator(num_items, seed=0, *, _current_index=0)
```
- Produces a *deterministic random permutation* of `[0, num_items)` parameterized by `seed`.
- **Codec relevance:** Not byteshuffle (that's a fixed pattern); this is for shuffling training data, sampling. Probably not useful for compression.

## `CacheModifiedInputIterator`
```python
CacheModifiedInputIterator(array, modifier='stream' | 'global' | 'volatile')
```
- Wraps a device pointer with PTX cache modifiers.
  - `'stream'` → `ld.global.cs` — no-reuse hint
  - `'global'` → `ld.global.cg` — L2-only cache
  - `'volatile'` → `ld.global.cv` — always go to memory
- Element types 1/2/4/8/16 bytes.
- **Codec relevance:** Performance knob for streaming decompression where input is consumed once. Skipping L1 cache pollution could be a meaningful Bruno-bandwidth win.

## `DiscardIterator`
```python
DiscardIterator(reference_iterator=None)
```
- Swallows writes — useful when you want a count without writing the actual selected values.
- **Codec relevance:** Two-phase compression: first pass with `DiscardIterator` as output to count compressed size; allocate exact output buffer; second pass for real.

---

# `cuda.coop._experimental` — Block- and Warp-level Primitives

These are **CUB BlockLoad/BlockStore/BlockReduce/BlockScan/BlockRadixSort/BlockMergeSort/BlockExchange and WarpReduce/WarpScan/WarpMergeSort exposed for use inside `@numba.cuda.jit` kernels**. You build the primitive with `coop.block.make_*(...)`, get a `.files` attribute (LTO-IR files to link), and call it from inside a `cuda.jit(link=...)` kernel.

**This is the path to writing custom tile kernels in Python.**

## General Invocation Pattern

```python
import numba
from numba import cuda
import cuda.coop._experimental as coop

block_reduce = coop.block.make_reduce(numba.int32, 128, lambda a,b: a if a>b else b)

@cuda.jit(link=block_reduce.files)
def kernel(d_in, d_out):
    val = block_reduce(d_in[cuda.threadIdx.x])
    if cuda.threadIdx.x == 0:
        d_out[0] = val

kernel[1, 128](d_input, d_output)
```

The primitive is **inlined into the JIT-compiled kernel via LTO-IR linking** — same compilation path as `cuda.compute`, just consumed at block scope instead of device scope.

## Block Primitives

### `block.make_load` / `block.make_store`
```python
block.make_load(dtype, threads_per_block, items_per_thread=1, algorithm='direct')
block.make_store(dtype, threads_per_block, items_per_thread=1, algorithm='direct')
```

- **Algorithms:** `'direct'`, `'striped'`, `'vectorize'`, `'transpose'`, `'warp_transpose'`, `'warp_transpose_timesliced'`. Same set as CUB's BlockLoad/BlockStore.
- **Codec relevance:** *Critical.* The whole point of `cuTile` was the tiled load/store primitive — and `make_load`/`make_store` is exactly that. We can write a byteshuffle tile kernel with `make_load(..., algorithm='vectorize')` (coalesced wide loads) → in-register byte permute → `make_store(..., algorithm='striped')` (transposed writes).

### `block.make_exchange`
```python
block.make_exchange(BlockExchangeType.StripedToBlocked,
    dtype, threads_per_block, items_per_thread, warp_time_slicing=False)
```

- Shared-memory transpose between striped and blocked layouts. Allocates `temp_storage_bytes` in shared memory.
- **`warp_time_slicing=True`** reduces shared memory at the cost of parallelism.
- **Only `StripedToBlocked = 1`** is currently in the enum (`BlockedToStriped` is missing; this is a gap).
- **Codec relevance:** Direct replacement for cuTile's transpose path. Shuffle byte-0/1/2/... → bytewise transposed planes via `make_exchange` is mechanically straightforward.

### `block.make_reduce` / `block.make_sum`
```python
block.make_reduce(dtype, threads_per_block, binary_op, items_per_thread=1, algorithm='warp_reductions')
block.make_sum(dtype, threads_per_block, items_per_thread=1, algorithm='warp_reductions')
```

- **Algorithms:** `'raking'`, `'raking_commutative_only'`, `'warp_reductions'`.
- Warning: *"The return value is undefined in threads other than thread 0."*
- Three invocation forms: single-item-per-thread, array-per-thread, single-item with `num_valid` partial.

### `block.make_scan` / `block.make_inclusive_sum` / `block.make_exclusive_sum`
```python
block.make_scan(dtype, threads_per_block, items_per_thread,
    initial_value=None, mode='exclusive'|'inclusive', scan_op='+',
    block_prefix_callback_op=None, algorithm='raking')
```

- **Algorithms:** `'raking'`, `'raking_memoize'`, `'warp_scans'`.
- **Scan ops:** `'add'/'plus'`, `'mul'/'multiplies'`, `'min'/'minimum'`, `'max'/'maximum'`, `'bit_and'`, `'bit_or'`, `'bit_xor'`, single-char aliases (`'+'`, `'*'`, `'&'`, `'|'`, `'^'`), or arbitrary Callable.
- **`block_prefix_callback_op`** — lets you supply a per-block running offset; chains multiple block scans into a device-wide scan. Equivalent to CUB's BlockScan callback pattern.

### `block.make_radix_sort_keys` / `block.make_radix_sort_keys_descending`
```python
block.make_radix_sort_keys(dtype, threads_per_block, items_per_thread)
```

- Block-scope radix sort, blocked arrangement. No keys/values variant documented yet — just keys.

### `block.make_merge_sort_keys`
```python
block.make_merge_sort_keys(dtype, threads_per_block, items_per_thread, compare_op)
```

- Block-scope merge sort with arbitrary comparator.

## Warp Primitives

### `warp.make_reduce` / `warp.make_sum` / `warp.make_exclusive_sum` / `warp.make_merge_sort_keys`
Same patterns as block-level but warp-scope (typically 32 threads). Result lives in lane 0 for reductions.

## What's NOT in `cuda.coop._experimental`

Compared to CUB, **missing**:
- BlockHistogram (CUB has it, coop doesn't)
- BlockDiscontinuity (run-detection — CUB has it, coop doesn't)
- BlockRunLengthDecode
- BlockAdjacentDifference
- Cooperative groups (the `cg::*` namespace — these are exposed via libcudacxx C++ but not in `coop` Python)
- `BlockedToStriped` direction in `BlockExchange` (only StripedToBlocked enum value exists)
- WarpExchange, WarpStore, WarpLoad

These gaps matter: BlockHistogram would be a direct win for entropy modeling, and BlockDiscontinuity is exactly what you want for RLE/delta passes. **Implication:** custom kernels for those need to compose `block.make_*` primitives manually (e.g. histogram = `make_exclusive_sum` + scatter), or be written in C++ and wrapped via `RawOp`.

---

# Operators (`cuda.compute.op`)

## `OpKind` enum (complete list)

```
PLUS, MINUS, MULTIPLIES, DIVIDES, MODULUS,
EQUAL_TO, NOT_EQUAL_TO,
GREATER, LESS, GREATER_EQUAL, LESS_EQUAL,
LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
BIT_AND, BIT_OR, BIT_XOR, BIT_NOT,
IDENTITY, NEGATE,
MINIMUM, MAXIMUM
```

22 predefined ops. Per the docs: *"Built-in OpKind values are preferred over user-defined equivalents for superior performance"* — they're directly compiled to CUB-equivalent code paths.

## `Operator` type alias

```python
Operator = Callable | OpKind | RawOp | _OpAdapter
```

So an algorithm's `op=` parameter accepts:
1. A Python `Callable` (lambda / `def`) — compiled via numba-cuda.
2. An `OpKind` enum member — uses the optimized CUB built-in.
3. A `RawOp` — pre-compiled C++ LTO-IR.
4. An `_OpAdapter` — internal, ignore.

## `RawOp` — C++ Escape Hatch

```python
class RawOp(*, ltoir: bytes, name: str, state: bytes = b'', state_alignment: int = 1,
            extra_ltoirs: list[bytes] | None = None):
    def compile(input_types, output_type=None) -> Op
    def get_state() -> bytes
```

**Calling convention for the LTO-IR function:**

Stateless:
```c
extern "C" __device__ void func(void* arg1, void* arg2, ..., void* result)
```

Stateful (first arg is state pointer):
```c
extern "C" __device__ void func(void* state, void* arg1, void* arg2, ...)
```

**Example: stateful predicate that atomically counts selected items**

```python
import struct
from cuda.compute.op import RawOp
from cuda.core import Device, Program, ProgramOptions

d_counter = cp.zeros(1, dtype=np.int32)
counter_ptr = d_counter.__cuda_array_interface__["data"][0]
state_bytes = struct.pack("P", counter_ptr)

cpp_source = """
extern "C" __device__ void select_even_with_count(void* state, void* input, void* result) {
    int* counter = *reinterpret_cast<int**>(state);
    int value = *static_cast<int*>(input);
    bool is_even = (value % 2 == 0);
    if (is_even) atomicAdd(counter, 1);
    *static_cast<unsigned char*>(result) = is_even ? 1 : 0;
}
"""

# Compile via cuda.core
opts = ProgramOptions(arch="sm_90", relocatable_device_code=True,
                     link_time_optimization=True)
prog = Program(cpp_source, "c++", options=opts)
ltoir = prog.compile("ltoir").code

select_op = RawOp(ltoir=ltoir, name="select_even_with_count",
                  state=state_bytes, state_alignment=np.dtype(np.intp).alignment)

cuda.compute.select(d_in=..., d_out=..., d_num_selected_out=..., cond=select_op,
                    num_items=...)
```

**Why this matters for czarr:** Any algorithm we *can't* write in Numba's subset (because of bit fiddling, intrinsics, atomics, warp shuffles, or library calls) can be written in C++, compiled to LTO-IR with `cuda.core`, and dropped into `cuda.compute` algorithms as a `RawOp`. This is the bridge from "compose with iterators" to "write a full custom kernel."

---

# Structs (`cuda.compute.struct`)

```python
@gpu_struct
class Pixel:
    r: np.int32
    g: np.int32
    b: np.int32
```

- `Pixel.dtype` → NumPy structured dtype, usable in `cp.empty(..., dtype=Pixel.dtype)`.
- Supports nested structs (see `nested_struct_zip_iterator.py` example).
- AoS (array-of-structs): allocate `cp.empty(N, dtype=Pixel.dtype)`.
- SoA (struct-of-arrays): allocate separate arrays per field, fuse via `ZipIterator(d_r, d_g, d_b)` — *no copy*.

**Codec relevance:** Useful when codecs need composite state (e.g. `(symbol, frequency)` pairs for Huffman, `(literal, match_offset, match_len)` for LZ outputs). Avoids the friction of NumPy structured dtypes.

---

# Typing (`cuda.compute.typing`)

- `DeviceArrayLike` — protocol for `__cuda_array_interface__`. Covers CuPy, PyTorch, Numba, RMM-allocated buffers.
- `GpuStruct` — TypeVar bound to gpu_struct'ed types.
- `IteratorT` — TypeVar bound to `IteratorBase`.
- `Operator = Callable | OpKind | RawOp | _OpAdapter`.

---

# Memory Management & Stream Model

## Allocation

- **Algorithms accept any `DeviceArrayLike` for inputs/outputs** — CuPy, RMM-allocated, PyTorch, raw numba device arrays all work.
- **Temp storage** in object-based API is user-allocated: query size with `temp_storage=None`, allocate via your allocator (CuPy/RMM/cuda.core's `MemoryResource`), pass via the `temp_storage` kwarg.
- **Immediate API auto-allocates** via the current allocator — no hook documented for overriding this, so it presumably uses CuPy/cudaMalloc.

**No explicit `cudaMallocAsync` / memory pool API.** No documented integration with `cuda.core.MemoryResource`. So for czarr's RMM/cuFile pipeline, **stick to the object-based `make_*` API** and allocate temp storage from our existing pool — that's the clean integration path.

## Streams

- Every immediate algorithm has `stream=None` kwarg accepting any object exposing the CUDA stream interface (cupy.cuda.Stream, numba.cuda.stream, raw cuStream_t int).
- Make-style callables accept the same `stream` kwarg at call time.
- **No documented synchronization between calls** — chained algorithms on the same stream queue up async.

## CUDA Graph Capture

**Not documented.** Stream kwargs would presumably allow capture-mode streams, but no examples or hints in the developer overview. This is a non-trivial gap for fixed-workflow Zarr reads where graph replay would help.

## Caching

- Compiled cubins cached in-process, keyed on (dtype, iterator kind, op bytecode + closure hash, compute capability, algorithm params).
- *Closure-captured device arrays cached by pointer/shape/dtype, not contents* — so swapping buffer contents does not invalidate cache.
- **`cuda.compute.clear_all_caches()`** to flush — manual; no automatic eviction.

For czarr-as-library-in-long-running-server: clear caches periodically or scope op definitions narrowly.

---

# Type System

- **Dtype-polymorphic via type inference.** Reductions infer types from `h_init.dtype`. Transforms infer from `d_in.dtype` for input and require explicit return annotations when the output dtype differs (`(x: np.int32) -> np.float32`).
- **Strongly typed at JIT time.** Mixing dtypes between `op` and arrays raises an error (or worse, silently corrupts — docs warn).
- **NumPy structured dtypes work** via `@gpu_struct`. Direct NumPy structured dtypes also work as `h_init` for reductions.
- **No support for fp8/fp4** documented (relevant for newer Hopper/Blackwell narrow-precision codecs).

---

# What's Missing vs CUB/Thrust

Explicit gaps observed:

| CUB/Thrust feature | `cuda.compute` status |
|---|---|
| `DeviceRunLengthEncode` | ❌ no Python binding (must compose via `unique_by_key` + count) |
| `DeviceHistogram::HistogramRange` (custom bin edges) | ❌ only even bins |
| `DeviceSpmv` (sparse matrix-vector) | ❌ |
| `DeviceCopy` / `DeviceFor` (parallel-for) | ❌ — use `unary_transform` with identity op |
| `BlockHistogram` | ❌ in `coop` (not exposed) |
| `BlockDiscontinuity` | ❌ in `coop` |
| `BlockRunLengthDecode` | ❌ in `coop` |
| `BlockAdjacentDifference` | ❌ in `coop` |
| `cudaGraph` capture | ❌ not documented |
| `MemoryResource` integration | ❌ — pass any `DeviceArrayLike`, but no allocator hook |
| `cuda::std::complex` / arbitrary user types in ops | ⚠️ partial — `@gpu_struct` works for POD; complex/non-POD likely fails |
| Tensor core / MMA primitives | ❌ (out of scope — use `cudax::experimental::matrix`) |
| TMA (Tensor Memory Accelerator) | ❌ (out of scope — separate `libcudacxx` API in C++) |
| Cooperative groups (`cg::*`) | ❌ in Python |

---

# Compose-This-Codec: czarr Codec Feasibility Map

For each codec currently in czarr's path, here is what `cuda.compute` + `cuda.coop._experimental` would give us.

## Filters (small, transform-shaped)

### **Shuffle / Byteshuffle** — *Compose with iterators*

**What it does:** For an array of N elements each S bytes wide, output is the array transposed bytewise: byte-0 of all elements, then byte-1, etc.

**Compose:**
- Strategy A (pure `cuda.compute`): `unary_transform` with a Python op that does the index math, captured into a `PermutationIterator`. Single pass, single kernel.
- Strategy B (tile kernel via `coop`): `@cuda.jit` kernel using `block.make_load(algorithm='vectorize')` → in-register byte permute → `block.make_exchange(StripedToBlocked)` → `block.make_store(algorithm='striped')`. Equivalent to cuTile's tile transpose, but in pure Python+Numba.

**Cost vs cuTile:** Strategy B is mechanically equivalent to what cuTile does. Likely matches or slightly underperforms (NVRTC + nvJitLink overhead vs cuTile's AOT path). But **escapes the sm_90 cuTile bug.**

**Recommendation:** Implement Strategy B as a cuda.compute-native byteshuffle filter for Hopper. Keep cuTile for Ampere if it remains slightly faster there.

### **Delta** — *Compose with iterators + scan*

**What it does:** Encode: `out[i] = in[i] - in[i-1]`. Decode: prefix-sum of `in`.

**Compose:**
- **Encode:** `binary_transform(d_in1=d_in[1:], d_in2=d_in[:-1], d_out=d_out[1:], op=OpKind.MINUS)` + copy `d_in[0]` to `d_out[0]`. Or use a `ZipIterator(d_in[:-1], d_in[1:])` and a custom subtract op.
- **Decode:** `inclusive_scan(d_in=d_in, d_out=d_out, op=OpKind.PLUS, init_value=None, num_items=N)`. **Single kernel.**

**Recommendation:** Trivial in `cuda.compute`. This is the cleanest mapping.

### **FixedScaleOffset** — *Compose with `unary_transform`*

**What it does:** Encode: `out[i] = round((in[i] - offset) * scale)`. Decode: `out[i] = in[i] / scale + offset`.

**Compose:**
- `unary_transform` with `op = lambda x: round((x - offset) * scale)` — closure captures host scalars; Numba inlines them as constants.

**Recommendation:** Single-kernel implementation in `cuda.compute`. Done.

### **BitRound** — *Compose with `unary_transform`*

**What it does:** Zero out the low N mantissa bits of each float, rounding to nearest.

**Compose:**
- `unary_transform` with bitwise op: reinterpret as int, mask out low bits with the rounding correction, reinterpret as float. Numba supports bit-cast and bit-AND.

**Recommendation:** Trivial in `cuda.compute`.

## Compressors (entropy + LZ)

These are **not** directly composable. They require parallel implementations of well-known sequential algorithms. CCCL gives us the building blocks (scan, sort, segmented operations, atomics via RawOp); we'd be implementing the *algorithm* on top.

### **Zstd** — *Infeasible to clone, partial reimpl possible via RawOp*

**What it does:** LZ77-style dictionary compression + FSE (tabled ANS) entropy coding + Huffman fallback + frames/blocks framing.

**Compose:** None of: LZ77 matching, FSE state machine, Huffman tree construction, block framing exist in cuda.compute. nvCOMP's GPU Zstd is a multi-thousand-line C++ project (closed-source).

**Recommendation:** **Infeasible without C++ kernel.** Even with `RawOp`, you'd be writing a full Zstd decoder in CUDA C++. Not a sane re-implementation target. **Stick with nvCOMP for Zstd.**

### **LZ4** — *Decoder feasible as custom C++ kernel; not directly composable*

**What it does:** Token-based LZ77 with byte-level encoding. Simpler than Zstd; many open-source GPU decoders exist (a few hundred lines).

**Compose:** Still need a custom kernel. CCCL's `unique_by_key`/`segmented_*` don't help with literal/match parsing. But:
- Could write the inner LZ4 decode loop in C++ → LTO-IR, expose as a `RawOp`, and use `cuda.compute.select` or `unary_transform` as the outer harness.
- More pragmatically: write the whole decoder as a `@cuda.jit` kernel using `coop.block.make_load(algorithm='vectorize')` for input streaming.

**Recommendation:** **Infeasible without C++ kernel.** Feasible to port nvCOMP's LZ4 decoder if open-source equivalents exist. Cost: weeks of work, validation effort.

### **Snappy** — *Same story as LZ4*

**What it does:** LZ77 variant from Google; tag bytes + 1/2/4-byte length encoding + copies.

**Compose:** Same as LZ4 — needs a custom decoder kernel. CCCL primitives don't help with the parsing state machine.

**Recommendation:** **Infeasible without C++ kernel.**

### **Deflate / GDeflate** — *Infeasible to clone*

**What it does:** Deflate = LZ77 + canonical Huffman. GDeflate = NVIDIA's GPU-friendly variant (parallel block-streaming).

**Compose:** Huffman decode is *especially* hard to parallelize because of bit-stream variable-length codes. nvCOMP's GDeflate is the state-of-the-art GPU implementation; cloning it is a research project.

**Recommendation:** **Infeasible without C++ kernel.** Keep nvCOMP.

### **Bitcomp** — *Infeasible to clone*

**What it does:** NVIDIA-internal bitplane-based compression. Closed-source.

**Compose:** No public algorithm specification. Even with CCCL's full toolbox, you can't reverse-engineer this in reasonable time.

**Recommendation:** **Infeasible without C++ kernel** (and we don't have the algorithm).

### **ANS** — *Partially feasible (research project)*

**What it does:** Asymmetric Numeral Systems entropy coder. Sequential by nature; parallel variants (rANS interleaved streams) are the active research area.

**Compose:** With `cuda.compute`, you have:
- `histogram_even` for symbol frequency tables
- `inclusive_scan` for cumulative frequencies (the heart of rANS state updates)
- `radix_sort` for code-table construction
- `select` for stream packing
- Custom `RawOp` for the inner per-symbol state update

**Recommendation:** Feasible *as a research project*, NOT a port. Estimate: 1-2 person-months for a working interleaved-rANS encoder + decoder, plus validation against a reference. **Defer until other agents recommend it.**

### **Cascaded** — *Mostly feasible*

**What it does:** Pipeline of RLE → Delta → BitPacking (per nvCOMP docs). All stages are amenable to GPU.

**Compose:**
- **RLE stage:** `unique_by_key` + `segmented_reduce(count=ones)` + prefix-sum offsets. Three CCCL calls, one fused pass with iterators.
- **Delta stage:** `binary_transform(op=MINUS)` over zipped iterators (see Delta filter above).
- **BitPacking stage:** Per-chunk bit-width detection (`segmented_reduce` with bitwise-OR + `__clz`), then per-chunk pack via custom `unary_transform` or `RawOp`.

**Recommendation:** **Feasible in `cuda.compute`.** The bit-packing is the hardest stage; the RLE+Delta stages drop straight in. Estimate: 2-4 weeks for a complete Cascaded replacement.

---

# Putting It Together: Recommended Use of CCCL in czarr

1. **Adopt `cuda.compute` for all filters** (Shuffle, Delta, FixedScaleOffset, BitRound). Drops nvCOMP/cuTile dependencies for the filter layer entirely. Resolves the sm_90 cuTile bug for Shuffle. Implementation cost: ~1-2 weeks total for all four filters.

2. **Keep nvCOMP for the heavy compressors** (Zstd, LZ4, Snappy, Deflate, GDeflate, Bitcomp). Cloning these in pure CCCL is multi-month work and we have no algorithmic advantage to offer.

3. **Prototype Cascaded in `cuda.compute`** as a self-contained study to confirm the iterator/scan composition model performs at parity with nvCOMP for simpler codecs. If it works, that's the template for any future custom codecs.

4. **Reserve `cuda.coop._experimental` for the byteshuffle tile kernel** if Strategy A (iterator-based) underperforms cuTile by >20%. Strategy B (block-level `make_load`/`make_exchange`/`make_store`) gives us the tile-kernel control plane in pure Python+Numba.

5. **Defer ANS and any GDeflate-class custom codec.** These are research projects, not engineering tasks. Re-evaluate after agent 3's analysis of alternative compression libraries.

6. **Production hardening considerations:**
   - **Cache management:** Call `cuda.compute.clear_all_caches()` at long-running-worker boundaries; budget for hundreds of MB of cached cubins in steady state.
   - **Stream binding:** All algorithms accept a `stream=` kwarg — ensure czarr's pipeline binds to RMM/CuPy stream context consistently.
   - **`__cuda_array_interface__` discipline:** Means any RMM-allocated buffer works directly. Match that to czarr's existing buffer abstraction.
   - **API churn risk:** `_experimental` namespace and "subject to change without notice" warning means we need a CI gate that pins `cuda-cccl==X.Y.Z` and re-checks compatibility on each upgrade. The "stable URL" still returns 404, so we are on the bleeding edge.

---

# Tactical Code Patterns That Should Work for czarr

## Byteshuffle as a Permutation + Transform Fused Pass (Strategy A)

```python
import cupy as cp
import numpy as np
import cuda.compute
from cuda.compute import PermutationIterator, CountingIterator

def make_shuffle_indices(num_elements: int, element_bytes: int) -> cp.ndarray:
    """Indices for transposing AoS bytes to SoA bytes."""
    n, s = num_elements, element_bytes
    # output position i corresponds to byte (i % s) of element (i // s).
    # input position for that byte = (i // s) * s + (i % s)  -- identity.
    # For SHUFFLE: byte b of element e moves from input[e*s + b] to output[b*n + e].
    # Inverse map: output[k] = input[(k % n) * s + (k // n)]
    k = cp.arange(n * s, dtype=cp.int64)
    return (k % n) * s + (k // n)

def byteshuffle_compute(d_in_bytes: cp.ndarray, n: int, s: int) -> cp.ndarray:
    indices = make_shuffle_indices(n, s)
    perm = PermutationIterator(d_in_bytes, indices)
    d_out = cp.empty_like(d_in_bytes)
    cuda.compute.unary_transform(d_in=perm, d_out=d_out,
        op=lambda b: b, num_items=n*s)  # identity transform; the work is in the permutation
    return d_out
```

## Delta Encode/Decode (One-Liners)

```python
import cuda.compute
from cuda.compute import OpKind

# Encode: out[i] = in[i] - in[i-1] for i > 0
cuda.compute.binary_transform(
    d_in1=d_in[1:], d_in2=d_in[:-1], d_out=d_out[1:],
    op=OpKind.MINUS, num_items=len(d_in)-1)
d_out[0:1] = d_in[0:1]

# Decode: out = inclusive_scan(in, +)
cuda.compute.inclusive_scan(
    d_in=d_in, d_out=d_out, op=OpKind.PLUS,
    init_value=None, num_items=len(d_in))
```

## BitRound via Bit-Cast Transform

```python
import numba
import numpy as np

def make_bitround_op(n_keep_bits: int):
    drop_mask = np.int32(~((1 << (23 - n_keep_bits)) - 1))  # for float32
    def op(x):
        # Numba supports view-casts on its int/float types.
        return float32(int32(view(x, int32)) & drop_mask)  # pseudo-code; actual numba syntax differs
    return op

cuda.compute.unary_transform(d_in=d_in, d_out=d_out,
    op=make_bitround_op(n_keep_bits=8), num_items=len(d_in))
```

## RLE Stage of Cascaded

```python
# Run-length encode: produce (symbol, run_length) pairs.
d_out_keys = cp.empty_like(d_in)
d_out_counts = cp.empty_like(d_in, dtype=cp.int32)
d_num_runs = cp.empty(1, dtype=cp.int32)

# Use unique_by_key with a ones-iterator as "items" and PLUS in segmented_reduce.
# Pseudo-flow:
ones = ConstantIterator(np.int32(1))
cuda.compute.unique_by_key(
    d_in_keys=d_in, d_in_items=d_in,  # dummy; we only want keys
    d_out_keys=d_out_keys, d_out_items=d_out_keys,
    d_out_num_selected=d_num_runs,
    op=OpKind.EQUAL_TO, num_items=len(d_in))
# Now d_out_keys[:d_num_runs[0]] is the unique symbol sequence.
# Use a second pass with segmented_reduce + offsets to count run lengths,
# or fuse this with a TransformOutputIterator that increments a counter per change.
```

---

# Bibliography (URLs visited)

- https://nvidia.github.io/cccl/unstable/python/compute/index.html  (landing)
- https://nvidia.github.io/cccl/unstable/python/compute_api.html  (full API ref)
- https://nvidia.github.io/cccl/unstable/python/coop_api.html  (coop API)
- https://nvidia.github.io/cccl/unstable/python/coop.html  (coop overview)
- https://nvidia.github.io/cccl/unstable/python/compute/developer_overview.html  (compilation pipeline)
- https://nvidia.github.io/cccl/unstable/python/setup.html  (install/versioning)
- https://nvidia.github.io/cccl/unstable/python/index.html  (Python landing)
- https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/cuda/compute  (source)
- https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/compute/examples  (compute examples)
- https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/coop/_experimental/examples  (coop examples)
- https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/compute/examples/scan/running_average.py  (fused scan example)
- https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/compute/examples/raw_op/cpp_stateful.py  (RawOp pattern)
- https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/coop/_experimental/examples/block/scan.py  (block scan kernel)
