# The two-tier architecture

czarr has two public read APIs over one implementation. This page describes what each is for, how a read flows through the stages, and which alternatives were measured and rejected.

## Why two APIs

Zarr reads on a GPU machine serve two audiences with conflicting needs. Existing zarr code wants GPU speed with no code changes and the full zarr surface: writes, fancy indexing, groups, arbitrary codec chains. Performance-critical readers, such as training loops and services reading one array thousands of times, want metadata parsed once, per-call control over threads and memory, and no global state. One API cannot satisfy both, so czarr ships one for each and keeps them consistent.

## The explicit tier

`czarr.lowlevel` splits a read into three stages:

```
zarr.json ──► DecodePlan ──► ranges(selection) ──► read() ──► decode() ──► cupy.ndarray
```

`DecodePlan` parses metadata once and caches parsed shard indexes. It does not import cupy, so plans build on hosts without a GPU. `plan.ranges()` maps a selection to chunks, resolves shard offsets, and fuses adjacent byte ranges: with cuFile costing about 1 ms per call, fusing 32 inner-chunk reads into one reduced a partial-shard read from 26.5 ms to 1.2 ms on H100. `lowlevel.read()` issues the fused reads through a threadpool of blocking cuFile calls. `lowlevel.decode()` decompresses every chunk in one batched nvCOMP call, applies shuffle kernels, and scatters into the output.

`czarr.core.Array` and `AsyncArray` wrap the stages in an object with property metadata and `retrieve_*` methods. Options travel per call; the async variant runs the sync path in worker threads.

## The zarr tier

`configure_gpu()` works entirely through zarr's extension points: codecs registered under the same names as their CPU equivalents, GPU buffer prototypes, a batched codec pipeline, and a coalescing replacement for the sharding codec. zarr's machinery does the orchestration. This tier keeps everything zarr can do, at the cost of zarr's per-read overhead and process-global configuration.

## The bridge

`CudaZarrArray.__getitem__` tries the explicit tier first and falls back to zarr:

```
CudaZarrArray[sel]
   ├─ basic indexing, supported codecs ──► cached core.Array, int axes squeezed
   └─ anything else ─────────────────────► zarr.Array.__getitem__
```

The cached plan is keyed by metadata identity; writes drop it, because a rewritten shard must not be read through stale shard indexes.

## Measurements

From the H100 gate (`bench/results/zarr-read.jsonl`), blosc `[bitshuffle, zstd]` fixture:

- At 128-256 MiB chunks, the fast path, lowlevel, and the zarr pipeline all read at about 20 GiB/s, against a raw GDS transfer ceiling of 24.6 GiB/s. Bulk reads are storage-concurrency-bound in every stack; the explicit tier's value there is the API contract, not throughput.
- kvikio `GDSStore` with GPU codecs reads the same fixture at 1.6-1.9 GiB/s. GDS primitives alone do not make a fast zarr reader; batching and coalescing do.
- GPU decode is 18-20x CPU decode on this workload.

## Rejected designs

Each of these was implemented and benchmarked before removal; see git history for the implementations.

| Design | Result |
|---|---|
| cuFile async and batch APIs | 1.8-14x slower than threaded sync on Bruno NFS |
| Per-stream nvCOMP decode overlap | 0.99-1.02x, no effect |
| Background prefetch threads | Redundant with CUDA's async queue for async consumers |
| Register-once buffer pools | 5% slower than stock cupy allocation on real GDS |
| Read/decode overlap within one read | Loses; reads are concurrency-bound and header parsing serializes |
| cuFile knob tuning | No effect; `max_request_parallelism` clamps at 8 |
