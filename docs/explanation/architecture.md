# The two-tier architecture

czarr's read machinery is deliberately split into two tiers that share one substrate. This page explains what each tier is for, how a read actually flows, and why the split exists.

## The problem shape

Zarr reads on a GPU machine have two distinct audiences:

1. **Existing zarr code** — notebooks, pipelines, and libraries already written against `zarr.Array`. These want GPU speed with zero code changes, and they need the full zarr surface: writes, fancy indexing, groups, arbitrary codec chains.
2. **Performance-critical readers** — training loops and services that read the same array thousands of times. These want to pay metadata parsing once, control threading and memory per call, and avoid global state entirely.

One API can't serve both without compromising one of them. So: two tiers.

## Tier 2 — the explicit path

`czarr.lowlevel` decomposes a read into three inspectable stages:

```
zarr.json ──► DecodePlan ──► ranges(selection) ──► read() ──► decode() ──► cupy.ndarray
              (parse once)    (coalesced byte      (threaded    (one batched
                               ranges + chunk       cuFile       nvCOMP call
                               mapping)             reads)       + scatter)
```

- **`DecodePlan`** parses metadata exactly once and caches parsed shard indexes. It is host-only — no cupy import — so planning runs on login nodes and in CPU-only tests.
- **`plan.ranges()`** maps a selection to chunks, resolves shard offsets, and fuses adjacent byte ranges into fewer, larger reads. With cuFile costing ~1 ms per call, fusing 32 inner-chunk reads into one is worth 22× on partial-shard reads.
- **`lowlevel.read()`** issues the fused reads through a threadpool of blocking cuFile calls (the GIL is released; parallel submission is the only lever that matters — measured, not assumed).
- **`lowlevel.decode()`** runs one batched nvCOMP call for all chunks, applies shuffle kernels, and scatters into the output array, filling missing chunks.

`czarr.core.Array` / `AsyncArray` wrap these stages in an object with plain-property metadata and `retrieve_*` methods. Knobs travel per call in a `ReadOptions` dict — nothing global.

## Tier 1 — the zarr-native path

`configure_gpu()` works entirely through zarr's public extension points: codecs registered under the same names as their CPU equivalents, GPU buffer prototypes, a batched codec pipeline, and a coalescing override of the sharding codec. Stock zarr machinery does the orchestration; czarr supplies the GPU parts.

This tier keeps everything zarr can do — writes, fancy indexing, groups, any codec chain — at the cost of zarr's per-read overhead and process-global configuration.

## The bridge

`CudaZarrArray.__getitem__` connects the tiers:

```
CudaZarrArray[sel]
   ├─ basic indexing + supported codecs ──► cached core.Array (tier 2)
   │                                        + squeeze int axes
   └─ anything else ──────────────────────► zarr fallback (tier 1)
```

The tier-2 plan is cached on the array keyed by metadata identity, so repeated reads skip re-parsing; writes drop the cache (a rewritten shard must not be read through stale indexes).

## What the measurements say

The H100 gate (`bench/results/zarr-read.jsonl`) keeps this design honest:

- **Tier-1 fast path ≈ lowlevel ≈ zarr pipeline** at 128-256 MiB chunks (~20 GiB/s). The explicit tier's value at bulk reads is the *API* — no global state, per-call control — not raw speed; bulk reads are storage-concurrency-bound whichever stack issues them.
- **kvikio `GDSStore` is 10-13× slower** on the same fixture — GDS primitives alone don't make a fast zarr reader; batched decode and read coalescing do.
- **GPU decode is 18-20× CPU** on blosc stores at real chunk sizes.

Several designs were tried and benched *out*: per-stream decode overlap (wash), background prefetch threads (redundant with CUDA's async queue), cuFile's async/batch APIs (1.8-14× slower than threaded sync on our storage), register-once buffer pools (~5% slower than stock cupy). The surviving architecture is the simple one because the alternatives lost on hardware.
