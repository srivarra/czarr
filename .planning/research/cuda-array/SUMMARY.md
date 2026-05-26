# CUDA-native Zarr Array — research synthesis

> Decision document. Inputs: agent reports 01-06 in `.planning/research/cuda-array/`, the LZ4 spike at `spikes/lz4_decoder.py`, and two stream-parallelism probes (`nvcomp_stream_parallelism.md`, `native_lz4_stream_parallelism.md`).

> **Revision (post-probes):** the original "per-microbatch overlap on a small stream pool" plan is dead — both nvCOMP and our native LZ4 kernel saturate the H200 from a single grid launch, so multi-stream lanes give zero compute speedup. The architectural win remaining is the **I/O pipeline** (removing the await-all-reads-before-decode barrier), not stream parallelism. v0.1 ships with one decode call per microbatch, a queued read producer, and the native LZ4 codec inline.

## TL;DR

Build a `CudaZarrArray(zarr.Array)` subclass with a private `_CudaArrayImpl` orchestrator. The fast path bypasses `BatchedCodecPipeline.read_batch` and uses a **bounded read-queue producer + single big decode call** per microbatch. Two backends behind one codec class: nvCOMP for the legacy bitstreams we keep, native cuda-python for LZ4 (default `backend="native"` after H200 bench validation).

The H200 stream-parallelism probes killed the lanes hypothesis: nvCOMP decodes serialise on the GPU regardless of how many streams we dispatch from, and our native LZ4 kernel already saturates 173 GiB/s from a single grid launch. Multi-stream is monotonically *worse*. So the design simplifies: no lane bookkeeping, one decoder per microbatch, the parallelism lives in the read side (32 OS threads via the existing pool) and in the codec kernel itself.

Keep nvCOMP for Zstd, Bitcomp, ANS, Cascaded permanently — three of those are proprietary formats we cannot re-implement, and Zstd is a 6-9 month research project we have no business taking on. **Native LZ4 unparks and merges into Phase 1** as `czarr.LZ4(backend="native"|"nvcomp")` with native as the default. Replace **all four filters** (Shuffle, Delta, FixedScaleOffset, BitRound) with `cuda.compute` versions in Phase 2 — ~2 engineer-weeks, removes the cuTile/sm_90 Hopper Shuffle bug, drops the cuTile dependency.

**Ballpark targets (revised after probes):**
- **nvCOMP path (Zstd workload, 1 GiB Z-slab, H200):** ~21 GiB/s end-to-end. Wall ≈ max(reads, decode) ≈ 47 ms once the await-barrier is removed. 1.8× the current 11.5 GiB/s baseline.
- **Native LZ4 path (LZ4 workload, 1 GiB, H200):** decode finishes in <1 ms (173 GiB/s kernel), wall ≈ reads ≈ 40 ms → **~25 GiB/s end-to-end on compat-mode storage**, much higher if reads land on a GDS-capable path.
- **Cost:** v0.1 = 6-8 engineer-weeks. Includes Phase 1 skeleton, I/O queue pipeline, native LZ4 productionisation (the spike's decode is already production-quality; we wrap + test + add encoder fallback to nvCOMP).

This bet is robust to GDS not firing: the nvCOMP-path win comes from *I/O overlap* (works identically in compat and direct modes); the native LZ4 win comes from *bypassing nvCOMP's per-call Python overhead* (works on any storage).

## Where the agents converge

High-confidence signal — independent agreement across reports:

1. **Subclass `zarr.Array`, do not fork.** Agent 4 §1 and the implicit assumptions in agent 5 §1 both arrive at this. Mirrors zarrs PR #147. Buys us metadata, group integration, attrs, resize, and a free fallback to `super().__getitem__` for any indexing path we don't fast-path.

2. **`cuda.core.VirtualMemoryResource(addr_align=4096, gpu_direct_rdma=True)` is the right substrate for cuFile-bound buffers.** Agent 1 §5.3, agent 4 §3, agent 5 §1.2 / §1.9 all agree. The only `MemoryResource` that yields 4 KiB-aligned device VAs marked GDR-capable.

3. **Per-microbatch overlap is the architectural win.** Agent 4 ("read/decode overlap — concretely") and agent 5 §8.3 describe the same pattern: round-robin read+decode stream pairs from a `StreamPool`, with `Event` recorded after each batch's read and waited on by the decode stream. This is what `CzarrPipeline` cannot do because the parent class `read_batch` *awaits* `concurrent_map(reads)` before kicking decode.

4. **Filters are trivially composable in `cuda.compute`.** Agent 2's per-codec verdict + agent 4's filter integration: Shuffle = `PermutationIterator`, Delta-decode = `inclusive_scan(PLUS)`, BitRound = `unary_transform` with a bit-cast op, FixedScaleOffset = `unary_transform` with closure-captured scalars. All four are single-kernel, one-pass.

5. **Zstd / Bitcomp / ANS / Cascaded must stay on nvCOMP.** Agent 2's feasibility map and agent 3's per-codec analysis both reach this. Three of the four are proprietary bitstreams (no published spec). Zstd is public but a 6-9 month research project with no open-source production GPU decoder.

6. **DLPack v1 + CAI dual-export is the right interop story.** Agent 1 §5.8, agent 4 §4, agent 5 §3 + §4. cupy / pytorch / numba accept both; nvCOMP accepts only CAI. Export both on `CzarrGpuBuffer` and on `_LazySlice`.

7. **`FileStreamProgramCache` answers "first import is slow because we JIT N kernels".** Agent 1 §6.9 + Appendix A4, agent 2 §"Architecture". Persistent on-disk cache at `$XDG_CACHE_HOME` — free warm-start.

8. **nvCOMP's per-call overhead is fixed, not amortisable below its batch-size floor.** The microbatch failure is independent evidence; agent 3's spike confirms ("the 2-43× win over nvcomp is not primarily kernel quality — it's that nvcomp's Python wrapper has substantial per-call setup cost"). Takeaway: stop trying to shrink batches; *overlap* full-size batches across streams.

## Where the agents disagree

1. **Agent 3 vs agent 2 on Snappy build cost.** Agent 3 says "BUILD, ~4 weeks total" (port cuDF `unsnap.cu`). Agent 2 says "Infeasible without C++ kernel" — but in context that means "not composable in `cuda.compute` primitives alone", which does not contradict porting a known-good Apache-2.0 source. **Resolution: agent 3 is right.** Agent 2's framing is about CCCL composability, not engineering effort.

2. **Agent 3 vs agent 4 on the bandwidth ceiling.** Agent 3's LZ4 spike hits 125 GiB/s kernel-only on A40. Agent 4 expects the slab bench to clear roughly 1.6-2× over baseline (~18-23 GiB/s) on H200. **Resolution: both correct, in different scopes.** Agent 3's number is decode-kernel-bound; agent 4's is end-to-end including cuFile compat-mode reads. The profile says nvCOMP dominates at 35-40 ms/call today, so codec improvements matter; agent 4's overlap design plus in-house LZ4 in v1 covers both stories.

3. **Agent 4 vs agent 5 on `Buffer.close(stream=...)` discipline.** Agent 4 §3 says "if we use cupy.ndarray directly, we need an explicit Event per batch"; agent 5 §1.9 + §2.3 says VMR/LegacyPinned `deallocate(stream=...)` does a host-side `stream.sync()` — which is the opposite of what you want for an overlap pipeline. **Resolution: agent 5 is more careful.** The encoded slab is VMR-backed; closing on the wrong stream produces a host stall that silently kills overlap. Add it to the test plan.

4. **Agent 2 vs agent 3 on Cascaded.** Agent 2 says "Feasible in cuda.compute, 2-4 weeks". Agent 3 says "BUY — proprietary framing format". **Resolution: agent 3 right about the format, agent 2 right about the components.** nvCOMP's Cascaded output is not bytewise-compatible with anyone else's RLE+Delta+BitPack. We could build *our own* composable equivalent under a different codec name, but it has no users today.

5. **Implicit disagreement on CUDA Graphs in v0.1.** Agent 1 §7 frames the graph API as the right primitive for the microbatch pipeline. Agent 5 §9.2-9.3 cautions that nvCOMP's variable-length-decode path is *not* capture-safe. **Resolution: keep graphs out of v0.1.** They're the right Phase 3+ knob once Python launch overhead becomes the bottleneck — which it isn't today.

## Per-codec verdict

Combines agent 2's CCCL feasibility, agent 3's nvCOMP-alternatives + LZ4 spike, agent 4's codec-chain integration, and the on-disk codec inventory at `src/czarr/codecs/compressors/native.py` + `src/czarr/codecs/filters/`. Decode-only engineer-weeks unless noted.

### Compressors (Bytes→Bytes)

| codec | verdict | path | eng-weeks (decode) | confidence | notes |
|---|---|---|---:|---|---|
| Zstd | **keep nvCOMP** | nvCOMP | 0 | high (a2, a3) | 6-9 mo research project; no open-source GPU production decoder |
| LZ4 | **rewrite from scratch** (post-v0.1) | in-house cupy.RawKernel from spike | 1.5 + 2 (encoder) | high (a3, spike) | spike at 125 GiB/s on A40, 17.8× nvCOMP; bitstream public + simple |
| Snappy | **rewrite from scratch** | port cuDF `unsnap.cu` (Apache-2.0) | 2 | medium (a3) | defer until LZ4 proves the model |
| Deflate (Gzip/Zlib) | **rewrite via fork** | vendor cuDF `gpuinflate.cu` (Apache-2.0) | 3.5 + CPU encode | medium (a3) | dynamic Huffman is hard; don't write from scratch |
| GDeflate | **drop** (or hybrid post-Deflate) | nvCOMP, or port DirectStorage HLSL | 0 (drop) or 2.5 | low | no user demand in scientific Python |
| Bitcomp | **keep nvCOMP** | nvCOMP | 0 | high (a3) | proprietary format, undocumented bitstream |
| ANS | **keep nvCOMP** | nvCOMP | 0 | high (a3) | proprietary gANS bitstream |
| Cascaded | **keep nvCOMP** | nvCOMP | 0 | high (a3) | proprietary framing format |

### Filters (Array→Array)

| filter | verdict | path | eng-weeks | confidence | notes |
|---|---|---|---:|---|---|
| Shuffle (byteshuffle) | **rewrite in CCCL** | `PermutationIterator` + `unary_transform`; fallback `coop.block.make_load`/`make_exchange`/`make_store` | 0.5 - 1 | high (a2) | escapes the sm_90 cuTile bug |
| Delta | **rewrite in CCCL** | encode: `binary_transform(MINUS)`; decode: `inclusive_scan(PLUS)` | 0.25 | high (a2) | one-liners |
| FixedScaleOffset | **rewrite in CCCL** | `unary_transform` with closure-captured scalars | 0.25 | high (a2) | trivial |
| BitRound | **rewrite in CCCL** | `unary_transform` with bit-cast op | 0.5 | high (a2) | Numba supports bit-cast |

**Filter total: ~1.5 - 2 engineer-weeks for all four.** Single highest-leverage CCCL work: removes the cuTile Hopper bug, removes dependence on `coop._experimental`'s churn (only need it as a perf fallback), gets us out of nvCOMP for the filter layer entirely.

**Net:** v0.1 in-house code = filters only (~2 eng-weeks). v1 = filters + LZ4 decoder (~4 more eng-weeks). Everything else stays on nvCOMP indefinitely.

## CudaZarrArray: go/no-go

**Go.** But with sharper expectations than agent 4's section implies.

The honest question: given that (a) the microbatch CodecPipeline experiment failed, (b) GDS does not fire on Bruno, and (c) decode dominates the profile at 35-40 ms per nvCOMP call, does the array facade buy us anything beyond what we already have? **Yes, but only because of the read↔decode barrier**, not because of any of the other reasons people argue for an array facade.

### Why it wins

`zarr.core.codec_pipeline.BatchedCodecPipeline.read_batch` is structured as:

```
await concurrent_map(reads)   # all chunk reads finish here
decoded = decode_batch(...)   # then decode starts
```

That `await` is a **synchronous wait point**. With 64 zstd chunks × ~40 ms cuFile read + ~47 ms decode, you cannot start decode of chunk 0 until chunk 63's read returns. The microbatch experiment tried to fix this by shrinking the batch — but a small batch hits the 35-40 ms fixed nvCOMP overhead on every batch, linearly hurting wall time.

`CudaZarrArray` fixes it differently: keep the full batch size, but **run K microbatches in flight** (round-robin read/decode stream pairs), so chunk-batch-0's decode overlaps chunk-batch-1's read. The fixed overhead is paid `N/microbatch_size` times in total (same as today), but `prefetch_depth - 1` of them happen in the shadow of disk I/O.

Quick arithmetic on H200 today (1 GiB slab, 11.5 GiB/s ≈ 87 ms wall):
- serial: 40 ms reads + 47 ms decode = 87 ms
- overlapped, prefetch_depth=2: steady-state per batch `≈ max(40, 47) = 47 ms`
- expected steady-state: **~47-55 ms per equivalent batch → ~18-22 GiB/s**

That's the 1.6-1.9× lift cited in the TL;DR. Agent 4 §11 expects "1.4-1.8× on H200 with the standard 1 GiB Z-slab" from this alone, with register-once cuFile compounding to 1.6-2× in phase 3.

### What we don't get

- **GDS-direct still doesn't fire.** Architecture doesn't fix compat-mode. Wins are in overlap, not in bypassing the bounce buffer.
- **The codec is not faster.** nvCOMP per-call overhead is unchanged — just hidden in the shadow of I/O.
- **No multi-GPU, no sharded v3 inner-chunk indexing.**

### When it would not be a go

If the profile were already read-bound (not decode-bound), the architecture wouldn't help. The profile says otherwise: 35-40 ms decode ≈ 40 ms read on this slab, so both phases are comparable and overlap shaves the smaller. **Architectural overlap is the structural win the microbatch experiment was reaching for**, with the right unit of parallelism (cross-batch instead of intra-batch).

## Risk register

Prioritised by likelihood × impact. P0 = would kill the project.

### P0 — kill scenarios

1. **`zarr.Array` internal-API churn breaks the subclass** (a4 §10). `_async_array._chunk_grid`, `_codec_pipeline`, `_get_selection` are underscore-private. *Mitigation*: wrap every internal access in a thin adapter; pin a minimum zarr version; nightly CI against zarr-python `main`; **do not** override methods we don't have to. Escape hatch `super().__getitem__` keeps correctness regardless of perf.

2. **VMR / LegacyPinned `deallocate(stream=...)` does a host-side `stream.sync()`** (a5 §2.3, footgun table). If the encoded slab's free is queued on the wrong stream, the entire overlap pipeline serialises silently. *Mitigation*: explicit close-on-stream discipline, asserted in a test that measures total wall time vs sum of phase times.

3. **nvCOMP decode is not CUDA-graph-safe** (a5 §9.2). Caps the win at the batch-overlap level. *Mitigation*: accept it. Phase 1-3 don't need graphs.

### P1 — likely degraders

4. **Pinned-buffer fragmentation** (a5 §7.3). Variable-length chunks miss exact-size buckets. *Mitigation*: jemalloc-style size-class promotion in `PinnedHostPool` (~30 LOC). Profile first; the buffer epic may already cover this.

5. **API churn in `cuda.coop._experimental`** (a2 §"Stability"). Underscore-experimental, docs literally say "subject to change without notice". *Mitigation*: only use `coop._experimental` as a perf fallback for Shuffle; primary impl via `cuda.compute` (less unstable).

6. **`cuda.core.TensorMapDescriptor` is private API** (a1 §10). *Mitigation*: no TMA dependency in v0.1 / v1.

7. **`PCH` options in `ProgramOptions` are CUDA 12.8+ only** (a1 §10). *Mitigation*: gate on driver version at startup; skip PCH on older drivers.

8. **DLPack v1 producer/consumer mismatch** (a5 §3.5). *Mitigation*: pin minimum torch/numpy/cupy in `pyproject.toml`; expose v0-compatible signature if necessary.

### P2 — watch list

9. **CCCL kernel cache grows unbounded in long-running workers** (a2). *Mitigation*: call `cuda.compute.clear_all_caches()` on long-lived-worker tick boundaries.
10. **Dangling `_LazySlice`s after parent close** (a4 §10). *Mitigation*: strong ref to `_CudaArrayImpl`; no `close()` in `__del__`.
11. **Multi-thread `Device.set_current()` thread-locality** (a5 §10.1). *Mitigation*: every worker thread calls `device.set_current()`. Already in `streams.py`.
12. **cuFile `read_async` on NFS silently fails without `buf_register`** (a5 §5.6, also project memory `cufile_async_constraints`). *Mitigation*: `ensure_buf_registered` discipline already documented.
13. **A40 + no `nvidia_fs` asserts inside libcufile on async read** (a5 §5.6). *Mitigation*: `is_async_available()` gate already exists; threaded sync fallback.
14. **RMM vs cuda.core DMR pool fragmentation** (a5 §6). *Mitigation*: don't introduce DMR in the hot path — VMR for cuFile, RMM-via-cupy for codec scratch (the split a5 §6.4 already recommends).

## v0.1 — proposed scope + bench target

**The one architectural bet:** *lanes architecture for codec overlap on the read path.* Everything in v0.1 is dedicated to validating that bet.

### Deliverables

- `src/czarr/array/cuda_array.py` — `CudaZarrArray(zarr.Array)` + `open_cuda_array(...)`
- `src/czarr/array/_impl.py` — `_CudaArrayImpl` with `retrieve_gpu(ranges, out)` only (read path)
- `src/czarr/array/_batch.py` — `_BatchHandles`, `_bucket_microbatches`, `_issue_read_batch`
- `src/czarr/array/_codec_chain.py` — compiled codec chain calling existing nvCOMP `CudaBytesBytesCodec` instances on a passed-in stream
- Reuses `CzarrGpuBuffer` (buffer epic) for encoded slabs; v0.1 still pays per-batch register cost in compat mode if buffer epic Phase 3 hasn't landed
- Reuses existing `StreamPool`, `PinnedHostPool`
- Tests: basic-indexing round-trip vs `zarr.Array`, fallback-indexing correctness (bool mask + strided slice), close discipline
- Bench: `bench/cuda_array/slab_compare.py` — 1 GiB Z-slab vs baseline

### Explicitly **not** in v0.1

- No write path (`store_gpu`, `__setitem__` fast path) — fall back to `super().__setitem__`
- No `.lazy` / `_LazySlice` — eager only
- No `copy_from` device-to-device
- No in-house codecs — still using nvCOMP for everything
- No CCCL filters yet (still using cuTile/Shuffle kernel — known sm_90 bug present)
- No CUDA Graphs, no sharding, no multi-device
- No `configure_gpu(use_cuda_array=True)` registry hook — explicit wrap only

### Bench target

| bench | platform | target |
|---|---|---|
| `slab_compare.py` (1 GiB Z-slab, 16×512×512 zstd chunks) | H200 | **≥ 18 GiB/s** (≥ 1.6× baseline) |
| `slab_compare.py` | A40 (compat-mode, no GDS) | parity ± 10% with baseline |
| `microbatch_sweep.py` | H200 | identify optimum; expect ~8 × 2 |

If we miss 18 GiB/s on H200, two diagnoses: (1) nvCOMP serialises decode across two streams sharing a Codec — fix by instantiating one Codec per stream; (2) overlap not firing due to encoded-slab close stream-discipline (P0-2 above) — fix by enforcing explicit close-on-stream.

### Sizing

Agent 4: ~1500 lines total for the full Phase 1-2 surface. v0.1 (Phase 1 read-only) is ~800-1000 lines including tests/benches. **6-8 engineer-weeks** including bench-driven tuning.

## v1 user-facing API sketch

```python
import czarr
import cupy as cp
import zarr

czarr.configure_gpu(rmm_pool_gb=4)

# Wrap or open.
arr = czarr.CudaZarrArray(zarr.open_array("data.zarr", mode="r"))
arr = czarr.open_cuda_array("data.zarr", mode="r")

# Eager basic indexing returns cupy on device, no host roundtrip.
slab: cp.ndarray = arr[0:8, :, :]
arr.read_into(prealloc_buf, np.s_[0:8])               # zero-alloc hot path

# Lazy: zero-copy handoff to downstream consumers.
lazy = arr.lazy[:, :, :]
torch_t = torch.utils.dlpack.from_dlpack(lazy)
view    = cp.asarray(lazy)

# Eager full-chunk write (RMW falls back to super().__setitem__).
arr[8:16, :, :] = some_cupy_ndarray

# Device-to-device chunk copy (skips decode/encode when codec chains match).
new = czarr.create_cuda_array(
    "rechunked.zarr", shape=arr.shape, chunks=(4, 4096, 4096),
    dtype=arr.dtype, compressors=[czarr.Zstd()],
    filters=[czarr.Shuffle()],   # cuda.compute-backed in-house
)
new.copy_from(arr, src_ranges=..., dest_ranges=...)
```

Contract:
- `arr[basic_key]` returns `cp.ndarray` (device).
- `arr[advanced_key]` falls back to `zarr.Array.__getitem__` and returns `np.ndarray` (host). Dual contract is documented.
- `arr.lazy[basic_key]` returns `_LazySlice` supporting `__cuda_array_interface__`, `__dlpack__`, `__array__`.
- `arr.read_into(out_cupy, key)` is the zero-allocation contract.

## Sequencing

Phase boundaries map to dex epics. Each phase is independently shippable: fast path either works (cupy) or falls back (numpy). No silent correctness regression.

### Phase 0 — foundation (must do first)

In flight. Blocks Phase 3 for cuFile register-once compounding.
- **Buffer epic** (`czarr.czarr-gpu-buffer` worktree) — `CzarrGpuBuffer` with VMR + register-once cuFile + DLPack v1 / CAI dual export (~80% done per context).
- `cufile_runtime` covers async/sync split + `nvidia_fs` gating.

### Phase 1 — CudaZarrArray skeleton + native LZ4 (the v0.1 ship)

Soft dep on Phase 0. Phase 6 LZ4 from the original plan merges here so v0.1 ships both backends.

- `_CudaArrayImpl` + `retrieve_gpu(ranges, out_cupy)`
- `CudaZarrArray.__getitem__` fast path + `super()` fallback
- `open_cuda_array` / `create_cuda_array` / `CudaZarrArray.wrap`
- **Bounded read-queue producer + single decode call per microbatch** (the I/O-overlap pattern). No stream lanes — the probes proved they don't win for either backend.
- **Native LZ4 codec** productionised from the spike. `czarr.LZ4(backend="native"|"nvcomp")` with `"native"` as the default once the v0.1 bench validates parity-or-better on H200. Encoder stays on nvCOMP; decoder is in-house.
- API surface follows `07-api-design.md` (PEP 695 generics, PEP 698 `@override`, PEP 692 `Unpack[TypedDict]`, PEP 688 `__buffer__`).
- Bench `slab_compare.py` with both backends.

**Bench targets:**
- Zstd workload (nvCOMP path), H200: ≥ 18 GiB/s (1.6× baseline). The I/O-overlap removes the 40 ms read barrier.
- LZ4 workload (native backend), H200: ≥ 25 GiB/s. Decode is sub-millisecond; wall is read-bound.
- A40 (compat-mode, no GDS): parity ± 10% with baseline on both paths.

Sizing: ~1000-1400 LOC including native LZ4 productionisation + tests + benches, 7-9 engineer-weeks.

### Phase 2 — CCCL filters (~2 engineer-weeks)

Depends on: Phase 1. Independent of buffer epic.
- `cuda.compute`-backed Shuffle, Delta, FixedScaleOffset, BitRound
- Removes cuTile/sm_90 Hopper bug
- Drops cuTile dependency
- Bitstream-compat tests against numcodecs reference (per `06-numcodecs-exploration.md` fixture pattern)

### Phase 3 — register-once cuFile (buffer epic merges)

Depends on: Phase 0 + Phase 1.
- `_CudaArrayImpl` allocates encoded slabs via `CzarrGpuBuffer.empty(stream=...)` which registers with cuFile once at allocation.
- Removes the ~1 ms (H200) / ~3 ms (A40) per-call register cost.
- **Marginal on top of Phase 1** — small absolute win since reads already parallelise.

### Phase 4 — write path

Depends on: Phase 1.
- `store_gpu` full-chunk path; `__setitem__` routes full-chunk writes; RMW falls back to `super()`.
- Encode side: native LZ4 encoder (Phase 4 deliverable, not Phase 1).
- Encode-side filter chain (uses Phase 2 CCCL filters if available).

### Phase 5 — lazy + copy_from

Depends on: Phase 1, Phase 4.
- `.lazy[key]` returns `_LazySlice` with DLPack + CAI + `__array__`.
- `copy_from` with chunk-aligned encoded-passthrough (skip decode + encode).
- Enables rechunk / coarsen / append workflows.

### Phase 6+ — out of scope; revisit only with a real user

- Snappy / Deflate / GDeflate in-house (per `03-nvcomp-alternatives.md` cost analysis)
- Zstd in-house (6-9 month research project; skip indefinitely)
- CUDA Graphs for inner pipeline (nvCOMP is not capture-safe per agent 5)
- Sharding v3 inner-chunk fast path (advanced indexing falls back to zarr.Array today)
- Multi-device, S3 / cloud stores, encoder rewrites beyond LZ4

## Out of scope

Deliberately deferred. Implementers should not be planning around these:

- **Multi-GPU.** Single device per process. `VMR(peers=[...])` plumbing exists (a5 §10.2) but is a separate epic.
- **S3 / cloud object stores.** File-system-bound (Bruno NFS, local NVMe).
- **Sharded Zarr v3 inner-chunk indexing.** Falls through to `super().__getitem__` (a4 §10).
- **Advanced / fancy indexing fast path.** Bool masks, ndarray indexers, strided slices, structured-dtype fields fall through to zarr (a4 §2).
- **Async (`asyncio`) API surface.** `retrieve_gpu` is sync-on-host (a4 §10); cuFile releases GIL.
- **Zstd / Bitcomp / ANS / Cascaded rewrites.** Permanent nvCOMP dependency.
- **Encoder rewrites in general.** czarr is read-dominated; encode goes through nvCOMP except where decode forces our hand (LZ4 in Phase 6).
- **CUDA Graphs in v0.1 / v1.** nvCOMP variable-decode is not capture-safe (a5 §9.2).
- **TensorMapDescriptor / TMA-bulk-copy codec kernels.** Private API (a1 §10).
- **Green Contexts / SM partitioning.** Not relevant to single-tenant codec workloads (a5 §1.7 / §1.8).
- **`configure_gpu(use_cuda_array=True)` global registry hook.** Return-type contract differs (cupy vs numpy); silent swap would surprise users.

## Open questions

1. **What is the actual win from per-microbatch overlap on the H200 slab bench, in compat mode?** 18-22 GiB/s estimate is back-of-envelope. *Experiment*: build v0.1 skeleton and run `slab_compare.py`. < 14 GiB/s indicates an orchestration problem (most likely encoded-slab close stream discipline).

2. **Does nvCOMP serialise decode across multiple decode streams sharing a Codec instance?** Agent 4 §10 calls this out. *Experiment*: instrument two `nvcomp.Codec` instances, launch concurrent decodes on independent streams, measure overlap with `Event` timing. If they serialise, cache one Codec per stream (cost: ~50 MiB scratch × streams).

3. **Does register-once `cuFileBufRegister` actually save ~30-50% in compat mode on Bruno VAST?** Agent 5 §5.5 cites this as "measurement-supported but not rigorously benched". *Experiment*: when buffer epic Phase 3 lands, register-once vs register-per-call bench on `/hpc/mydata`. < ~10% win = not worth the additional buffer-lifetime complexity.

4. **Does the LZ4 spike's 17.8× win hold on H200?** Agent 3 only had A40 access. *Experiment*: run `spikes/lz4_decoder.py` on H200 vs `nvidia-nvcomp-cu13` 5.x. If < 3×, defer Phase 6 until a real user asks.

5. **Does CCCL's `unique_by_key` produce per-segment run lengths in one pass for a Cascaded-style RLE stage?** Agent 2's sketch plausible but unverified. *Experiment*: spike RLE on 1 GiB of integer data; compare wall time to a single hand-written `@cuda.jit` kernel. Within 2× of the hand kernel = adopt composable approach.

6. **What happens to `_async_array._chunk_grid` in zarr-python 3.1.x when their pipeline refactor lands?** *Experiment*: nightly CI job against zarr-python `main`. Document breaking-change protocol.

7. **Does the existing `CzarrPipeline` need to disappear, or does it remain the fallback?** Agent 4 §7 implies it stays as the fallback. *Resolution*: keep both, document that the fast-path bypasses `CzarrPipeline` entirely and the fallback uses it. No code merge.

## References

Agent reports, all in `/hpc/mydata/sricharan.varra/Dev/czarr/.planning/research/cuda-array/`:

- `01-cuda-core-api.md` — `cuda.core` API inventory; load-bearing for VMR / Program / Linker / FileStreamProgramCache / GraphBuilder
- `02-cccl-compute.md` — `cuda.compute` + `cuda.coop._experimental` filter feasibility and per-codec composability map
- `03-nvcomp-alternatives.md` — per-codec build/hybrid/buy analysis with engineer-week estimates; LZ4 spike numbers (125 GiB/s on A40, 17.8× nvCOMP)
- `04-cuda-array-architecture.md` — `CudaZarrArray(zarr.Array)` design including `_CudaArrayImpl`, indexing model, lazy view, copy_from, phase plan
- `05-lifetime-interop.md` — `MemoryResource` hierarchy, stream-ordered allocation rules, DLPack v1 / CAI interop, cuFile registration lifecycle, RMM coexistence, segfault footgun table

Spike code:

- `spikes/lz4_decoder.py` — 311-line pure-cupy LZ4 block decoder with CPU oracle, batched launcher, four fixtures, benchmark; A40 at 125 GiB/s for 1024×64KiB batches.

External references cited inline:

- zarrs-python PR #147: https://github.com/zarrs/zarrs-python/pull/147
- nvCOMP 2.2 BSD-3-Clause source tree: https://github.com/NVIDIA/nvcomp/tree/branch-2.2
- cuDF GPU codec sources (Apache-2.0): https://github.com/rapidsai/cudf/tree/main/cpp/src/io/comp
- Microsoft DirectStorage GDeflate (Apache-2.0): https://github.com/microsoft/DirectStorage/tree/main/GDeflate
- CCCL Python landing: https://nvidia.github.io/cccl/unstable/python/
- DLPack v1 spec: https://dmlc.github.io/dlpack/latest/python_spec.html
- CAI v3 spec: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
- cuFile API guide: https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html
