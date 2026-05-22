# cross-chunk batching at the pipeline layer

Batch ALL chunk reads of one `arr[...]` selection into a single cuFile call so per-chunk Python dispatch becomes per-selection. This is the next blocker for H100/H200 throughput.

## Motivation

**Phase 0 measurement** (cProfile on the 2048-chunk 32 KiB workload, A40):

```
ncalls  tottime    percall  function
4096    3.216 s    0.8 ms   cufile_runtime.registered_handle   ← 50% of wall time
2048    1.591 s    0.8 ms   cufile_runtime.read_into
2048    1.140 s    0.6 ms   posix.open
   1    0.051 s   51.0 ms   _batch_sync (codec)                ← codec is fine
```

The bottleneck is **not in the codec**. Decode finishes in 51 ms for the whole batch. The fixed Python overhead is **per-chunk cuFile handle register/deregister + open/close**, totalling 6 seconds for 2048 chunks at ~3 ms each.

Today's read flow:

```
concurrent_map(N chunks):
    for each chunk i (in a thread):
        byte_getter_i.get(prototype)
          └─► GPULocalStore.get(key, prototype)
               └─► _gds_get_sync:
                    fd = os.open(path)                       ← 0.6 ms / chunk
                    with registered_handle(fd):              ← 0.8 ms / chunk
                        cufile.read(h, dev_ptr, size, ...)   ← 0.8 ms / chunk
                    os.close(fd)
                    return gpu.Buffer(...)

(N × ~3 ms overhead, all before any codec work runs)

decode_batch:
    bb_codec.decode([N items])    ← ONE nvCOMP call, 51 ms ✓
```

CzarrPipeline already batches at the codec layer. The wall is the **storage layer** doing per-chunk fd management.

## Locked decisions

1. **Batch at the storage layer**, not the codec layer (codec is already fine).
2. **Add `GPULocalStore.get_many(keys, prototype)`** that opens all fds, registers all handles, issues parallel cuFile reads, then deregisters/closes — *one* batched I/O round-trip for an entire selection.
3. **Override `CzarrPipeline.read_batch`** to call `get_many` instead of zarr's per-chunk `concurrent_map`. Single override; existing per-chunk `get` path stays for non-CzarrPipeline callers.
4. **Try cuFile's batched API** (`cuFileBatchIOSetUp` + `Submit` + `GetStatus`) as the inner mechanism. Project memory says batched lost on Bruno NFS for *large* chunks; per-chunk overhead dominates for *small* chunks, so the trade-off swings the other way here. Validate empirically in Phase 1.
5. **Fall-back path**: if cuFile batched I/O doesn't win, use a threadpool with **pre-registered handles** (open + register once for all fds before any read, then read in parallel, deregister + close at the end). Same algorithmic shape, dumber implementation.
6. **Symmetric for writes** — `set_many` plus `CzarrPipeline.write_batch`.

## Architecture

```
CzarrPipeline.read_batch(batch_info, out, drop_axes):
    │
    ├── Extract all ByteGetter keys + prototypes (one pass over batch_info).
    │
    ├── If all byte_getters resolve to the same GPULocalStore:
    │       chunk_bytes = await store.get_many(keys, prototype)
    │   else:
    │       chunk_bytes = await concurrent_map(...)        ← existing path
    │
    ├── decode_batch(zip(chunk_bytes, specs))              ← unchanged; already batched
    │
    └── per-chunk assembly into out[...]                    ← unchanged

GPULocalStore.get_many(keys, prototype):
    │
    ├── if cuFile batched API available and `len(keys) > 1`:
    │       _batched_cufile_read(keys, prototype)
    │   else:
    │       _threaded_get_many_with_handle_reuse(keys, prototype)
    │
    └── return list[Buffer]

_batched_cufile_read(keys, prototype):
    fds        = [os.open(path) for path in keys]            ← still per-fd, batched syscall would be nicer
    handles    = cufile_runtime.handle_register_many(fds)
    bufs       = [prototype.buffer.empty(size) for size in sizes]   ← allocated up-front
    cufile_runtime.batch_submit(handles, bufs, offsets, sizes)
    cufile_runtime.batch_wait()
    cufile_runtime.handle_deregister_many(handles)
    [os.close(fd) for fd in fds]
    return bufs
```

The expensive bit (register/deregister) goes from 0.8 ms × N to 0.8 ms × 1 (one batched syscall).  The cuFile read itself happens in parallel inside the driver.

## Phases

### Phase 0 — measure the gap ✅

- `bench/zarr/profile_smallchunk.py` cProfile run on A40, 2048-chunk workload.
- **Finding**: cuFile register/deregister/open/close = ~6 s of 6.4 s wall time. Codec is 51 ms. Bottleneck is the **storage** layer, not the codec layer.
- Phase 1 plan pivoted from "decode_batch coalescing" to "batched cuFile reads via GPULocalStore.get_many".

Deliverable: profile + redirected plan. **DONE.**

### Phase 1 — `GPULocalStore.get_many` (threaded with pre-registered handles)

- New method `GPULocalStore.get_many(keys: Sequence[str], prototype) -> Sequence[Buffer | None]`.
- Inner mechanism: thread-pool that opens + registers all handles upfront, issues parallel `cufile.read` calls, then deregisters + closes in one pass at the end.
- Avoids the cuFile batched API for now — simpler, same algorithmic shape, easy to validate.
- Tests: byte-equality vs the existing `get` loop on a real store.

Deliverable: 2048-chunk read goes from 6.4 s → expected sub-1 s on A40.

### Phase 2 — `CzarrPipeline.read_batch` override

- Override `read_batch` in `CzarrPipeline`.
- When all byte-getters resolve to the same `GPULocalStore`, call `store.get_many(...)` once.
- Otherwise fall through to zarr's default `concurrent_map`-based path.
- Re-use the existing `decode_batch` / assembly downstream.

Deliverable: pipeline routes multi-chunk reads through the batched storage path automatically.

### Phase 3 — cuFile batched API attempt

- Try `cuFileBatchIOSetUp` + `cuFileBatchIOSubmit` + `cuFileBatchIOGetStatus`.
- Compare vs the threaded handle-reuse path from Phase 1 (same workload).
- If batched API wins, swap; if it loses, document the regime where threaded wins and keep the threaded path. Project memory says batched lost on Bruno NFS for *large* chunks; this is the opposite size regime, so the answer may flip.

Deliverable: empirical decision on which inner mechanism wins for small-chunk workloads.

### Phase 4 — `set_many` symmetry

- Same architecture for writes: `GPULocalStore.set_many(keys, buffers)` + `CzarrPipeline.write_batch` override.
- Tests: round-trip on multi-chunk writes.

Deliverable: writes also batched at the storage layer.

### Phase 5 — bench

- Rerun `bench.zarr.pipeline_sweep` on A40 / H100 / H200 with the batched store path.
- Expected wins:
  - **Small chunks: ~6× speedup** (the bottleneck Phase 0 found gets cleared).
  - Medium chunks: 1.5-2× (proportional cuFile overhead, smaller).
  - Large chunks: marginal (cuFile overhead amortised across few large reads).
- Rerun `bench.zarr.slice_compare` — headline expected to climb above 6.41× on H100.

Deliverable: documented numbers in `docs/planning/batching-results.md`.

## Out of scope

- Cross-batch coalescing (across multiple `arr[...]` calls) — too speculative; one call's batch is plenty.
- Async batching — current zarr pipeline is async at the call boundary; we batch sync inside `_batch_sync`.
- Sharding inner-chunk dispatch — handled separately by zarr's `ShardingCodec` which already feeds us a batched call.

## Risks

- nvcomp.Codec.decode requires inputs + outputs to be `nvcomp.Array` instances; pre-sliced cupy views might not match the API directly. Mitigate: wrap each view in `nvcomp.as_array(view)` — cheap.
- Stream synchronization at the macro-batch boundary: all chunks share one stream's completion event. If a downstream caller wants per-chunk async semantics, we've removed that granularity. Document as a deliberate trade-off.
- Memory peak goes up: a `sum(nbytes)` alloc per call. Worst case for a 1024-chunk decode at 256 KiB each = 256 MiB transient. RMM pool absorbs it, but visible in `rmm.statistics`. Document the new peak in the bench results.
