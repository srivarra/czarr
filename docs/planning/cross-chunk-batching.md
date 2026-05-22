# cross-chunk batching at the pipeline layer

Batch ALL chunks of one `arr[...]` selection into a single per-codec call so per-chunk Python dispatch becomes per-selection. This is the next blocker for H100/H200 throughput.

## Motivation

Today's read flow inside `zarr.core.codec_pipeline.BatchedCodecPipeline.read_batch` looks like:

```
concurrent_map: get_byte_getter(chunk_i).get()  ─► chunk_bytes_i   ← parallel I/O ✓
decode_batch:   bb_codec.decode([(bytes_0, spec_0), ..., (bytes_N, spec_N)])
                  └─► CudaBytesBytesCodec._batch_sync handles them as one nvcomp.decode call ✓
```

That's already batched in theory. **In practice** the per-chunk loop in `_batch_sync` still dominates because:

- `bytes(chunk.to_bytes())` per chunk → host materialise (CzarrPipeline skips this for gpu.Buffer, but the loop still allocates an nvcomp.Array per chunk).
- `nvcomp.as_array(...)` per chunk: ~50 µs Python overhead per chunk.
- `cp.empty(spec.dtype...)` per chunk for the output buffer: ~10 µs each.
- Output wrapping via `prototype.buffer.from_array_like(dev)` per chunk: ~20 µs each.

At 1024 chunks × 80 µs per chunk = 80 ms of fixed Python overhead, regardless of how fast the underlying nvCOMP call is. On A40 with slow decode this is hidden; on H100/H200 with 20 GiB/s nvCOMP, it's the wall.

H100 small-chunk slowdown (0.86×) measured in `pipeline_compare.py` was this exact effect.

## Locked decisions

1. **Don't change zarr's BatchedCodecPipeline API** — override `decode_batch` / `encode_batch` only inside `CzarrPipeline`.
2. **Coalesce by chunk-spec-equality**: chunks with identical specs (same dtype, shape, prototype) share one big device output buffer; we view-slice on the way out.
3. **Single nvcomp.Codec.decode call** per BB-codec for the entire batch — already the contract of nvcomp, just need to feed it a flat list of inputs/outputs.
4. **Output sliced views**, not per-chunk Buffer objects, until the very end of the pipeline. Saves the `from_array_like` per-chunk cost.
5. **Keep `_batch_sync` semantics** for callers that pass a single chunk — the batching is purely an internal coalescing.

## Architecture

```
CzarrPipeline.decode_batch(chunks_and_specs):
    │
    ├── Group by (codec_id, dtype, output_shape, prototype) key.
    │   Each group becomes one "macro batch".
    │
    ├── For each macro batch:
    │   │
    │   ├── Build a flat list of nvcomp.Array views from the input chunk
    │   │   buffers (1 cp.array_from_dlpack OR 1 batched H2D if host inputs).
    │   │
    │   ├── Allocate ONE device buffer of size = sum(chunk.nbytes for c in batch).
    │   │   Hand nvcomp pre-sliced device output views into it.
    │   │
    │   ├── codec.decode(nv_inputs, out=nv_output_views)   ← single call
    │   │
    │   └── Slice the device buffer back into per-chunk Buffer wrappers
    │       (cheap — these are zero-copy views, not allocations).
    │
    └── Return assembled list in original order.
```

Allocation strategy:

```
device_buffer_pool.acquire(total_size, stream)
    └── single RMM alloc, lives for the call
    └── per-chunk views = uint8 slices into it
    └── codec writes directly into the views (nvcomp respects out= ptrs)
```

## Phases

### Phase 0 — measure the gap

- Profile current `_batch_sync` with `nsys`:
  - One frame: arr[...] on a 256-small-chunk workload on H100.
  - Confirm Python overhead per chunk vs nvcomp_decode duration.
- Document baseline in `docs/planning/batching-baseline.md` with the nsys timeline screenshot or pre/post stats table.

Deliverable: numbers proving where the time goes. Sanity-check the hypothesis before refactoring.

### Phase 1 — `decode_batch` coalescing

- Override `CzarrPipeline.decode_batch`.
- Implement chunk-spec grouping + single-buffer allocation + sliced output views.
- Pass single-group path through unchanged (1 chunk = same as today).
- Tests: end-to-end byte equality vs current path on multi-chunk reads (no chunk-spec variation).

Deliverable: same correctness, fewer Python calls per selection.

### Phase 2 — `encode_batch` symmetry

- Same coalescing for write path.
- Tests: round-trip on multi-chunk writes.

Deliverable: writes also batched at the macro level.

### Phase 3 — handle heterogeneous batches

- When chunks have differing specs (rare but possible — edge chunks in a non-uniform shard, or mixed-dtype filter outputs):
  - Multiple macro batches, each homogeneous.
  - Still one nvcomp call per macro batch.

Deliverable: correctness in the edge-chunk case.

### Phase 4 — bench

- Rerun `bench.zarr.pipeline_sweep` on H100 + H200 with the new path.
- Expected wins:
  - H100 small chunks: 0.86× → 1.5× or better (eliminate the small-chunk regression).
  - H100 medium chunks: 1.07× → 1.3-1.5×.
  - Large chunks: marginal — nvCOMP decode already dominates there.
- Rerun `bench.zarr.slice_compare` for the headline GPU-vs-CPU number — expected to climb above 6.41×.

Deliverable: documented headline numbers in `docs/planning/batching-results.md`.

### Phase 5 — propagate to filter chain

- Filters (`Shuffle`, `Delta`, `FixedScaleOffset`, `BitRound`) also go through `_decode_single` per chunk.
- Same coalescing applied to ArrayArrayCodec batches.

Deliverable: filter chain also batched.

## Out of scope

- Cross-batch coalescing (across multiple `arr[...]` calls) — too speculative; one call's batch is plenty.
- Async batching — current zarr pipeline is async at the call boundary; we batch sync inside `_batch_sync`.
- Sharding inner-chunk dispatch — handled separately by zarr's `ShardingCodec` which already feeds us a batched call.

## Risks

- nvcomp.Codec.decode requires inputs + outputs to be `nvcomp.Array` instances; pre-sliced cupy views might not match the API directly. Mitigate: wrap each view in `nvcomp.as_array(view)` — cheap.
- Stream synchronization at the macro-batch boundary: all chunks share one stream's completion event. If a downstream caller wants per-chunk async semantics, we've removed that granularity. Document as a deliberate trade-off.
- Memory peak goes up: a `sum(nbytes)` alloc per call. Worst case for a 1024-chunk decode at 256 KiB each = 256 MiB transient. RMM pool absorbs it, but visible in `rmm.statistics`. Document the new peak in the bench results.
