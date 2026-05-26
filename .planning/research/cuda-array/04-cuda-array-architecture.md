# 04 — `CudaZarrArray` architecture sketch

**Status**: design doc, no code yet.
**Goal**: a GPU-native Zarr v3 Array class that replaces the
`zarr.Array → CodecPipeline → GPULocalStore` call chain with a single
`_CudaArrayImpl` orchestrator running end-to-end on the device — modelled
on the `ZarrsArray(zarr.Array)` pattern from
[zarrs/zarrs-python#147](https://github.com/zarrs/zarrs-python/pull/147)
but in cuda-python / CUDA instead of Rust.

**Why this exists**: today on H200, a 1 GiB Z-slab read with 64 zstd
chunks costs ~93 ms wall — ~47 ms decode + ~40 ms reads, *serial*,
because `BatchedCodecPipeline.read_batch` awaits `concurrent_map(reads)`
before starting `decode_batch(...)`. Even with the in-flight microbatch
work on the standard pipeline, the surface we can squeeze through zarr's
abstractions stays bounded by their batch boundary. A side-channel
`CudaZarrArray` lets us:

1. start decode as soon as **the first chunk** lands;
2. write decoded data directly into the caller's output buffer (no
   per-chunk NDBuffer hop, no host roundtrip);
3. expose a `.lazy` view + `__cuda_array_interface__` / `__dlpack__` so
   downstream cupy / pytorch / cuTile pipelines fuse instead of forcing
   a materialise.

Cross-references:
- Agent 1's `01-cuda-core-api.md` for cuda.core primitives we lean on.
- Agent 2's `02-cufile-batched.md` for the cuFile call surface we
  schedule against.
- Agent 3's `03-overlap-microbatching.md` for the read/decode overlap
  policy we plug into here.
- Agent 5's `05-lifetime-interop.md` for VMR-backed buffer lifetime,
  DLPack handoff, cuFile registration.
- Existing planning: `docs/planning/buffer-handoff.md` (CzarrGpuBuffer),
  `docs/planning/pipeline-refactor.md` (CzarrPipeline phase plan).

---

## 1. Class signature & inheritance

### Decision: **subclass `zarr.Array`**

Mirror the zarrs PR. The `class CudaZarrArray(zarr.Array)` form gives
us, for free:

- `.shape`, `.dtype`, `.chunks`, `.metadata`, `.store_path`, `.attrs`,
  `.path`, `.basename`, `.info`, `.resize(...)`, group integration,
  `.__repr__`, the `ArrayConfig` plumbing.
- Fallback to `super().__getitem__(key)` for any indexing path we don't
  fast-path. This is the *escape hatch* — the day someone does
  `arr[bool_mask]` or `arr[[0, 5, 3]]` we don't crash, we just stop
  being fast.
- Round-trip with `zarr.open(...)`: a user can re-open a CudaZarrArray
  store as a plain `zarr.Array` from another process and everything
  still works. We are not a new format; we are a new *reader/writer*.

### How we handle `zarr.Array._async_array`

`zarr.Array` is a `@dataclass(frozen=False)` whose only state is
`_async_array: AsyncArray`. The synchronous Array delegates almost
every property to `self._async_array.metadata`, `...config`,
`...store_path`, etc. — and `_get_selection` / `_set_selection` ask
`self._async_array.codec_pipeline` to do the IO.

That means: **as long as we pass through to `super().__init__(...)`
with a valid `_async_array`, the inherited machinery keeps working**.
We don't touch `_async_array`'s codec pipeline; we just bypass it on
the fast path.

```python
class CudaZarrArray(zarr.Array):
    """Zarr v3 Array with a GPU-native fast path for basic indexing.

    Drop-in for ``zarr.Array``: same metadata, same store, same on-disk
    layout. The difference is that ``arr[...]`` (basic indexing) routes
    through ``_CudaArrayImpl`` instead of ``BatchedCodecPipeline``, so:

    * reads + decodes overlap (microbatched, see agent 3)
    * decoded chunks land directly in the user-supplied output cupy
      ndarray — no per-chunk NDBuffer materialisation
    * lazy views let downstream consumers extend the pipeline rather
      than forcing a materialise.

    Advanced / fancy indexing still works — it falls back to
    ``zarr.Array.__getitem__`` exactly like the zarrs PR's
    ``ZarrsArray``.
    """

    _impl: _CudaArrayImpl

    def __init__(
        self,
        array: zarr.Array,
        *,
        # tuning — passed through to _CudaArrayImpl
        stream_pool_size: int = 4,
        microbatch_size: int = 8,
        prefetch_depth: int = 2,
        cufile_register_handles: bool = True,
    ) -> None:
        super().__init__(array._async_array)
        self._impl = _CudaArrayImpl(
            store=array.store_path.store,
            zarr_path=array.store_path.path or "",
            metadata=array.metadata,          # ArrayV3Metadata
            chunk_grid=array._async_array._chunk_grid,
            codec_pipeline=array._async_array.codec_pipeline,
            stream_pool_size=stream_pool_size,
            microbatch_size=microbatch_size,
            prefetch_depth=prefetch_depth,
            cufile_register_handles=cufile_register_handles,
        )
```

### What we lose vs. a standalone class

If we instead made `CudaArray` a sibling of `zarr.Array` (not a
subclass), we'd lose:

- transparent participation in `zarr.Group['name']` — the group resolver
  always returns a `zarr.Array`;
- the `info_complete()`, `attrs`, `resize`, dimension-name etc. surface
  area, which we'd have to re-implement or proxy;
- any tooling that does `isinstance(x, zarr.Array)` would skip us.

These costs are not worth saving in exchange for "cleaner ownership",
particularly since the zarrs project landed on the same conclusion
and they had a *much* larger Rust runtime they could have wrapped
independently.

### Risks of subclassing

- zarr's internals (`_async_array`, `_chunk_grid`, `_get_selection`)
  are not public API. The pipeline refactor going on in zarr-python
  4.x will probably change names. **Mitigation**: do not override
  anything; use `super().__getitem__(key)` as the escape hatch and keep
  the surface area of what we override tiny (`__getitem__`,
  `__setitem__`, `lazy`, optional `copy_from`).
- `@dataclass(frozen=False)` parent means adding fields requires us to
  declare them at class scope, not in `__init__`. We accept that and
  declare `_impl: _CudaArrayImpl` as a class annotation.
- `Array.with_config(...)` returns `type(self)(self._async_array.with_config(...))`
  — uses our subclass's `__init__`, which now requires a `zarr.Array`,
  not an `AsyncArray`. We override `with_config` to wrap the result
  back into `CudaZarrArray`:

  ```python
  def with_config(self, config: ArrayConfigLike) -> CudaZarrArray:
      base = zarr.Array(self._async_array.with_config(config))
      return type(self)(base, stream_pool_size=..., ...)  # reuse our knobs
  ```

---

## 2. Indexing model

### `_parse_key` — generalised from the zarrs PR

The zarrs PR's `_parse_key` returns `(ranges, region_shape,
squeeze_dims)` where `ranges` is a per-dimension `(start, stop)` list.
We use the same shape because the *index → chunk mapping* logic on the
GPU side is identical to what zarr.indexing's `BasicIndexer` does
internally:

```python
def _parse_key(
    self,
    key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...],
) -> tuple[
    list[tuple[int, int]],   # per-dim (start, stop) inclusive-of-start
    list[int],               # region_shape (post-broadcast)
    list[int],               # squeeze_dims (int-indexed axes)
]:
    """Lift a basic-indexing key into a per-dim half-open range list.

    Matches zarrs-python ZarrsArray._parse_key. See zarrs PR #147 for
    the exact rules; in short:

    * int k         -> (k, k+1), squeeze this axis
    * slice         -> (start, stop), step must be 1 or None
    * Ellipsis      -> expand to enough slice(None)s
    * missing dims  -> implicit slice(None)
    """
```

We **deliberately** keep the contract identical to the zarrs PR so the
output buffer layout below maps trivially.

### Index → chunk coords → byte ranges

Given `ranges = [(s0, e0), (s1, e1), ...]` and the v3 regular chunk
grid `chunks = (c0, c1, ...)`:

```
chunk_range_d = (s_d // c_d, ceil(e_d / c_d))
```

The set of intersecting chunks is `itertools.product` over these
per-dim ranges. For each chunk coord `cc`:

- chunk file key = `metadata.chunk_key_encoding.encode_chunk_key(cc)`
  (e.g. `"c/0/3/2"` for default v3).
- in-chunk slice = clipped `(s_d - cc_d*c_d, e_d - cc_d*c_d)` per dim.
- out-region slice = `(cc_d*c_d - s_d + clip_lo, cc_d*c_d - s_d + clip_hi)`.

This is *exactly* what `zarr.core.indexing.BasicIndexer.__iter__`
yields as `ChunkProjection(chunk_coords, chunk_selection, out_selection,
is_complete_chunk)`. So in the first cut **we just reuse
`BasicIndexer`** — same correctness guarantees, free regression coverage
against the rest of zarr.

```python
def _projections(self, ranges, region_shape):
    """Per-chunk projection iterator. Reuses zarr.BasicIndexer for the
    arithmetic; we only consume the resulting ChunkProjection."""
    sel = tuple(slice(s, e) for s, e in ranges)
    return zarr.core.indexing.BasicIndexer(sel, self.shape, self._chunk_grid)
```

The byte-range-in-chunk question only arises for **sharded** chunks
(Zarr v3 sharding) — the outer "chunk" is a shard file; the inner
chunk lives at an offset+size encoded in the shard index. **Phase 1
scope: regular grid only.** Sharding is a separate epic — see §10.

### When we fall back to `super().__getitem__`

The zarrs PR considers basic indexing to be: int, step-1 slice, and a
single ellipsis. **We adopt the same.** Specifically we fall back when:

1. Any axis uses a bool array (`arr[mask]`) — orthogonal/fancy.
2. Any axis uses a list / ndarray of ints — vindex / orthogonal.
3. Any slice has `step not in (None, 1)` — strided.
4. Multiple ellipses (already an error in numpy; we re-raise).
5. The output `dtype` is structured (`fields`-aware indexing).
6. The selection produces a scalar (`arr[0,0]` returning python int).
   The zarrs PR handles this via `squeeze`. We can match — see below.

For each case we have a clean test:

```python
def _is_basic_indexing(key) -> bool:
    # identical to zarrs PR; reproduced for self-containment
    ...
```

```python
def __getitem__(self, key):
    if not _is_basic_indexing(key):
        return super().__getitem__(key)        # zarr fallback
    if isinstance(key, tuple) and any(...structured field stuff...):
        return super().__getitem__(key)
    ranges, region_shape, squeeze_dims = self._parse_key(key)
    out = cp.empty(region_shape, dtype=self.dtype)
    if out.size > 0:
        self._impl.retrieve_gpu(ranges, out)
    if squeeze_dims:
        out = out.squeeze(axis=tuple(squeeze_dims))
    return out
```

**Scalar reads (`arr[0,0,0]`)**: zarrs returns a 0-d ndarray and lets
the caller `.item()`. We do the same — cupy 0-d arrays work fine.

### What about `out=` and `fields=`?

`zarr.Array.__getitem__` doesn't take these — they're on
`AsyncArray.getitem(..., out=, ...)`. We don't proxy that path. If a
user wants to write into a pre-allocated cupy buffer, they go through
`arr.lazy[...].__array__()`-style explicit materialisation, or we add
an explicit `arr.read_into(out, key)` method:

```python
def read_into(self, out: cp.ndarray, key) -> None:
    """Fill ``out`` (cupy ndarray) with arr[key]. Out-shape must match."""
    ranges, region_shape, _ = self._parse_key(key)
    if tuple(region_shape) != tuple(out.shape):
        raise ValueError(...)
    if out.size > 0:
        self._impl.retrieve_gpu(ranges, out)
```

This is the single most useful API we add on top of zarrs's pattern,
because GPU users often have a pre-allocated workspace.

---

## 3. `retrieve_gpu` / `store_gpu` hot paths

### `_CudaArrayImpl` — what it owns

```python
class _CudaArrayImpl:
    """Per-array GPU IO orchestrator.

    Owns: open store, chunk grid + key encoding, codec chain
    (Array-Array filters -> Array-Bytes codec -> Bytes-Bytes
    compressors as a tuple of czarr CudaXxxCodec instances), a
    StreamPool, a CzarrGpuBuffer alloc pool, optional cuFile registered
    handles (one per chunk path or one per file when sharded).

    Methods:
      retrieve_gpu(ranges, out)
      store_gpu(ranges, value)
      copy_from(source_impl, source_ranges, dest_ranges)
      close()  -- deregister cuFile handles, free buffers
    """

    def __init__(self, *, store, zarr_path, metadata, chunk_grid,
                 codec_pipeline, stream_pool_size, microbatch_size,
                 prefetch_depth, cufile_register_handles):
        self._store = store
        self._zarr_path = zarr_path
        self._metadata = metadata
        self._chunk_grid = chunk_grid
        self._codec_chain = self._compile_codec_chain(codec_pipeline.codecs)
        self._streams = StreamPool(size=stream_pool_size)
        self._microbatch_size = microbatch_size
        self._prefetch_depth = prefetch_depth
        self._cufile_register = cufile_register_handles
        self._encoded_pool = DeviceBufferPool()      # CzarrGpuBuffer-backed
        self._scratch_pool = DeviceBufferPool()      # for filter pipeline temps
```

### `retrieve_gpu(ranges, out_cupy)` — the hot path

The single most important method. Step by step:

```python
def retrieve_gpu(self, ranges, out_cupy: cp.ndarray) -> None:
    """Read+decode the chunks intersecting `ranges` straight into `out_cupy`.

    No host roundtrip; no per-chunk NDBuffer; reads and decodes overlap
    on the StreamPool.
    """
    # 1. Enumerate chunk projections (chunk_coord, chunk_sel, out_sel,
    #    is_complete_chunk). Reuse BasicIndexer.
    projections = list(self._projections(ranges, _region_shape(ranges)))

    if not projections:
        return

    # 2. Stat all chunk files in parallel (or use store.exists_many) so
    #    we know which are "missing" (=> fill_value) vs present, and
    #    their on-disk sizes for cuFile allocations.
    paths_and_sizes = self._stat_chunks([p.chunk_coords for p in projections])

    # 3. Bucket into microbatches of `microbatch_size` chunks.
    #    Why microbatch instead of one big batch?
    #      - amortise cuFile open+register cost (need >=N chunks per batch
    #        before threading helps).
    #      - keep nvCOMP scratch memory bounded.
    #      - start nvcomp.decode on batch K while cuFile reads batch K+1.
    #
    #    See agent 3's report for the chosen `microbatch_size` and
    #    `prefetch_depth` heuristics. We assume both are passed in.
    batches = _bucket_microbatches(projections, paths_and_sizes,
                                   self._microbatch_size)

    # 4. Pipeline: for each batch
    #       (a) alloc encoded-buffer slab on stream S_read
    #       (b) cuFile read encoded chunks into slab
    #          - if `cufile_register_handles`: use pre-registered handles
    #            stored in self._fh_cache (LRU keyed by path)
    #          - sync semantics for compat mode (no async); on real-GDS
    #            hosts use read_async to inherit stream ordering
    #       (c) record event E_read
    #       (d) on stream S_decode, wait E_read, then nvcomp.decode
    #          INTO an scratch ND-buffer view (because we still need to
    #          apply array-array filters + slice into out_cupy)
    #       (e) on stream S_decode, run filters in reverse, copy the
    #           in-chunk slice into out_cupy[out_selection]
    #       (f) record event E_done
    #
    # Stream scheduling: round-robin (S_read, S_decode) pairs from the
    # StreamPool. Default pool size 4 = 2 read/decode pairs running
    # concurrently. See agent 1's stream-pool doc.
    in_flight: deque[_BatchHandles] = deque()
    for batch in batches:
        # Respect prefetch_depth: at most N in-flight batches before
        # we drain one to keep memory bounded.
        while len(in_flight) >= self._prefetch_depth:
            in_flight.popleft().wait_decode_complete()

        handles = self._issue_batch(batch, out_cupy)
        in_flight.append(handles)

    # 5. Drain.
    for h in in_flight:
        h.wait_decode_complete()
```

### Read / decode overlap — concretely

```
microbatch_size = 8, prefetch_depth = 2, stream_pool_size = 4

batch 0  read on S0 ─────────────────► event R0
         decode on S1 (wait R0) ─────► event D0
batch 1  read on S2 ─────────────────► event R1   (concurrent with S1)
         decode on S3 (wait R1) ─────► event D1
batch 2  read on S0 (after D0) ──────► event R2   (S0 free again)
         ...
```

The "after D0" condition on S0 is automatic *iff* the encoded-buffer
slab is freed back to the device pool only after D0 — see §5
(buffer-prototype discussion). If we use `CzarrGpuBuffer` (VMR-backed,
4 KiB aligned) and let the pool track per-stream events, no manual
`Event.wait` is needed; the allocator inserts the dependency.

If we use `cupy.ndarray` directly, we need an explicit `Event` per
batch and a `stream.wait(event)` to safely recycle the slab.

### Stream scheduling — one stream per micro-batch? StreamPool?

**Decision: dedicated read/decode stream pair per in-flight microbatch,
drawn round-robin from a single StreamPool of size 2 × prefetch_depth.**

Rationale:
- nvCOMP's batched decode can saturate one stream's compute easily;
  a second concurrent decode stream is what gives the chip a second
  workload to hide gaps (memory-bound zstd) or chip kicks.
- A second concurrent *read* stream is what gives cuFile a longer
  queue to schedule against (matters mostly in real-GDS mode; in
  compat mode the bottleneck is the host bounce-buffer
  `cufile_posix_read`, which serialises anyway).
- With `prefetch_depth=2` and `stream_pool_size=4` (two read/decode
  pairs), we get the same 2× decode + 2× read concurrency as zarr's
  default `concurrent_map(reads)` but without the *barrier* between
  the read and decode phases.

This decision *should* be re-checked against agent 3's overlap
microbatching report — they may have different numbers.

### Buffer prototype — `CzarrGpuBuffer` vs `cupy.ndarray`

**Decision: `CzarrGpuBuffer` for encoded slabs, `cupy.ndarray` for the
caller-owned `out_cupy`.**

Why `CzarrGpuBuffer` for encoded slabs:
- 4 KiB aligned + GPU-direct-RDMA tagged → cuFile direct path on
  real-GDS hosts; on compat-mode hosts it still works, just goes
  through the kernel-bounce buffer. No measured downside to alignment
  there.
- `cuda.core.VirtualMemoryResource` gives clean per-stream lifetime
  ordering via DLPack (see agent 5). cupy's pool is per-device, not
  per-stream — stream-ordered allocs require `CUDA_PYTHON_ASYNC_ALLOC=1`
  and the user to opt in.
- Phase 3 of the buffer epic registers the device pointer with cuFile
  *at allocation time* — register-once-per-pool — which kills the
  per-read `buf_register`/`buf_deregister` cost. That cost was
  measured at ~1 ms/call on real GDS and ~3 ms on compat (see
  `docs/planning/buffer-handoff.md`). It's the dominant fixed
  overhead today.

Why `cupy.ndarray` for `out_cupy`:
- The user passed it in. We don't get to choose its allocator.
- nvCOMP can write to any device pointer; alignment of the output
  is fine for any reasonable ND-array (cupy returns ≥256 alignment
  from its pool — well above what nvcomp's vector loads require
  on the *output* side).
- Decoded data is *plain decompressed bytes* that we then scatter
  into `out_cupy[out_selection]` via a cupy slice assignment
  — the kernel that does that scatter is whatever cupy generates
  for `__setitem__`, no special alignment needed.

### cuFile direct-into-output-buffer — when?

For an *uncompressed* zarr array (codec chain is just bytes-encoding,
no compressor), and the requested range covers full chunks with
chunk_size that's a multiple of the dtype itemsize, we *could*
cuFile-read directly into a slice of `out_cupy`. Skipping the
intermediate slab is a win.

**Conditions for direct-into-output**:
1. Codec chain has zero `Bytes-Bytes` compressors.
2. Codec chain's `Array-Bytes` codec is `BytesCodec` (no transcoding).
3. There are no `Array-Array` filters (shuffle / delta / bitround).
4. The requested range covers *whole* chunks in every dim except
   possibly the contiguous innermost (since the kernel can DMA a
   contiguous run into a contiguous slice of `out_cupy`).
5. `out_cupy` is C-contiguous.
6. `out_cupy.dtype` matches `metadata.dtype` (or is a view of bytes).

When all six hold:
```
out_chunk_slice = out_cupy[out_selection]   # contiguous slice
cufile.read(fh, int(out_chunk_slice.data.ptr), size, file_offset, 0)
```

No staging, no decode. **This is the killer feature** for
uncompressed scratch arrays (cuML model checkpoints, intermediate
tensors written to scratch). Even with a single chunk it eliminates
two allocations + one device copy.

In the common-case **compressed** path (the demo target), step 1
fails immediately so we always go through the slab path.

```python
def _can_direct_io(self, projection) -> bool:
    return (
        not self._codec_chain.has_compressors
        and not self._codec_chain.has_filters
        and isinstance(self._codec_chain.serializer, BytesCodec)
        and projection.is_complete_chunk
    )
```

### `store_gpu(ranges, value_cupy)` — write path

The inverse of retrieve. For each intersecting chunk:

1. Read existing chunk if the write doesn't cover it whole
   (read-modify-write) — same logic zarr uses.
2. Decode (only when RMW). Same path as `retrieve_gpu` for the read
   half.
3. Patch `decoded[chunk_selection] = value_cupy[out_selection]` on
   stream.
4. Encode the patched chunk on stream → produces an encoded
   `CzarrGpuBuffer`.
5. cuFile-write the encoded buffer back to disk.

The write path is harder to overlap because encode is data-dependent
on the patch, which is data-dependent on the prior decode for RMW
chunks. For *full-chunk writes* (the only common case for
high-throughput consumers), there's no read step:

```python
def store_gpu(self, ranges, value):
    projections = list(self._projections(ranges, _region_shape(ranges)))
    batches = _bucket_microbatches(projections, [], self._microbatch_size)
    in_flight = deque()
    for batch in batches:
        if all(p.is_complete_chunk for p in batch):
            handles = self._issue_full_chunk_writes(batch, value)
        else:
            handles = self._issue_rmw_writes(batch, value)
        in_flight.append(handles)
        while len(in_flight) >= self._prefetch_depth:
            in_flight.popleft().wait_write_complete()
    for h in in_flight:
        h.wait_write_complete()
```

Phase 1 scope: support full-chunk writes only. RMW falls back to
`super().__setitem__(key, value)`.

---

## 4. Lazy indexing API

### Why lazy?

Two reasons:
1. Compose. `arr.lazy[:, 0:128] @ kernel(...)` lets us fuse the
   read+decode with the kernel — same as the zarrs PR's intent but
   *more powerful* on GPU because zero-copy interop matters more.
2. Defer. A user might describe a workflow as `out = arr.lazy[...]`
   but never materialise it because they only need it as a
   `__cuda_array_interface__` source for something else.

### Shape

Identical to the zarrs PR with two additions: `__cuda_array_interface__`
and `__dlpack__`.

```python
class _LazyIndexer:
    """``arr.lazy`` — proxy that captures __getitem__ keys lazily."""

    __slots__ = ("_arr",)

    def __init__(self, arr: CudaZarrArray):
        self._arr = arr

    def __getitem__(self, key) -> _LazySlice:
        if not _is_basic_indexing(key):
            raise IndexError("lazy indexing requires basic indexing; "
                             "use arr[key] for advanced indexing fallback")
        ranges, region_shape, squeeze_dims = self._arr._parse_key(key)
        return _LazySlice(
            impl=self._arr._impl,
            ranges=ranges,
            region_shape=tuple(region_shape),
            dtype=self._arr.dtype,
            squeeze_dims=tuple(squeeze_dims),
        )


@dataclass(frozen=True, slots=True)
class _LazySlice:
    impl: _CudaArrayImpl
    ranges: list[tuple[int, int]]
    region_shape: tuple[int, ...]
    dtype: np.dtype
    squeeze_dims: tuple[int, ...]

    # ----- numpy-compatible materialisation -----
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out_gpu = self._materialise()
        out = cp.asnumpy(out_gpu)
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype, copy=False)
        return out

    # ----- cupy / pytorch / cuTile / nvCOMP zero-copy -----
    @property
    def __cuda_array_interface__(self) -> dict:
        return self._materialise().__cuda_array_interface__

    def __dlpack__(self, stream=None):
        return self._materialise().__dlpack__(stream=stream)

    def __dlpack_device__(self):
        # cupy device ordinal; safe to compute without IO
        return (2, self.impl.device_id)   # 2 = kDLCUDA

    # ----- public materialise (single-shot, memoised on slice) -----
    @functools.cached_property
    def _cached(self) -> cp.ndarray:
        out = cp.empty(self.region_shape, dtype=self.dtype)
        if out.size > 0:
            self.impl.retrieve_gpu(self.ranges, out)
        if self.squeeze_dims:
            out = out.squeeze(axis=self.squeeze_dims)
        return out

    def _materialise(self) -> cp.ndarray:
        return self._cached
```

Notes:
- `_cached` makes a `_LazySlice` *one-shot* per object — if you call
  `__array__()` then `__cuda_array_interface__`, the second hit is
  free. If you want a fresh read, make a new slice.
- The cached cupy result keeps the underlying device memory alive
  for the lifetime of the `_LazySlice`. Users who want to drop it
  delete the slice.
- We expose **both** `__cuda_array_interface__` and `__dlpack__` —
  cupy and pytorch prefer different protocols, and our buffer epic
  uses DLPack as its native handoff. See agent 5.

### Risks

- **Dangling slices** — a `_LazySlice` outlives its parent
  `CudaZarrArray`, and the impl is closed (cuFile handles released).
  Materialisation then explodes with a confusing error.
  **Mitigation**: `_LazySlice` holds a strong reference to `impl`
  (via the dataclass field). `_CudaArrayImpl.close()` is the only
  way to release resources, and it's *not* in `__del__` — we
  document explicit close-on-array-close and reference-count by
  Python.
- **Stale data** — if a user writes into the same array, the cached
  `_LazySlice._cached` won't see the new bytes. Document: a
  `_LazySlice` is a read-time snapshot. For fresh data, recreate.

---

## 5. `copy_from` — chunk-aligned GPU-to-GPU

The zarrs PR's `ArrayImpl.copy_from(source_impl, source_ranges,
dest_ranges)` exists to short-circuit `dest[ranges] = source.lazy[ranges]`
when both arrays share a codec chain and the ranges align to chunks:
it copies the *encoded* chunk bytes directly.

### Our version: decode-once, scatter-many

The GPU equivalent is more useful than the Rust version because the
**device-resident decoded buffer** can be reused if it's chunk-aligned.
Case analysis:

1. **Identical codec chain + chunk_size + aligned ranges**: copy the
   encoded chunk bytes via cuFile read+write (skip decode/encode
   entirely). Same as zarrs.

2. **Same dtype + aligned ranges + different codec**: decode source
   on GPU, encode into dest. Skips host roundtrip. This is the
   re-sharding workflow (`coarsen`, `rechunk`).

3. **Same dtype + same codec + non-aligned ranges**: decode source
   chunks, scatter into dest, re-encode affected dest chunks. RMW
   per dest chunk.

```python
def copy_from(self, src_impl, src_ranges, dest_ranges):
    """Device-side chunk copy.

    Equivalent to ``dest[dest_ranges] = src.lazy[src_ranges].__cuda_array_interface__``
    but skips intermediate host copies and, when the chunk grids align
    + codec chains match, skips decode/encode too.
    """
    if self._can_passthrough_copy(src_impl, src_ranges, dest_ranges):
        return self._encoded_passthrough(src_impl, src_ranges, dest_ranges)
    # General case: decode src once, set into dest as a contiguous
    # cupy array. Path through retrieve_gpu + store_gpu but with a
    # shared device buffer instead of round-tripping to user code.
    region_shape = _region_shape_from_ranges(src_ranges)
    staging = cp.empty(region_shape, dtype=self._metadata.dtype.to_native_dtype())
    src_impl.retrieve_gpu(src_ranges, staging)
    self.store_gpu(dest_ranges, staging)
```

`_can_passthrough_copy` checks:
- `src_impl._codec_chain == self._codec_chain`
- `src_impl._metadata.chunks == self._metadata.chunks`
- `src_impl._metadata.dtype == self._metadata.dtype`
- `src_ranges` aligns to `src_impl`'s chunk grid (every range a chunk
  boundary)
- `dest_ranges` aligns to our chunk grid

When can we *invoke* `copy_from`? Two integration points:
- `dest_arr[ranges] = src_arr.lazy[ranges]` — overridden in
  `__setitem__` to detect the lazy-on-rhs case and route here.
- Explicit API: `dest_arr.copy_from(src_arr, src_ranges, dest_ranges)`.

### Workflow examples

- **rechunk**: open source v3 array, open destination v3 array with a
  different chunk shape, iterate over dest chunks, copy_from
  source-aligned regions. Saves decode-encode on every chunk-aligned
  dest chunk.
- **coarsen / downsample**: dest chunks are a multiple of src chunks
  in one dim; same encoded-passthrough condition.
- **append**: dest is a v3 array being extended; new region is
  chunk-aligned by definition; same encoded passthrough.
- **resize-and-fill**: dest grows, fill_value is the same as src;
  copy_from existing region, leave new region to default.

---

## 6. Public API

```python
import czarr
import cupy as cp
import zarr

czarr.configure_gpu(rmm_pool_gb=2)            # existing call, no change

# Three constructors:
arr = czarr.CudaZarrArray(zarr.open_array("data.zarr", mode="r"))   # wrap
arr = czarr.open_cuda_array("data.zarr", mode="r")                  # open
arr = czarr.create_cuda_array(                                       # create
    "out.zarr", shape=(10_000, 4096, 4096), chunks=(1, 4096, 4096),
    dtype="float32", compressors=[czarr.Zstd()])

# Use.
slab: cp.ndarray = arr[0:8, :, :]            # eager basic indexing → cupy
arr[8:16, :, :] = some_cupy_array            # eager full-chunk write
lazy = arr.lazy[:, :, :]                     # _LazySlice, no IO yet
torch_t = torch.utils.dlpack.from_dlpack(lazy)
host_arr = np.asarray(lazy)
buf = cp.empty((8, 4096, 4096), dtype="float32")
arr.read_into(buf, np.s_[0:8])               # zero-alloc

# Device-to-device chunk copy.
new = czarr.create_cuda_array("new.zarr", shape=arr.shape,
                              chunks=(4, 4096, 4096), dtype=arr.dtype,
                              compressors=[czarr.Zstd()])
new.copy_from(arr, src_ranges=..., dest_ranges=...)

```

### `configure_gpu(use_cuda_array=...)`?

No, not in Phase 1. The semantics differ enough (sync vs async, cupy
vs numpy return) that silent swap would surprise users. A
`czarr.configure(default_array_class=CudaZarrArray)` hook is an
option later, but `arr = czarr.CudaZarrArray(...)` is the right
default verbosity for now.

---

## 7. Integration with existing czarr machinery

| Component | Reused? | How |
|---|---|---|
| `GPULocalStore` | Yes | as the `store` argument; we still go through `store.get` for any non-fast-path reads (advanced indexing falls back), and we use `store.get_many` as the slow lane when cuFile direct isn't available. |
| `cufile_runtime.{read_into, read_into_many, read_async}` | Yes | direct calls from `_CudaArrayImpl._issue_batch`. We skip `GPULocalStore.get` to avoid the Buffer/Prototype layer when we know we want a `CzarrGpuBuffer`-backed slab. |
| `CudaBytesBytesCodec` family | Partially | we call `_get_codec()` / `_codec_kwargs()` to instantiate nvCOMP, then call `nvcomp.Codec.decode(...)` ourselves with our own stream and output buffer. We *don't* go through `BytesBytesCodec.decode(chunks_and_specs)` because that path allocates a fresh output buffer per chunk and returns a list of Buffers — we want one slab. The codec config (algorithm, chunk_size, bitstream_kind, checksum_policy) round-trips unchanged. |
| `CzarrPipeline` (the existing CodecPipeline subclass) | Bypassed | the standard `arr[:]` path through `BatchedCodecPipeline.read_batch` is not invoked from `CudaZarrArray.__getitem__`. It remains as the *fallback* path the inherited `super().__getitem__` calls for advanced indexing. |
| `CzarrGpuBuffer` (from buffer epic worktree) | Yes | for encoded slabs in `retrieve_gpu`. When the buffer epic merges, we get the 4 KiB-aligned + cuFile-registered substrate for free. |
| `StreamPool`, `PinnedHostPool`, `DeviceBufferPool` (`czarr.pipeline`) | Yes | `_CudaArrayImpl` instantiates per-array StreamPool + DeviceBufferPool. PinnedHostPool is unused unless we add a host-staging fallback for the compat-mode-on-NFS case (Bruno VAST). |
| `czarr._buffer.buffer_to_nvarray` / `nvarray_to_buffer` | Partially | we still need them for the *codec encode path* in `store_gpu`, and we may call `nvcomp.as_array(cp_arr)` directly in `retrieve_gpu` since we already have a cupy view. |
| `czarr._nvtx.nvtx_range` | Yes | wrap each phase (read, decode, scatter) so nsys timelines line up with the existing benches. |
| Filters (`Delta`, `Shuffle`, `BitRound`, `FixedScaleOffset`) | Yes | we apply them in `_CudaArrayImpl._apply_filters_reverse` on the decode output. Same kernels they currently run; we just call them inline instead of through Zarr's batched-pipeline plumbing. |

### What we **bypass**

- `zarr.core.codec_pipeline.BatchedCodecPipeline` (the serial-barrier
  parent of CzarrPipeline). This is the entire point.
- `zarr.core.indexing` for the per-chunk slice math: we reuse
  `BasicIndexer` as a *correctness oracle* but the actual byte-level
  IO is ours.
- `Buffer.from_bytes(...) / to_bytes()` round-trips. The whole point
  is no host roundtrip.
- `asyncio.to_thread` for the codec call — we run nvCOMP synchronously
  on a CUDA stream and let CUDA do the host/device decoupling.

---

## 8. Migration / coexistence

### Opt-in model

**Per-array, explicit, default off.** This matches zarrs's PR and is
the right default because:
- The return type changes (cupy vs numpy) — silently swapping in
  `CudaZarrArray` would break any caller that calls `.tobytes()` or
  expects a numpy buffer.
- Some codec chains we don't accelerate yet (no GPU implementation
  of the filter, sharding codec, fancy v2-style filters). Letting
  the user opt in means they implicitly assert "my codec chain is
  GPU-friendly".
- The benchmark dance (cold-vs-warm, slab size sweeps) needs the
  user to explicitly hit our path so we can compare.

### How

```python
# Wrap existing array.
arr = czarr.CudaZarrArray(zarr.open_array(...))

# Open a new array directly.
arr = czarr.open_cuda_array(...)

# Per-array opt-in via zarr's existing extension hook (codec_pipeline kwarg)?
# Doesn't quite fit -- we replace the whole array surface, not just the pipeline.
# Skip.
```

### When `czarr.configure_gpu(use_cuda_array=True)` could exist

Long-term, once the API has settled and the fallback path is robust:

```python
czarr.configure_gpu(
    use_cuda_array=True,   # zarr.open(...) returns CudaZarrArray
    cuda_array_strict=False,  # advanced indexing falls back vs raises
)
```

This would patch `zarr.api.synchronous.open_array` (or use zarr's
registry hook if one exists) to wrap with `CudaZarrArray`. Out of
scope for Phase 1.

### Coexistence with the existing pipeline

A CudaZarrArray and a plain zarr.Array can refer to the same on-disk
store at the same time. They are not exclusive. The `_async_array`
underneath has the standard `BatchedCodecPipeline` (or `CzarrPipeline`
if `configure_gpu` set it); `CudaZarrArray` bypasses it on the fast
path and uses it on the slow fallback path. Coherence is maintained
because both write through the same `store.set` semantics eventually.

---

## 9. Tests

**Unit** (`tests/array/`):

- `test_basic_indexing.py` — all `arr[i, j, k]` / `arr[a:b]` / `...`
  forms, compare to `np.asarray(zarr_arr[key])`.
- `test_fallback_indexing.py` — bool mask, int list, strided slice;
  spy `zarr.Array.__getitem__` is called and result matches.
- `test_lazy.py` — `.lazy[...]` returns `_LazySlice`; `np.asarray(lazy)`
  matches eager; `cp.asarray(lazy)` is zero-copy (CAI ptr match);
  `torch.utils.dlpack.from_dlpack(lazy)` works (skip if torch absent).
- `test_roundtrip_vs_zarr.py` — for each compressor (zstd, lz4, gzip,
  zlib, snappy, deflate, ans, bitcomp, gdeflate): write with
  zarr.Array, read with CudaZarrArray, assert equal. And vice versa.
- `test_filters.py` — same matrix with each filter in the chain.
- `test_copy_from.py` — chunk-aligned passthrough (no decode happens);
  non-aligned falls back to decode path; both correct.
- `test_setitem.py` — full-chunk write, RMW (assert fallback), lazy
  RHS (`dest[r] = src.lazy[r]`).
- `test_close.py` — open + close, assert cuFile `buf_deregister`
  called for all registered ptrs.
- `test_dangle.py` — `_LazySlice` outlives parent array; still
  materialises until both drop.

**Bench** (`bench/cuda_array/`):

- `slab_compare.py` — 1 GiB Z-slab, 64 zstd chunks; compare
  `zarr.Array[:]` baseline, `configure_gpu` (CzarrPipeline),
  `CudaZarrArray[:]`. Target: ≥1.5× on H200.
- `microbatch_sweep.py` — sweep `microbatch_size` ∈ {1, 4, 8, 16, 64}
  × `prefetch_depth` ∈ {1, 2, 4} on H200 and A40.
- `lazy_compose.py` — `.lazy[...]` feed to downstream cuTile/cupy
  reduction without intermediate materialise; measure vs
  eager-then-feed.

**Bruno constraints**: compat-mode-only (no `nvidia_fs`). We gate the
`read_async` tests on `cufile_runtime.is_async_available()` and add a
`@pytest.mark.real_gds` marker. Perf benchmarks require local NVMe or
Lustre.

---

## 10. Risks

**Subclassing brittleness**. zarr-python 3.x hasn't stabilised:
`_async_array._chunk_grid`, `._codec_pipeline`, `_get_selection` are
underscore-private. Mitigation: wrap every access in a small adapter
(`_get_chunk_grid(async_arr)`), pin a minimum zarr version, run our
suite against zarr-python `main` weekly.

**Dangling lazy slices**. A `_LazySlice` may outlive its parent
`CudaZarrArray` and crash when materialising if `arr.close()` already
released cuFile handles. Mitigation: `_LazySlice` holds a strong ref
to `_CudaArrayImpl`, no `close()` in `__del__`, document explicit
cleanup. Phase 2: ref-count `_LazySlice`s against the impl so
`close()` becomes a no-op until slices drop.

**GPULocalStore on VAST (compat mode)**. Bruno's cuFile is compat-only;
every read goes through `cufile_posix_read` (kernel bounce buffer).
Register-once still saves the per-call register cost, but absolute
throughput is bounded by VAST's POSIX read. Fundamental — can't fix at
the czarr layer. We must verify compat-mode performance regresses no
worse than `zarr.Array` baseline. Demo runs on H200 local NVMe.

**cuFile + decode overlap — natural ordering vs explicit events**.
Within a stream, ops serialise in submit order, so
`cufile.read_async(stream=S)` then `nvcomp.decode(stream=S)` needs
no event. *Between* streams (read on `S_read`, decode on `S_decode`)
we need an `Event()`:

```python
cufile.read_async(fh, dev, size, S_read, args=...)
read_event = Event(); read_event.record(S_read)
S_decode.wait_event(read_event)
nvcomp.decode(..., stream=S_decode)
```

`cuda.core.Event` + `Stream.wait_event` cover this — see agent 1.
Standardise on `cuda.core.Stream` internally; accept foreign streams
via `__cuda_stream__` (same as `CudaBytesBytesCodec.cuda_stream`).

**Sharding (v3)**. Inner chunks live at `(shard_file, byte_offset,
byte_length)`; our chunk-key → file-key assumption breaks. Out of
scope for Phase 1; sharded reads fall through to `super().__getitem__`.

**nvCOMP scratch memory**. nvCOMP holds stream-bound scratch on each
`Codec` instance. With `stream_pool_size=4` we cache one Codec
per-stream (not per-thread, unlike today's `CudaBytesBytesCodec`).
Scratch scales linearly: zstd ~50 MiB × 4 streams = ~200 MiB. Fine on
H200 (140 GiB); flag for smaller cards.

**Error propagation through async**. cuFile async errors surface only
after stream sync — the submit call succeeds, the actual IO fails.
`_BatchHandles.wait_decode_complete()` syncs, then checks each
`args.bytes_done`; short reads (other than legitimate EOF) raise a
typed `czarr.CudaReadError` with chunk coord + path + bytes_done.

**Output-buffer allocation**. `cp.empty(region_shape, ...)` per
`__getitem__` is fine because cupy's pool (RMM-backed if
`configure_gpu(rmm_pool_gb=...)`) recycles. Document `arr.read_into`
as the zero-allocation path for hot loops.

**Threading model**. `retrieve_gpu` is sync-on-host (no `await`). In
an async context we'd block the event loop during cuFile reads, but
cuFile releases the GIL during the syscall, so the practical impact
is small. Phase 2 polish: `retrieve_gpu_async()` returning a
`_BatchHandles`-like awaitable.

---

## 11. Implementation phases (suggested)

Concrete dex epics this maps to:

- **Phase 1**: `_CudaArrayImpl` skeleton + retrieve_gpu (no overlap,
  no microbatch) → unit tests pass. Bench vs zarr.Array baseline.
  Expect parity, maybe -10%/+10% (we lose async-pipeline overlap but
  gain no-host-roundtrip).
- **Phase 2**: microbatch + read/decode overlap (depends on agent 3's
  parallel CzarrPipeline work). Bench expects 1.4-1.8× on H200 with
  the standard 1 GiB Z-slab.
- **Phase 3**: cuFile register-once integration with the buffer epic.
  Bench expects 1.6-2× on H200 (compounding with phase 2).
- **Phase 4**: `store_gpu` full-chunk write path. Round-trip tests.
- **Phase 5**: `.lazy` + `copy_from`. Compose tests.
- **Phase 6**: sharding fast path (if scoped in).

Each phase is independently shippable: the fast path either works
(returns cupy) or falls back to `super()` (returns numpy). No silent
correctness regression.

---

## 12. Open questions

- **Return type on fallback path**: eager fast path returns cupy,
  fallback returns numpy. Document the dual contract; `arr.lazy[...]`
  is the always-device alternative for code that needs the guarantee.
- **`arr.attrs.set(...)` / `arr.resize(...)`** — through `super()`,
  untouched. Resize invalidates `_fh_cache`; add a
  `_on_metadata_changed` hook in Phase 4.
- **Per-array vs class-level stream pool**: per-array — per-array
  overlap policy may differ (small chunks → more streams). The
  CzarrPipeline class-level pool stays for codec-only fallback calls.
- **`zarr.Group[name]` returns plain `zarr.Array`**: user has to
  wrap (`czarr.CudaZarrArray(group['x'])`) for now. A
  `configure_gpu(use_cuda_array=True)` registry hook automates this
  later — out of scope for Phase 1.
- **`force_gpu` / `host_fallback`**: `CudaZarrArray` refuses to
  construct unless the store supports cuFile (or `host_fallback=True`
  is set, which routes through cupy H2D — slow but correct).

---

## 13. Skeleton — what the file tree looks like after Phase 1

```
src/czarr/
  array/                       # new package
    __init__.py
    cuda_array.py              # CudaZarrArray, _LazyIndexer, _LazySlice,
                               # open_cuda_array, create_cuda_array
    _impl.py                   # _CudaArrayImpl
    _codec_chain.py            # compiled codec chain (filters + bytes +
                               # compressors), per-stream codec cache
    _batch.py                  # _BatchHandles, _bucket_microbatches,
                               # _issue_batch
  ...
tests/
  array/
    test_basic_indexing.py
    test_fallback_indexing.py
    test_lazy.py
    test_roundtrip_vs_zarr.py
    test_filters.py
    test_copy_from.py
    test_setitem.py
    test_close.py
bench/
  cuda_array/
    slab_compare.py
    microbatch_sweep.py
    lazy_compose.py
```

Total new code estimate: ~1500 lines, of which ~600 is the
`_CudaArrayImpl` hot path, ~400 is `CudaZarrArray` + lazy, ~500 is
tests + benches.

---

## 14. Sketch — `_issue_read_batch` + codec chain orchestration

The earlier sections already showed `CudaZarrArray`'s top-level shape
(§1), `_parse_key` (§2), `retrieve_gpu` skeleton (§3), `_LazySlice`
(§4), `copy_from` (§5), and the `_CudaArrayImpl` public surface (§3).
The one piece left is the *inside* of `_issue_read_batch` — i.e.
the place where streams, cuFile, nvCOMP, and the output buffer all
meet. That's:

```python
def _issue_read_batch(self, batch, out_cupy) -> "_BatchHandles":
    # 1. Resolve paths + on-disk sizes. _stat returns the bytes
    #    actually on disk per chunk; missing chunks are filtered out
    #    (caller will fill from metadata.fill_value).
    paths_sizes = [(self._chunk_path(p.chunk_coords), self._stat(p))
                   for p in batch]

    # 2. Allocate ONE encoded slab covering all chunks in this batch.
    #    Using CzarrGpuBuffer = 4 KiB aligned + RDMA tagged + sized
    #    so cuFile-buf-register is amortised across the whole batch.
    sizes = [sz for _p, sz in paths_sizes]
    offsets = list(np.cumsum([0] + sizes))
    from czarr.core.buffer import CzarrGpuBuffer    # buffer epic
    slab = CzarrGpuBuffer.empty(offsets[-1])

    # 3. Pick a read stream + a decode stream from the pool.
    s_read = self._streams.acquire()
    s_decode = self._streams.acquire()
    read_event = Event()

    # 4. cuFile reads. Two paths: async (real GDS) vs sync (compat).
    if self._cufile_register and cufile_runtime.is_async_available():
        cufile_runtime.ensure_buf_registered(slab.device_ptr, offsets[-1])
        cufile_runtime.ensure_stream_registered(int(s_read.handle))
        io_args = []
        for (path, size), off in zip(paths_sizes, offsets):
            fh = self._get_handle(path)  # cached; one-time register
            args = cufile_runtime.make_io_args()
            cufile_runtime.read_async(
                fh, slab.device_ptr + off, size, int(s_read.handle),
                args=args, file_offset=0)
            io_args.append(args)
        read_event.record(s_read)
    else:
        # Compat-mode fallback: threaded sync reads. Pretty-much
        # identical to today's GPULocalStore._gds_get_many_sync, but
        # writing into our slab instead of per-chunk buffers.
        requests = [(path, slab.device_ptr + off, size, 0)
                    for (path, size), off in zip(paths_sizes, offsets)]
        cufile_runtime.read_into_many(requests)
        # No host-event needed: read_into_many is sync; record now.
        read_event.record(s_read)
        io_args = None

    # 5. Decode on the decode stream once reads land. The codec
    #    chain (compiled once at __init__) does:
    #      (a) nvcomp.decode(slab[offsets..]) -> per-chunk byte buffers
    #      (b) apply Array-Array filters in reverse (delta/shuffle/bitround)
    #      (c) scatter each decoded chunk into out_cupy[out_selection]
    s_decode.wait_event(read_event)
    decode_event = self._codec_chain.decode_into(
        slab=slab,
        offsets=offsets, sizes=sizes,
        out_cupy=out_cupy,
        projections=batch,
        stream=s_decode,
    )
    return _BatchHandles(slab=slab, decode_event=decode_event,
                         io_args=io_args)
```

And `_CodecChain.decode_into` is the hot loop:

```python
def decode_into(self, slab, offsets, sizes, out_cupy,
                projections, stream) -> Event:
    with cp.cuda.ExternalStream(int(stream.handle)):
        # (a) batched nvcomp.decode of the whole microbatch in one call.
        slab_cp = slab.as_array_like()    # cupy view, no copy
        nv_inputs = [nvcomp.as_array(slab_cp[off:off + sz])
                     for off, sz in zip(offsets, sizes)]
        decoded = [cp.empty(self._chunk_bytes(proj), dtype=cp.uint8)
                   for proj in projections]
        self._compressor.decode(nv_inputs, out=decoded)

        # (b) typed view + reverse filters.
        typed = [self._apply_filters_reverse(d, proj) for d, proj in
                 zip(decoded, projections)]

        # (c) scatter into output. cupy slice-assign respects the
        # current-stream context (set above by ExternalStream).
        for chunk, proj in zip(typed, projections):
            out_cupy[proj.out_selection] = chunk[proj.chunk_selection]

    done = Event()
    done.record(stream)
    return done
```

`_BatchHandles` wraps the slab + decode_event + per-chunk io_args.
Its `wait_decode_complete()` calls `decode_event.sync()` then
validates each `io_args.bytes_done` matches the expected size; any
short read raises `czarr.CudaReadError`.

Two subtleties:

- The `cp.cuda.ExternalStream(int(stream.handle))` context is what
  binds cupy slice-assign to our chosen stream. Without it, cupy
  uses the default per-thread stream and we lose all overlap.
- The slab's lifetime is held by `_BatchHandles.slab`. Once
  `wait_decode_complete()` returns, the slab is dropped; the
  `CzarrGpuBuffer` deleter returns the VMR allocation to the pool.
  No explicit "recycle slab" call.

---

## 15. Summary

A `CudaZarrArray(zarr.Array)` subclass that wraps a
`_CudaArrayImpl` orchestrator. The impl re-uses zarr's
`BasicIndexer` for the chunk-projection math, and goes around
`BatchedCodecPipeline.read_batch` to issue cuFile reads + nvCOMP
decodes with **per-microbatch overlap** on a small `StreamPool`.
Output bytes land directly in the user's `cp.ndarray` — no
host roundtrip, no per-chunk NDBuffer allocation, no serial
read-then-decode barrier.

The class subclasses `zarr.Array` for compatibility (existing
metadata, attrs, group integration, write semantics, and a clean
escape hatch for advanced indexing). A `.lazy` view exposes
`__cuda_array_interface__` + `__dlpack__` for zero-copy handoff
to downstream cupy/torch/cuTile pipelines.

`copy_from` skips decode/encode when both arrays share a codec
chain and the ranges align — useful for rechunk and coarsen
workflows.

Phase 1 ships the skeleton (no overlap), phase 2 adds the
microbatch overlap (depends on agent 3's CzarrPipeline parallel
work), phase 3 integrates the register-once cuFile pattern from
the buffer epic.
