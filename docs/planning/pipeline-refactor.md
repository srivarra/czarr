# czarr pipeline refactor

Total refactor: czarr becomes a GPU-native zarr v3 pipeline that handles compressors, filters, sharding, and checksums in one orchestrated path. Drop zarr v2 support entirely. Drop the `numcodecs.registry` adapter. Restructure files by codec ABC kind. Add a `CzarrPipeline` (CodecPipeline subclass) that owns staging, streams, batched decompress, and gather.

## Locked design decisions

1. **Two base classes** — mirror zarr-python's split: one base for `BytesBytesCodec` compressors, another for `ArrayArrayCodec` filters. No artificial unification.
2. **Pipeline registration** — support both global (via `zarr.config.set({"codec_pipeline.path": ...})` inside `configure_gpu()`) and per-array opt-in (`zarr.create_array(..., codec_pipeline=CzarrPipeline)`).
3. **StreamPool size** — default 4, configurable via `configure_gpu(stream_pool_size=N)`.
4. **PinnedHostPool** — pre-allocate a default slab at `configure_gpu()` time + grow on demand if exhausted.
5. **Sharding** — `ShardingCodec` uses a *sub-pipeline* instance (slimmer than the outer one; no further sharding nesting expected).

## Architecture

```
zarr v3 store
   ↓
CzarrPipeline (registered globally or per-array)
   │
   ├── PHASE 1: stage compressed bytes
   │     PinnedHostPool ──► one big pinned buffer per selection
   │     StreamPool      ──► N=4 cuda.core.Stream (configurable)
   │     DeviceBufferPool── RMM-backed slabs
   │
   ├── PHASE 2: shard-aware dispatch
   │     parse shard index (CPU)
   │     enqueue inner-chunks via sub-pipeline
   │
   ├── PHASE 3: BytesBytesCodec chain (decompress)
   │     batched nvCOMP across all inner-chunks, on stream pool
   │
   ├── PHASE 4: ArrayBytesCodec (dtype/endian view)
   │
   ├── PHASE 5: ArrayArrayCodec chain (filter inverse, reversed)
   │     bitshuffle / shuffle / delta / fixedscaleoffset / bitround
   │
   └── PHASE 6: gather into selection output
```

## Target file layout

```
src/czarr/
├── __init__.py
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py          # CzarrPipeline
│   ├── streams.py           # StreamPool
│   ├── pinned.py            # PinnedHostPool
│   ├── device.py            # DeviceBufferPool
│   └── sharding.py          # ShardingCodec
├── codecs/
│   ├── __init__.py
│   ├── base.py
│   ├── compressors/         # BytesBytesCodec
│   │   ├── blosc.py
│   │   ├── zstd.py
│   │   ├── lz4.py
│   │   ├── gzip.py
│   │   ├── zlib.py
│   │   └── native.py
│   ├── filters/             # ArrayArrayCodec
│   │   ├── bitshuffle.py
│   │   ├── shuffle.py
│   │   ├── delta.py
│   │   ├── fixedscaleoffset.py
│   │   └── bitround.py
│   ├── checksum/
│   │   └── crc32c.py
│   └── _blosc1.py           # Blosc1 container parser
├── kernels/
│   ├── __init__.py
│   ├── bitshuffle.py
│   ├── byteshuffle.py
│   └── crc32c.py
├── storage/
├── alloc.py
└── _buffer.py
```

## What gets deleted

- `czarr.codecs.compat.BloscNumcodec` (numcodecs adapter)
- `numcodecs.registry.register_codec(BloscNumcodec)` block in `configure_gpu()`
- iohub-based path in `bench/talon_slice.py` (rewrite around plain zarr v3)
- Existing `codecs/compat.py` and `codecs/native.py` (split into new tree)

## Phases

### Phase 0 — File restructure + v2 removal

Mechanical refactor. No behavior changes besides v2 drop.

- Move existing 10 codec classes into `codecs/compressors/{blosc,zstd,lz4,gzip,zlib,native}.py`
- Move `_blosc_format.py` → split into `codecs/_blosc1.py` (container parser) + `kernels/bitshuffle.py` + `kernels/byteshuffle.py`
- Delete `BloscNumcodec` + numcodecs registration
- Rewrite `bench/talon_slice.py` to use plain zarr v3 (skip iohub which is v2)
- Update tests for new module paths
- Verify all existing tests still pass post-restructure

Deliverable: new file tree compiles, tests green, no v2 surface.

### Phase 1 — cuda.core substrate

- `czarr.pipeline.streams.StreamPool` — N `cuda.core.Stream`s, round-robin `acquire()/release()` API, default N=4
- `czarr.pipeline.pinned.PinnedHostPool` — pre-allocated slab via `cuda.core.PinnedMemory`, free-list reuse, grow-on-demand fallback
- `czarr.pipeline.device.DeviceBufferPool` — thin wrapper around RMM, segment-based allocation aligned to nvCOMP scratch requirements
- Migrate `alloc.py` and `_buffer.py` to use cuda.core types where it's cleaner
- Add unit tests for each primitive

Deliverable: reusable substrate independent of any specific codec.

### Phase 2 — CzarrPipeline (no sharding)

- `czarr.pipeline.pipeline.CzarrPipeline` subclasses `zarr.abc.codec.CodecPipeline`
- Implements `read`, `write`, `decode`, `encode`, `decode_partial_batch`, `encode_partial_batch`
- For a selection across N chunks: stage all N compressed buffers in one pinned-host alloc, schedule per-chunk decompress across the stream pool, batched nvCOMP call where the codec supports it
- Wire global registration via `configure_gpu()` + per-array opt-in argument
- Update existing per-codec `_batch_sync` to delegate to the pipeline rather than re-do staging

Deliverable: all existing compressor codecs run through the new pipeline; speedup over the current per-call serial flow.

### Phase 3 — ArrayArrayCodec filters

- `czarr.codecs.filters.bitshuffle.Bitshuffle` — wraps `kernels/bitshuffle.py`
- `czarr.codecs.filters.shuffle.Shuffle` — wraps `kernels/byteshuffle.py`
- `czarr.codecs.filters.delta.Delta` — cumsum/diff via cupy
- `czarr.codecs.filters.fixedscaleoffset.FixedScaleOffset` — elementwise mul/add
- `czarr.codecs.filters.bitround.BitRound` — bit-mask kernel
- All register via zarr v3 entry-points so v3 stores using these filters decode on the GPU
- Add forward (encode) kernels where they don't exist yet (e.g., bitshuffle forward)

Deliverable: any zarr v3 store with these filter chains decodes fully on GPU.

### Phase 4 — Sharding

- `czarr.pipeline.sharding.ShardingCodec` — `ArrayBytesCodec` that holds an inner `CzarrPipeline` instance
- Parse shard index from last bytes of shard
- For each requested inner-chunk in shard: byte-range extract, feed through inner pipeline
- Inner pipeline configured slimmer (smaller stream pool, no further sharding hooks)

Deliverable: OME-NGFF v3 sharded stores read correctly on GPU.

### Phase 5 — Crc32c checksum

- `czarr.codecs.checksum.crc32c.Crc32c` BytesBytesCodec
- Use nvCOMP's built-in crc32c or a hand kernel
- Trailing 4 bytes appended on encode, validated + stripped on decode

Deliverable: v3 stores using crc32c codec validated on GPU.

### Phase 6 — Benchmark suite

- Rewrite bench/ to use only zarr v3 (no iohub dependency)
- Benchmark Phase 0 baseline → Phase 2 (pipeline) → Phase 3+ (filters) → Phase 4 (sharding)
- Include comparison vs default zarr CPU pipeline + GDS reads where available

Deliverable: documented perf numbers across the migration; identifies remaining bottlenecks.

## Out of scope

- zarr v2 compatibility (explicitly dropped)
- Multi-GPU (nvshmem4py, etc.) — single-GPU pipeline only
- Encode/write path for Blosc (deferred from prior work; revisit after read pipeline is solid)
- iohub OME-zarr v0.4 / v0.5 wrappers — bench/utilities should use plain zarr v3
