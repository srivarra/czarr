# CzarrGpuBuffer — cuda.core.Buffer-backed zarr Buffer

Replace the cupy-ndarray-backed `zarr.core.buffer.gpu.Buffer` with a buffer wrapping `cuda.core.Buffer` so we get explicit memory-class semantics, 4 KiB alignment, stream-ordered lifetime, and a clean handoff path to cuFile direct I/O.

## Motivation

Today's GPU buffer is `zarr.core.buffer.gpu.Buffer(cupy_ndarray)`. That delivers `__cuda_array_interface__` but:

1. **No alignment guarantee** beyond cupy's allocator default (256 bytes). cuFile direct I/O wants 4 KiB-aligned device pointers; we currently get them by luck.
2. **No memory-class introspection** — `is_device_accessible` / `is_host_accessible` flags aren't first-class; codecs branch on `isinstance(chunk, gpu_buffer.Buffer)` which is fragile.
3. **Lifetime is global RMM-pool-bound** — no per-stream `allocate(size, stream=s)` semantics. False dependencies between in-flight chunks.
4. **Awkward bridge to cuFile** — current path goes `cupy.ndarray.data.ptr` (`int`), works but the buffer object itself can't be passed to a cuFile call directly.

`cuda.core.Buffer` has all four properties native: `__dlpack__` for cupy interop, `is_device_accessible` / `is_host_accessible`, stream-ordered `MemoryResource.allocate(size, stream=...)`, and 4 KiB-aligned pointers from `DeviceMemoryResource`.

## Locked decisions

1. **New class** `czarr.core.buffer.CzarrGpuBuffer` — subclass of `zarr.core.buffer.core.Buffer` (the abstract), implementing the same protocol so zarr-pipeline code that handles `zarr.gpu.Buffer` also handles ours.
2. **Backing store**: `cuda.core.Buffer` via either `DeviceMemoryResource` (default) or `PinnedMemoryResource` (host-staging path).
3. **`__cuda_array_interface__` exposure**: synthesise CAI on the wrapper so existing nvCOMP / cupy interop keeps working zero-copy.
4. **Allocator hookup**: `CzarrGpuBuffer.create(size, *, stream)` uses the active `czarr.pipeline.DeviceBufferPool` substrate.
5. **Backwards compatibility**: keep `zarr.core.buffer.gpu.Buffer` workable too. Codecs accept either via duck-typing on CAI.

## Architecture

```
czarr.core.buffer
    │
    ├── CzarrGpuBuffer(zarr.core.buffer.core.Buffer)
    │   │   wraps cuda.core.Buffer (device or pinned)
    │   │
    │   ├── .as_array_like()           -> cupy.ndarray via DLPack (zero copy)
    │   ├── .__cuda_array_interface__  -> synth from cuda.core.Buffer.handle + size
    │   ├── .to_bytes()                -> host copy via cuda.core.Buffer.copy_to(...)
    │   ├── .as_numpy_array()          -> same
    │   ├── .combine(others)           -> stream-ordered concat (allocate new big buffer)
    │   └── .from_array_like(arr)      -> dlpack import wrapping arbitrary CAI
    │
    └── CzarrGpuNDBuffer(zarr.core.buffer.core.NDBuffer)
            Same idea for the n-dimensional variant.

czarr.core.buffer.prototype
    BufferPrototype(buffer=CzarrGpuBuffer, nd_buffer=CzarrGpuNDBuffer)
```

cuFile integration:

```
storage.cufile_runtime
    │   today: takes int device pointer + size
    │
    ├── read_into_buffer(path, buf: CzarrGpuBuffer, offset=0)
    │       cuFile reads directly into buf's aligned 4 KiB-page-boundary pointer
    │       no extra register-buffer step (cuFile auto-registers aligned pages
    │       on first use within a process)
    │
    └── write_from_buffer(path, buf: CzarrGpuBuffer)
```

GPULocalStore swap:

```
GPULocalStore.get(key, prototype) -> Buffer
    if prototype.buffer is CzarrGpuBuffer:
        buf = CzarrGpuBuffer.empty(size, stream=current_stream)
        cufile_runtime.read_into_buffer(path, buf, offset=byte_range.start)
        return buf
    else:
        # existing cupy-backed path
        ...
```

## Phases

### Phase 0 — survey + spike

- Probe `cuda.core.Buffer` actual fields/methods, dlpack interop with cupy.
- Verify 4 KiB alignment empirically across sizes (1 KiB up to 128 MiB).
- Compare alignment behaviour: cuda.core `DeviceMemoryResource` vs cupy default vs RMM pool.
- Document in `docs/planning/buffer-spike.md`.

Deliverable: a 30-line spike script + decision: is the alignment / stream-ordered behaviour worth the wrap?

### Phase 1 — `CzarrGpuBuffer` skeleton

- New module `czarr.core.buffer` with `CzarrGpuBuffer` + `CzarrGpuNDBuffer`.
- Implement the `zarr.core.buffer.core.Buffer` ABC: `__init__`, `as_array_like`, `to_bytes`, `from_array_like`, `combine`, etc.
- Register the prototype via `zarr.registry.register_buffer` / `register_ndbuffer` with a unique qualname.
- Unit tests against the protocol surface: round-trip via DLPack, byte equality, combine + slice.

Deliverable: buffer class that passes the zarr buffer ABC contract.

### Phase 2 — codec wiring

- `CudaBytesBytesCodec._batch_sync` adds a third branch:
  - GPU-via-CzarrGpuBuffer  → nvcomp.as_array on the buffer's synthesised CAI (no host trip).
  - GPU-via-zarr-gpu.Buffer → existing path.
  - Host buffer             → host bytes → .cuda().
- Codec output side: when prototype is CzarrGpuBuffer, allocate via `CzarrGpuBuffer.empty(size, stream=current_stream)` from `DeviceBufferPool`.
- Filter codecs (Bitshuffle, Shuffle, etc.) need the same plumbing.

Deliverable: codecs decode end-to-end through CzarrGpuBuffer with no host round-trips.

### Phase 3 — GPULocalStore cuFile direct I/O

- Add `cufile_runtime.read_into_buffer(path, buf, offset)` accepting a `CzarrGpuBuffer`.
- `GPULocalStore.get` and `set` branch on buffer prototype; when CzarrGpuBuffer is in use, cuFile uses the aligned device pointer directly.
- Skip the registered-handle dance per call — register the buffer once during allocation, keep the registration across reads.

Deliverable: cuFile direct I/O without per-call register/deregister overhead.

### Phase 4 — `configure_gpu` integration

- New kwarg `buffer_backend: Literal["cupy", "cuda-core"] = "cuda-core"`.
- When `"cuda-core"`, set `zarr.config.set("buffer", "czarr.core.buffer.CzarrGpuBuffer")` + ndbuffer counterpart.
- Document the switch + the trade-offs in the docstring.

Deliverable: opt-in via `configure_gpu(buffer_backend="cuda-core")`.

### Phase 5 — bench

- Extend `bench/zarr/slice_compare.py` with a third path: czarr with `buffer_backend="cuda-core"`.
- Measure on A40 / H100 / H200.
- Expected wins: H100/H200 GDS path where aligned-pointer + register-once savings show up.

Deliverable: numbers in `docs/planning/buffer-results.md`. If wins are >5%, make `"cuda-core"` the default in a follow-up.

## Out of scope

- `ManagedBuffer` integration — addressed and ruled out separately (managed memory wrong tool for hot path).
- Multi-GPU buffer sharing (nvshmem4py). Single-GPU only.
- Replacing nvCOMP's internal scratch allocator (RMM stays).

## Risks

- DLPack round-trip subtleties: stream-ordering between cuda.core stream and cupy's default stream needs careful handling (use `cp.from_dlpack(buf, stream=...)`).
- Some zarr-internal code path may bypass `as_array_like` and assume the underlying object IS a cupy array. Catch in Phase 1 tests; widen `_data` type or add adapter as needed.
- Register-once cuFile path means handle leaks if buffer is freed without deregister. Wrap in RAII (`__del__`) carefully.
