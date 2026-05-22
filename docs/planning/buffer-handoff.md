# buffer epic — handoff context

Standalone context for the next session picking up dex epic **`bnwyxani`**
(CzarrGpuBuffer — cuda.core.Buffer-backed zarr Buffer).  This document
captures everything that came out of the broader perf-work session that
preceded this worktree so the next agent doesn't need to re-derive
findings or re-read the bench logs.

## What this epic is

Per `docs/planning/czarr-gpu-buffer.md`: replace the cupy-ndarray-backed
`zarr.core.buffer.gpu.Buffer` with `CzarrGpuBuffer` wrapping
`cuda.core.Buffer`.  Six phases:

1. survey + spike — already part-done; see "Established facts" below.
2. `CzarrGpuBuffer` skeleton (new `czarr.core.buffer` module).
3. codec wiring in `CudaBytesBytesCodec._batch_sync`.
4. `GPULocalStore` cuFile direct I/O using the aligned buffer.
5. `configure_gpu(buffer_backend=…)` integration.
6. Bench (cold-vs-warm of the buffer backend, slice_compare 4th column).

Dex IDs for the phases:
```
pf3es13e  Phase 0 — survey + spike
jc84qa13  Phase 1 — CzarrGpuBuffer skeleton
mn4gkxb3  Phase 2 — codec wiring
yj0senev  Phase 3 — GPULocalStore cuFile direct I/O
226m2a1c  Phase 4 — configure_gpu integration
b5wz83u9  Phase 5 — bench buffer-backend comparison
```

## Why this is the remaining epic

The other two follow-up epics (cross-chunk batching `z3hd9ph7`, kernel
cache `dscxrw7c`) closed in the previous session:

* `z3hd9ph7` — Phase 0 cProfile located the bottleneck (cuFile per-call
  overhead, not codec).  Phase 1 building blocks shipped
  (`cufile_runtime.read_into_many` + `GPULocalStore.get_many`).  Phase 2
  pipeline override regressed 2-4× on H100 (reverted in commit
  `6c7d0fc`).  Phase 3 cuFile batched I/O API capped at 128 IOCBs and
  was 6.5× slower than threaded_sync in earlier probe — scrapped.
* `dscxrw7c` — cuTile, cupy, and CUDA driver all auto-cache by default
  to `~/.cache/cutile-python`, `~/.cupy/kernel_cache`, `~/.nv/ComputeCache`
  respectively.  No wiring needed.  Cold-vs-warm bench blocked by an
  unrelated cuTile/Hopper bug.

The buffer epic is the only one with a real architectural change left.

## Established facts from earlier probing

These were verified in the session before this worktree:

1. **`cuda.core.Buffer.__dlpack__` works**.  `cp.from_dlpack(buf)` yields
   a zero-copy cupy ndarray sharing the same pointer.  Path:
   `cuda.core.DeviceMemoryResource.allocate(size, stream=...)` →
   `Buffer` → DLPack → `cupy.ndarray` → `nvcomp.as_array(...)`.

2. **Alignment is 4 KiB** out of `DeviceMemoryResource.allocate(4096)`.
   Verified by `(buf.handle % 4096) == 0`.  This is exactly what
   cuFile direct I/O requires (no internal staging buffer).  cupy's
   default allocator gives 256-byte alignment.

3. **No `__cuda_array_interface__` on cuda.core.Buffer** — only DLPack.
   We'd need to synthesise CAI on our wrapper class from `handle + size`
   so nvCOMP / zarr code that branches on CAI keeps working.

4. **`is_device_accessible` / `is_host_accessible`** flags are
   first-class properties on cuda.core.Buffer.  cupy hides this.

5. **PinnedMemoryResource** is stream-ordered (needs a stream at
   `.allocate`).  `LegacyPinnedMemoryResource` accepts `stream=None` —
   that's what Phase 1's `PinnedHostPool` uses.

6. **cuFile handle register/deregister cost** dominates per-chunk
   reads on A40 (compat mode, ~3 ms / call).  On H100 with real GDS
   it's ~1 ms / call but zarr's `concurrent_map` already runs the
   work 32-way parallel; the buffer epic's register-once-at-allocation
   pattern is the right architectural fix because it eliminates the
   register/deregister cost from the per-read path entirely.

## Existing infrastructure to lean on

`czarr.pipeline` already provides:

* `StreamPool` — `cuda.core.Stream` × N (default 4).
* `PinnedHostPool` — `LegacyPinnedMemoryResource`-backed pool of
  pinned host buffers, exact-size bucketed.
* `DeviceBufferPool` — thin RMM wrapper today; the buffer epic likely
  swaps this to `DeviceMemoryResource` + `CzarrGpuBuffer.empty(size,
  stream=)` semantics.

`czarr.storage.GPULocalStore` already provides:

* `get(key, prototype, byte_range)` — single-key cuFile read.
* `get_many(keys, prototype)` — batched fetch.  Inner mechanism is
  serial open + register, parallel reads.  Phase 3 of the buffer epic
  will rewrite the per-call cuFile call site to use registered handles
  on the new buffer.

`czarr.codecs.base.CudaBytesBytesCodec._batch_sync` is where the
codec wiring lands.  Currently has two branches (gpu.Buffer fast-path,
host buffer slow-path).  Phase 2 of the buffer epic adds a third
branch for `CzarrGpuBuffer`.

## Open known issues (not in this epic's scope, just useful context)

* **`og6bzlnb`** — cuTile / `tileiras 13.2.78` rejects `sm_90` (H100).
  Affects `czarr.Shuffle` only.  Workaround: write a `cupy.RawKernel`
  byteshuffle for Hopper, or pin newer `nvidia-cuda-tileiras`.

* **`test_alloc.py` segfault** on process exit when combined with the
  full suite.  RMM pool reinit races with cuda.core / nvCOMP teardown.
  Workaround: exclude `test_alloc` from full suite runs; tracked in
  conftest as a known limitation.

## Important constraints

* **No zarr v2 surface** — entire pipeline targets v3 only.
* **Two codec ABCs** — mirror zarr-python's `BytesBytesCodec` /
  `ArrayArrayCodec` split.
* **Bruno HPC** has H100 + H200 nodes with real nvidia_fs; A40 is
  interactive partition + compat-mode cuFile only (no GDS for tests).
* **CUDA 13.1 on cluster** but **CuPy is cu12** — kernel JIT errors
  happen for any cupy operation that compiles against CUDA 13 headers.
  Workaround in tests / benches: avoid cupy JIT in hot paths
  (`cp.array_equal` triggers it; use `cp.asnumpy` + `np.array_equal`).

## Headline benchmark numbers (baseline for comparison)

From `bench/zarr/slice_compare.py` — read a 1 GiB Z-slab `(16, 4096,
4096)` of float32 zstd into GPU memory:

```
| GPU         | czarr (GPU)   | zarr+h2d (CPU) | speedup |
|-------------|---------------|----------------|---------|
| A40 (no GDS)|  4.43 GiB/s   |  1.40 GiB/s    | 3.15x   |
| H100 + GDS  | 11.18 GiB/s   |  1.74 GiB/s    | 6.41x   |
| H200 + GDS  | 11.88 GiB/s   |  1.95 GiB/s    | 6.11x   |
```

The buffer epic's Phase 5 bench should beat these on H100 / H200 by
**eliminating per-call cuFile register/deregister** (register-once at
allocation), and by **guaranteeing 4 KiB alignment** so cuFile takes
the direct path instead of going through its internal staging buffer.

## Suggested start

1. `dex start pf3es13e` — Phase 0 spike.  Already partly done in the
   prior session (see "Established facts" above).  Write a small
   `bench/buffer/alignment_probe.py` that captures the verified facts
   as a reproducible script.  Then mark complete.
2. `dex start jc84qa13` — Phase 1 skeleton.  Real code starts here.
   New file `src/czarr/core/buffer.py` with `CzarrGpuBuffer` +
   `CzarrGpuNDBuffer`.  Match the protocol surface zarr's
   `gpu.Buffer` exposes.

## Commits that landed in this epic's preamble (main branch)

```
d61c922  fix(filters): unpack v3 metadata wrapper in from_dict
8cbca5b  bench(kernels): shell-driven cold-vs-warm + tracked Hopper cuTile bug
c5cc001  bench(storage): probe cuFile batch_io_submit for small-chunk regime
6c7d0fc  revert: CzarrPipeline.read_batch override (epic z3hd9ph7 Phase 2)
e6b66d6  feat(storage): batched cuFile reads + pipeline override (epic z3hd9ph7 phases 0-2)
652b053  docs(planning): three follow-up epics — kernel cache, GPU buffer, cross-chunk batching
8a0fada  bench: cross-GPU + czarr-vs-CPU+h2d measurements on Bruno H100/H200
abf22d2  feat(bench): pipeline_compare + phase6 perf write-up (Phase 6)
```

Resume from `d61c922`.
