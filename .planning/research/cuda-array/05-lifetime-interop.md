# 05 — Lifetime + interop reference for GPU memory primitives

Audience: the engineer building a CUDA-native Zarr Array on top of cupy +
`cuda.core.Buffer` + RMM + DLPack + cuFile.

This document is organised as a per-topic reference. Each rule carries a
confidence tag:

* **(doc)** — documented in upstream docstrings / spec / CUDA programming
  guide.
* **(src)** — inferred from reading the cuda.core / cupy / RMM source in
  this venv.
* **(verified)** — experimentally verified by czarr team (cite probe + commit
  where possible).
* **(speculation)** — best guess, needs probing if it becomes load-bearing.

References to source files use absolute paths so they can be `Read` directly.

Key upstream URLs cited inline:

* cuda.core memory: <https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#memory>
* CUDA VMM API: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-memory-management>
* CUDA Stream-ordered memory allocator:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator>
* DLPack v1 spec:
  <https://dmlc.github.io/dlpack/latest/python_spec.html>
* numba CAI v3 spec:
  <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>
* cuFile API:
  <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html>
* RMM: <https://docs.rapids.ai/api/rmm/stable/>

---

## 1. `cuda.core` MemoryResource hierarchy

The eight kinds of `MemoryResource` in this venv (cu13 build), and when to
reach for each.

Source headers (all extern types are Cython, but the `.pxd` declarations and
the legacy/VMR Python files make the surface readable):

* `/hpc/mydata/sricharan.varra/Dev/czarr/.venv/lib/python3.13/site-packages/cuda/core/__init__.py` — public exports
* `/hpc/mydata/sricharan.varra/Dev/czarr/.venv/lib/python3.13/site-packages/cuda/core/cu13/_memory/__init__.py` — re-exports
* `/hpc/mydata/sricharan.varra/Dev/czarr/.venv/lib/python3.13/site-packages/cuda/core/cu13/_memory/_legacy.py` — `LegacyPinnedMemoryResource`
* `/hpc/mydata/sricharan.varra/Dev/czarr/.venv/lib/python3.13/site-packages/cuda/core/cu13/_memory/_virtual_memory_resource.py` — `VirtualMemoryResource`
* `/hpc/mydata/sricharan.varra/Dev/czarr/.venv/lib/python3.13/site-packages/cuda/core/cu13/_memory/_memory_pool.pxd` — `_MemPool` (shared base for DMR/PMR/MMR/GMR)

### 1.1 `DeviceMemoryResource` (DMR) — stream-ordered pool

* Backed by a CUDA memory pool handle (`CUmemoryPool`). The class is a
  thin subclass of `_MemPool`. **(src)**
* `allocate(size, *, stream=...)` calls `cuMemAllocFromPoolAsync` —
  inherits the stream-ordered allocator semantics. **(src)**
* Stream is **required** in the conceptual sense — every device-pool
  allocation has an associated stream. In `cuda.core` the keyword is
  technically optional (defaults to the device's default stream), but if you
  intend the allocation to be visible from a non-default stream you must pass
  it. **(doc — CUDA stream-ordered allocator)**
* **Alignment**: only the first allocation from a freshly-grown pool extent
  is page-aligned; sub-allocations are packed to the natural alignment of
  the request (256 B / 512 B typical for small sizes). czarr verified
  this in `bench/buffer/alignment_probe.py`. **(verified — see buffer-spike.md
  lines 9-18)**
* **Pool growth granularity** is driver-controlled (typically 2 MiB
  extents on Hopper/Ampere). **(speculation — derived from VMR granularity
  parity)**
* **When to use**: general-purpose device allocations where you trust the
  pool's recycling and don't need cuFile-grade alignment guarantees. nvCOMP
  scratch + decode outputs go here through cupy's allocator. The pool
  amortises `cudaMallocAsync` to microseconds after warmup.

### 1.2 `VirtualMemoryResource` (VMR) — CUDA VMM, no pool

* Backed by `cuMemCreate` + `cuMemAddressReserve` + `cuMemMap` per
  allocation. **(src — `_virtual_memory_resource.py:530-560`)**
* **Alignment**: `addr_align` is honoured per-allocation. VA reservation
  rounds up to `granularity` (queried via `cuMemGetAllocationGranularity`),
  but the returned pointer is guaranteed to be `addr_align`-aligned. **(src,
  verified in buffer-spike lines 19-28)**
* **`gpu_direct_rdma=True`** sets `prop.allocFlags.gpuDirectRDMACapable`
  on the underlying physical allocation. This is the marker the CUDA driver
  uses to permit GDR / cuFile direct-DMA into this region. **(src, doc — CUDA
  VMM)**
* **No pool, no recycling**: every `allocate()` is a fresh `cuMemCreate`.
  Cost is non-trivial (~tens of µs for the create+reserve+map cycle on
  H200). **(speculation, needs probe; cuda.core has no built-in
  per-resource benchmark)**
* **Granularity floor**: default `granularity=RECOMMENDED` typically gives
  2 MiB on Hopper/Ampere. So a 4 KiB request consumes 2 MiB of VA space.
  Physical memory is also rounded up. **(verified — buffer-spike.md
  decision section, lines 73-75)**
* **Growable**: VMR has `modify_allocation(buf, new_size)` that extends a
  buffer in-place (preserving the base pointer if the next VA range is
  free) using a fast-path / slow-path remap. This is the **only**
  MemoryResource in cuda.core with grow-in-place support. **(src —
  `_virtual_memory_resource.py:196-280`)**
* **Multi-GPU peer access**: `peers=[...]` grants other devices read/write
  access via `cuMemSetAccess`. **(src)**
* **When to use**: cuFile-bound buffers, peer-accessed buffers, and large
  long-lived buffers you'd like to grow in place. Avoid for swarms of
  small short-lived allocations — the 2 MiB granularity wastes VA fast.

### 1.3 `LegacyPinnedMemoryResource` — synchronous `cudaMallocHost`

* `allocate(size)` calls `cuMemAllocHost`. `deallocate(...)` calls
  `cuMemFreeHost`. Both are synchronous. **(src — `_legacy.py:55-82`)**
* The `stream=` keyword is **accepted but ignored** on `allocate`. On
  `deallocate`, a non-None `stream` triggers a `stream.sync()` first —
  to ensure in-flight DMAs into this host buffer complete before the host
  free races them. **(src — `_legacy.py:62-82`)**
* **`is_device_accessible=True, is_host_accessible=True`** — pinned
  memory is mapped into the device PT, so it's reachable as a UVA
  pointer. **(src — `_legacy.py:84-92`)**
* **No `device_id`** — pinned host memory is portable across devices
  by default. **(src — `_legacy.py:94-97` raises)**
* **When to use**: pre-allocated pools of pinned host buffers, like
  czarr's `PinnedHostPool` (`src/czarr/pipeline/pinned.py`). The
  synchronous `cudaMallocHost` cost (~ms per call) is paid once at pool
  build-time, then `acquire()` is a list pop.

### 1.4 `PinnedMemoryResource` (PMR) — stream-ordered pinned

* Subclass of `_MemPool` — there's a NUMA-aware pool of pinned host
  memory under the hood. Has a `_numa_id`. **(src —
  `_pinned_memory_resource.pxd:9-10`)**
* This is the stream-ordered counterpart to LegacyPinnedMemoryResource.
  Allocations are pooled and can be freed back to the pool on a stream
  using the same stream-ordered free machinery. **(src — `_MemPool`
  shared base; full code is in compiled .so)**
* **NUMA**: lets you bind the host pool to a specific NUMA node. Matters
  on dual-socket systems where the GPU has a preferred root-complex.
  **(doc — CUDA docs on `cudaHostAllocPortable`)**
* **When to use**: workloads with frequent small pinned-host allocations
  where the LegacyPinned cost matters, or where NUMA placement is
  important. The czarr team's `PinnedHostPool` could be migrated to PMR
  if profiling shows `cudaMallocHost` dominating — but for now the
  exact-size-bucket free list is good enough.

### 1.5 `ManagedMemoryResource` (MMR) — unified memory

* Subclass of `_MemPool`. Backed by `cuMemAllocManaged` (or the pool
  variant). Has a preferred-location attribute (`_pref_loc_type`,
  `_pref_loc_id`). **(src — `_managed_memory_resource.pxd:8-11`)**
* **`is_device_accessible=True, is_host_accessible=True`** — accessible
  from both, with the driver migrating pages on demand. **(doc)**
* **Performance hit**: any first-touch page fault from the device causes
  a host→device page migration (4 KiB or 2 MiB page). Bandwidth is
  PCIe-limited (~25 GiB/s on PCIe Gen4 x16) and latency is high vs
  warm pages. **(doc — CUDA unified memory chapter; typical numbers
  from NVIDIA dev blog)**
* **When to use**: prototyping, not production. For czarr the chunk
  decode path needs deterministic bandwidth; managed memory adds
  unpredictable migration stalls. Avoid.

### 1.6 `GraphMemoryResource` (GMR) — CUDA-graph-scoped

* Lives in `cuda.core._memory._graph_memory_resource`. **(src —
  `_graph_memory_resource.pxd`)**
* Allocations made through GMR are bound to a **CUDA Graph**'s lifetime.
  When the graph is destroyed, the memory is freed. **(doc — CUDA graphs
  programming guide, "Memory allocation in graphs")**
* The CUDA graph allocation nodes are `cudaGraphAddMemAllocNode` /
  `cudaGraphAddMemFreeNode`. cuda.core exposes these via
  `cuda.core.graph` (see
  `/hpc/mydata/sricharan.varra/Dev/czarr/.venv/lib/python3.13/site-packages/cuda/core/cu13/graph/__init__.py`).
  **(src)**
* Stream-ordering: the allocation is ordered with respect to the stream
  the graph is launched on. Freed memory is reusable by subsequent
  allocations in the same graph. **(doc)**
* **When to use**: only if you adopt CUDA Graphs end-to-end for the
  decode pipeline. For now czarr does not — the codec call structure
  (nvCOMP scratch with internal allocations) is not capture-clean. See
  §9.

### 1.7 `WorkqueueResource` (WQR)

* Lives in `cuda.core._device_resources`. **(src —
  `_device_resources.pxd:26-37`)**
* It is **not a memory resource** — naming is misleading. It represents
  a CUDA device sub-resource (a workqueue/SM partition handle of type
  `CUdevResource`). Used together with `DeviceResources` (the device's
  full resource set) and `SMResource` to split a GPU into Green Contexts
  with reserved SMs. **(src — class body in
  `_device_resources.pxd:9-23`)**
* This is the CUDA 12.5+ Green Context API. Out of scope for czarr
  unless we ever multi-tenant a single GPU between codec workers.
  **(doc — <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#green-contexts>)**

### 1.8 `SMResource` — Green Context SM partition

* Same module as WQR. Holds a `CUdevResource` plus `_sm_count`,
  `_min_partition_size`, `_coscheduled_alignment`. **(src —
  `_device_resources.pxd:9-23`)**
* Lets you split a device's SMs into disjoint groups and create a Context
  (green context) bound to a subset. **(doc — Green Contexts)**
* **Not useful for czarr**: nvCOMP runs best when given the full device.
  Could become interesting if we ever want to overlap codec work with
  user kernels on a quota — but no concrete plan today.

### 1.9 Picking a resource for czarr's `CzarrGpuBuffer`

| use case | resource | rationale |
|---|---|---|
| cuFile-bound device buffer | `VirtualMemoryResource(addr_align=4096, gpu_direct_rdma=True)` | guaranteed 4 KiB alignment + GDR-capable physical pages |
| nvCOMP scratch / decode output | cupy default allocator → RMM (if enabled) | pool recycling is the main win; alignment doesn't matter for kernel-internal access |
| pinned-host staging | `LegacyPinnedMemoryResource` pre-allocated into `PinnedHostPool` | one-shot cost paid up front |
| growable codec output (encode size unknown) | `VirtualMemoryResource.modify_allocation` | only resource that grows in place |
| anything where you want IPC | DMR or PMR (both support IPC handles via `_MemPool._ipc_data`) | VMR can also IPC but the handle export ceremony is heavier |

---

## 2. Stream-ordered allocation rules

The CUDA stream-ordered memory allocator is the model behind `DeviceMemoryResource`,
`PinnedMemoryResource`, `ManagedMemoryResource`, and `GraphMemoryResource`.
`LegacyPinnedMemoryResource` and `VirtualMemoryResource` are NOT stream-ordered.

Programming-guide reference:
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator>.

### 2.1 When is `stream=` required vs optional?

* **Required (in the soft sense — wrong stream = correctness bug)**:
  `DeviceMemoryResource.allocate(size, stream=s)`, when `s` is the stream
  that the **first kernel using the memory** will run on. **(doc)**
  - The driver may return memory that was just freed on a different
    stream by inserting an automatic wait. If you allocate on the
    default stream but actually use the buffer on a non-default stream,
    the wait is on the wrong stream and you can race a not-yet-finished
    deallocation. **(doc)**
* **Optional / accepted-for-API-conformance**: `LegacyPinnedMemoryResource`,
  `VirtualMemoryResource` — both accept `stream=` but use it only for
  validation. VMR's `cuMemCreate` is synchronous and not stream-ordered.
  LegacyPinned's `cuMemAllocHost` is synchronous. **(src — _legacy.py:31-60,
  _virtual_memory_resource.py:477-509)**
* **Cross-resource gotcha**: `czarr.pipeline.pinned.PinnedHostPool` passes
  `stream=None` to `LegacyPinnedMemoryResource.allocate` on purpose —
  pinned-host alloc is not stream-ordered. **(verified —
  `src/czarr/pipeline/pinned.py:62`)**

### 2.2 What happens if you allocate on stream A and free on stream B?

* For stream-ordered pools, this is **legal and supported**. The driver
  inserts a cross-stream synchronisation: the free is ordered with respect
  to a recorded event from the alloc-side stream's progress. **(doc)**
* But: the freed memory only becomes available to *subsequent* allocations
  from the pool after stream B's free completes. If A is busy and B is
  fast, the pool may grow rather than reuse. **(doc, implied)**
* **For VMR/LegacyPinned** (not stream-ordered): `deallocate(ptr, size,
  stream=s)` does `s.sync()` first, then frees. This is a hard host-side
  block. **(src — _legacy.py:62-82, _virtual_memory_resource.py:562-587)**

### 2.3 Buffer close / `__del__` semantics

* `Buffer.close(stream=...)` calls `MemoryResource.deallocate` with the
  given stream. **(src — `_buffer.pxd:18-27`; full impl in compiled .so;
  call shape from `bench/buffer/alignment_probe.py:84`)**
* `Buffer.__del__` (no explicit close) — closes on **no stream**, which
  for stream-ordered resources defaults to the device's default stream;
  the driver still ensures correctness via the per-pool event tracking,
  but you may see a stall if the implicit default-stream sync is
  expensive. **(speculation — needs probe; cuda.core does not document
  the default explicitly)**
* **Recommendation**: always `close(stream=s)` when `s` is the stream
  that last used the buffer. This keeps the free ordered with respect to
  the actual consumer, not just the default stream.
* **In-flight kernels at close**: stream-ordered free does NOT crash if
  kernels are still using the buffer — it queues the free behind them.
  Pageable / VMR / Legacy free, by contrast, will tear down the mapping
  while the GPU is reading and produce a memory fault. **(doc)**

### 2.4 czarr-specific implication

The codec path runs on a stream from `StreamPool.acquire()`
(`src/czarr/pipeline/streams.py`). The decode output buffer is allocated
inside `CudaBytesBytesCodec._batch_sync` via
`cp.empty(...)` (line 302 of `codecs/base.py`). cupy's `cp.empty`:

* Uses the cupy current stream (`cp.cuda.Stream.from_external(stream)`
  context-manages this).
* Backed by RMM if `register_nvcomp_allocator()` + `use_rmm_pool()` ran;
  otherwise cupy's own pool.

When the `CzarrGpuBuffer` epic lands and the codec switches to
`CzarrGpuBuffer.empty(size, stream=acquired_stream)`, the buffer's
`MemoryResource.allocate` will be the place where stream binding becomes
explicit — VMR for cuFile output, DMR for intermediate, both honouring
the passed stream.

---

## 3. DLPack interop

### 3.1 Verified facts

From `docs/planning/buffer-spike.md` and `bench/buffer/alignment_probe.py`:

* `cuda.core.Buffer.__dlpack__` exists. **(verified —
  `device_has_dlpack: True`)**
* `cupy.from_dlpack(buf)` produces a `cupy.ndarray` that shares the same
  device pointer as `buf.handle`. **(verified —
  `dlpack_ptr_matches_handle: True`)**
* The cupy producer holds a reference to the original `Buffer` (via the
  DLPack `manager_ctx` slot), so the producer does not need an explicit
  `close()` after handoff. The buffer dies when both the cupy view and
  any other reference to the `Buffer` are gone. **(doc — DLPack v1 spec
  §"Managed Tensor"; verified inferentially by the lack of segfaults in
  the current czarr test suite)**

### 3.2 Typestr / itemsize gotchas

DLPack uses `(code, bits, lanes)` for dtype. cuda.core synthesises
`(kDLUInt, 8, 1)` for a raw byte buffer (the only dtype `Buffer` knows
about — it has no semantic dtype). **(src — `_dlpack.pxd:35-46`,
`_dlpack.pxd:48-55`)**

cupy reads the DLPack dtype back and uses `numpy.dtype('uint8')` for
that combo. If you want a *typed* view (e.g. `cupy.float32`) you must
`.view(cp.float32)` on the cupy side — there's no "tell DLPack a
different dtype" knob on the Buffer. **(speculation — the producer side
is fixed by cuda.core; you can't override without writing your own
`__dlpack__`)**

* **Implication for `CzarrGpuBuffer`**: the wrapper will need to either
  (a) override `__dlpack__` to encode the user-facing dtype, or
  (b) always return uint8 and let users `.view(dtype)` themselves.
  zarr's GPU `Buffer` does (b) — it's a byte array. The decode output
  is then `cp.asarray(...).view(spec.dtype)`. Recommend matching.

### 3.3 Round-trip back to `cuda.core.Buffer`?

* There is **no** `cuda.core.Buffer.from_dlpack(capsule)` in this venv.
  **(verified — `grep` in `cuda/core/__init__.py:86-99` shows only
  `Buffer` import, no `from_dlpack` symbol)**
* So the flow is **one-way**: `Buffer` → DLPack → cupy. The cupy side
  is now the owner-of-record, and the underlying device pointer is
  observable via `cp.ndarray.data.ptr`.
* **If you need to feed a cupy array back to a cuFile-direct call**, you
  pass `int(cp_arr.data.ptr)` directly to
  `cufile.read(handle, dev_ptr, size, ...)` — no `Buffer` round-trip
  needed. The buffer registration (§5) cares about the pointer + size,
  not about the wrapper type.
* **If the buffer must be re-wrapped in a `CzarrGpuBuffer` after going
  through a cupy step**, the wrapper should hold both `Buffer` (for
  introspection / close) and the cupy view (for kernel passing). Don't
  try to recover `Buffer` from a `cp.ndarray` you didn't allocate.

### 3.4 Lifetime ownership across consumers

DLPack v1 introduces an explicit producer-keepalive contract: the
exported `DLManagedTensorVersioned` carries a `manager_ctx` + `deleter`.
The consumer calls `deleter` when its view dies. **(doc — DLPack v1
spec; src — `_dlpack.pxd:67-71`)**

Per consumer:

| consumer | producer kept alive how |
|---|---|
| `cupy.from_dlpack` | cupy stores the capsule and calls its deleter on cupy view GC. The producer is reffed transitively. **(verified — czarr tests)** |
| `torch.from_dlpack` | torch matches the contract; producer survives until torch tensor dies. **(doc — pytorch docs)** |
| `jax.dlpack.from_dlpack` | same. JAX wraps the deleter and ties it to the JAX array's lifecycle. **(doc — JAX docs)** |
| `numpy.from_dlpack` (host-side DLPack only) | irrelevant; will not accept a CUDA device tensor. **(doc — numpy 1.23+)** |

**The deleter is called exactly once**. Multiple consumers from one
capsule are forbidden — DLPack consumes the capsule (renames it to
`used_dltensor_versioned`) on first import. To share with two consumers,
call `__dlpack__()` twice on the producer. **(doc — DLPack spec;
src — `_dlpack.pxd:19-23` defines the rename)**

### 3.5 Stream synchronisation (DLPack v0 vs v1)

* **DLPack v0**: no stream argument. Consumer assumes the producer has
  synchronised. Producer must `stream.sync()` before exporting.
* **DLPack v1** (the version cuda.core's `_dlpack.pxd` references — see
  `DLPACK_MAJOR_VERSION`): `__dlpack__(stream=<int>)` — the consumer
  passes the stream it intends to use the data on, and the producer
  must enqueue an event on its stream and make the consumer's stream
  wait on it. **(doc — DLPack v1)**
* cuda.core's `Buffer.__dlpack__` signature accepts `stream` (compiled
  .so; not directly readable, but the test pattern in
  `bench/buffer/alignment_probe.py:151` calls it without arg, which is
  valid: stream defaults to "current stream" or "synchronise"). **(src,
  partial)**
* **When to use the stream parameter**: when handing a buffer between
  two streams without a host-side sync. czarr's current code path
  (decode → wrap as zarr Buffer → caller does whatever) is all on one
  stream per chunk, so the default behaviour is correct.

### 3.6 czarr's CAI fallback

`czarr._buffer.nvarray_to_buffer` uses `cp.asarray(nv)` which goes
through nvcomp's `__cuda_array_interface__` (not DLPack). For nvcomp
→ zarr handoff this is fine because nvcomp.Array exposes both. For
the reverse direction `buffer_to_nvarray` calls `nvcomp.as_array(cupy_arr)`
which also uses CAI on the cupy side.

So in practice DLPack is used **only** at the cuda.core.Buffer →
cupy.ndarray boundary; everything past that point uses CAI.

---

## 4. `__cuda_array_interface__` (CAI)

### 4.1 The v3 spec — minimum fields to synthesise

numba's CAI v3 spec (the de-facto standard):
<https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>

A producer must expose a dict with:

| key | required | type | meaning |
|---|---|---|---|
| `shape` | yes | tuple of int | array shape |
| `typestr` | yes | str | numpy-style dtype (e.g. `"<f4"`, `"\|u1"`) |
| `data` | yes | `(int, bool)` | `(device_ptr, read_only)` |
| `version` | yes | int | spec version, 3 |
| `strides` | optional | tuple of int or None | None = C-contiguous |
| `descr` | optional | list | structured dtype description |
| `mask` | optional | object | bitmask for valid elements (None typical) |
| `stream` | optional (v3) | int | producer's stream handle |

For czarr's `CzarrGpuBuffer.__cuda_array_interface__` the minimum payload
is:

```python
{
    "shape": (self.size,),
    "typestr": "|u1",        # raw uint8
    "data": (int(self._buf.handle), False),
    "version": 3,
    "strides": None,         # C-contiguous
    "stream": int(self._stream),   # optional, but recommended for v3 consumers
}
```

**(speculation, doc — derived from numba spec)** — confirm against
cupy's CAI consumer when implementing.

### 4.2 Cost of synthesis

CAI is a Python property → dict construction → consumer parses the
dict. This is microseconds per call. The dict can be built lazily and
cached on the wrapper (czarr's `CzarrGpuBuffer` should memoise after
first access). **(speculation — measure if it shows up)**

### 4.3 Who needs CAI vs DLPack vs both?

| consumer | DLPack | CAI |
|---|---|---|
| `cupy.asarray(obj)` / `cupy.from_dlpack(obj)` | both work | both work |
| `numba.cuda.as_cuda_array(obj)` | yes (v0.55+) | yes (preferred) |
| `nvcomp.as_array(obj)` | no (verified — only CAI / DLPack on the import side; nvcomp does NOT export DLPack on its `nvcomp.Array`) | **yes** |
| `torch.from_dlpack(obj)` | yes | no |
| `cudf` / `rmm.DeviceBuffer` | both | both |

**(speculation, partial doc — verify nvcomp once if we depend on DLPack
side; czarr's current code uses CAI through `cp.asarray`)**

### 4.4 czarr's existing reliance on CAI

* `_buffer.nvarray_to_buffer` does `cp.asarray(nv)` → CAI-based wrapping
  of an nvcomp.Array into a cupy.ndarray. **(src — `_buffer.py:62`)**
* `codecs/base.py:221` reads `encoded.__cuda_array_interface__["stream"]`
  to recover nvcomp's internal stream handle for `get_stream()`. This is
  the only place we extract the optional `stream` field from CAI today.
  **(verified — src)**

---

## 5. cuFile registration lifecycle

Source surface: `/hpc/mydata/sricharan.varra/Dev/czarr/.venv/lib/python3.13/site-packages/cuda/bindings/cufile.pyx` (lines around the
`cpdef`s — full list above in research note). czarr's wrapper:
`src/czarr/storage/cufile_runtime.py`.

### 5.1 `cuFileHandleRegister` vs `cuFileBufRegister`

* **`cuFileHandleRegister`**: registers an OS **file descriptor** with the
  cuFile driver. Returns an opaque `CUfileHandle_t`. This is what
  binds the kernel-side file open to the DMA-able state. Per-file. **(doc
  — cuFile API guide)**
* **`cuFileBufRegister`**: registers a **device or pinned-host buffer**
  with cuFile (pins it for DMA, marks it as cuFile-ready). One call per
  buffer per process. **(doc)**
* Both are required for the **async** path (`cuFileReadAsync`).
* For the **sync** path (`cuFileRead`), buf-register is NOT strictly
  required — sync read works with an unregistered buffer, but it's
  faster with registration. **(doc — cuFile programming guide)**
* **`cuFileStreamRegister`**: a third registration required only for the
  async path. **(doc — `czarr/storage/cufile_runtime.py:251-259`)**

### 5.2 Lifecycle pattern

```text
driver_open                           process-wide, once
  ├── handle_register(fd) ──► CUfileHandle    per-file
  │     [reads happen here]
  │   handle_deregister(CUfileHandle)
  ├── buf_register(dev_ptr, size)              per-device-region
  │     [reads using this region happen here]
  │   buf_deregister(dev_ptr)
  └── stream_register(stream_ptr)              per-stream (async only)
        [async reads happen here]
      stream_deregister(stream_ptr)
driver_close                          on process exit
```

### 5.3 Buffer registration semantics

* **Can we register 1 GiB once and use for many reads at different
  offsets?** Yes. `buf_register(ptr, size, 0)` registers the entire
  region; subsequent `cufile.read(handle, ptr + offset, length, file_offset, 0)`
  calls work as long as `ptr + offset + length <= ptr + size`. **(doc —
  cuFile docs; verified by czarr's `cufile_runtime.ensure_buf_registered`
  which assumes this — `src/czarr/storage/cufile_runtime.py:262-279`)**
* **Re-registration**: if you register the same `dev_ptr` with a larger
  size, you must `buf_deregister` first. czarr handles this — see
  `cufile_runtime.py:271-279`. **(verified)**

### 5.4 Cost (anchored to nsys data)

* czarr's measurements (recorded in `buffer-handoff.md:73-80`):
  - **A40 compat mode**: ~3 ms per `read_into` call (open + register +
    read + deregister + close).
  - **H100 + GDS**: ~1 ms per call.
  - **H200 + GDS**: similar to H100.
* These figures are dominated by `handle_register` + `handle_deregister`
  + the open() syscall, not by `buf_register`. `buf_register` is
  ~sub-ms on H200 with real GDS, but the **A40 compat path has been
  measured at ~3.77 ms median for `cuFileBufRegister`** — this is the
  "sub-ms on H200 GDS" baseline contrast cited in the task brief.
  **(verified — `buffer-spike.md` + task brief context)**
* **Implication**: register-once-at-allocation amortises the register
  cost across the buffer's lifetime, taking it out of the per-chunk
  hot path. This is the architectural win the buffer epic targets.

### 5.5 Direct vs compat mode

* **Direct mode**: GPU does the DMA directly via the `nvidia_fs` kernel
  module. Requires a supported filesystem (ext4, xfs with
  GDS-capable backing; not VAST NFS on Bruno's `/hpc/mydata`). **(doc)**
* **Compat mode**: cuFile falls back to a pinned host bounce buffer +
  `cudaMemcpyAsync`. This is what runs on Bruno A40 (no nvidia_fs) and
  on VAST NFS (no GDS support on the FS side). **(verified — task brief
  states `cufile_posix_read` is the code path on `/hpc/mydata`)**
* **Does `cuFileBufRegister` help in compat mode?** Yes, though the
  benefit is smaller. The registration still pins the device buffer
  (so the bounce DMA can target it directly without first walking PTs),
  and it caches per-region metadata in the cuFile driver. czarr's
  measurements show register-once still saves ~30-50 % of the per-read
  cost in compat mode. **(speculation — measurement-supported but not
  rigorously benched)**

### 5.6 Async path constraints (cited from project memory)

From the cuFile-async-constraints memory note (referenced in
`MEMORY.md`):

* `cuFileReadAsync` on NFS silently fails (does nothing) without
  `cuFileBufRegister` on the dest buffer. **(verified — czarr team)**
* A40 with no `nvidia_fs` hits an internal assertion in the cuFile
  worker thread on the first async call. czarr's
  `cufile_runtime.is_async_available()` gates this by checking
  `/proc/driver/nvidia-fs`. **(verified —
  `src/czarr/storage/cufile_runtime.py:79-86`)**
* The size/offset/bytes-completed pointers passed to `read_async` must
  be **pinned host memory** — they're read by the GDS kernel when the
  op runs. czarr's `_AsyncIOArgs` (lines 296-337) allocates four pinned
  scalars per async submission. **(verified)**
* **Batch I/O API** (`cuFileBatchIO*`): tested by czarr, found to be
  1.8-14× slower than threaded sync on VAST NFS. Scrapped. **(verified
  — MEMORY.md project note + commit c5cc001)**

### 5.7 Recommended pattern for `CzarrGpuBuffer`

```text
At buffer allocation:
  buf = VMR.allocate(size)
  cufile.buf_register(buf.handle, size, 0)
  buf._cufile_registered = True

At buffer use:
  cufile.read(handle, buf.handle + chunk_offset, chunk_size, file_offset, 0)

At buffer close:
  if buf._cufile_registered:
      cufile.buf_deregister(buf.handle)
  buf.close()
```

This puts the registration cost on the allocation hot path (which is
already amortised by VMR / DMR pools) and removes it from the per-chunk
read path entirely.

---

## 6. RMM + cuda.core coexistence

### 6.1 The cupy hook is the only place they meet today

czarr's allocator wiring:

* `register_nvcomp_allocator()` routes nvCOMP's internal alloc through
  cupy (via `nvcomp.set_device_allocator(_cupy_alloc)` —
  `src/czarr/alloc.py:77`).
* `use_rmm_pool()` routes cupy through RMM (via
  `cp.cuda.set_allocator(rmm_cupy_allocator)` — `alloc.py:109`).
* `cuda.core.DeviceMemoryResource` is **independent** — it talks
  directly to `cuMemAllocFromPoolAsync` on whatever pool was created
  for it. **(verified — src)**

So **no conflict, but no sharing either**. If a user calls
`configure_gpu(rmm_pool_gb=8)`, RMM owns the cupy + nvcomp allocations;
DMR-allocated buffers (if czarr starts using them) sit in a separate
pool. Both are on the same device and compete for the same physical
memory, but they cannot reuse each other's freed blocks. **(verified —
src; speculation on the GC interaction)**

### 6.2 Is the RMM pool visible from `cuda.core.Device(0)`?

No. `cuda.core.Device(0).memory_resource` (if you query it via
`DeviceMemoryResource(0)`) creates a **new** memory pool with default
properties, not RMM's pool. **(verified — `DeviceMemoryResource.__init__`
calls `MP_init_create_pool` per
`_device_memory_resource.pxd:9-11`)**

`Device.memory_resource` could in principle return RMM's pool if RMM
were the **current pool** (`cudaDeviceSetMemPool`). RMM does call
`cudaDeviceSetMemPool` when initialised in pool mode — but cuda.core's
`MP_init_current_pool` (an alternative initializer in
`_memory_pool.pxd:29-34`) is not the path used by `DeviceMemoryResource`.
**(speculation — needs an empirical probe; if true, `cuda.core` could
be made to share RMM's pool by switching init path)**

### 6.3 Memory accounting

* RMM `rmm.statistics.enable_statistics()` tracks allocations through
  the RMM allocator. cuda.core's DMR allocations bypass it entirely.
  **(doc — RMM docs)**
* If we want unified accounting we either:
  1. Route DMR through RMM by setting RMM's pool as the device current
     pool (per §6.2), or
  2. Avoid DMR and stay on RMM exclusively, or
  3. Wrap DMR with a counter (additive accounting only).

### 6.4 Recommendation

For now, keep the split:

* **VMR** for cuFile-bound buffers (czarr-owned, not RMM-routed).
* **RMM** (via cupy) for codec scratch + decode output.

This matches the buffer-epic decision in `buffer-spike.md:76-84`.

---

## 7. Pinned host buffer patterns

### 7.1 When pinned vs pageable?

* **H2D / D2H bandwidth**: pinned gives ~25 GiB/s on PCIe Gen4 x16;
  pageable peaks at ~10 GiB/s because the driver does a copy-to-pinned
  internally and then DMAs. **(doc — NVIDIA dev blog "How to Optimize
  Data Transfers in CUDA C/C++")**
* **Async transfers** (`cudaMemcpyAsync`): only pinned host memory
  participates in real async DMA. Pageable forces a sync. **(doc)**
* **For czarr**: any time we stage compressed bytes through host to
  reach the GPU (e.g. host-prototype path in `codecs/base.py:298-299`),
  pinned is mandatory if we want to overlap H2D with subsequent decode.

### 7.2 cuFile compat-mode bounce buffer

* In compat mode, cuFile's internal `cufile_posix_read` reads from disk
  into a **driver-managed pinned bounce buffer**, then issues a
  `cudaMemcpyAsync` to the destination. **(doc — cuFile compat-mode
  section)**
* The bounce buffer size is configurable via
  `cufile.set_parameter_size_t(SizeTConfigParameter.MAX_PINNED_MEMORY_SIZE, ...)`.
  Default is typically 32 MiB total across the driver. **(doc, src —
  `cufile.pyx` exposes `set_parameter_size_t`)**
* **Can we make it our buffer?** No — the bounce buffer is internal to
  the cuFile driver. You can't substitute your own. The only knob is
  the total size cap. **(doc)**
* **Can we skip the bounce by registering our own pinned host buffer
  and reading into it directly (then memcpy to device ourselves)?**
  That sidesteps cuFile entirely and just uses POSIX `pread` + an
  explicit `cudaMemcpyAsync`. On VAST NFS this is **the same code path
  cuFile compat would have taken**, just with the bounce-buffer
  size now under our control. czarr could measure this — likely no
  win on the happy path but a useful escape hatch when the cuFile
  bounce buffer is saturated.

### 7.3 Sizing the existing `PinnedHostPool`

Current strategy (`src/czarr/pipeline/pinned.py`): exact-size buckets, no
size-class promotion. Buffers are returned to the free list keyed on
exact byte count.

* **Works for**: uniform-chunk-size workloads (the usual zarr case —
  imaging data with one chunk shape per array).
* **Pathological for**: variable-length compressed chunks. Each
  compressed chunk is a different size; the exact-size match nearly
  always misses, leading to a fresh `cudaMallocHost` per acquire.
* **jemalloc-style size-class promotion** (round up to nearest power of
  2 or to a small set of buckets like {64K, 256K, 1M, 4M, 16M, 64M})
  would mitigate this:
  - Worst-case waste: ~2× (one bucket size up).
  - Free-list hit rate: high once warm.
  - Implementation: ~30 LOC change in `PinnedHostPool` —
    `_size_class(size)` rounds up + the dict is keyed on the class.

**Recommendation**: switch when we hit the variable-chunk pinned-alloc
path; profile to confirm `cudaMallocHost` is the bottleneck before
spending the LOC.

---

## 8. Device events / stream synchronisation

### 8.1 `cuda.core.Event` vs `cupy.cuda.Event`

* Both wrap `CUevent`. **(src — `_event.pxd:9-22`; cupy docs)**
* They are **not the same object class**, so:
  - `cuda.core.Stream.wait_event(cuda_core_event)` works (same package
    family).
  - `cupy.cuda.Stream(...).wait_event(cuda_core_event)` requires
    going through the raw handle: `cp_stream.wait_event(cp.cuda.Event(0, handle=ev._h_event.handle))`.
    **(speculation — cupy doesn't expose a from-handle event constructor
    cleanly; you may have to drop to raw `cudart.cudaStreamWaitEvent`)**
* **Easiest interop pattern**: keep one stream type per layer. czarr
  uses `cuda.core.Stream` in `StreamPool`; the codec layer accepts the
  stream's `__cuda_stream__()` handle and re-wraps it in
  `cp.cuda.ExternalStream(handle)` when entering cupy code (the
  `_resolve_stream` pattern in `codecs/base.py:158-179`). This avoids
  needing to bridge event types — work on the stream is enqueued from
  whatever framework owns the call site, and synchronisation flows
  through the stream.

### 8.2 `Stream.wait_event` and multi-device

* `cudaStreamWaitEvent(stream, event)` works across devices since
  CUDA 11. The event must have been recorded on a stream whose device
  context is current at record time, but a stream on a different
  device can wait on it without an explicit context push. **(doc — CUDA
  programming guide §"Stream and Event Behaviour Across Devices")**
* **In czarr**: today we only target single-device per process. Even on
  a multi-GPU box (Bruno gpu-h-* nodes have 8 H200s), the runtime per
  process binds to one device. So this is mostly academic for now.

### 8.3 Multi-stream read/decode overlap pattern

The classic pattern (cuFile-bound + decode-bound):

```text
streams = StreamPool(size=N)
for chunk in chunks:
    s = streams.acquire()      # round-robin
    cufile.read_async(handle, buf_ptr[chunk], size, s)
    nvcomp_decode_on_stream(buf_ptr[chunk], out_ptr[chunk], s)
streams.sync_all()
```

The key insight: keep each chunk's read+decode on **one** stream. Then
the only synchronisation is at the end (`sync_all`). N concurrent
chunks → N-way overlap of disk I/O and decode kernels.

This is the pattern the buffer epic will reach for in Phase 3 / 4. Today
the codec is per-batch (not per-chunk) so the overlap is between batches,
not within a batch.

---

## 9. CUDA Graphs

### 9.1 Capture semantics

`cuda.core.Stream.begin_capture()` / `Stream.end_capture()` follow the
CUDA Graphs API: any kernel launch + memcpy + memset enqueued on the
stream between begin/end is recorded into a `Graph`. Then you
`Graph.instantiate()` and launch repeatedly with lower per-launch
overhead. **(doc — CUDA Graphs programming guide;
src — `cuda/core/cu13/graph/`)**

### 9.2 Can we capture an nvCOMP decode call?

**Probably not, today**. nvCOMP's batched decode API internally:

* Allocates per-batch scratch (cudaMallocAsync).
* Launches a sequence of kernels including a CPU-side decision step
  for variable-length chunks.

The CPU-side decisions plus the malloc make this **not** capture-safe
in the simple sense. nvCOMP 4.0+ has explicit graph-capture support for
the **fixed-shape** path (uniform chunk size, known output size); the
variable path is still a no-go. **(doc — nvCOMP "Working with Graphs"
section; speculation on which specific paths are safe — needs a small
spike to confirm)**

### 9.3 What captures cleanly?

* Memcpy chains (D2D, D2H, H2D).
* Custom kernel launches (e.g. czarr's byteshuffle kernel in
  `src/czarr/kernels/byteshuffle.py`).
* nvCOMP **encode** when used in the "single shape, known output bound"
  pattern. **(doc — nvCOMP)**

So the smallest useful capture target for czarr is: **the per-chunk
post-decode pipeline** (memcpy decode output to a slab, run any filter
kernels, sync). Building a graph for that gets you sub-µs per-chunk
launch instead of ~10 µs per launch (per-launch overhead is the typical
benchmark improvement from CUDA Graphs). **(doc)**

### 9.4 GraphMemoryResource ties in here

If you capture an allocation **with** a `GraphMemoryResource`, the
graph's instantiate/launch knows about the allocation node and can
free + reuse memory across iterations. This is the only way to capture
work that itself allocates without leaking on every graph launch.
**(doc — CUDA Graphs §"Memory allocation in graphs")**

For czarr this is a Phase-N concern — only worth it if Python overhead
becomes the bottleneck. Today the bottleneck is cuFile per-call
overhead (per-chunk register/deregister), not kernel launch.

---

## 10. Multi-device

### 10.1 `Device(N).set_current()`

* CUDA's `cuCtxSetCurrent` is **thread-local**, not process-wide. **(doc)**
* `cuda.core.Device.set_current()` calls this. So if you `set_current(1)`
  in thread A, thread B still sees device 0 (or whatever its last call
  set). **(src — Device.set_current ends up in compiled .so but the
  CUDA semantics are unambiguous)**
* **Implication for czarr's pipeline**: `StreamPool` calls
  `device.set_current()` in `__init__` (`pipeline/streams.py:57`). If
  the pool is constructed on the main thread and then used from worker
  threads, those workers need their own `device.set_current()` first.
  cupy hides this with `cp.cuda.Device(0)` context managers; cuda.core
  is more explicit. **(verified — read of `streams.py`)**

### 10.2 Peer access / NVLink

* Bruno gpu-h-* nodes have 8× H200 SXM5 with full NVLink mesh
  (NVL5 fabric). Peer access between any two devices is supported.
  **(doc — H200 datasheet)**
* `cuda.core` exposes peer access through `Device.enable_peer_access(peer)`
  (in the compiled .so; signature is in cuda-python docs). **(doc)**
* For czarr today: single-device-per-process. Multi-device is out of
  scope for the buffer epic. The note in `VirtualMemoryResource`'s
  `peers=[...]` option (§1.2) is the future plumbing if we ever
  fan out one VMR-backed buffer across multiple H200s — would let one
  GPU read a chunk and a peer GPU decode it without going through host.

---

## Cross-references and confidence summary

| section | dominant confidence | source |
|---|---|---|
| §1 MemoryResource | doc + src | cuda.core source in venv |
| §2 stream-ordered rules | doc | CUDA programming guide |
| §3 DLPack | doc + verified | spec + buffer-spike + DLPack v1 src |
| §4 CAI | doc + speculation | numba spec; czarr usage in codec base |
| §5 cuFile registration | doc + verified | cuFile guide + czarr nsys data |
| §6 RMM + cuda.core | src + speculation | reading both allocator code paths |
| §7 pinned patterns | doc + verified | dev blog + czarr PinnedHostPool |
| §8 events / streams | doc | CUDA programming guide |
| §9 CUDA Graphs | doc + speculation | CUDA Graphs guide + nvCOMP notes |
| §10 multi-device | doc | CUDA programming guide |

---

## Implementer's quick-reference (the "why did this segfault?" table)

| symptom | likely cause | fix |
|---|---|---|
| `cudaErrorMisalignedAddress` deep in nvCOMP | nvcomp.Array source is not 16-byte aligned; common with sharding-codec sub-buffers | `_buffer._ensure_aligned` already handles this for cupy arrays. New code paths should round dev pointers up to 16 manually or use VMR with `addr_align=16` (or 4096). **(verified — `src/czarr/_buffer.py:31-36`)** |
| Async cuFile read returns 0 bytes silently | buffer not registered with `cuFileBufRegister` (NFS) | Always call `ensure_buf_registered(ptr, size)` before `read_async`. **(verified — MEMORY.md cuFile-async-constraints)** |
| Process segfaults at exit when running full test suite incl. test_alloc | RMM pool reinit races cuda.core / nvCOMP teardown | Exclude `test_alloc` from full-suite runs (already done in conftest). **(verified — `buffer-handoff.md:112-115`)** |
| `read_async` worker thread asserts inside libcufile (A40) | nvidia_fs not loaded; cuFile in compat mode does not support async | Gate on `is_async_available()` which checks `/proc/driver/nvidia-fs`. **(verified — `cufile_runtime.py:79-86`)** |
| Buffer freed but kernel using it crashes | non-stream-ordered free (VMR / LegacyPinned) called while kernel still in flight | Always `stream.sync()` before close on VMR/LegacyPinned buffers, or migrate to DMR/PMR which queue the free. |
| cupy view of cuda.core.Buffer crashes after producer goes out of scope | the DLPack consumer should keep the producer alive but didn't | Verify the consumer is DLPack v1 compliant (cupy is). If you wrote your own consumer, make sure it stores the capsule deleter. |
| Pool grows unboundedly under streamed allocations | stream-ordered alloc + free on different streams without sync; pool can't reuse blocks until the free-side stream catches up | Either alloc + free on the same stream, or insert a periodic `pool.trim_to(size)` (`_MemPoolAttributes.set_release_threshold`). **(doc — CUDA pool docs)** |
| Codec output is on a different stream than next user op | nvcomp.Codec uses its internal stream unless you pass `cuda_stream=` | Always pass a known stream to `nvcomp.Codec(cuda_stream=...)` or use `Codec.get_stream()` to read it back. **(verified — `codecs/base.py:204-223`)** |
| `Device.set_current()` succeeded on main thread but worker sees device 0 | thread-local context | Each worker thread must call `device.set_current()` itself. **(doc)** |
| 2 MiB of VA "lost" per small VMR alloc | granularity floor on default `RECOMMENDED` granularity | For tiny allocs use `granularity=MINIMUM` (queries `cuMemGetAllocationGranularity(MINIMUM)` — typically 4 KiB on Linux with FABRIC handle type). **(doc — CUDA VMM)** |

---

## Appendix A — concrete API shapes for `CzarrGpuBuffer`

Sketched here to remove guesswork; the implementer can deviate:

```python
class CzarrGpuBuffer:
    # Backing primitives
    _buf: cuda.core.Buffer        # the VMR-backed allocation
    _size: int                     # byte size
    _stream: int                   # cudaStream_t for current ops
    _cufile_registered: bool       # tracks buf_register state

    # Interop surface
    def __dlpack__(self, *, stream=None) -> capsule: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...   # (kDLCUDA, device_id)
    @property
    def __cuda_array_interface__(self) -> dict: ...

    # Lifetime
    def close(self, *, stream=None) -> None: ...
    def __del__(self) -> None: ...   # falls back to close(stream=None)

    # cuFile binding (idempotent)
    def cufile_register(self) -> None: ...
    def cufile_deregister(self) -> None: ...

    # Zero-copy view as cupy
    def as_cupy(self, dtype=cp.uint8) -> cp.ndarray:
        return cp.from_dlpack(self).view(dtype)
```

The crucial invariants:

1. `int(__cuda_array_interface__["data"][0]) == int(__dlpack__-exported ptr) == int(_buf.handle)` — single device pointer through all three views.
2. `close(stream=s)` is the right call after the last consumer on stream `s`.
3. `cufile_register()` is called once at allocation time (in `_create()`), `cufile_deregister()` once at close.

---

End of document. ~720 lines.
