# cuda.core API inventory for a CUDA-native Zarr Array

> Agent 1 / 5 — Background research for the GPU-native Zarr v3 array decision.
> Sources: <https://nvidia.github.io/cuda-python/cuda-core/latest/api.html> plus
> every linked `generated/cuda.core.*` page; the official examples on
> <https://nvidia.github.io/cuda-python/cuda-core/latest/examples.html>; and the
> live czarr usage in `src/czarr/core/buffer.py` /
> `bench/buffer/alignment_probe.py` (worktree `czarr.czarr-gpu-buffer`).

---

## Executive summary

* `cuda.core` (the new high-level façade on top of `cuda.bindings`) gives us a
  near-complete substrate for a GPU-native zarr v3 Array path. We can stay in
  pure Python/CUDA from chunk discovery, through cuFile read, through codec
  pipeline, into the final cupy / DLPack tensor — without dropping to CUDA C
  for the orchestration layer.
* The pieces we already use are stable and exactly the right primitives:
  `Device`, `Stream`, `Buffer`, `DeviceMemoryResource`, `LegacyPinnedMemoryResource`,
  `VirtualMemoryResource` (with `VirtualMemoryResourceOptions(addr_align=4096,
  gpu_direct_rdma=True)` — verified as the only allocator that yields fresh
  4 KiB-aligned device VAs).
* The pieces we should adopt next are the **JIT compilation toolchain**
  (`Program` → `ObjectCode` → `Kernel` + `launch` + `LaunchConfig`) and the
  **graph capture API** (`Stream.create_graph_builder` → `GraphBuilder` →
  `Graph.upload` / `Graph.launch`). The graph API is the right tool for the
  microbatch overlap pipeline once Python launch overhead becomes the dominant
  cost.
* For caching compiled kernels across processes there is a first-class
  `FileStreamProgramCache` rooted at `$XDG_CACHE_HOME/cuda-python/program-cache`
  — we get persistent JIT caching for free as long as we pass `cache=…` to
  `Program.compile()`.
* For codec orchestration we get `prefetch_batch` / `discard_batch` (managed
  memory) and `ManagedBuffer.prefetch` / `accessed_by` / `read_mostly`. These
  are interesting for chunk-prefetch scheduling but only with managed memory,
  which we are not using for the GDS path. They are noted but not load-bearing.
* The pieces that are **partially documented or surface-experimental** and
  introduce API risk are: `cuda.core.checkpoint.Process` (stub), `cuda.core.TensorMapDescriptor`
  (private `_from_*` helpers — "this API surface settles"), `cuda.core.graph.*` host
  callbacks (no explicit safety contract listed), `discard_prefetch_batch` (CUDA 13+),
  and PCH options in `ProgramOptions` (CUDA 12.8+). We should call these out in any design.
* What `cuda.core` does **not** give us natively:
  * No CAI on `Buffer` — we already synthesise `__cuda_array_interface__` in
    `CzarrGpuBuffer`.
  * No first-class p2p memcpy across devices (the bindings expose it, the
    high-level surface does not yet).
  * No async cuFile / GDS bridge — we still need `cuda.bindings.cufile`
    directly (covered by Agent 4).
  * No `Stream.is_done` / non-blocking query — synchronization is sync-or-wait,
    not poll. `Event.is_done` does exist; that's our knob.

The recommendation flowing out of this inventory is: build the CUDA-native
Array on **(`VirtualMemoryResource` allocator)** × **(`Program`/`Kernel`/`launch`
codec kernels)** × **(`GraphBuilder` microbatch pipelines)** × **(`Event`
ordering)**, with `FileStreamProgramCache` for warm-start. Everything except
the codec kernel set is shipped and stable.

---

## 1. The complete `cuda.core` index (as of build from `main`)

The API ref renders from a development branch; this list is what `api.html`
returns today. Citations: <https://nvidia.github.io/cuda-python/cuda-core/latest/api.html>.

### Devices and execution
| Symbol | Role |
| --- | --- |
| `cuda.core.Device` | GPU singleton; entry point for almost everything |
| `cuda.core.Host` | Symmetric "CPU location" for managed-memory advice/prefetch |
| `cuda.core.Context` | Wrapped `CUcontext` (primary + green) |
| `cuda.core.Stream` | Queue of GPU operations |
| `cuda.core.Event` | Execution point within a stream |
| `cuda.core.SMResource`, `SMResourceOptions` | SM partition / green context source |
| `cuda.core.WorkqueueResource`, `WorkqueueResourceOptions` | Workqueue config (green-context related) |
| `cuda.core.LaunchConfig` | Grid / cluster / block / shmem launch params |
| `cuda.core.StreamOptions` | `nonblocking`, `priority` |
| `cuda.core.EventOptions` | `timing_enabled`, `blocking_sync`, `ipc_enabled` |
| `cuda.core.ContextOptions` | Context creation options |
| `cuda.core.launch` | The kernel launch function |
| `cuda.core.LEGACY_DEFAULT_STREAM`, `PER_THREAD_DEFAULT_STREAM` | Sentinels for the two default-stream regimes |

### Memory
| Symbol | Role |
| --- | --- |
| `cuda.core.Buffer` | Owns / references a device or pinned allocation |
| `cuda.core.ManagedBuffer` | `Buffer` subclass with unified-memory advice API |
| `cuda.core.MemoryResource` | Abstract base |
| `cuda.core.DeviceMemoryResource` | Pool-based device allocator (`cuMemAllocAsync`) |
| `cuda.core.PinnedMemoryResource` | Pool-based pinned host allocator |
| `cuda.core.ManagedMemoryResource` | Pool-based managed (unified) allocator |
| `cuda.core.LegacyPinnedMemoryResource` | Sync `cuMemAllocHost` allocator |
| `cuda.core.VirtualMemoryResource` | VMM-based, addr-aligned, GDR-tagged allocations |
| `cuda.core.GraphMemoryResource` | Allocations whose lifetime is the enclosing graph |
| `cuda.core.DeviceMemoryResourceOptions` | `ipc_enabled`, `max_size` |
| `cuda.core.PinnedMemoryResourceOptions` | `ipc_enabled`, `max_size`, `numa_id` |
| `cuda.core.ManagedMemoryResourceOptions` | (options for managed pool) |
| `cuda.core.VirtualMemoryResourceOptions` | `addr_align`, `gpu_direct_rdma`, `handle_type`, `granularity`, `addr_hint`, `peers`, `self_access`, `peer_access`, `allocation_type`, `location_type` |

### CUDA Graphs
| Symbol | Role |
| --- | --- |
| `cuda.core.graph.Graph` | Instantiated executable (`CUgraphExec`) |
| `cuda.core.graph.GraphBuilder` | Stream-capture front-end |
| `cuda.core.graph.GraphDefinition` | Explicit (node-by-node) front-end |
| `cuda.core.graph.GraphNode` | Base node |
| `cuda.core.graph.GraphCondition` | Conditional variable |
| `cuda.core.graph.GraphCompleteOptions` | Instantiation options (`auto_free_on_launch`, `upload_stream`, `device_launch`, `use_node_priority`) |
| `cuda.core.graph.GraphDebugPrintOptions` | DOT print toggles |
| Node types: `EmptyNode`, `KernelNode`, `AllocNode`, `FreeNode`, `MemsetNode`, `MemcpyNode`, `ChildGraphNode`, `EventRecordNode`, `EventWaitNode`, `HostCallbackNode`, `ConditionalNode`, `IfNode`, `IfElseNode`, `WhileNode`, `SwitchNode` |

### Graphics / TMA
| Symbol | Role |
| --- | --- |
| `cuda.core.GraphicsResource` | OpenGL buffer / texture interop |
| `cuda.core.TensorMapDescriptor` | Hopper TMA descriptor (128-byte opaque) |
| `cuda.core.TensorMapDescriptorOptions` | TMA descriptor config |

### Compilation toolchain
| Symbol | Role |
| --- | --- |
| `cuda.core.Program` | NVRTC / NVVM compilation front-end |
| `cuda.core.ProgramOptions` | ~50 NVCC-equivalent knobs |
| `cuda.core.Linker` | Link cubin / ltoir |
| `cuda.core.LinkerOptions` | Link-time knobs |
| `cuda.core.ObjectCode` | Compiled artifact (cubin / ptx / ltoir / fatbin) |
| `cuda.core.Kernel` | Loaded kernel + occupancy / attribute introspection |

### `cuda.core.utils`
| Symbol | Role |
| --- | --- |
| `cuda.core.utils.ProgramCacheResource` | Cache base class |
| `cuda.core.utils.InMemoryProgramCache` | LRU in-process cache for compiled kernels |
| `cuda.core.utils.FileStreamProgramCache` | LRU on-disk cache (multi-process safe) |
| `cuda.core.utils.make_program_cache_key` | Key builder |
| `cuda.core.utils.StridedMemoryView` | DLPack/CAI/`__array_interface__` triple-source view |
| `cuda.core.utils.args_viewable_as_strided_memory` | Decorator that proxies arg N as a `StridedMemoryView` |
| `cuda.core.utils.prefetch_batch` | Batched `cuMemPrefetchAsync` over managed buffers |
| `cuda.core.utils.discard_batch` | Batched managed-memory page discard (CUDA 13+) |
| `cuda.core.utils.discard_prefetch_batch` | Combined discard+prefetch (CUDA 13+) |

### `cuda.core.system` (NVML)
| Symbol | Role |
| --- | --- |
| `cuda.core.system.get_user_mode_driver_version` / `get_kernel_mode_driver_version` / `get_driver_branch` / `get_nvml_version` | Driver introspection |
| `cuda.core.system.get_num_devices` | Device count |
| `cuda.core.system.get_process_name(pid)` | Look up process name |
| `cuda.core.system.get_topology_common_ancestor` / `get_p2p_status` | Topology |
| `cuda.core.system.register_events` | NVML event registration |
| `cuda.core.system.Device` | NVML device wrapper (utilization, temperature, clocks…) |
| `cuda.core.system.NvlinkInfo` | NVLink topology |

### `cuda.core.checkpoint` (experimental-feel)
| Symbol | Role |
| --- | --- |
| `cuda.core.checkpoint.Process(pid)` | Lockable / checkpointable process — page is a stub |

---

## 2. `Device` — the entry point

Source: <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Device.html>.

### Construction

```python
Device(device_id: int | None = None)
```

* Thread-local **singleton**: `Device(0)` returns the same Python object every
  time on the calling thread.
* `Device()` / `Device(None)` returns the currently active device on this thread.
* `Device.get_all_devices() -> tuple[Device, ...]` enumerates everything.

### Properties

| Name | Type | Notes |
| --- | --- | --- |
| `device_id` | `int` | Ordinal |
| `name` | `str` | "NVIDIA H100 80GB HBM3", etc. |
| `arch` | `str` | "90" for sm_90 — used as `f"sm_{dev.arch}"` in every example |
| `compute_capability` | named tuple `(major, minor)` | |
| `uuid` | `str` | Includes MIG UUID in MIG mode |
| `pci_bus_id` | `str` | |
| `context` | `Context` | Primary context (device must be initialised) |
| `default_stream` | `Stream` | Per-thread or legacy default based on env |
| `memory_resource` | `MemoryResource` | The device's currently-active MR |
| `properties` | `DeviceProperties` | 100+ attributes — see §2.1 |
| `resources` | `DeviceResources` | Hardware resource namespace (sm / workqueue) |

### Methods

* `set_current(ctx: Context | None = None) -> Context | None` — initialise the
  device and make it current; if `ctx` is given it's pushed too.
* `create_stream(obj: IsStreamType | None = None, options: StreamOptions | None = None) -> Stream`
  — `obj` is for adopting a foreign `cudaStream_t`.
* `create_event(options: EventOptions | None = None) -> Event`
* `create_context(options: ContextOptions | None = None) -> Context`
* `create_graph_builder() -> GraphBuilder`
* `allocate(size: int, *, stream: Stream | GraphBuilder) -> Buffer` — shortcut
  for `device.memory_resource.allocate(...)`.
* `sync()` — `cuCtxSynchronize`.
* `can_access_peer(peer: Device | int) -> bool`
* `to_system_device() -> cuda.core.system.Device` — bridge to NVML.

### 2.1 `DeviceProperties` — what we get for free

`show_device_properties.py` enumerates 108 attributes (full list in §A1). The
ones that matter for an async-IO / GDS design:

* `gpu_direct_rdma_supported` (bool) — does the card support GDR at all
* `gpu_direct_rdma_flush_writes_options` (bitmask)
* `gpu_direct_rdma_writes_ordering` (int)
* `concurrent_kernels`, `gpu_overlap`, `kernel_exec_timeout`
* `memory_pools_supported`, `mempool_supported_handle_types`,
  `handle_type_posix_file_descriptor_supported`,
  `virtual_memory_management_supported`
* `l2_cache_size`, `max_persisting_l2_cache_size`, `multiprocessor_count`,
  `max_blocks_per_multiprocessor`, `max_threads_per_multiprocessor`,
  `max_shared_memory_per_block_optin`, `warp_size`
* NUMA: `numa_config`, `numa_id`, `pageable_memory_access`,
  `pageable_memory_access_uses_host_page_tables`
* `multicast_supported` (Hopper switch multicast — useful for collective codecs)

These are queried lazily from the driver; cheap to read in `__init__` of an
array class.

> **API risk note**: there's no documented schema for `DeviceProperties` — the
> 108 attributes are inferred from the example. Treat fields as "may exist on
> some driver versions" and use `getattr(dev.properties, name, None)`.

---

## 3. `Stream` — async submission

Source: <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html>.

### Construction

* Direct `Stream(...)` is unsupported. Use `Device.create_stream(options=...)`
  or `Context.create_stream()` (green contexts only).
* `Stream.from_handle(handle: int) -> Stream` — wraps a foreign
  `cudaStream_t`. **Lifetime is not managed** — the foreign owner must
  outlive the wrapper. This is what we'd use to adopt a torch / cupy stream
  or to expose our stream to another library.

### Properties

* `handle` — `CUstream` object; `int(stream.handle)` gives the C pointer.
* `device` → `Device` singleton this stream belongs to.
* `context` → `Context`.
* `priority` — int; lower number = higher priority (CUDA convention).
* `is_nonblocking` — bool; true means "does not synchronise with the NULL stream".
* `resources` → green-context resources provisioned for this stream's context.

### Methods

| Method | Notes |
| --- | --- |
| `sync()` | Block until the queue drains. |
| `record(event: Event = None, options: EventOptions = None) -> Event` | Records an event into the stream. If `event` is None a fresh `Event` is created (using `options`). Returned event is the canonical "wait point". |
| `wait(event_or_stream: Event \| Stream)` | Establishes stream ordering. Anything supporting `__cuda_stream__` is accepted, so a torch / cupy stream can be passed directly. |
| `create_graph_builder() -> GraphBuilder` | Bind a builder to this stream. |
| `close()` | Destroys / releases the stream. |

### `StreamOptions`

* `nonblocking: bool = True` — default is NON-blocking. That matters: the
  default stream returned by `Device.create_stream()` is **not** ordered against
  the implicit NULL stream. This is the right default for our overlap pipeline,
  but anyone porting C++ code that relied on default-stream sync needs to know.
* `priority: int | None = None` — None means "lowest priority". `[lo, hi]` is
  device-specific; the driver query gives us the valid range (`cuCtxGetStreamPriorityRange`,
  not exposed at the high level — we'd reach into `cuda.bindings`).

### What the Stream page does NOT expose

* No `Stream.is_done` / `query()` — we have to use `Event.is_done` instead.
* No `Stream.begin_capture` / `end_capture` at the Stream level. Capture is
  encapsulated in `GraphBuilder` (`begin_building` / `end_building`), which
  internally does the `cuStreamBeginCapture` work.
* No `Stream.add_host_callback`. Host callbacks live in the graph API as
  `HostCallbackNode`, or you can use the `callback()` chainable on any graph
  builder / node.
* No `Stream.attach_mem_async`. For managed memory you go through
  `ManagedBuffer.prefetch(location, *, stream)`.

### Stream-ordering rules (the part the docs gloss over)

* Every Buffer-returning API (`MR.allocate`, `Buffer.copy_to/copy_from/fill`,
  `Buffer.close`) takes a `stream=` keyword. Operations are stream-ordered.
* `Buffer.close(stream=...)` queues an async free on that stream. If you close
  with `stream=None` the buffer falls back to the deallocation stream stored
  inside the handle — **which can be a different stream than where the buffer
  was last used**. For an overlap pipeline we want to always pass an explicit
  stream and ideally one that has just consumed the buffer.
* Cross-stream ordering: `consumer.wait(producer.record())` is the canonical
  pattern.

---

## 4. `Event` — fine-grained sync + timing

Source: <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Event.html>.

### Construction

```python
event = device.create_event(options=EventOptions(timing_enabled=True))
# or
event = stream.record(options=EventOptions(...))
```

* No direct constructor. Always via `Device.create_event` or `Stream.record`.

### `EventOptions`

* `timing_enabled: bool = False` — without it, `e2 - e1` raises (or returns
  nothing meaningful). Enabling it costs the timer / disables blocking-sync
  optimisations.
* `blocking_sync: bool = False` — when False, `Event.sync()` busy-waits the
  CPU. When True it blocks on a futex. For overlap pipelines we want True
  (we shouldn't burn a CPU core spinning while the GPU works).
* `ipc_enabled: bool = False` — requires `timing_enabled=False`.

### Properties

| Name | Type | Notes |
| --- | --- | --- |
| `is_done` | `bool` | Non-blocking query — the poll knob. |
| `is_timing_enabled` | `bool` | |
| `is_blocking_sync` | `bool` | |
| `is_ipc_enabled` | `bool` | |
| `device` | `Device` | |
| `context` | `Context` | |
| `handle` | `CUevent` | |
| `ipc_descriptor` | opaque | For exporting cross-process |

### Methods

* `sync()` — block / busy-wait until the event completes.
* `close()` — destroy.
* `Event.from_ipc_descriptor(descriptor)` (classmethod) — adopt an event
  exported from another process.

### Stream-ordering / lifetime

* `e2 - e1` (Python `__sub__`) returns elapsed milliseconds — both events
  must have `timing_enabled=True`.
* Events stay alive while *any* stream still has work waiting on them.
* Event re-use is fine: `stream.record(event=existing_event)` overwrites the
  recorded point. Caller is responsible for ensuring prior consumers are done
  with the old recording.

### What we'd use it for

* **Stream barrier without a `sync()`**: `consumer.wait(producer.record())`.
* **Polling**: `Event.is_done` is the only non-blocking query in the API.
* **Per-stage timing in the overlap pipeline**: rec at the start, rec at the
  end of every microbatch stage, subtract for true GPU time excluding Python.

---

## 5. Memory: the resource hierarchy

The most important class hierarchy for us. Sources:
<https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.MemoryResource.html>
and the per-subclass pages.

### 5.1 `MemoryResource` (abstract)

```python
class MemoryResource:
    @abstractmethod
    def allocate(self, size: int, *, stream: Stream | GraphBuilder) -> Buffer: ...
    @abstractmethod
    def deallocate(self, ptr, size: int, *, stream: Stream | GraphBuilder) -> None: ...
    # read-only properties
    device_id: int           # or -1 for host-only resources
    is_device_accessible: bool
    is_host_accessible: bool
    is_managed: bool
```

Buffers proxy their `is_*_accessible` / `is_managed` flags to the parent MR —
so a `Buffer` returned by `LegacyPinnedMemoryResource.allocate` reports BOTH
device-accessible and host-accessible.

### 5.2 `DeviceMemoryResource` — pool, stream-ordered

```python
DeviceMemoryResource(device_id: Device | int, options: DeviceMemoryResourceOptions | None = None)
```

* Wraps `cuMemPool*` (stream-ordered async allocator).
* If `options=None`: uses the driver's current/default pool (does NOT own it).
* If `options` provided: creates a new pool, owns it, will destroy on `close()`.
* `DeviceMemoryResourceOptions`:
  * `ipc_enabled: bool = False`
  * `max_size: int = 0` (0 = system-dependent default).
  * No `release_threshold` / `allocation_handle_type` documented at the high
    level — those are configured by the underlying pool which we don't see.
* IPC: `is_ipc_enabled`, `is_mapped`, `uuid`, `allocation_handle`,
  `from_allocation_handle()`, `from_registry()`, `register()`,
  `peer_accessible_by` (set-like proxy).
* **Behavioural gotcha (verified in our bench)**: `DeviceMemoryResource`
  sub-allocates from a pool. The *first* allocation on a fresh extent is
  page-aligned; subsequent allocations are packed at natural alignment (often
  256 B / 512 B). So **DMR pointers are NOT generally 4 KiB aligned**.

### 5.3 `VirtualMemoryResource` — what we use

```python
VirtualMemoryResource(device_id: Device | int, config: VirtualMemoryResourceOptions | None = None)
```

* Wraps `cuMem*` (Virtual Memory Management) APIs — reserve VA + allocate
  physical + map + grant.
* Allocation is **transactional**: if any step fails the resource cleans up
  what's already reserved/mapped.
* `modify_allocation(...)` — **grows an existing allocation in place** while
  preserving the base pointer (reserves more VA and maps more physical).
  This is potentially huge for streaming append.
* The docs warn: *"This is a low-level API that is provided only for
  convenience. Make sure you fully understand how CUDA Virtual Memory
  Management works before using this."* Not marked experimental, but flagged
  as "advanced — know what you're doing".

#### `VirtualMemoryResourceOptions` (full)

| Field | Type | Default |
| --- | --- | --- |
| `allocation_type` | `VirtualMemoryAllocationType \| str` | `PINNED` |
| `location_type` | `VirtualMemoryLocationType \| str` | `DEVICE` |
| `handle_type` | `VirtualMemoryHandleType \| str` | `POSIX_FD` |
| `granularity` | `VirtualMemoryGranularityType \| str` | `RECOMMENDED` |
| `gpu_direct_rdma` | `bool` | `False` |
| `addr_hint` | `int \| None` | `0` (driver picks) |
| `addr_align` | `int \| None` | `None` (use queried granularity) |
| `peers` | `Iterable[int]` | `()` |
| `self_access` | `VirtualMemoryAccessType \| str` | `READ_WRITE` |
| `peer_access` | `VirtualMemoryAccessType \| str` | `READ_WRITE` |

* `gpu_direct_rdma=True` is documented as "a hint" that requests GDR support.
  Combined with `addr_align=4096` it is the only path that yields:
    * Fresh 4 KiB-aligned device VA per allocation.
    * `cuFile`-direct compatibility.
    * Inter-device peer access if we set `peers=[…]`.
* `handle_type="posix_fd"` is recommended on Linux only if you intend to
  import/export the allocation across processes. Otherwise set to None.
* Granularity cost: each allocation rounds up VA to the device's VMM
  granularity (≈ 2 MiB on Hopper/Ampere). The granularity overhead is in **virtual
  address space, not physical memory**. We've already accepted this trade.

### 5.4 `PinnedMemoryResource` — pool, stream-ordered host

Wraps `cuMemPool*` for host-pinned memory. Same API shape as DMR.
`PinnedMemoryResourceOptions`:

* `ipc_enabled: bool = False`
* `max_size: int = 0`
* `numa_id: int | None = None` (None + ipc_enabled=True picks the device's
  `host_numa_id`).

### 5.5 `LegacyPinnedMemoryResource` — what we use today

```python
LegacyPinnedMemoryResource()  # no args
buf = mr.allocate(size)                # stream=None is fine
mr.deallocate(ptr, size)               # syncs the stream if one is given
```

* Synchronous (wraps `cuMemAllocHost` = `cudaMallocHost`).
* No flags — no portable / mapped / write-combined knobs.
* `device_id = -1`. `is_host_accessible = True`. `is_device_accessible = True`
  (it's automatically mapped into the device address space; the docs are
  explicit about this — handy for kernel args without a separate copy).
* Per the page: "This resource ignores any supplied stream" — the stream
  arg is accepted for API parity but unused.
* Our code uses this with `dev` passed in (`LegacyPinnedMemoryResource(dev)`),
  which contradicts the doc's no-arg signature. Likely a private overload —
  worth a regression note.

### 5.6 `ManagedMemoryResource` + `ManagedBuffer`

* `ManagedMemoryResource(options=None)` → returns `ManagedBuffer`, a `Buffer`
  subclass.
* IPC is **not** supported for managed pools (today).
* `ManagedBuffer` adds:
  * `read_mostly` (property: get/set)
  * `preferred_location` (Device / Host / Host(numa_id=…))
  * `accessed_by` (live mutable set-like view over locations)
  * `prefetch(location, *, stream)`
  * `discard(*, stream)` (CUDA 13+)
  * `discard_prefetch(location, *, stream)` (CUDA 13+)
* If we ever wanted "give me one chunk-sized buffer that automatically
  migrates from host to device when the kernel hits it", this is the door.
  But for a deterministic GDS pipeline managed memory is the wrong tool.

### 5.7 `GraphMemoryResource`

* For allocations done **inside a graph capture** — they become `AllocNode`s
  and are reclaimable on graph relaunch (with `GraphCompleteOptions(auto_free_on_launch=True)`).
* Not interesting until/unless we move chunk-output buffers inside a graph.

### 5.8 `Buffer`

| Property | Notes |
| --- | --- |
| `handle` | `int(buf.handle)` → device pointer (or pinned-host VA). |
| `size` | bytes. |
| `memory_resource` | The MR that minted it. |
| `device_id` | -1 for pure-host pinned. |
| `is_device_accessible`, `is_host_accessible`, `is_managed`, `is_mapped` | |
| `ipc_descriptor` | For sharing across procs. |
| `owner` | External holder if Buffer is a non-owning wrapper. |

Methods (all stream-keyword):
* `copy_to(dst: Buffer = None, *, stream) -> Buffer` — D2D / H2D / D2H.
  If `dst=None`, allocates from `self.memory_resource` and returns the new
  buffer.
* `copy_from(src: Buffer, *, stream)`
* `fill(value: int | BufferProtocol, *, stream)` — async memset; value is
  1/2/4 bytes; int range `[0, 256)`.
* `close(stream=None)` — async free. With `stream=None` it uses the
  deallocation stream stored in the buffer's handle (which may have been set
  at allocation time).
* `Buffer.from_handle(ptr, size, mr=None, owner=None)` (static) — wrap a raw
  pointer. Without `mr` / `owner`, the wrapper is non-owning and won't free.
* `Buffer.from_ipc_descriptor(mr, desc, *, stream)` — adopt a cross-proc
  buffer.

**DLPack**: `Buffer.__dlpack__` is supported and zero-copy (we have confirmed
this — `cp.from_dlpack(buf).data.ptr == int(buf.handle)`). `Buffer` does NOT
expose `__cuda_array_interface__` — we synthesise that on `CzarrGpuBuffer`.

### 5.9 Summary table for the array path

| Need | Recommended MR |
| --- | --- |
| 4 KiB aligned, GDR-tagged device buffer (cuFile / IB / NIC paths) | `VirtualMemoryResource(addr_align=4096, gpu_direct_rdma=True)` |
| Hot path device buffer with stream-ordered async free, alignment OK | `DeviceMemoryResource` (default pool) |
| Pinned host bounce buffer, one-shot | `LegacyPinnedMemoryResource` |
| Pinned host bounce buffer with stream-ordered free and NUMA pinning | `PinnedMemoryResource(options=...)` |
| Unified memory experimentation | `ManagedMemoryResource` |
| Buffers whose lifetime is a graph | `GraphMemoryResource` (auto-managed) |

---

## 6. Compilation toolchain: `Program` / `Linker` / `ObjectCode` / `Kernel`

This is the path we'd use to JIT-compile a pure-CUDA-Python LZ4 / Blosc /
custom codec.

### 6.1 `Program`

Source: <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Program.html>.

```python
Program(code: str | bytes | bytearray,
        code_type: Literal["c++", "ptx", "nvvm"],
        options: ProgramOptions | None = None)
```

* `code_type="c++"` → NVRTC backend. `"ptx"` → driver / nvJitLink. `"nvvm"` →
  NVVM compiler.
* `.compile(target_type, *, name_expressions=None, logs=None, cache=None)`
  * `target_type ∈ {"ptx", "cubin", "ltoir"}`.
  * `name_expressions=(...)` — required to instantiate templated kernels.
    Pass the mangleable C++ expression strings (e.g. `"vector_add<float>"`),
    then look them up unmangled via `ObjectCode.get_kernel(name)`.
  * `logs` — any object with `.write()` — the compiler will stream warnings.
  * `cache` — a `ProgramCacheResource` (in-mem or file). Identical compiles
    are de-duplicated.
* `.backend` → `CompilerBackendType` enum.
* `.handle` → underlying compiler handle.
* `.pch_status` (NVRTC + C++ only) → "created" / "not_attempted" / "failed".

### 6.2 `ProgramOptions` — 50+ NVCC-equivalent knobs

Source: <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.ProgramOptions.html>.
Full list with defaults in §A2. The ones to know:

* `arch: str` (e.g. `"sm_90"`) — required for cubin.
* `std: str = "c++17"`
* `relocatable_device_code: bool = False` — must be True if you'll link.
* `link_time_optimization: bool = False`
* `device_code_optimize: bool | None = None`
* `max_register_count: int | None`
* `ptxas_options: str | list | None`
* `lineinfo`, `debug`, `disable_warnings`
* `define_macro`, `undefine_macro`, `include_path`, `pre_include`
* `use_fast_math`, `prec_sqrt`, `prec_div`, `fma`, `ftz`
* `split_compile: int = 1` (parallel compile)
* `extra_sources` (NVVM only) — extra LLVM IR modules.
* `use_libdevice` (NVVM only) — link NVIDIA's libdevice math builtins.

**CUDA 12.8+ only**: PCH machinery (`pch`, `create_pch`, `use_pch`, `pch_dir`,
`pch_verbose`, `pch_messages`, `instantiate_templates_in_pch`). API risk: an
older driver will choke.

### 6.3 `ObjectCode`

Source: <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.ObjectCode.html>.

* **No default constructor**. Factory loaders:
  * `ObjectCode.from_cubin(...)`, `.from_fatbin(...)`, `.from_ptx(...)`,
    `.from_ltoir(...)`, `.from_object(...)`, `.from_library(...)`.
* `code`, `code_type`, `handle`, `name`, `symbol_mapping` (unmangled → mangled).
* `.get_kernel(name) -> Kernel` — the canonical lookup. For templates, pass
  the unmangled expression and the symbol_mapping resolves it.
* Docs say *"Constructing directly from all other possible code types should
  be avoided in favor of compilation through `Program`"* — so for our path we
  drive everything through `Program.compile()` + `cache=…`.

### 6.4 `Linker`

```python
Linker(options: LinkerOptions = None, *object_codes: ObjectCode)
linker.link(target_type)   # "cubin" or "ptx"
```

* Backend: `nvJitLink` (≥ 12.3) or driver `cuLink`. `Linker.which_backend()`
  reports which is in use.
* `get_error_log()`, `get_info_log()`.
* For LTO across user codecs the pattern from `jit_lto_fractal.py` is:
  1. Compile the codec stub `Program(...).compile("ltoir")` with
     `ProgramOptions(link_time_optimization=True)`.
  2. `Linker(main_objcode, user_objcode, options=LinkerOptions(link_time_optimization=True))`.
  3. `.link("cubin").get_kernel("main_workflow")`.

### 6.5 `LinkerOptions` — defaults

`name='<default linker>'`, `arch=None`, `max_register_count=None`,
`time=False`, `verbose=False`, `link_time_optimization=False`, `ptx=False`,
`optimization_level=None`, `debug=False`, `lineinfo=False`, `ftz=False`,
`prec_div=True`, `prec_sqrt=True`, `fma=True`, `kernels_used=None`,
`variables_used=None`, `optimize_unused_variables=False`, `ptxas_options=None`,
`split_compile=1`, `split_compile_extended=1`, `no_cache=False`.

### 6.6 `Kernel`

* Cannot be instantiated; obtain via `ObjectCode.get_kernel(name)` or
  `Kernel.from_handle(handle, mod=None)` (advanced — wraps a foreign `CUkernel`).
* Properties:
  * `handle` — `int(kernel.handle)` → C ptr.
  * `num_arguments` — int.
  * `arguments_info` — list of `ParamInfo(offset, size)` tuples (we can use
    this to validate Python-side arg packing).
  * `attributes` — read-only attribute namespace.
  * `occupancy` — namespace for occupancy queries.

### 6.7 `launch`

```python
cuda.core.launch(stream: Stream | GraphBuilder | IsStreamType,
                 config: LaunchConfig,
                 kernel: Kernel,
                 *kernel_args)
```

* `*kernel_args` — packed positionally. From the examples, scalars are passed
  as cupy scalars (`cp.uint64(size)`) and pointers as `cupy.ndarray.data.ptr`
  or `Buffer` objects directly (the launch machinery unwraps both).
* **The same `launch()` call works against a `GraphBuilder`** — it queues a
  `KernelNode` instead of an immediate launch. This is what makes the graph
  API drop-in for our existing code.

### 6.8 `LaunchConfig`

| Field | Default | Notes |
| --- | --- | --- |
| `grid` | — | When `cluster` is unset, the block count. When `cluster` is set, the cluster count. |
| `cluster` | None | Hopper thread-block-cluster size. |
| `block` | — | Threads per block. |
| `shmem_size` | 0 | Bytes of dynamic shared memory per block. |
| `is_cooperative` | False | Cooperative launch (cross-block sync). |

Cluster support: `thread_block_cluster.py` shows the Hopper pattern.

### 6.9 Caching: `InMemoryProgramCache` / `FileStreamProgramCache`

* `InMemoryProgramCache(*, max_size_bytes: int | None = None)` — LRU on
  total payload bytes; bytes/bytearray/memoryview/ObjectCode are all
  accepted as values. `get(key, default=None)` is the recommended lookup
  (`__contains__` deliberately not implemented — race safety).
* `FileStreamProgramCache(path=None, *, max_size_bytes=None)`:
  * Default path: `$XDG_CACHE_HOME/cuda-python/program-cache` on Linux
    (`%LOCALAPPDATA%\cuda-python\program-cache` on Windows).
  * Stores the **raw** compiled binary (cubin / PTX / LTO-IR) — readable by
    other tools.
  * Atomic-for-readers (uses `os.replace`); evicts on read-LRU.
  * Multi-process safe but **not crash-durable** (no directory fsync).
* Both have `get(key, default=None)`, `update(items)`, `clear()`, `close()`.
* `make_program_cache_key(...)` builds the key from `(source, options)`
  consistently.
* Hooks straight into `Program.compile(..., cache=cache)` — first call
  compiles & stores, subsequent calls fetch.

This is **the** answer for the "first import is slow because we JIT five
kernels" problem.

---

## 7. CUDA Graphs

Sources:
<https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.graph.Graph.html>,
<https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.graph.GraphBuilder.html>,
<https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.graph.GraphDefinition.html>,
example `cuda_graphs.py`.

There are **two** front-ends.

### 7.1 `GraphBuilder` — stream-capture front-end

The simpler, drop-in path. Pattern from the example:

```python
gb = stream.create_graph_builder()
gb.begin_building()                       # cuStreamBeginCapture under the hood
launch(gb, config, kernel_a, *args_a)     # builds a KernelNode
launch(gb, config, kernel_b, *args_b)     # builds another KernelNode
buf.copy_to(other, stream=gb)             # builds a MemcpyNode
graph = gb.end_building().complete()      # cuStreamEndCapture + cuGraphInstantiate
graph.upload(stream)                      # cuGraphUpload (warm up exec)
graph.launch(stream)                      # cuGraphLaunch (cheap, repeatable)
```

* `begin_building(mode='relaxed')` — `'global' | 'thread_local' | 'relaxed'`.
* `complete(options: GraphCompleteOptions | None = None) -> Graph`.
* `create_condition(default_value=None)` — runtime condition variable.
* `if_then(cond)`, `if_else(cond)`, `while_loop(cond)`, `switch(cond, count)`
  — control flow (Hopper+).
* `callback(fn, user_data=None)` — host callback. **Docs warning**: "Callbacks
  must avoid CUDA API calls to prevent deadlocks or driver corruption."
* `embed(child_graph_builder)` — child graph.
* `split(count)` — fan out into N sibling builders.
* `GraphBuilder.join(*builders)` (static) — fan back in.
* `debug_dot_print(path, options=None)` — GraphViz output for debugging.
* `is_building`, `is_join_required`, `stream` properties.

### 7.2 `GraphDefinition` — explicit DAG construction

For when you want to manually build the dependency tree:

```python
gd = GraphDefinition(device)        # presumably; constructor not in docs
n_alloc = gd.allocate(...)
n_memcpy = gd.memcpy(dst, src, size)
n_kern = gd.launch(config, kernel, *args)
n_join = gd.join(n_memcpy, n_kern)
graph = gd.instantiate()
graph.launch(stream)
```

Every node returned has chainable methods (`launch`, `memcpy`, `memset`,
`allocate`, `deallocate`, `record`, `wait`, `callback`, `embed`, `if_then`,
`if_else`, `while_loop`, `switch`, `join`, `destroy`).

* `nodes()` / `edges()` for inspection.
* `debug_dot_print(...)`.

### 7.3 `Graph` — the instantiated executable

* `handle` — `CUgraphExec`.
* `launch(stream)`, `upload(stream)`.
* `update(builder_or_definition)` — refresh an existing exec; **requires
  identical topology** (cuGraphExecUpdate). This is how we'd swap kernel args
  between batches without rebuilding.
* `close()`.

### 7.4 `GraphCompleteOptions`

* `auto_free_on_launch: bool = False` — auto-reclaim graph-internal allocations
  on relaunch.
* `upload_stream: Stream | None = None` — auto-upload after `complete()`.
* `device_launch: bool = False` — graph can be launched from device code
  (incompatible with `auto_free_on_launch`).
* `use_node_priority: bool = False` — per-node priorities override the
  enclosing stream priority.

### 7.5 Node types

| Class | Purpose |
| --- | --- |
| `EmptyNode` | Pure sync barrier. |
| `KernelNode` | Kernel launch. `kernel`, `grid`, `block`, `shmem_size`, `config` properties; chainable mutators (`update_args`-style via re-build for now). |
| `AllocNode` / `FreeNode` | Stream-ordered alloc/free inside the graph. |
| `MemsetNode`, `MemcpyNode` | I/O between buffers. `dst`, `src`, `size` props. |
| `ChildGraphNode` | Embedded sub-graph. |
| `EventRecordNode` / `EventWaitNode` | Event sync at graph node granularity. |
| `HostCallbackNode` | Python callable / ctypes function pointer. |
| `ConditionalNode` and its subclasses (`IfNode`, `IfElseNode`, `WhileNode`, `SwitchNode`) | Control flow. |

### 7.6 Implications for the microbatch pipeline

* If Python launch overhead dominates (`launch()` + arg marshalling + Python
  attribute lookups ~ tens of µs per call), capture the whole per-microbatch
  fetch→decode→write subgraph once and replay. We get O(1)-Python-per-batch
  instead of O(N-kernels-per-batch)-Python.
* `Graph.update(builder)` makes it possible to keep one graph instance and
  rebind input/output pointers between batches as long as the topology is
  fixed.
* Host callbacks (HostCallbackNode) could in principle drive Python codec
  decode steps from inside the graph — but the "no CUDA API calls in
  callback" rule means we'd just be running CPU work; not useful for GDS.

---

## 8. Utilities

### 8.1 `StridedMemoryView` + `args_viewable_as_strided_memory`

Source: <https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.utils.StridedMemoryView.html>.

```python
@args_viewable_as_strided_memory((1,))
def encode(stream, src, dst):
    view = src.view(int(stream.handle))     # -> StridedMemoryView
    # view.ptr, view.shape, view.strides, view.dtype, view.device_id, view.is_device_accessible
```

* Constructor: `StridedMemoryView(obj=None, stream_ptr=None)` — auto-detects
  DLPack v1.0, then `__cuda_array_interface__` v3, then `__array_interface__`.
* Properties: `ptr`, `shape` (tuple), `strides` (tuple, in element counts),
  `dtype` (numpy dtype, supporting `ml_dtypes` narrow types), `device_id`
  (-1 = CPU), `is_device_accessible`, `readonly`.
* Classmethod constructors: `from_dlpack`, `from_cuda_array_interface`,
  `from_array_interface`, `from_any_interface` (PyTorch fast path).
* `copy_from(...)` / `copy_to(...)` / `as_tensor_map()` (Hopper TMA).
* `stream_ptr=-1` skips automatic stream sync — useful when we know the
  caller has already established ordering.

This is exactly the right primitive for codec kernels that need to be
**array-library-agnostic**: a single kernel wrapper that accepts cupy,
PyTorch, numpy, CzarrGpuBuffer interchangeably.

### 8.2 `prefetch_batch` / `discard_batch` / `discard_prefetch_batch`

```python
prefetch_batch(stream, buffers: Sequence[Buffer], locations: Device | Host | Sequence[...])
discard_batch(stream, buffers)
discard_prefetch_batch(stream, buffers, locations)
```

* **Managed memory only.** Not applicable to our VMR path.
* On CUDA 12 falls back to a Python loop calling `cuMemPrefetchAsync` per
  buffer; CUDA 13+ uses the native `cuMemPrefetchBatchAsync`.
* `discard_*` are **CUDA 13+ only**. API risk for older drivers.

### 8.3 `Host` — the CPU symmetry of `Device`

```python
Host()                       # any NUMA node
Host(numa_id=N)              # pin to node N
Host.numa_current()          # caller's NUMA node (per-thread singleton)
```

* Singletons keyed by args (`Host() is Host()`).
* Properties: `numa_id: int | None`, `is_numa_current: bool`.
* Use as the `locations=` arg to `prefetch_batch`, the
  `ManagedBuffer.preferred_location` setter, etc.

### 8.4 NVML — `cuda.core.system`

* `get_num_devices()`, `get_user_mode_driver_version()`,
  `get_kernel_mode_driver_version()`, `get_driver_branch()`,
  `get_nvml_version()`, `get_process_name(pid)`.
* `get_topology_common_ancestor(d1, d2, ...)`, `get_p2p_status(d1, d2)`.
* `register_events(events)` — start NVML-side event recording.
* `cuda.core.system.Device` (NVML wrapper, not to be confused with
  `cuda.core.Device`):
  * Properties: `memory_info`, `bar1_memory_info`, `uuid`, `pci_bus_id`,
    `name`, `index`, `utilization`, `temperature`, `performance_state`,
    `pci_info`, `numa_node_id`, `cuda_compute_capability`.
  * Methods: `get_clock()`, `get_fan()`, `get_cpu_affinity()`,
    `get_topology_nearest_gpus()`, `register_events()`, `to_cuda_device()`.
* `Device.to_system_device()` bridges from compute → NVML.
* Use cases for us: log GDR status, temperature, memory headroom, NUMA
  affinity (to inform `PinnedMemoryResource(numa_id=...)`).

### 8.5 `Context` and green contexts

* `Context` is *only* obtained via `Device.context` (primary) or
  `Device.create_context(options)`. No direct constructor.
* `is_green` (bool) — green contexts come from SM partitions or workqueue
  configs.
* `create_stream()` — **only on green contexts**. For primary contexts use
  `Device.create_stream()`.
* `resources` — hardware resource namespace.
* Green contexts give us a path to *partition SMs* — e.g. dedicate half the
  GPU to codec kernels and half to the user's compute. Useful for the
  "embedded inside a larger workload" use case.

### 8.6 SM / workqueue resources

* `Device.resources.sm` → `SMResource`. `.split()` it into groups with
  `SMResourceOptions(...)`. Properties: `sm_count`, `min_partition_size`,
  `coscheduled_alignment`, `handle`, `flags`.
* `Device.resources.workqueue` → `WorkqueueResource`. `.configure(options)`
  for sharing-scope settings.
* Both are advanced and orthogonal to the chunk-read pipeline; mentioned for
  completeness.

### 8.7 `TensorMapDescriptor` (Hopper TMA)

* Obtain via `StridedMemoryView.as_tensor_map()`.
* `device` property, `replace_address(tensor)` method.
* Pass directly to `launch()` as a kernel argument.
* **Docs say**: *"specialized `_from_*` helpers remain private while this API
  surface settles."* — flag for API risk.
* Relevance: a future cuTile-style decoder kernel for compressed Zarr chunks
  could use TMA bulk copy for the chunk → shared-memory move.

### 8.8 `cuda.core.checkpoint.Process`

* Constructor: `Process(pid: int)`.
* Page is a stub — no methods documented.
* Not in scope for the array path; just noting it exists.

### 8.9 `GraphicsResource`

* OpenGL interop only (no Vulkan in the high-level API yet).
* `from_gl_buffer`, `from_gl_image`, `map(stream)` → `Buffer`, `unmap`.
* Irrelevant for our array path.

---

## 9. Specific question answers

### Q1. Full Stream API — record, wait_event, sync, priority, capture?

* `record(event=None, options=None) -> Event` — yes.
* `wait(event_or_stream)` — yes; accepts any `__cuda_stream__` provider, so
  torch / cupy streams interop.
* `sync()` — yes; blocking.
* `priority` — read-only property; the value is set at creation via
  `StreamOptions(priority=N)` and **cannot be changed** later (driver API
  limitation).
* **Capture is NOT directly on Stream** — it lives on `GraphBuilder`
  (`begin_building` / `end_building`). You get a builder from
  `Stream.create_graph_builder()`.

### Q2. Full Event API — record, sync, elapsed_time, wait?

* Record: via `Stream.record(event=..., options=...)`.
* Sync: `Event.sync()` (blocking or busy-wait per `blocking_sync`).
* Wait on event: not a method on Event — call `stream.wait(event)`.
* Elapsed time: `event_late - event_early` returns milliseconds (Python
  `__sub__`) — both events must have `timing_enabled=True`.
* Non-blocking query: `Event.is_done` (the only poll knob in the API).

### Q3. Is there a CUDA Graph API? Capture + replay?

Yes. Two front-ends (`GraphBuilder` stream-capture; `GraphDefinition` explicit).
Pattern:

```python
gb = stream.create_graph_builder()
gb.begin_building()
# normal cuda.core ops with `gb` instead of a Stream
graph = gb.end_building().complete()      # exec graph
graph.upload(stream); graph.launch(stream)
graph.update(other_builder)               # cuGraphExecUpdate
```

* Conditional execution and loops are first-class via `if_then` / `if_else` /
  `while_loop` / `switch` (Hopper+ for the device-side variants).
* `Graph.update(...)` enables rebinding without re-instantiate as long as
  topology is identical.

### Q4. JIT compile from a string?

Yes, exactly:

```python
opts = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
prog = Program(my_lz4_kernel_src, code_type="c++", options=opts)
mod  = prog.compile("cubin",
                    name_expressions=("decode<u8>", "decode<u16>"),
                    cache=FileStreamProgramCache())
kernel = mod.get_kernel("decode<u8>")
launch(stream, LaunchConfig(grid=g, block=b), kernel, src_buf, dst_buf, cp.uint64(n))
```

* `code_type` is one of `"c++"`, `"ptx"`, `"nvvm"`. **There is no `"cu"` or
  `"fatbin"` source path** at this layer — use `ObjectCode.from_fatbin` to
  load pre-built fatbins.
* For LTO across multiple translation units use `compile("ltoir")` +
  `Linker(...).link("cubin")`.
* `FileStreamProgramCache` persists compiles across processes / runs.

### Q5. Full MemoryResource hierarchy?

See §5. Summary:

* `MemoryResource` (ABC).
* `DeviceMemoryResource` — pool-based device.
* `PinnedMemoryResource` — pool-based pinned host (modern, NUMA-aware).
* `ManagedMemoryResource` — pool-based unified.
* `LegacyPinnedMemoryResource` — sync `cuMemAllocHost`.
* `VirtualMemoryResource` — VMM, aligned, GDR-tagged.
* `GraphMemoryResource` — graph-scoped allocations.

(`SMResource` / `WorkqueueResource` are NOT MemoryResources — different
hierarchy under `Device.resources`.)

### Q6. `prefetch_batch` / `FileStreamProgramCache`?

* **Yes** to both, both stable in the API ref.
* `prefetch_batch(stream, buffers, locations)` — managed memory only. Falls
  back to a Python `for` loop calling `cuMemPrefetchAsync` on CUDA 12; uses
  `cuMemPrefetchBatchAsync` natively on CUDA 13+.
* `FileStreamProgramCache(path=..., max_size_bytes=...)` — XDG-default
  location, atomic-for-readers, LRU eviction, raw cubin / PTX storage.

---

## 10. API risk callouts (experimental / unstable)

* **`cuda.core.TensorMapDescriptor`** — "specialized `_from_*` helpers remain
  private while this API surface settles." Subject to change.
* **`cuda.core.checkpoint.Process`** — page is a stub; assume experimental.
* **`cuda.core.utils.discard_batch` / `discard_prefetch_batch`** — CUDA 13+
  only; older drivers raise.
* **`ProgramOptions.pch*`** — CUDA 12.8+ only.
* **`ProgramOptions` NVVM-only fields**: `extra_sources`, `use_libdevice`,
  `device_float128`, `fdevice_time_trace`, `frandom_seed`, `ofast_compile`.
* **`Linker` backend selection** — `nvJitLink` vs driver `cuLink` depending
  on availability; behaviour and supported targets differ slightly. Use
  `Linker.which_backend()` to verify in tests.
* **`Stream.from_handle`** — lifetime is **not** managed; foreign owner must
  outlive the wrapper. Be careful when adopting torch / cupy streams that
  might be re-pooled.
* **`VirtualMemoryResource`** — flagged as "low-level API… know how CUDA VMM
  works." Not experimental, but make sure tests cover granularity overflow.
* **`Buffer.close(stream=None)`** — falls back to a deallocation stream
  stored in the buffer's handle, which may not be the stream where the
  buffer was last used. Best practice: pass an explicit stream.
* **`DeviceMemoryResource` page references** `release_threshold` /
  `allocation_handle_type` as concepts but doesn't surface them in
  `DeviceMemoryResourceOptions` — so the high-level API is less expressive
  than the underlying `cuMemPool*`. To set those we'd have to drop to
  `cuda.bindings`.
* **`LegacyPinnedMemoryResource()`** signature in the docs is no-arg; our
  code calls `LegacyPinnedMemoryResource(dev)`. Both work today — either
  there's an undocumented private overload or the docs are stale. Worth
  testing on the next cuda-python release.
* **Host callbacks** in graphs: "must not call CUDA APIs" but the docs don't
  spell out the consequences. Treat as a hard rule.
* **No `Stream.is_done` / `query()`** at the high level — fall back to
  `Event.is_done` for non-blocking polling.

---

## 11. What we'd actually use for the CUDA-native Array path

The "would help us" list, ranked by importance:

* **`VirtualMemoryResource(addr_align=4096, gpu_direct_rdma=True)`** — already
  the load-bearing allocator for `CzarrGpuBuffer`. Keep using it; if the GDS
  / cuFile path settles on a single shared per-Array buffer we should also
  use `modify_allocation()` to grow without changing the base pointer.
* **`LegacyPinnedMemoryResource`** — already in use. For one-shot host
  bounce buffers (e.g. read-staging for very small chunks where cuFile is in
  compat mode). Consider switching to `PinnedMemoryResource(numa_id=…)` once
  we know we want NUMA pinning.
* **`Buffer.copy_to(dst, *, stream)` / `Buffer.copy_from`** — for H2D / D2H
  / D2D in the codec / fallback paths. Stream-ordered, no extra plumbing.
* **`Buffer.fill(value, *, stream)`** — for zero-filling chunk regions
  (e.g. `fill_value` semantics in Zarr v3) without a kernel launch.
* **`Stream` with `StreamOptions(nonblocking=True, priority=…)`** — one
  high-priority stream for cuFile reads, one normal-priority stream per
  codec worker, joined via `stream.wait(other.record())`.
* **`Event(timing_enabled=False, blocking_sync=True)`** — sync points
  between cuFile read / decode / consumer stages; `blocking_sync=True` so
  the Python orchestrator doesn't burn a core spinning.
* **`Event.is_done`** — the polling primitive for the orchestrator's main
  loop (decide whether to start the next microbatch).
* **`Program` + `ProgramOptions(std='c++17', arch=f'sm_{dev.arch}', link_time_optimization=…)`**
  — JIT-compile codec kernels (LZ4 decode, Blosc, ZFP, our own filters).
* **`Program.compile("cubin", name_expressions=(...), cache=FileStreamProgramCache())`**
  — instantiate templated codec kernels per chunk dtype, with persistent
  on-disk caching keyed by source + options. **This is the warm-start
  primitive — first import will cost the JIT, subsequent imports are loads.**
* **`Kernel.attributes` + `Kernel.occupancy`** — auto-tune block size for a
  given codec kernel at startup; avoid hard-coding `block=256`.
* **`LaunchConfig(grid, block, shmem_size)`** — codec kernel launch.
* **`launch(stream, config, kernel, *args)`** — single call site for both
  immediate launch and graph capture (because `launch` accepts a
  `GraphBuilder` in place of a `Stream`).
* **`StridedMemoryView` + `args_viewable_as_strided_memory((idx,))`** —
  array-library-agnostic codec kernel wrappers. Take a `CzarrGpuBuffer`,
  cupy array, or `Buffer` interchangeably; pull out `.ptr / .shape /
  .strides / .dtype` for the launch.
* **`Stream.create_graph_builder()` + `GraphBuilder.begin_building` /
  `end_building` / `complete`** — capture the per-microbatch
  fetch→decode→write pipeline once, replay per batch to amortize Python
  overhead.
* **`Graph.upload(stream)` then `Graph.launch(stream)`** — keep the
  pipeline warm; replay is ~free.
* **`Graph.update(builder)`** — rebind input/output pointers between
  batches without re-instantiating.
* **`GraphCompleteOptions(auto_free_on_launch=True)`** — only if we move
  intermediate buffers inside the graph (probably out of scope for v1).
* **`Device.properties.gpu_direct_rdma_supported` / `gpu_direct_rdma_writes_ordering`**
  — startup gate for the GDS path; bail to compat-mode if unavailable.
* **`Device.properties.multiprocessor_count` / `max_blocks_per_multiprocessor`**
  — for codec kernel grid sizing.
* **`cuda.core.system.Device(...).numa_node_id`** — feed into
  `PinnedMemoryResource(numa_id=...)` once we move off `LegacyPinnedMemoryResource`.
* **`cuda.core.system.get_p2p_status(d1, d2)`** — for multi-GPU read
  distribution.
* **`Linker(options=LinkerOptions(link_time_optimization=True))`** —
  if/when we ship pre-compiled LTO-IR codec stubs that users can override
  with their own decode functions (the `jit_lto_fractal.py` pattern,
  applied to a "user-provided filter" Zarr extension).

What we'd **explicitly NOT** use (for the array path, with reasons):

* `ManagedBuffer` / `ManagedMemoryResource` — defeats the GDS direct-IO
  story; managed memory pages are not pre-mapped to device.
* `GraphicsResource` — irrelevant.
* `cuda.core.checkpoint.Process` — too new / stub.
* `cuda.core.TensorMapDescriptor` — keep an eye on it for a future
  TMA-bulk-copy codec kernel, but don't depend on it for v1 (private
  `_from_*` API).
* `prefetch_batch` / `discard_batch` — managed-only.
* `cuda.core.Host.numa_current()` — fine for advice, but we should pin
  NUMA explicitly via `PinnedMemoryResource(numa_id=…)` rather than rely
  on "wherever the calling thread is".

---

## Appendix A1 — Full `DeviceProperties` attribute list (108)

`can_map_host_memory`, `can_use_host_pointer_for_registered_mem`,
`clock_rate`, `memory_clock_rate`, `single_to_double_precision_perf_ratio`,
`compute_capability_major`, `compute_capability_minor`, `compute_mode`,
`compute_preemption_supported`, `concurrent_kernels`,
`concurrent_managed_access`, `deferred_mapping_cuda_array_supported`,
`direct_managed_mem_access_from_host`, `ecc_enabled`,
`generic_compression_supported`, `global_l1_cache_supported`,
`local_l1_cache_supported`, `global_memory_bus_width`,
`gpu_direct_rdma_supported`, `gpu_direct_rdma_flush_writes_options`,
`gpu_direct_rdma_writes_ordering`, `gpu_overlap`,
`handle_type_posix_file_descriptor_supported`,
`handle_type_win32_handle_supported`, `handle_type_win32_kmt_handle_supported`,
`host_native_atomic_supported`, `integrated`, `kernel_exec_timeout`,
`l2_cache_size`, `max_persisting_l2_cache_size`, `managed_memory`,
`max_access_policy_window_size`, `max_block_dim_x`, `max_block_dim_y`,
`max_block_dim_z`, `max_blocks_per_multiprocessor`, `max_grid_dim_x`,
`max_grid_dim_y`, `max_grid_dim_z`, `max_pitch`, `max_registers_per_block`,
`max_registers_per_multiprocessor`, `max_shared_memory_per_block`,
`max_shared_memory_per_block_optin`, `max_shared_memory_per_multiprocessor`,
`max_threads_per_block`, `max_threads_per_multiprocessor`, `warp_size`,
`maximum_surface1d_width`, `maximum_surface1d_layered_layers`,
`maximum_surface1d_layered_width`, `maximum_surface2d_width`,
`maximum_surface2d_height`, `maximum_surface2d_layered_layers`,
`maximum_surface2d_layered_width`, `maximum_surface2d_layered_height`,
`maximum_surface3d_width`, `maximum_surface3d_height`,
`maximum_surface3d_depth`, `maximum_surfacecubemap_width`,
`maximum_surfacecubemap_layered_layers`, `maximum_surfacecubemap_layered_width`,
`maximum_texture1d_width`, `maximum_texture1d_linear_width`,
`maximum_texture1d_layered_layers`, `maximum_texture1d_layered_width`,
`maximum_texture1d_mipmapped_width`, `maximum_texture2d_width`,
`maximum_texture2d_height`, `maximum_texture2d_linear_width`,
`maximum_texture2d_linear_height`, `maximum_texture2d_linear_pitch`,
`maximum_texture2d_layered_layers`, `maximum_texture2d_layered_width`,
`maximum_texture2d_layered_height`, `maximum_texture2d_mipmapped_width`,
`maximum_texture2d_mipmapped_height`, `maximum_texture3d_width`,
`maximum_texture3d_height`, `maximum_texture3d_depth`,
`maximum_texture3d_width_alternate`, `maximum_texture3d_height_alternate`,
`maximum_texture3d_depth_alternate`, `maximum_texturecubemap_width`,
`maximum_texturecubemap_layered_layers`, `maximum_texturecubemap_layered_width`,
`texture_alignment`, `texture_pitch_alignment`, `memory_pools_supported`,
`mempool_supported_handle_types`, `multi_gpu_board`, `multi_gpu_board_group_id`,
`multicast_supported`, `multiprocessor_count`, `numa_config`, `numa_id`,
`pageable_memory_access`, `pageable_memory_access_uses_host_page_tables`,
`pci_bus_id`, `pci_device_id`, `pci_domain_id`,
`read_only_host_register_supported`, `reserved_shared_memory_per_block`,
`sparse_cuda_array_supported`, `tcc_driver`, `total_constant_memory`,
`unified_addressing`, `virtual_memory_management_supported`.

## Appendix A2 — Full `ProgramOptions` field list with defaults

```
name='default_program', arch=None,
relocatable_device_code=False, extensible_whole_program=False,
debug=False, lineinfo=False, device_code_optimize=None,
ptxas_options=None, max_register_count=None,
ftz=False, prec_sqrt=True, prec_div=True, fma=True,
use_fast_math=False, extra_device_vectorization=False,
link_time_optimization=False, gen_opt_lto=False,
define_macro=None, undefine_macro=None,
include_path=None, pre_include=None, no_source_include=False,
std='c++17', builtin_move_forward=True, builtin_initializer_list=True,
disable_warnings=False, restrict=False,
device_as_default_execution_space=False, device_int128=False,
optimization_info=None, no_display_error_number=False,
diag_error=None, diag_suppress=None, diag_warn=None,
brief_diagnostics=False, time=None, split_compile=1,
fdevice_syntax_only=False, minimal=False, no_cache=False,
fdevice_time_trace=None, device_float128=False,
frandom_seed=None, ofast_compile=None,
# CUDA 12.8+ PCH
pch=False, create_pch=None, use_pch=None, pch_dir=None,
pch_verbose=False, pch_messages=False,
instantiate_templates_in_pch=False,
# NVVM only
extra_sources=None, use_libdevice=False, numba_debug=None
```

## Appendix A3 — Full `LinkerOptions` defaults

```
name='<default linker>', arch=None, max_register_count=None,
time=False, verbose=False, link_time_optimization=False, ptx=False,
optimization_level=None, debug=False, lineinfo=False,
ftz=False, prec_div=True, prec_sqrt=True, fma=True,
kernels_used=None, variables_used=None,
optimize_unused_variables=False,
ptxas_options=None, split_compile=1, split_compile_extended=1,
no_cache=False
```

## Appendix A4 — Pattern: `Program.compile` with persistent cache

```python
from cuda.core import Device, Program, ProgramOptions
from cuda.core.utils import FileStreamProgramCache, make_program_cache_key

dev = Device(); dev.set_current()
cache = FileStreamProgramCache(max_size_bytes=128 * 1024 * 1024)

opts = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}",
                      use_fast_math=True,
                      link_time_optimization=False)
prog = Program(SRC, code_type="c++", options=opts)
mod  = prog.compile(
    "cubin",
    name_expressions=("lz4_decode<unsigned char>", "lz4_decode<unsigned short>"),
    cache=cache,
)
k_u8  = mod.get_kernel("lz4_decode<unsigned char>")
k_u16 = mod.get_kernel("lz4_decode<unsigned short>")
```

## Appendix A5 — Pattern: microbatch capture-replay

```python
stream = dev.create_stream(options=StreamOptions(nonblocking=True, priority=-1))
gb = stream.create_graph_builder()
gb.begin_building()

cufile_read(gb, fd, offsets, sizes, staging_buf)        # imagined wrapper
launch(gb, decode_cfg, k_decode, staging_buf, decoded_buf, n_chunks)
decoded_buf.copy_to(dst_buf, stream=gb)

graph = gb.end_building().complete(GraphCompleteOptions(upload_stream=stream))
# subsequent microbatches:
graph.launch(stream)
event = stream.record()
# orchestrator: while not event.is_done: schedule_more_work()
```

— end —
