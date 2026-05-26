# Two-tier codec architecture — nvCOMP + cuda-python native

> Design doc for the codec subsystem czarr ships in v0.1 and grows
> through subsequent phases. Inputs: the agent reports `01-cuda-core-api.md`
> through `07-api-design.md`, the LZ4 spike at
> [`spikes/lz4_decoder.py`](spikes/lz4_decoder.py), the
> [`native_lz4_stream_parallelism.md`](native_lz4_stream_parallelism.md)
> probe, the existing nvCOMP-wrapping codec set in
> [`src/czarr/codecs/`](../../../src/czarr/codecs/) on the `czarr-v0.1`
> branch, and the public discussion in `06-numcodecs-exploration.md`.
>
> Audience: the engineer who picks up the codec subsystem in Phase 1.
>
> Constraint: Python 3.12 idioms (PEP 695 type parameters, PEP 698
> `@override`, PEP 692 `Unpack[TypedDict]`); inline code, no
> pseudocode. The document is normative for the codec layer — when it
> diverges from `07-api-design.md` the diff is called out explicitly
> in section 1.

## 0. Why two tiers

The codec subsystem has to deliver three things at once:

1. **Coverage** — eight codec families ship on day one (Zstd, LZ4,
   Snappy, Deflate, GDeflate, Bitcomp, ANS, Cascaded), plus filters
   (Shuffle, Delta, FixedScaleOffset, BitRound) and checksums (CRC32C,
   Adler32, Fletcher32). nvCOMP gives us all of them today via
   `src/czarr/codecs/base.py:CudaBytesBytesCodec`.
2. **Performance and transparency for the codecs that matter most.**
   LZ4 is the numcodecs-compat compressor czarr's largest user
   population already writes with. Agent 3's spike beats nvCOMP 17.8×
   on A40 and clears 173 GiB/s on H200 — see
   `native_lz4_stream_parallelism.md`. Bypassing nvCOMP's per-call
   Python overhead is the single biggest decode-side win available.
3. **Permanence for the long tail.** Zstd is 6-9 engineer-months of
   research for a correct GPU decoder; Bitcomp / ANS / Cascaded are
   proprietary bitstreams with no published spec. Per
   [`03-nvcomp-alternatives.md`](03-nvcomp-alternatives.md), those four
   stay on nvCOMP forever.

The two-tier shape is the answer: a default tier (`backend="nvcomp"`)
that covers everything and a high-performance tier
(`backend="native"`) we grow codec by codec. Both tiers must produce
and consume the same on-disk bitstream for any codec where both exist,
so a write on one tier reads back on the other without surprise.

The rest of this document is the implementation contract.

---

## 1. Tier interface — the bridging abstraction

Three shapes are plausible. The chooser is at the end of this section.

### 1a. Backend-as-instance-field

The `_BackendAware` sketch from `07-api-design.md:248-310`. A frozen
dataclass field with a `Literal["native", "nvcomp"]` type carries the
runtime backend choice; the per-codec subclass dispatches on it.

```python
# czarr/codecs/_base.py

from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal, override

from zarr.abc.codec import BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer

type CodecBackend = Literal["native", "nvcomp"]


@dataclass(frozen=True, slots=True)
class _BackendAware(BytesBytesCodec):
    """Bytes-bytes codec that may route through nvCOMP or a native kernel."""

    is_fixed_size: ClassVar[bool] = False
    codec_name: ClassVar[str] = ""
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("nvcomp",)
    _native_default: ClassVar[bool] = False

    # Runtime-only field. Persisted? See section 2.
    backend: CodecBackend | None = field(default=None, compare=False, repr=True)

    def __post_init__(self) -> None:
        # Resolve None -> the configured default at construction time.
        from czarr.codecs._registry import resolve_default_backend

        chosen = self.backend or resolve_default_backend(
            self.codec_name,
            supported=self._supported_backends,
            native_preferred=self._native_default,
        )
        if chosen not in self._supported_backends:
            allowed = ", ".join(self._supported_backends)
            raise ValueError(
                f"codec={self.codec_name!r} backend={chosen!r} unsupported; allowed: {allowed!r}"
            )
        object.__setattr__(self, "backend", chosen)

    # Subclasses implement these halves; the dispatcher lives here.
    def _encode_native(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        raise NotImplementedError

    def _decode_native(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        raise NotImplementedError

    def _encode_nvcomp(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        raise NotImplementedError

    def _decode_nvcomp(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        raise NotImplementedError

    @override
    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        items_with_idx = [(i, c, s) for i, (c, s) in enumerate(chunks_and_specs) if c is not None]
        if not items_with_idx:
            return [None] * (len(items_with_idx))
        impl = self._encode_native if self.backend == "native" else self._encode_nvcomp
        results = impl([(c, s) for _, c, s in items_with_idx])
        return _splice_back(results, items_with_idx)

    @override
    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        items_with_idx = [(i, c, s) for i, (c, s) in enumerate(chunks_and_specs) if c is not None]
        if not items_with_idx:
            return [None] * len(items_with_idx)
        impl = self._decode_native if self.backend == "native" else self._decode_nvcomp
        results = impl([(c, s) for _, c, s in items_with_idx])
        return _splice_back(results, items_with_idx)
```

The dispatcher is sync-on-async — the `await` boundary stays
(zarr's `BatchedCodecPipeline` calls `await codec.decode(...)`), but
the actual codec work happens synchronously on a CUDA stream inside
the per-backend `_decode_native` / `_decode_nvcomp`. This matches the
existing nvCOMP wrapper at `src/czarr/codecs/base.py:327-341` where
`encode`/`decode` wrap `_batch_sync` in `asyncio.to_thread`.

For native codecs we drop `asyncio.to_thread` — see section 3.2.

**Runtime:** one class, one dataclass; the backend is a runtime field.

**Type checker:** `LZ4(backend="native")` and `LZ4(backend="nvcomp")`
are the same class. The type checker validates the `Literal[...]`
string but does not differentiate the instances. `isinstance(codec,
LZ4)` is true regardless of backend.

### 1b. Backend-as-type-parameter

PEP 695 generic; `LZ4[Native]` and `LZ4[Nvcomp]` are distinct types.

```python
type Native = Literal["native"]
type Nvcomp = Literal["nvcomp"]
type CodecBackend = Native | Nvcomp


class LZ4[B: CodecBackend = Native](_BackendAware):
    backend: B

    @overload
    def __init__(self: "LZ4[Native]", *, backend: Native = "native", **kw): ...
    @overload
    def __init__(self: "LZ4[Nvcomp]", *, backend: Nvcomp, **kw): ...
```

**Runtime:** same as 1a (Python erases the type parameter). The
constructor overload pair is the only mechanical bookkeeping.

**Type checker:** discriminates. `LZ4[Native]` is incompatible with
`LZ4[Nvcomp]` — you cannot accidentally pass one where the other is
expected. Useful only if a function genuinely wants to accept "any
native codec" and refuse the nvCOMP-backed variant. czarr's call
graph (Zarr v3 `CodecPipeline` consumes any `BytesBytesCodec`) does
not benefit from that distinction; the dispatch happens inside the
codec at decode time.

### 1c. Backend-as-separate-class with mixin

`LZ4Native` and `LZ4Nvcomp` are distinct concrete classes sharing a
`_LZ4Base` mixin holding the config fields.

```python
@dataclass(frozen=True, slots=True)
class _LZ4Base:
    codec_name: ClassVar[str] = "lz4"
    acceleration: int = 1


@dataclass(frozen=True, slots=True)
class LZ4Native(_LZ4Base, _BackendAware): ...


@dataclass(frozen=True, slots=True)
class LZ4Nvcomp(_LZ4Base, _BackendAware): ...
```

**Runtime:** two classes per codec. The Zarr v3 registry now needs to
know both — or, more practically, one canonical name maps to the
default class, and the alternate is constructed manually. Constructor
ergonomics suffer (`czarr.LZ4Native(acceleration=1)` vs
`czarr.LZ4(backend="native", acceleration=1)`).

**Type checker:** clean discrimination; ergonomics worse.

### 1d. Recommendation: backend-as-instance-field (1a)

Adopt 1a. The two-class flavour (1c) doubles the class surface for a
distinction the runtime already encodes; the generic flavour (1b)
adds overload pairs without buying anything our call graph asks for
(no function signature in czarr or zarr-python wants "native LZ4
specifically" — they want `BytesBytesCodec`). 1a matches the
`backend=` kwarg pattern in `07-api-design.md`, keeps a single class
per codec, and the runtime dispatch is a one-line `if` in
`_encode_impl` / `_decode_impl`.

**Where this differs from `07-api-design.md`:** the sketch there has
`_encode_impl` / `_decode_impl` as the subclass override points. We
split those into `_encode_native` / `_encode_nvcomp` /
`_decode_native` / `_decode_nvcomp` on `_BackendAware` itself, with
the runtime `if self.backend == "native"` dispatch in
`encode`/`decode`. This:

* Forces subclasses to declare both halves separately rather than
  hiding the dispatch in a single override (which makes
  `_supported_backends` more declarative).
* Centralises the `None`-input filtering in the base class — no
  per-codec boilerplate around the `chunk is None` skip Zarr v3
  passes for empty chunks.
* Leaves `_encode_native` / `_decode_native` as `NotImplementedError`
  by default, so a codec that does not have a native implementation
  silently raises a clear error if someone forces `backend="native"`.
  See section 5 for the fallback policy.

The other design tweak: `backend` is `field(compare=False)` so it
does not participate in `__eq__` (a codec with `backend="native"` is
still equal to the same codec with `backend="nvcomp"` — they encode
the same bitstream). This matches the metadata round-trip
requirement in section 2.

---

## 2. Registry and metadata round-trip

### 2.1 The bitstream is the spec; the backend is not

A reader on tier 1 must decode a stream written by tier 2 and vice
versa. The on-disk Zarr v3 metadata format
([`{"name": "lz4", "configuration": {...}}`](06-numcodecs-exploration.md))
carries only the *bitstream identity* — the codec_id and its tuning
parameters. The backend choice is a runtime implementation knob; it
does not belong in metadata.

```python
@dataclass(frozen=True, slots=True)
class LZ4(_BackendAware):
    codec_name: ClassVar[str] = "lz4"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("native", "nvcomp")
    _native_default: ClassVar[bool] = True

    acceleration: int = 1

    @override
    def to_dict(self) -> dict[str, JSON]:
        # Bitstream-identifying fields only. `backend` is runtime, not persisted.
        return {
            "name": self.codec_name,
            "configuration": {"acceleration": int(self.acceleration)},
        }

    @classmethod
    @override
    def from_dict(cls, data: dict[str, JSON]) -> "LZ4":
        cfg = dict(data.get("configuration", {}))
        # Defensive: a writer might have leaked `backend` into a persisted dict.
        # Strip it; the runtime backend comes from the override/default.
        cfg.pop("backend", None)
        return cls(**cfg)
```

The existing nvCOMP wrapper at `src/czarr/codecs/base.py:351-375`
already filters non-`compare` fields from `to_dict`; the equivalent
behaviour is reproduced here with the additional defensive strip in
`from_dict` to tolerate broken writers.

### 2.2 Default backend selection in `from_dict`

When the metadata says `{"name": "lz4", "configuration": {"acceleration": 1}}`
and we hand it to `LZ4.from_dict`, the resulting codec needs a
backend. The decision tree:

```python
# czarr/codecs/_registry.py

_BACKEND_OVERRIDES: dict[str, CodecBackend] = {}

def _set_overrides(overrides: dict[str, CodecBackend]) -> None:
    _BACKEND_OVERRIDES.clear()
    _BACKEND_OVERRIDES.update(overrides)


def resolve_default_backend(
    codec_name: str,
    *,
    supported: tuple[CodecBackend, ...],
    native_preferred: bool,
) -> CodecBackend:
    # 1. Process-global override wins. configure_gpu sets this.
    override = _BACKEND_OVERRIDES.get(codec_name)
    if override is not None:
        if override not in supported:
            raise ValueError(
                f"override codec_backend_overrides[{codec_name!r}]={override!r}; "
                f"codec only supports {supported!r}"
            )
        return override
    # 2. Per-codec class default: native if the codec opted in.
    if native_preferred and "native" in supported:
        return "native"
    # 3. Otherwise nvCOMP.
    return "nvcomp"
```

* **Per-instance `backend=` kwarg wins** over everything (handled in
  `__post_init__`).
* **Process-global override wins** over the class default. Set via
  `czarr.configure_gpu(codec_backend_overrides={"lz4": "nvcomp"})`.
* **Class default** picks the preferred tier — `native` if the codec
  has a native implementation flagged with `_native_default = True`.
* **Fallback** is always `nvcomp` if the per-codec class supports it.

This matches the open question 7 in `07-api-design.md:1012-1023` and
the override mechanism mooted at `07-api-design.md:580-583`.

### 2.3 Registry — one entry per codec

The Zarr v3 registry keys on the bitstream identity (`codec_name`).
Backend variants share that identity, so each codec class registers
**once**:

```python
# czarr/codecs/__init__.py

from zarr.registry import register_codec

# One class per codec; backend is a per-instance kwarg.
for _cls in (
    ANS, Bitcomp, Cascaded, Deflate, GDeflate, Snappy, Zstd, LZ4,
    Gzip, Zlib, Shuffle, Delta, FixedScaleOffset, BitRound,
    CRC32C, Adler32, Fletcher32,
):
    register_codec(_cls.codec_name, _cls)
```

Same shape as today's `src/czarr/codecs/__init__.py:37-53`. The native
backend is **never** a separate registry entry — that would imply two
codecs with the same bitstream but different on-disk identities,
which contradicts the Zarr v3 metadata model.

### 2.4 Cross-backend metadata round-trip

The acceptance test (see section 6):

```python
# encode with native; reload; backend defaults differ; decode with nvcomp
written = czarr.LZ4(backend="native", acceleration=1)
meta = written.to_dict()
assert meta == {"name": "lz4", "configuration": {"acceleration": 1}}

# Reader on the other backend pins:
read_through_nvcomp = czarr.LZ4.from_dict({"name": "lz4", "configuration": {"acceleration": 1}})
object.__setattr__(read_through_nvcomp, "backend", "nvcomp")  # hypothetical pin
# (in real code: czarr.configure_gpu(codec_backend_overrides={"lz4": "nvcomp"}))

# Both codecs must decode the same bitstream identically.
```

The lookup pipeline is:

```
zarr metadata -> {"name": "lz4", "configuration": {...}}
              -> zarr.registry.get_codec_class("lz4") -> czarr.LZ4
              -> czarr.LZ4.from_dict(meta) -> LZ4(acceleration=1, backend=<resolved>)
              -> codec.decode(...) -> dispatch by backend field
```

---

## 3. Codec implementation contract

The native backend's `_decode_native` / `_encode_native` is the new
API surface we introduce. The contract has to satisfy three users:

1. Zarr v3's `BatchedCodecPipeline.decode_batch` — passes
   `Iterable[tuple[Buffer | None, ArraySpec]]`.
2. czarr's own fast-path `_CudaArrayImpl` (eventual subsequent
   subtask) — wants to bind the codec to a specific stream.
3. The native kernel itself — wants the input as a device pointer +
   length tuple, the simpler the better.

The spike's `decode_lz4_blocks_gpu(compressed_blocks: list[bytes],
uncompressed_sizes: list[int]) -> list[bytes]` shape
([`spikes/lz4_decoder.py:245-283`](spikes/lz4_decoder.py)) gives us
the concrete starting point. We promote it to a typed contract.

### 3.1 The native-codec function signature

```python
# czarr/codecs/_native/_contract.py

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import cupy as cp
from cuda.core import Stream
from cuda.core.experimental._memoryview import StridedMemoryView

from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer


type DeviceInput = cp.ndarray | StridedMemoryView | Buffer
"""Anything the native kernel can pull a device pointer + length from.

In practice the kernel calls a helper (`_view_as_uint8`) that accepts
any of the three and returns a (ptr: int, size: int, stream: int)
triple. cupy ndarrays via `__cuda_array_interface__`; StridedMemoryView
via `.ptr` / `.bytesize` / `.stream_ptr`; Buffer via `as_array_like()`.
"""


@runtime_checkable
class NativeDecoder(Protocol):
    """Sync, stream-bound, batched decode for one codec."""

    def __call__(
        self,
        compressed: Sequence[DeviceInput],
        uncompressed_sizes: Sequence[int],
        *,
        stream: Stream,
        scratch: "ScratchAllocator | None" = None,
    ) -> list[cp.ndarray]:
        """Decode N compressed buffers into N cupy ndarrays of uint8.

        Returns ndarrays of dtype uint8 with the *exact* sizes given in
        `uncompressed_sizes`. Caller reinterprets via `.view(target_dtype)`
        or constructs a `CzarrGpuBuffer` around them.

        Raises:
            CodecDecodeError: on corrupted input.
            CodecScratchError: on scratch-allocator failure.
        """


@runtime_checkable
class NativeEncoder(Protocol):
    """Sync, stream-bound, batched encode for one codec."""

    def __call__(
        self,
        raw: Sequence[DeviceInput],
        *,
        stream: Stream,
        scratch: "ScratchAllocator | None" = None,
    ) -> list[cp.ndarray]:
        """Encode N uncompressed buffers into N cupy ndarrays of uint8."""
```

The contract differs from the spike's surface in five ways:

1. **`compressed: Sequence[DeviceInput]`** instead of
   `list[bytes]`. The spike accepts host bytes and uploads internally
   — fine for a benchmark, wrong for production. The codec wrapper
   that consumes Zarr v3's `Buffer` argument is responsible for
   making sure the input is a device buffer (or coercing host
   buffers via `cupy.asarray`) *before* the kernel sees it.
2. **`uncompressed_sizes` becomes a parameter**, not an argument the
   kernel infers. For LZ4 with `WITH_UNCOMPRESSED_SIZE` framing the
   size is encoded in the first 4 bytes of each chunk; the wrapper
   strips it before calling the kernel. For nvCOMP's RAW LZ4 there's
   no size prefix. The wrapper handles both — see section 3.4.
3. **`stream: Stream`** is required, not derived. czarr's fast path
   binds the codec to one of the streams in `pipeline.streams.StreamPool`;
   the codec must run on that stream so subsequent kernels (filters,
   the user's compute) can chain on the same handle.
4. **`scratch: ScratchAllocator | None`** is a hook for the kernel
   to ask the codec layer for device memory without owning an
   allocator. See section 4.3.
5. **Returns `list[cp.ndarray]`** of uint8. The codec wrapper at
   `_BackendAware` level wraps these in the spec's prototype buffer
   (`spec.prototype.buffer.from_array_like(...)`), so the dual
   GPU-buffer / host-buffer paths in
   `src/czarr/codecs/base.py:289-310` stay in one place.

### 3.2 Sync vs async

The existing nvCOMP codecs wrap `_batch_sync` in `asyncio.to_thread`
(`src/czarr/codecs/base.py:332-341`) so that the codec call does not
block the asyncio loop while nvCOMP serialises in C++. For native
codecs:

* The kernel launch is **non-blocking on the host** — `cp.RawKernel`
  enqueues onto the stream and returns immediately.
* The only host-blocking call would be a `stream.sync()` to wait for
  completion. In the fast path we **never sync** inside the codec; we
  let downstream kernels chain on the same stream. The
  `BatchedCodecPipeline` does a stream sync at the end of
  `read_batch` regardless.
* `asyncio.to_thread` therefore adds latency without benefit. The
  thread offload is pure overhead for the native path.

The recommended shape on `_BackendAware.decode`:

```python
@override
async def decode(self, chunks_and_specs):
    items = [(c, s) for c, s in chunks_and_specs if c is not None]
    if not items:
        return []
    if self.backend == "native":
        # Native path: kernel launches don't block; no thread offload.
        return self._decode_native_wrapped(items)
    # nvCOMP path: keep the thread offload — nvCOMP does block.
    return await asyncio.to_thread(self._decode_nvcomp_wrapped, items)
```

This means native codecs run on the *same* thread as the event loop.
For an `await codec.decode(...)` in `read_batch` that's fine — the
work is queued onto the GPU and the host thread returns ~instantly.
The Phase 1 `_CudaArrayImpl` queue producer/consumer pattern is
independent of asyncio anyway; it drives the loop synchronously via
`zarr.core.sync.sync` as `src/czarr/array/_impl.py:85` already does.

### 3.3 Input/output types

The native kernel sees device-side data. The wrapper bridges:

* **Buffer (Zarr v3 prototype)**: when called via Zarr's pipeline,
  the codec receives `Buffer` instances. Path A — GPU prototype: the
  buffer's `as_array_like()` returns a `cupy.ndarray` view already
  on device. Path B — host prototype: `.to_bytes()` lands a host
  bytes object; upload to device via `cupy.asarray`. Both cases
  exist already in `src/czarr/codecs/base.py:290-300`; the native
  wrapper takes the same two branches.
* **StridedMemoryView**: for codec consumers that drive the codec
  directly without going through Zarr v3 (e.g. the
  `_CudaArrayImpl.retrieve_gpu` fast path eventually consuming an
  encoded slab). Not required for v0.1 — Zarr's `Buffer` is the only
  caller — but the kernel function accepts it for symmetry with
  `cuda.core`'s preferred buffer abstraction (per
  `01-cuda-core-api.md`).
* **Raw `cp.ndarray`**: hot path. The wrapper unwraps the buffer
  argument to a cupy view and the kernel runs on it directly. The
  size and offset arrays go to device as `cupy.asarray(...)` of
  int64.

Output is always `cp.ndarray` of dtype uint8. The wrapper reshapes
or views it according to the spec.

### 3.4 Stream handling

nvCOMP codecs in `src/czarr/codecs/base.py:138-145` accept a
`cuda_stream` field at construction:

```python
cuda_stream: Any | None = field(default=None, compare=False, repr=False)
```

That's a per-instance binding — the `nvcomp.Codec` object lives in a
thread-local cache (`base.py:196-202`) and runs on the stream the
codec was constructed with. The native codec **must not** bind a
stream at construction. The native kernel is a stateless
`cupy.RawKernel`; launching it on a different stream every call is
free.

The reconciliation: keep `cuda_stream` on `_BackendAware` as a
*default* stream for the codec when called without one, and accept a
per-call `stream=` override that the dispatcher threads through to
the native function. The nvCOMP path ignores the per-call stream
(nvCOMP cannot retarget on the fly) and falls back to the codec's
construction stream. This is asymmetric but mirrors the underlying
asymmetry of the two backends.

```python
def _decode_native_wrapped(
    self,
    items: list[tuple[Buffer, ArraySpec]],
    *,
    stream: Stream | None = None,
) -> list[Buffer]:
    from czarr.codecs._native.lz4 import decode_lz4_native

    target_stream = stream or self._resolve_default_stream()
    cp_inputs, sizes = self._unwrap_inputs(items, framing=self._native_framing)
    cp_outs = decode_lz4_native(cp_inputs, sizes, stream=target_stream)
    return [spec.prototype.buffer.from_array_like(o) for o, (_, spec) in zip(cp_outs, items)]
```

### 3.5 Bitstream framing — where the LZ4 4-byte prefix lives

LZ4 has two on-disk forms in our world:

* **`WITH_UNCOMPRESSED_SIZE`** (the numcodecs `LZ4` format): 4 bytes
  little-endian uncompressed size followed by the LZ4 block payload.
  This is what czarr's `LZ4` produces today via
  `_BitstreamKind.WITH_UNCOMPRESSED_SIZE` at
  `src/czarr/codecs/compressors/lz4.py:20`.
* **RAW LZ4 block**: no prefix; the uncompressed size is conveyed
  out-of-band (e.g. the chunk's expected size from the spec).

The spike's kernel
([`spikes/lz4_decoder.py:130-235`](spikes/lz4_decoder.py)) decodes
**raw blocks only** — the wrapper at
[`spikes/lz4_decoder.py:255-283`](spikes/lz4_decoder.py) accepts
`compressed_blocks: list[bytes]` and `uncompressed_sizes: list[int]`
explicitly. The size prefix is the wrapper's responsibility.

In production czarr, the framing lives in the **codec wrapper layer**
— the `LZ4` dataclass's `_decode_native` method — not in the kernel
and not in `_BackendAware`:

```python
@dataclass(frozen=True, slots=True)
class LZ4(_BackendAware):
    codec_name: ClassVar[str] = "lz4"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("native", "nvcomp")
    _native_default: ClassVar[bool] = True

    acceleration: int = 1

    @override
    def _decode_native(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        from czarr.codecs._native.lz4 import decode_lz4_native

        cp_inputs: list[cp.ndarray] = []
        sizes: list[int] = []
        for chunk, spec in items:
            cp_arr = _buffer_to_cupy(chunk)
            # Strip the 4-byte LE uncompressed-size prefix.
            uncompressed = int(cp_arr[:4].view(cp.uint32).item())
            cp_inputs.append(cp_arr[4:])
            sizes.append(uncompressed)
        cp_outs = decode_lz4_native(cp_inputs, sizes, stream=self._resolve_stream())
        return [
            spec.prototype.buffer.from_array_like(o)
            for o, (_, spec) in zip(cp_outs, items, strict=True)
        ]

    @override
    def _decode_nvcomp(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        # The existing nvCOMP path — calls into `_batch_sync` with
        # `_bitstream_kind=WITH_UNCOMPRESSED_SIZE` so nvCOMP parses the prefix.
        return _nvcomp_decode_batch(items, algorithm="LZ4", bitstream_kind="WITH_UNCOMPRESSED_SIZE")
```

The two backends see different views of the same bytestream:

* Native: strip the prefix host-side (cheap — one 4-byte view + an
  ndarray slice on the GPU), pass the raw block to the kernel.
* nvCOMP: pass the whole thing including the prefix; nvCOMP's
  `WITH_UNCOMPRESSED_SIZE` mode parses it internally.

Encoding is the inverse (see section 4.1).

`_buffer_to_cupy` is the existing branch that picks
`as_array_like()` for GPU prototypes vs `cupy.asarray(.to_bytes())`
for host prototypes. Pulled out of the nvCOMP wrapper in section
8.3.

---

## 4. Native-LZ4 production gaps

The spike is decode-only and intentionally minimal. Productionising
for `czarr.LZ4(backend="native")` needs the following.

### 4.1 Encoder strategy

The spike has no encoder. Two options to ship `backend="native"`
end-to-end:

**Option A — symmetric native**: write a CUDA LZ4 encoder. Cost per
`03-nvcomp-alternatives.md:142-146`: 2 engineer-weeks, ~400 lines of
kernel + hash table. Nice property: the codec is fully self-contained
in czarr; no nvCOMP dependency for LZ4.

**Option B — asymmetric**: native decode, nvCOMP encode. The encoder
already exists at `src/czarr/codecs/base.py:313-325`. Saves the 2
weeks. Cost: czarr still depends on nvCOMP for LZ4 writes, but
writes are a fraction of the workload (per `SUMMARY.md:18-20`) and
the wire format is bit-identical to numcodecs.LZ4 regardless of
which side produced it.

**Recommendation for v0.1: Option B.** Native decode (where the win
is); nvCOMP encode (where the cost is). The codec dataclass:

```python
class LZ4(_BackendAware):
    @override
    def _encode_native(self, items):
        # Encode falls back to nvCOMP for v0.1. Documented in the
        # docstring; revisit in v0.2 if a real user asks.
        return self._encode_nvcomp(items)
```

Future Option A is one PR adding a CUDA encode kernel and switching
the body — no surface change. Document the asymmetry in the
docstring so users do not assume "native" means "no nvCOMP
dependency" today.

### 4.2 Robustness gaps in the spike kernel

The spike skips bounds checks in non-critical paths
(`spikes/lz4_decoder.py:148` and the elided literal copy bounds
check). Production needs:

* **Bounds checks on every load** — the kernel comment at line 28-29
  acknowledges this. Add `src_end` / `dst_end` guards to every
  `compressed[sp + i]` and `decompressed[dp + i]` access. The
  performance penalty is roughly a 4-byte-aligned predicated load;
  in practice <5% on the 173 GiB/s headline number, well within the
  17.8× margin over nvCOMP.
* **Status output** — the kernel already writes `status[block_id]`
  on corruption (line 232-234). The wrapper checks
  `(status != 0).any()` and raises (spike line 277-280); production
  uses a dedicated `CodecDecodeError` carrying the bad block index +
  a hex dump of the first 32 bytes.
* **Empty-input edge case** — `numcodecs.LZ4.encode(b"")` returns
  one byte (the literal-block header). Verify the kernel handles a
  1-byte input with `uncompressed_size=0` correctly; if not, special-
  case at the wrapper level.
* **The RLE overlap path** (line 217-228) is byte-by-byte on lane 0,
  which is correct but ~32× slower than the offset-≥-32 fast path.
  Acceptable for now — RLE-heavy data is rare in scientific arrays
  — but worth measuring on the BitRound + Shuffle pipeline output
  where small repeats appear.

### 4.3 Scratch memory

The spike allocates output buffers via `cp.empty(int(dst_offs[-1]),
dtype=cp.uint8)` and offset arrays via `cp.asarray(...)` of int64
(lines 270-272). For production:

* **Output buffer**: callers allocate. The codec wrapper receives
  the spec, computes the expected size via
  `_expected_decoded_bytes(spec)` (already on
  `CudaBytesBytesCodec`), and creates one `cp.empty(...)` per chunk
  before the kernel runs.
* **Offset arrays**: device-side int64 buffers of length N+1.
  Allocate via cupy (default RMM-backed when
  `czarr.configure_gpu(rmm_pool_gb=...)` is set; see
  `src/czarr/__init__.py:125-126`). Per-call allocation is fine —
  the arrays are <1 KiB for the batch sizes we see.
* **Kernel scratch**: the spike has none beyond the offset arrays.
  When we add the shared-memory prefetcher
  (`03-nvcomp-alternatives.md:140-142` mentions ~150 lines mirroring
  nvCOMP 2.2 `BufferControl`), the kernel will need
  `__shared__` memory configured at launch. cupy's `RawKernel.__call__`
  accepts a `shared_mem` kwarg; sized at 8 KiB per block (covers
  one LZ4 block's worst-case prefetch window).

A future `ScratchAllocator` Protocol gives advanced callers a hook
to pass a pinned-pool-backed scratch buffer:

```python
@runtime_checkable
class ScratchAllocator(Protocol):
    def allocate(self, size: int, *, stream: Stream) -> cp.ndarray: ...
    def deallocate(self, buf: cp.ndarray, *, stream: Stream) -> None: ...
```

For v0.1 the default is `None` → cupy/RMM allocator. The hook lets
the eventual `_CudaArrayImpl` pass a pool-backed allocator without
the codec needing to know about it.

### 4.4 Multi-chunk batching

The spike's launch shape (`grid=(n_blocks,)`, `block=(32,)`) is
already the production shape per the H200 probe
(`native_lz4_stream_parallelism.md`): one warp per LZ4 block,
~173 GiB/s with a single grid launch over 1024 × 64 KiB = 64 MiB.

The probe shows the kernel scales linearly to 1024 blocks per
launch with no measurable degradation. Practical safe batch ceiling
on H200:

* GPU SM count: 132 (H200 SXM).
* Resident warps per SM (assuming 32 regs/thread, no shared mem):
  ~64.
* Warps per kernel = 1 per block → ~8000 concurrent warps maxes
  occupancy.
* Block count = up to ~8000 in practice; beyond that, the scheduler
  serialises and the throughput plateau holds.

The codec wrapper has no need to cap; chunks arrive from
`BatchedCodecPipeline` already batched (czarr sets the batch size to
`sys.maxsize` in `configure_gpu`). The kernel handles them all in
one launch.

### 4.5 JIT cache

The spike calls `cp.RawKernel(...)` on every call to `_get_kernel()`
(line 240-242) — implicitly cached by cupy at the source-string level,
but no on-disk persistence.

Production uses `cuda.core.utils.FileStreamProgramCache` (per
`01-cuda-core-api.md` §6.9 and the cuda-core docs). The cache lives
at `$XDG_CACHE_HOME/czarr/kernels/`; first import compiles
LZ4 + future codecs into it; subsequent process starts read PTX
straight from disk.

```python
# czarr/codecs/_native/_kernel_cache.py

from pathlib import Path
import os

from cuda.core.experimental import Program, ProgramOptions
from cuda.core.experimental.utils import FileStreamProgramCache


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "czarr" / "kernels"


_CACHE = FileStreamProgramCache(_cache_dir())


def get_kernel(name: str, source: str, options: ProgramOptions) -> "Kernel":
    prog = _CACHE.get(name=name, code=source, options=options)
    if prog is None:
        prog = Program(source, "c++", options=options)
        prog.compile("cubin")
        _CACHE.put(name=name, code=source, options=options, program=prog)
    return prog.get_kernel(name)
```

The Program is compiled per (source, options) pair; options include
the `arch=...` flag, so a multi-arch deployment generates per-arch
entries automatically.

---

## 5. Fallback strategy

Three failure modes; three behaviours.

### 5.1 No native implementation for this codec

`czarr.Zstd(backend="native")` → the class's
`_supported_backends = ("nvcomp",)` excludes `"native"`, so the
constructor's validation in `__post_init__` raises immediately:

```
ValueError: codec='zstd' backend='native' unsupported; allowed: ('nvcomp',)
```

This is the *only* loud-fail in the matrix. Silent fallback to
nvCOMP here would hide a user mistake — they explicitly asked for
native because they wanted the perf characteristic; giving them
nvCOMP silently means they cannot find out why their bench numbers
do not match the LZ4 spike's.

**Per-codec class default** for codecs without a native impl is
always `"nvcomp"` (`_native_default = False` on the class).
`czarr.Zstd()` with no kwarg picks nvCOMP without warning — that's
the intended default.

### 5.2 Native impl exists but fails at runtime

Three sub-cases:

**Corrupted bitstream** — kernel writes nonzero to `status[block_id]`.
The wrapper raises `CodecDecodeError` carrying the block index and a
hex dump. Do **not** retry under nvCOMP — corrupted data is
corrupted data; both decoders should fail. If they don't, we have a
deeper bug to find.

**Output buffer too small** — never happens in czarr because the
output size comes from `spec`, which is the spec's invariant. If it
does happen (caller bug or codec-config mismatch), it's a `ValueError`
in the wrapper.

**Hardware doesn't support the kernel** — sm_<70 might lack some
warp intrinsics; sm_<60 lacks shuffle entirely. Native LZ4 uses
`__shfl_sync` which is sm_70+. Detection at codec construction:

```python
def __post_init__(self) -> None:
    super().__post_init__()
    if self.backend == "native":
        cc = cp.cuda.Device().compute_capability  # ("7", "0") -> "70"
        if int(cc[0] + cc[1]) < 70:
            # Force-fall-back; warn so the user knows.
            import warnings
            warnings.warn(
                f"native LZ4 requires sm_70+; got sm_{cc[0]}{cc[1]}. "
                f"Falling back to nvcomp. Pass backend='nvcomp' to silence.",
                stacklevel=2,
            )
            object.__setattr__(self, "backend", "nvcomp")
```

The fall-back is **silent except for the warning**. This is the
opposite policy from 5.1: here the user did not pick "native" with
intent; the runtime is degraded. Falling forward to nvCOMP is the
charitable interpretation, and the warning gives them an out.

### 5.3 The codec class chose "native" by default but native is broken

Edge case: `czarr.LZ4()` (no kwarg) picks `"native"` because
`_native_default = True`. The native kernel then segfaults. Two
mitigations:

* **Static gate**: pre-flight the kernel at first import. On import,
  run a one-block decode through `decode_lz4_native` against a
  fixture. If it fails, set a module-level flag that
  `resolve_default_backend` reads as "native broken for this codec
  on this host" and silently defaults to nvCOMP. This is the
  paranoid path.
* **No gate**: trust the kernel. If it fails for a real user, treat
  it as a real bug and fix it.

**Recommendation for v0.1: no gate.** The kernel is small enough to
audit; the spike already runs in CI. Add the gate only if a
real-world segfault appears.

---

## 6. Test strategy

Five categories, three priority tiers.

### 6.1 Bitstream-compat fixtures (P0)

The pattern from `06-numcodecs-exploration.md:751-797`:

```
tests/fixture/lz4/
    array.00.npy             # np.random rounds-trip-safe float32
    array.01.npy             # all-zeros (RLE-heavy)
    array.02.npy             # ascending uint32 (delta-friendly)
    array.03.npy             # uniform random bytes (~incompressible)
    codec.00/
        config.json          # {"id": "lz4", "acceleration": 1}
        encoded.00.dat       # numcodecs.LZ4().encode(array.00)
        encoded.01.dat
        encoded.02.dat
        encoded.03.dat
    codec.01/
        config.json          # {"id": "lz4", "acceleration": 12}
        ...
```

A `scripts/regen_fixtures.py` (CPU-only; uses numcodecs to write the
oracle) regenerates the dir. Fixtures committed to git. The test:

```python
@pytest.mark.parametrize("backend", ["native", "nvcomp"])
@pytest.mark.parametrize("arr_path", FIXTURE.glob("array.*.npy"))
def test_lz4_decode_against_numcodecs_oracle(backend, arr_path):
    arr = np.load(arr_path)
    for codec_dir in sorted(FIXTURE.glob("codec.*")):
        cfg = json.loads((codec_dir / "config.json").read_text())
        encoded_path = codec_dir / f"encoded.{arr_path.stem.split('.')[-1]}.dat"
        encoded = encoded_path.read_bytes()

        codec = czarr.LZ4(backend=backend, acceleration=cfg["acceleration"])
        decoded = _drive_single_chunk(codec, encoded, expected_dtype=arr.dtype, expected_shape=arr.shape)
        np.testing.assert_array_equal(cp.asnumpy(decoded), arr)
```

This runs across all combinations (4 fixtures × 2 backends × N
codec configs) — typically <16 tests per codec. Fast (single chunk,
single decode call); the GPU side adds <1 ms each.

### 6.2 Cross-backend round-trip (P0)

For every codec where both backends exist, encode with one, decode
with the other, byte-compare the decoded array. Differs from 6.1 in
that we test the *czarr-to-czarr* path, not the *numcodecs-to-czarr*
path.

```python
@pytest.mark.parametrize("write_be,read_be", [("native", "nvcomp"), ("nvcomp", "native")])
def test_lz4_cross_backend_round_trip(write_be, read_be):
    arr = np.random.default_rng(0).bytes(65536)
    write_codec = czarr.LZ4(backend=write_be)
    read_codec = czarr.LZ4(backend=read_be)
    encoded = _drive_single_chunk_encode(write_codec, arr)
    decoded = _drive_single_chunk_decode(read_codec, encoded, expected_size=len(arr))
    assert bytes(decoded) == arr
```

Note that for LZ4 in v0.1, both `write_be` flavours hit nvCOMP
encode (per section 4.1 Option B), so the test is effectively
"nvCOMP encode -> {native, nvCOMP} decode" — which is exactly the
useful case. When the native encoder lands, the test starts
exercising native encode without any change.

### 6.3 Pure-Python oracle for the kernel (P1)

The spike's `_decode_lz4_block_cpu`
([`spikes/lz4_decoder.py:70-112`](spikes/lz4_decoder.py)) is the
algorithm-level oracle. Promote it to `czarr/codecs/_native/_oracle.py`
and use it in two ways:

* **As a pre-flight self-test** at first call (off in production via
  a flag).
* **As a slow-path test target**. For random fuzz inputs, run both
  the CPU oracle and the GPU kernel; assert byte equality.

```python
@hypothesis.given(st.binary(min_size=0, max_size=65536))
def test_native_lz4_oracle_matches_kernel(raw):
    encoded = numcodecs.LZ4().encode(raw)
    cpu_decoded = _decode_lz4_block_cpu(encoded[4:], len(raw))  # strip 4B prefix
    gpu_decoded = decode_lz4_native([cp.asarray(encoded[4:])], [len(raw)], stream=...)
    assert bytes(gpu_decoded[0].get()) == cpu_decoded == raw
```

This catches kernel divergences from spec without needing
numcodecs round-trips on every fuzz iteration (the oracle is the
spec).

### 6.4 Performance regression bench (P1)

A small bench at `bench/codecs/native_vs_nvcomp.py` that records:

* Native LZ4 decode throughput on `[64, 1024, 16384] × [16 KiB, 64 KiB]`
  block configurations.
* nvCOMP LZ4 decode throughput on the same.
* Ratio (native / nvcomp).

A regression threshold in CI: fail if native throughput drops below
50 GiB/s on H200 for the 1024 × 64 KiB regime. The probe established
~173 GiB/s; 50 is a comfortable lower bound that catches kernel
regressions without false-positives from machine variance.

The bench file mirrors the existing
`bench/experiments/native_lz4_stream_parallelism.py` shape — the
data points populate a small Markdown table the CI artifact ships
with every PR.

### 6.5 Filter bit-exactness (P2)

For filters (Shuffle, Delta, FixedScaleOffset, BitRound), bit-equality
is the contract — see `06-numcodecs-exploration.md:937-944`. Once
those move to `cuda.compute` in Phase 2, the same fixture pattern
applies but with `assert_equal` on the byte payload, not just the
decoded array. The BitRound round-to-even tiebreaker bug surfaced in
`06-numcodecs-exploration.md:548-568` is a test case here.

---

## 7. Iterative addition path

Per `03-nvcomp-alternatives.md`'s cost matrix, the queue for adding
native codecs:

| codec | weeks | priority | path |
|---|---:|---|---|
| LZ4 decode | 1.5 | v0.1 | spike productionisation |
| Snappy decode | 2.0 | post-v0.1 | port cuDF `unsnap.cu` |
| Deflate decode (Gzip/Zlib) | 3.5 | post-v0.1 | fork cuDF `gpuinflate.cu` |
| Filters (CCCL Shuffle/Delta/FSO/BitRound) | 2.0 | Phase 2 | `cuda.compute` |
| LZ4 encode | 2.0 | post-v0.1 | CUDA kernel from spec |
| GDeflate | 2.5 | gated on user demand | port DirectStorage HLSL |
| Snappy encode | 2.0 | gated on user demand | port cuDF `snap.cu` |
| Zstd, Bitcomp, ANS, Cascaded | — | never | proprietary or 6-9 mo work |

### 7.1 The minimum viable PR shape

For each codec added to the native tier:

1. **One file at `src/czarr/codecs/_native/<codec>.py`** containing:
   * The kernel source string.
   * The Python wrapper function (the `NativeDecoder` /
     `NativeEncoder` contract).
   * A docstring linking to the spec and the spike file.
2. **One bitstream-compat fixture suite** generated by extending
   `scripts/regen_fixtures.py`.
3. **One bench entry** in `bench/codecs/native_vs_nvcomp.py` and a
   regression threshold in CI.
4. **One toggle** on the codec class: add `"native"` to
   `_supported_backends` and (optionally) flip `_native_default` to
   `True`.
5. **One docstring update** describing what's native vs nvCOMP.

Total per codec: ~5 file diffs, one new file. The
`_BackendAware` machinery is reused; the per-codec class gains a
~30-line `_decode_native` method.

### 7.2 Kernel directory layout

```
src/czarr/codecs/_native/
    __init__.py            # exports the decoder fns; not the kernel sources
    _common.py             # shared helpers — see 7.3
    _oracle.py             # pure-Python reference decoders (test/dev only)
    _kernel_cache.py       # FileStreamProgramCache wrapper
    lz4.py                 # kernel + Python wrapper
    snappy.py              # later
    deflate.py             # later — possibly compiles cuDF C++ via ProgramOptions cpp_std="c++17"
```

The kernels live as Python string constants inside the `.py` files
for v0.1 — the spike's `LZ4_DECODE_KERNEL_SRC` shape transfers
directly. This keeps the single-file-per-codec contract and avoids
build-system complexity. When kernel size exceeds ~500 lines (likely
for deflate), promote the source to a sibling `.cu` file loaded via
`importlib.resources` at import; the kernel cache treats them
identically.

### 7.3 Shared infrastructure — `_common.py`

```python
# src/czarr/codecs/_native/_common.py

from collections.abc import Sequence
from typing import Any

import cupy as cp
from cuda.core import Stream

from zarr.core.buffer import Buffer
from zarr.core.buffer import gpu as gpu_buffer


def buffer_to_cupy(buf: Buffer) -> cp.ndarray:
    """Pull a uint8 cupy view out of any Zarr v3 Buffer.

    GPU prototype: returns the underlying ndarray view (no copy).
    Host prototype: uploads to device via cupy.asarray (one H2D copy).
    """
    if isinstance(buf, gpu_buffer.Buffer):
        return buf.as_array_like().view(cp.uint8)
    return cp.asarray(buf.to_bytes(), dtype=cp.uint8)


def build_offsets(sizes: Sequence[int]) -> cp.ndarray:
    """Prefix-sum sizes into an int64 device array of length len(sizes)+1."""
    arr = cp.zeros(len(sizes) + 1, dtype=cp.int64)
    arr[1:] = cp.asarray(sizes, dtype=cp.int64).cumsum()
    return arr


def assert_supported_cc(min_cc: int) -> None:
    """Raise if the active device's compute capability is below `min_cc`."""
    cc = cp.cuda.Device().compute_capability  # "70", "75", "80", ...
    if int(cc) < min_cc:
        raise RuntimeError(f"native codec requires sm_{min_cc}+; active device is sm_{cc}")


def stream_handle(stream: Stream) -> int:
    """Coerce a cuda.core.Stream to a raw cudaStream_t int.

    Mirrors `CudaBytesBytesCodec._resolve_stream` on the nvCOMP side but
    is the symmetric helper for native kernels — accepts a Stream
    directly without the cuda_stream-field overhead.
    """
    return int(stream.handle)
```

These three helpers cover the common patterns. As more native codecs
land we add: vectorised alignment helpers (`align_to_4` etc.), a
shared scratch-pool wrapper, the `assert_supported_cc` per-codec
threshold.

### 7.4 Kernel cache layout

`_kernel_cache.py` (already sketched in section 4.5) lives once at
the package level. Every native codec module gets its kernel via
`get_kernel(name=..., source=..., options=...)`. The cache is
content-addressed on (source, options) so a code change invalidates
automatically.

For multi-arch deployment, the options include `arch=` set from
`Device().compute_capability` at import time. First import on each
distinct arch triggers a one-time compile; subsequent imports hit
the disk cache.

---

## 8. Implementation details

### 8.1 Pre-compiled PTX vs runtime JIT

For v0.1, JIT-at-runtime with a `FileStreamProgramCache` is
sufficient — first import is a ~1 s compile; subsequent imports are
~10 ms cache hits. The disk artefacts are CUBINs sized for the host's
specific arch.

For distribution (wheels):

* **Single PTX in the wheel**: PTX is forward-compatible across
  arches; the driver JIT-compiles to the running arch's SASS on
  first load. Wheel size impact: ~50 KiB per codec.
* **Pre-compiled CUBINs per arch**: faster first load (no JIT), but
  the wheel ships N CUBINs per codec for the arches we declare. For
  v0.1 we don't need this; revisit when we have a real "ship the
  wheel" deadline.

Recommendation: ship PTX in the wheel, prime the disk cache on
first use, run a Python-startup "warm the cache" step in
`czarr/__init__.py` that compiles per the active arch in a
background thread. The user's first import is fast; first codec call
hits the cache.

### 8.2 Kernel source: Python string vs `.cu` file

For v0.1, Python strings (matches the spike). Three reasons:

* No build-system hooks needed; `pip install czarr` produces a usable
  artifact.
* Source is grep-able alongside the wrapper.
* Diffs are reviewable in the same PR.

When deflate lands (~1500 lines vendored from cuDF), promote the
deflate kernel to `deflate.cu` loaded via `importlib.resources`. Mix
freely — LZ4 stays inline, deflate becomes a sidecar — based on
what's readable. The kernel cache abstracts the source loading.

### 8.3 Extracting the nvCOMP-specific machinery

`src/czarr/codecs/base.py:CudaBytesBytesCodec` mixes the
nvCOMP-wrapping logic (`_create_codec`, `_get_codec`, `_batch_sync`
with the `nv_inputs` shape) with the generic codec contract
(`to_dict`, `from_dict`, framing hooks). The refactor:

```
src/czarr/codecs/
    base.py                # _BackendAware + the bytes-bytes generic shape
    _native/
        __init__.py
        _common.py
        _oracle.py
        _kernel_cache.py
        lz4.py             # kernel + decode_lz4_native()
    _nvcomp/
        __init__.py
        _adapter.py        # the old CudaBytesBytesCodec body, renamed
                           # _NvcompBytesBytesAdapter, used internally by
                           # `_BackendAware._decode_nvcomp`
        compressors/
            lz4.py         # the nvCOMP LZ4 binding (passes WITH_UNCOMPRESSED_SIZE)
            zstd.py        # etc.
            ...
    compressors/
        lz4.py             # the public LZ4 class (_BackendAware subclass)
        zstd.py            # the public Zstd class (nvcomp-only)
        ...
```

The public `czarr.LZ4` is the `compressors/lz4.py` class — a
`_BackendAware` subclass whose `_decode_native` calls into
`_native/lz4.py` and whose `_decode_nvcomp` calls into the
`_nvcomp/_adapter.py` machinery (which is mostly today's
`CudaBytesBytesCodec._batch_sync` body). The
nvCOMP-thread-local-cache and stream-binding logic stays inside the
adapter; the codec class is just a config dataclass + two
dispatch methods.

This puts ~250 lines of refactor on the Phase 1 budget, but it
yields a clean separation where nvCOMP can be a soft dependency
later (`import nvidia.nvcomp` is gated inside `_nvcomp/_adapter.py`
behind a try/except; the native tier still works without nvCOMP
installed).

### 8.4 Equality and hashing

The `backend` field is `compare=False` (section 1). Two implications:

* `czarr.LZ4(backend="native") == czarr.LZ4(backend="nvcomp")` is
  **True** if all other fields match. This is correct — they encode
  the same bitstream and so are functionally the same codec.
* `hash(czarr.LZ4(backend="native")) == hash(czarr.LZ4(backend="nvcomp"))`.
  Codecs are not commonly used as dict keys, but if they are, this
  is the right behaviour.

If a downstream caller wants to distinguish, they check
`.backend` explicitly.

### 8.5 Eager validation of unsupported configurations

The `__post_init__` raise on unsupported backend (section 5.1) is
eager — it fires at codec construction. This matters because the
codec is typically constructed at array-creation time (well before
the first read or write); a delayed failure deep inside
`decode_batch` after an hour of I/O is worse than a clear
`ValueError` upfront.

The validation is cheap: a tuple membership check. No reason to
defer.

---

## 9. Migration / coexistence

### 9.1 The transition for existing users

Today: `czarr.LZ4()` constructs a codec that decodes through nvCOMP
with the `WITH_UNCOMPRESSED_SIZE` bitstream
(`src/czarr/codecs/compressors/lz4.py:20`).

After Phase 1: `czarr.LZ4()` constructs a codec whose default
backend is `"native"`. The same bitstream. Decode is now the
spike's kernel; encode still goes through nvCOMP. All existing
arrays read back unchanged.

The migration is a no-op for users who do not set
`backend=` explicitly. Users who *do* want to pin to nvCOMP can:

```python
codec = czarr.LZ4(backend="nvcomp")          # per-instance
czarr.configure_gpu(codec_backend_overrides={"lz4": "nvcomp"})  # global
```

### 9.2 nvCOMP-only codecs

Zstd, Bitcomp, ANS, Cascaded never get a native backend. Their
classes:

```python
@dataclass(frozen=True, slots=True)
class Zstd(_BackendAware):
    codec_name: ClassVar[str] = "zstd"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("nvcomp",)
    _native_default: ClassVar[bool] = False

    level: int = 0
    checksum: bool = False
```

`czarr.Zstd(backend="native")` raises. `czarr.Zstd()` works as
before. The Zarr v3 registry sees `"zstd"` map to `czarr.Zstd`
unchanged.

### 9.3 Codec-by-codec rollout

Adding Snappy native (post-v0.1 example):

1. Land `src/czarr/codecs/_native/snappy.py` with the ported cuDF
   `unsnap.cu` kernel.
2. Update `czarr.Snappy` (currently nvCOMP-only): add `"native"` to
   `_supported_backends`, flip `_native_default = True` once the
   bench validates throughput.
3. Generate Snappy fixtures via `scripts/regen_fixtures.py`.
4. Land the bench entry; set the regression threshold.

No change to call sites, no migration. Users who created
`czarr.Snappy()` get the new native default; those who pinned
`backend="nvcomp"` keep nvCOMP.

### 9.4 Bitstream incompatibilities to watch

Two existing czarr codecs use nvCOMP-native bitstreams that the
native tier cannot match without changing the on-disk format:

* `czarr.Snappy` (under `czarr.snappy` codec_name in
  `src/czarr/codecs/compressors/native.py:63-71`) uses the nvCOMP
  *chunked* Snappy bitstream — not standard Snappy block.
  Productionising a native Snappy means deciding: do we move
  `czarr.Snappy` to standard Snappy block format (breaking change),
  or keep `czarr.snappy` as nvCOMP-only and add a separate
  `czarr.SnappyBlock` (under codec_name `"snappy"`) for the standard
  format? Recommendation: the latter. Standard Snappy gets a new
  class name and the standard codec_name; the nvCOMP-native version
  stays at its czarr-prefixed codec_name.
* `czarr.Deflate` / `czarr.GDeflate` — same story. Raw deflate
  (RFC 1951) is the standard bitstream; the existing classes use
  the nvCOMP-native chunked variants. Add separate classes for the
  standard formats when the native impl lands.

Document this in the codec module docstrings so the asymmetry is
not surprising.

### 9.5 `configure_gpu` integration

The new `codec_backend_overrides` kwarg fits into the existing
`configure_gpu` shape (`src/czarr/__init__.py:63-152`):

```python
def configure_gpu(
    *,
    batch_size: int | None = None,
    async_concurrency: int = 32,
    rmm_pool_gb: float | None = None,
    ...
    codec_backend_overrides: dict[str, CodecBackend] | None = None,
) -> None:
    ...
    if codec_backend_overrides:
        from czarr.codecs._registry import _set_overrides
        _set_overrides(codec_backend_overrides)
```

The override map is process-global and applies to every codec
constructed *after* the `configure_gpu` call (i.e. every codec from
`from_dict`, every direct construction).

Codecs constructed *before* `configure_gpu` snapshot their backend at
construction; they do not retroactively change. This is intentional
— it preserves the per-instance contract — but worth documenting.

---

## 10. Concrete code sketch — putting it all together

For reference, the full shape of `czarr.LZ4` after the refactor:

```python
# src/czarr/codecs/compressors/lz4.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from czarr.codecs._base import CodecBackend, _BackendAware

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON


@dataclass(frozen=True, slots=True)
class LZ4(_BackendAware):
    """LZ4 block format — bitstream-compatible with numcodecs.LZ4.

    Two backends:

    * ``backend="native"`` (default) — in-house ``cupy.RawKernel`` decode,
      ~17.8× nvCOMP on A40 / ~173 GiB/s on H200 per the
      ``native_lz4_stream_parallelism`` probe. Encode falls back to
      nvCOMP for v0.1.
    * ``backend="nvcomp"`` — closed-source nvCOMP path; useful as a
      regression-compare baseline.

    Both backends write the same on-disk bitstream (the numcodecs.LZ4
    4-byte LE uncompressed-size prefix followed by the LZ4 block).
    A reader on either backend decodes a stream written by the other
    byte-for-byte identically.
    """

    codec_name: ClassVar[str] = "lz4"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("native", "nvcomp")
    _native_default: ClassVar[bool] = True

    acceleration: int = 1

    @override
    def _decode_native(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        from czarr.codecs._native.lz4 import decode_lz4_native
        from czarr.codecs._native._common import buffer_to_cupy

        import cupy as cp

        cp_inputs: list[cp.ndarray] = []
        sizes: list[int] = []
        for chunk, _ in items:
            cp_arr = buffer_to_cupy(chunk)
            sizes.append(int(cp_arr[:4].view(cp.uint32).item()))
            cp_inputs.append(cp_arr[4:])
        cp_outs = decode_lz4_native(cp_inputs, sizes, stream=self._resolve_stream_or_default())
        return [
            spec.prototype.buffer.from_array_like(o)
            for o, (_, spec) in zip(cp_outs, items, strict=True)
        ]

    @override
    def _decode_nvcomp(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        from czarr.codecs._nvcomp._adapter import nvcomp_decode_batch

        return nvcomp_decode_batch(
            items,
            algorithm="LZ4",
            bitstream_kind="WITH_UNCOMPRESSED_SIZE",
            chunk_size=65536,
        )

    @override
    def _encode_native(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        # v0.1: native encode is nvcomp encode. Documented in the docstring.
        return self._encode_nvcomp(items)

    @override
    def _encode_nvcomp(self, items: list[tuple[Buffer, ArraySpec]]) -> list[Buffer]:
        from czarr.codecs._nvcomp._adapter import nvcomp_encode_batch

        return nvcomp_encode_batch(
            items,
            algorithm="LZ4",
            bitstream_kind="WITH_UNCOMPRESSED_SIZE",
            chunk_size=65536,
        )

    @override
    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": self.codec_name,
            "configuration": {"acceleration": int(self.acceleration)},
        }

    @classmethod
    @override
    def from_dict(cls, data: dict[str, JSON]) -> "LZ4":
        cfg = dict(data.get("configuration", {}))
        cfg.pop("backend", None)  # never honour persisted backend
        return cls(**cfg)
```

And `czarr.Zstd` (nvCOMP-only, for contrast):

```python
@dataclass(frozen=True, slots=True)
class Zstd(_BackendAware):
    """Zstd — RFC 8478 frame compatible with libzstd / numcodecs.Zstd.

    Native backend not available — Zstd's FSE + Huffman decoders are
    6-9 engineer-months of work. Permanent nvCOMP dependency. See
    ``03-nvcomp-alternatives.md`` §5 for the analysis.
    """

    codec_name: ClassVar[str] = "zstd"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("nvcomp",)
    _native_default: ClassVar[bool] = False

    level: int = 0
    checksum: bool = False

    @override
    def _decode_nvcomp(self, items):
        from czarr.codecs._nvcomp._adapter import nvcomp_decode_batch
        return nvcomp_decode_batch(
            items, algorithm="Zstd", bitstream_kind="RAW", chunk_size=65536,
        )

    @override
    def _encode_nvcomp(self, items):
        from czarr.codecs._nvcomp._adapter import nvcomp_encode_batch
        return nvcomp_encode_batch(
            items, algorithm="Zstd", bitstream_kind="RAW", chunk_size=65536,
        )

    @override
    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": self.codec_name,
            "configuration": {"level": int(self.level), "checksum": bool(self.checksum)},
        }
```

The two classes are structurally identical; the only difference is
that `LZ4` declares `"native"` in its supported set and implements
`_decode_native`. Adding a codec to the native tier is a matter of
extending the supported set and writing one method.

---

## 11. Open questions and explicit punts

### 11.1 Punted

* **`backend="auto"`** — would pick per-call based on payload size
  (native LZ4 may lose vs nvCOMP for very small payloads due to
  kernel launch cost). Per `07-api-design.md:984-986`, skip until
  profiling shows it's worth. The probe already shows native wins
  monotonically at the batch sizes we care about.
* **Per-codec backend pickers via type parameters** (`LZ4[Native]` /
  `LZ4[Nvcomp]` as distinct types) — per section 1, skipped. Plain
  `Literal["native", "nvcomp"]` is enough.
* **Pre-compiled CUBINs in the wheel** — section 8.1. JIT-at-runtime
  + persistent cache is enough for v0.1.
* **CPU-fallback bridge to `zarr.codecs.numcodecs`** — per
  `06-numcodecs-exploration.md:697-723`, that bridge already exists
  in zarr-python; czarr does not need its own. Users on hosts
  without GPU read CPU-written stores via the standard zarr
  registry; the native tier is purely additive.

### 11.2 Open

* **Native Zstd in any form**: probably never. `cuDF` does not have
  one, NVIDIA Zstd is closed, the research literature has no
  production decoder. If a research collaboration produces one we
  reconsider.
* **`czarr.Snappy` rename** (section 9.4): does the existing
  nvCOMP-native bitstream stay at `codec_name="czarr.snappy"`, and
  the new standard-format codec take `codec_name="snappy"`? Or do
  we break the existing name? I lean rename (the existing codec is
  ~4 weeks old; users are few). Defer until Snappy native lands.
* **CRC32C / Adler32 / Fletcher32 native**: per
  `06-numcodecs-exploration.md:858-866` and the open issue in
  `03-nvcomp-alternatives.md:298`, GPU CRC32C is a custom kernel
  (~1 week). Reasonable v1 addition; not v0.1.
* **The encoder asymmetry in section 4.1**: ship Option B for v0.1
  but track Option A as a "good first issue". The encoder is the
  same on-disk format regardless of which side produced it, so
  switching later is mechanical.

---

## 12. References

* [`spikes/lz4_decoder.py`](spikes/lz4_decoder.py) — the source of
  truth for the LZ4 kernel and its CPU oracle.
* [`native_lz4_stream_parallelism.md`](native_lz4_stream_parallelism.md)
  — H200 throughput probe; gates the "no lanes" decision.
* [`nvcomp_stream_parallelism.md`](nvcomp_stream_parallelism.md) —
  the symmetric nvCOMP probe.
* [`SUMMARY.md`](SUMMARY.md) — the overarching plan.
* [`07-api-design.md`](07-api-design.md) — public API surface;
  section 1 of this doc replaces the `_BackendAware` sketch there.
* [`06-numcodecs-exploration.md`](06-numcodecs-exploration.md) —
  numcodecs ABC, registry, bridge, and the fixture-based
  bitstream-compat test pattern.
* [`03-nvcomp-alternatives.md`](03-nvcomp-alternatives.md) —
  per-codec rewrite cost analysis.
* [`01-cuda-core-api.md`](01-cuda-core-api.md) §6.9 + Appendix A4 —
  `FileStreamProgramCache` and kernel-cache discipline.
* `src/czarr/codecs/base.py` — the existing nvCOMP-wrapping
  `CudaBytesBytesCodec`; the `_nvcomp/_adapter.py` rename is a
  refactor of this file's body.
* `src/czarr/codecs/compressors/lz4.py` — the existing LZ4 wrapper;
  the new public class supersedes it.
* `src/czarr/codecs/compressors/zstd.py` —
  `src/czarr/codecs/compressors/native.py` — examples of the codecs
  that stay nvCOMP-only.
* `src/czarr/__init__.py:63-152` — `configure_gpu` is where
  `codec_backend_overrides` plugs in.
* `src/czarr/array/_impl.py` — the orchestrator that will consume
  per-call stream binding; section 3.4's `stream=` kwarg threads
  through `_decode_native`.
