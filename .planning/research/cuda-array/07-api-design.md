# czarr API design — surface integration with numcodecs + zarr-python

> Design exploration, not a spec. Captures the public surface czarr exposes to
> users coming from numcodecs and zarr-python. Integrates the post-synthesis
> findings: nvCOMP decode does not parallelise across streams (`nvcomp_stream_parallelism.md`),
> native LZ4 may or may not (probe in flight), I/O-overlap (not lanes) is the
> Phase 1 win.

## Python 3.12 features czarr should adopt

Lower bound is Python 3.12 (per `pyproject.toml`). The following 3.12 features
shape the API surface and the implementation idioms:

### PEP 695 — type parameter syntax + `type` statement

```python
# Old
from typing import TypeVar, Generic
T = TypeVar("T", bound=np.dtype)
class Slab(Generic[T]): ...

# New (3.12)
class Slab[T: np.dtype]: ...

type ChunkBytes = bytes | memoryview | bytearray
type Selector = int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...]
type CodecBackend = Literal["native", "nvcomp"]
```

Use cases in czarr:

- `class CzarrGpuBuffer[Dtype: np.dtype]: ...` for dtype-aware buffer typing
- `type DeviceArray = cp.ndarray` aliases for clarity at boundaries
- `type LaneId = int` and `type StreamHandle = int` for newtype-ish hints

### PEP 698 — `typing.override`

```python
from typing import override

class CudaZarrArray(zarr.Array):
    @override
    def __getitem__(self, key: Selector, /) -> cp.ndarray | np.ndarray: ...
```

Use cases:

- Every method we override from `zarr.Array` carries `@override` so future
  zarr-python internal-API changes show up as a static-typing failure rather
  than a silent fallback to the parent class.
- Every concrete `Codec.encode` / `decode` we ship gets `@override` against
  the zarr v3 ABCs.

### PEP 692 — `Unpack[TypedDict]` for `**kwargs`

```python
from typing import TypedDict, Unpack, NotRequired

class ConfigureGpuKwargs(TypedDict):
    batch_size: NotRequired[int | None]
    async_concurrency: NotRequired[int]
    rmm_pool_gb: NotRequired[float | None]
    stream_pool_size: NotRequired[int]
    queue_depth: NotRequired[int]
    cufile_poll_mode: NotRequired[bool]
    cufile_poll_threshold_kb: NotRequired[int]
    pinned_prealloc: NotRequired[Iterable[tuple[int, int]] | None]
    pipeline: NotRequired[bool]
    use_cuda_array: NotRequired[bool]

def configure_gpu(**kwargs: Unpack[ConfigureGpuKwargs]) -> None: ...
```

Replaces today's positional+keyword sprawl in `configure_gpu` with a single
typed mapping. Editors complete the keys. Callers can pass `**preset_dict`.

### PEP 688 — `__buffer__` Python-level buffer protocol

```python
from collections.abc import Buffer  # new in 3.12

class CzarrGpuBuffer:
    def __buffer__(self, flags: int) -> memoryview:
        # Lazy H2D for the host-bytes-required protocol callers
        # (e.g. zarr.core.Buffer.to_bytes()).
        return memoryview(self.as_numpy_array())

# Now `bytes(czarr_gpu_buffer)`, `memoryview(czarr_gpu_buffer)`, and any
# code expecting `collections.abc.Buffer` JustWorks.
```

Use cases:

- `CzarrGpuBuffer` becomes a `Buffer` per PEP 688, so any zarr-internal
  code calling `bytes(buf)` or `memoryview(buf)` works without us
  monkeypatching `to_bytes`.
- We can type buffer-accepting helpers as `Buffer` and accept cupy
  arrays, our wrapper, raw bytes, or pinned host arrays uniformly.

### PEP 701 — f-strings in the grammar

Use for cleaner error messages, esp. with nested quotes:

```python
raise ValueError(f"codec={self.codec_name!r} got unexpected backend={backend!r} (allowed: {", ".join(allowed)!r})")
```

### PEP 709 — comprehension inlining

No code change required; comprehensions in the hot path (chunk-iteration,
range expansion in `_parse_key`) get a free ~2× speedup vs 3.11.

## Design constraints from numcodecs + zarr-python

From agent 6's deep-dive (`06-numcodecs-exploration.md`):

1. **Zarr v3 splits codecs into three ABCs**: `ArrayArrayCodec`,
   `ArrayBytesCodec`, `BytesBytesCodec`. numcodecs has one `Codec` base. The
   v3 bridge lives at `zarr.codecs.numcodecs` (the legacy `numcodecs.zarr3`
   module is deprecated). czarr already follows the three-ABC split.

2. **Codec config round-trips through a `{"name": ..., "configuration": {...}}`
   dict** in Zarr v3 metadata. numcodecs uses a flat dict with `codec_id`.
   czarr's `to_dict`/`from_dict` already handles both shapes.

3. **Registry**: `zarr.registry.register_codec(name, cls, qualname=...)` is the
   contract. Same for `register_buffer` / `register_ndbuffer` /
   `register_pipeline`. numcodecs has its own entry-point group
   `numcodecs.codecs` that the deprecated bridge consumed.

4. **Entry-point plugin discovery is the idiomatic way** for third-party
   codec libraries to register without forking. czarr should publish entry
   points in `pyproject.toml` so a `pip install czarr` makes its codecs
   discoverable, but **must not shadow numcodecs.codecs** ids — only
   `zarr.codecs` ids, since the codec dispatch is per-store.

5. **Bit-exact compatibility** between czarr-encoded and numcodecs-encoded
   bitstreams is the contract. agent 6 surfaced one real bug
   (`czarr.BitRound` misses round-to-even) and showed the fixture pattern
   in `numcodecs/tests/common.py:155-244` is the right template for our
   compatibility tests.

## Current API state (czarr today, before this exploration)

```python
import czarr
import zarr

# Configuration — one entry point sets EVERYTHING.
czarr.configure_gpu(
    batch_size=None,         # None → sys.maxsize (one big nvCOMP call)
    async_concurrency=32,
    rmm_pool_gb=4,
    stream_pool_size=4,
    pipeline=True,
)

# Codec construction — direct.
codec_zstd = czarr.Zstd(level=3, chunk_size=65536)
codec_lz4 = czarr.LZ4(acceleration=1, chunk_size=65536)
codec_shuffle = czarr.Shuffle(elementsize=4)

# Array creation — via zarr.create_array.
arr = zarr.create_array(
    store=czarr.GPULocalStore("data.zarr"),
    shape=(16, 4096, 4096),
    chunks=(16, 512, 512),
    dtype="float32",
    compressors=[codec_zstd],
    filters=[codec_shuffle],
)

# Read — through zarr.Array's standard path. Returns cupy.ndarray because
# `configure_gpu` set the global buffer prototype.
out: cp.ndarray = arr[:]
```

Surface-level pain points worth fixing:

1. **Backend selection is implicit and global.** Today every `czarr.LZ4`
   instance routes through nvCOMP. We can't easily ask "use the native
   LZ4 backend for this codec only" without rewiring globally.
2. **No first-class `CudaZarrArray`.** The fast path goes through
   `zarr.Array.__getitem__` → `BatchedCodecPipeline.read_batch`. We have no
   ergonomic way to bypass that pipeline for the basic-indexing case
   (the architectural bet from the research synthesis).
3. **`configure_gpu` is a kitchen-sink.** Eight kwargs, growing. No
   typed signature; IDE autocomplete is mediocre.
4. **No structured way to ship a "preset"** (e.g. "the H200 preset"
   would set `stream_pool_size=4`, `async_concurrency=32`, `batch_size=
   sys.maxsize` because lanes are dead, `queue_depth=16`).

## Proposed surface

### Public namespace (`czarr.__init__`)

```python
# czarr/__init__.py

# Codecs — names match Zarr v3 codec_ids (lowercase).
from czarr.codecs import (
    # Compressors (BytesBytesCodec)
    ANS, Bitcomp, Cascaded, Deflate, GDeflate, Gzip, LZ4, Snappy, Zlib, Zstd,
    # Filters (ArrayArrayCodec)
    BitRound, Delta, FixedScaleOffset, Shuffle,
    # Checksums (BytesBytesCodec)
    CRC32C, Adler32, Fletcher32,
)

# Array facade — the architectural bet.
from czarr.array import CudaZarrArray, open_cuda_array, create_cuda_array

# Storage — already exists.
from czarr.storage import GPULocalStore, cufile_runtime

# Buffer prototype — from buffer epic.
from czarr.core.buffer import CzarrGpuBuffer, CzarrGpuNDBuffer, buffer_prototype

# Configuration — typed surface.
from czarr.configure import configure_gpu, presets, Preset

# Type aliases — common at boundaries (PEP 695).
from czarr.types import (
    Selector,           # int | slice | EllipsisType | tuple[...]
    CodecBackend,       # Literal["native", "nvcomp"]
    StreamHandle,       # int (cudaStream_t)
    ChunkBytes,         # bytes | memoryview | bytearray
)

__all__ = [...]  # explicit
```

### Codec base — consolidated shared features

The current `CudaBytesBytesCodec` mixes nvCOMP-specific logic with generic
codec contract. With native backends entering Phase 1, we split:

```python
# czarr/codecs/_base.py

from typing import ClassVar, Literal, override, Self
from dataclasses import dataclass, field, fields
from zarr.abc.codec import BytesBytesCodec

type CodecBackend = Literal["native", "nvcomp"]


@dataclass(frozen=True, slots=True)
class _BackendAware(BytesBytesCodec):
    """Common surface for codecs that can route through multiple backends.

    Subclasses declare ``_supported_backends`` and supply backend-specific
    encode/decode methods (``_encode_native`` / ``_decode_native`` /
    ``_encode_nvcomp`` / ``_decode_nvcomp``).  The dispatch lives here once.
    """

    codec_name: ClassVar[str] = ""
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("nvcomp",)
    _default_backend: ClassVar[CodecBackend] = "nvcomp"

    backend: CodecBackend = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.backend is None:
            # frozen dataclass — bypass setattr
            object.__setattr__(self, "backend", self._default_backend)
        if self.backend not in self._supported_backends:
            allowed = ", ".join(self._supported_backends)
            raise ValueError(
                f"codec={self.codec_name!r} backend={self.backend!r} unsupported; "
                f"allowed: {allowed!r}"
            )

    # The two halves subclasses implement.
    def _encode_impl(self, chunks, specs): ...  # required
    def _decode_impl(self, chunks, specs): ...  # required

    @override
    async def encode(self, chunks_and_specs):
        # Dispatch lives here; subclasses implement the per-backend halves.
        return await asyncio.to_thread(self._encode_impl, chunks_and_specs)

    @override
    async def decode(self, chunks_and_specs):
        return await asyncio.to_thread(self._decode_impl, chunks_and_specs)


@dataclass(frozen=True, slots=True)
class LZ4(_BackendAware):
    codec_name: ClassVar[str] = "lz4"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("native", "nvcomp")
    _default_backend: ClassVar[CodecBackend] = "native"  # post-spike, this wins
    acceleration: int = 1
    chunk_size: int = 65536

    def _decode_impl(self, items):
        if self.backend == "native":
            from czarr.codecs._lz4_native import decode_batch_lz4_native
            return decode_batch_lz4_native(items, self.chunk_size)
        from czarr.codecs._lz4_nvcomp import decode_batch_lz4_nvcomp
        return decode_batch_lz4_nvcomp(items, self.chunk_size, self.acceleration)

    def _encode_impl(self, items):
        # Encode stays on nvCOMP for v0.1 — the spike covers decode only.
        # Document this explicitly.
        from czarr.codecs._lz4_nvcomp import encode_batch_lz4_nvcomp
        return encode_batch_lz4_nvcomp(items, self.chunk_size, self.acceleration)
```

Three nice things from this shape:

- `backend` is per-instance and round-trips in Zarr v3 metadata, so writers
  on the native backend produce identical bitstreams to nvCOMP readers
  (czarr's existing `WITH_UNCOMPRESSED_SIZE` covers this for LZ4).
- Codecs that don't support multiple backends (Zstd, Bitcomp, etc.) just
  leave `_supported_backends = ("nvcomp",)` and skip the `_decode_native`
  half.
- The user never has to know about the backend split unless they want to.

For Zstd:

```python
@dataclass(frozen=True, slots=True)
class Zstd(_BackendAware):
    codec_name: ClassVar[str] = "zstd"
    # Per research synthesis: native zstd is 6-9 months of work; keep nvCOMP
    # permanently for now.
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("nvcomp",)
    _default_backend: ClassVar[CodecBackend] = "nvcomp"
    level: int = 3
    chunk_size: int = 65536
```

### CudaZarrArray — entry points

Modeled on the zarrs PR #147 pattern but adapted for our async/sync surface:

```python
# czarr/array/_facade.py

import zarr
import numpy as np
import cupy as cp
from typing import Any, Literal, override, TypedDict, Unpack, NotRequired
from types import EllipsisType

from czarr.types import Selector

type _BasicKey = int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...]


class CudaArrayKwargs(TypedDict):
    """Tuning knobs surfaced from configure_gpu — explicit override per array."""
    queue_depth: NotRequired[int]
    stream_pool_size: NotRequired[int]
    microbatch_size: NotRequired[int]
    backend: NotRequired[Literal["auto", "native-where-possible", "nvcomp"]]


class CudaZarrArray(zarr.Array):
    """zarr.Array subclass with a GPU-resident basic-indexing fast path.

    Public API surface beyond zarr.Array:

    - ``arr[basic_key]`` returns ``cp.ndarray`` on device.  Advanced
      indexing falls through to ``super().__getitem__`` and returns
      ``np.ndarray`` on host.  The dual contract is intentional — users
      who want consistency wrap the result in ``cp.asarray(...)``.
    - ``arr.read_into(out: cp.ndarray, key=...)`` — zero-allocation hot
      path; the caller owns the output buffer.
    - ``arr.lazy[key]`` — returns a ``_LazySlice`` that defers IO until
      materialised via ``__array__`` (numpy), ``cp.asarray(...)``,
      ``torch.utils.dlpack.from_dlpack(...)``, etc.
    """

    @classmethod
    def wrap(cls, array: zarr.Array, /, **kwargs: Unpack[CudaArrayKwargs]) -> "CudaZarrArray":
        """Wrap an existing zarr.Array.

        Preferred over the constructor because the existing array's
        metadata + store + codec chain are reused — no second open.
        """
        ...

    @override
    def __getitem__(self, key: _BasicKey, /) -> cp.ndarray | np.ndarray:
        if _is_basic_indexing(key):
            ranges, region_shape, squeeze = self._parse_key(key)
            out = cp.empty(region_shape, dtype=self.dtype)
            if out.size > 0:
                self._impl.retrieve_gpu(ranges, out)
            return out.squeeze(axis=tuple(squeeze)) if squeeze else out
        return super().__getitem__(key)  # fallback to zarr.Array, returns numpy

    def read_into(self, out: cp.ndarray, key: _BasicKey = (...,), /) -> None:
        """Zero-allocation read into ``out``.  ``out.shape`` must match."""
        ...

    @property
    def lazy(self) -> "_LazyIndexer":
        """Capture indexing without IO.  Materialise via __array__/DLPack/CAI."""
        ...


def open_cuda_array(
    store: zarr.abc.store.Store | str,
    path: str | None = None,
    *,
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
    **kwargs: Unpack[CudaArrayKwargs],
) -> CudaZarrArray:
    """Open an existing array path and wrap it as a CudaZarrArray.

    Equivalent to ``CudaZarrArray.wrap(zarr.open_array(store, path, mode))``
    but bundles the open + wrap into one call.
    """
    ...


def create_cuda_array(
    store: zarr.abc.store.Store | str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | None = None,
    dtype: np.typing.DTypeLike,
    compressors: list[Any] | None = None,
    filters: list[Any] | None = None,
    fill_value: Any | None = None,
    **kwargs: Unpack[CudaArrayKwargs],
) -> CudaZarrArray:
    """Create + wrap in one call.  Mirrors zarr.create_array's signature."""
    ...
```

Single canonical entry point per usage shape, no surprise. Wrap an existing
array, open a new one, create a new one. All return the same `CudaZarrArray`.

### Buffer prototype — `__buffer__` integration

```python
# czarr/core/buffer.py

from collections.abc import Buffer  # 3.12 — Python-level buffer protocol

class CzarrGpuBuffer[T: np.dtype]:  # PEP 695 generic param
    """Device-resident 1-D byte buffer backed by cuda.core.VirtualMemoryResource.

    Implements ``zarr.core.buffer.core.Buffer`` (the v3 ABC) **and** the
    ``collections.abc.Buffer`` protocol via ``__buffer__``, so it works
    interchangeably with code that expects raw bytes.
    """

    def __buffer__(self, flags: int) -> memoryview:
        # H2D copy lazily.  Most callers want this for to_bytes() /
        # legacy host-bytes-required protocol paths.
        if flags & inspect.BufferFlags.WRITABLE:
            raise BufferError("CzarrGpuBuffer host views are read-only")
        return memoryview(self.as_numpy_array())

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """v3 CAI synthesised from our handle + size."""
        ...

    def __dlpack__(self, *, stream: int | None = None) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
```

The `Buffer` protocol + `__cuda_array_interface__` + DLPack triple makes our
buffer accepted by every consumer we care about (zarr, cupy, torch, nvCOMP,
cuTile, numba, jax).

### Configuration — `configure_gpu` + presets

```python
# czarr/configure.py

from typing import TypedDict, Unpack, NotRequired, Literal
import sys

class ConfigureGpuKwargs(TypedDict):
    # Pipeline
    batch_size: NotRequired[int | None]
    async_concurrency: NotRequired[int]
    queue_depth: NotRequired[int]              # NEW: I/O→decode queue depth (replaces dead microbatch knob)

    # Memory
    rmm_pool_gb: NotRequired[float | None]
    pinned_prealloc: NotRequired[Iterable[tuple[int, int]] | None]

    # Streams + cuFile
    stream_pool_size: NotRequired[int]
    cufile_poll_mode: NotRequired[bool]
    cufile_poll_threshold_kb: NotRequired[int]

    # Toggles
    pipeline: NotRequired[bool]
    use_cuda_array: NotRequired[bool]          # NEW: auto-wrap zarr.open_array in CudaZarrArray
    codec_backend_overrides: NotRequired[dict[str, CodecBackend]]  # NEW: per-codec backend


@dataclass(frozen=True, slots=True)
class Preset:
    """Named tuning preset for a known GPU class."""
    name: str
    kwargs: ConfigureGpuKwargs

    def apply(self) -> None:
        configure_gpu(**self.kwargs)


class presets:
    """Named presets for common deployments."""

    H200 = Preset("h200", {
        "batch_size": sys.maxsize,         # nvCOMP wants one big batch (research finding)
        "queue_depth": 16,                  # I/O→decode pipeline depth
        "stream_pool_size": 4,
        "async_concurrency": 32,
        "rmm_pool_gb": 4.0,
    })

    H100 = Preset("h100", {**H200.kwargs, "rmm_pool_gb": 4.0})

    A40_COMPAT = Preset("a40-compat", {       # cuFile in compat mode
        "batch_size": sys.maxsize,
        "queue_depth": 8,                     # less aggressive — compat-mode reads are slower
        "stream_pool_size": 2,
        "async_concurrency": 16,
    })


def configure_gpu(**kwargs: Unpack[ConfigureGpuKwargs]) -> None:
    """One entry point.  Typed kwargs; IDE autocompletes.  No positional args.

    Use a ``Preset`` for known hardware::

        czarr.presets.H200.apply()
    """
    ...
```

### Storage — `GPULocalStore` stays as-is (already ergonomic)

The store API is already well-shaped; just `__buffer__` plumbing on the
buffer prototype makes it interop better with non-czarr callers. No
intentional API surface change here.

## Worked examples — what the user actually writes

### Smallest possible: open + read

```python
import czarr
import cupy as cp

# One-line config; the preset picks all the knobs.
czarr.presets.H200.apply()

# Open + wrap.
arr = czarr.open_cuda_array("data.zarr")

# Read returns cupy.ndarray on device.
slab: cp.ndarray = arr[0:8, :, :]
```

### Backend choice for LZ4

```python
import czarr

# Default — native (post-spike, the spike beat nvCOMP 17.8× on A40).
codec = czarr.LZ4()

# Explicit native — same as default.
codec = czarr.LZ4(backend="native")

# Opt-out to nvCOMP — for debugging / regression compare.
codec = czarr.LZ4(backend="nvcomp")

# Globally pin the backend (e.g. in CI to test nvCOMP path).
czarr.configure_gpu(codec_backend_overrides={"lz4": "nvcomp"})
```

### Mix-and-match

```python
import czarr
import zarr

czarr.presets.H200.apply()

arr = czarr.create_cuda_array(
    "rechunked.zarr",
    shape=(64, 4096, 4096),
    chunks=(16, 512, 512),
    dtype="float32",
    filters=[czarr.Shuffle(elementsize=4)],   # cuda.compute-backed (Phase 2)
    compressors=[
        czarr.Delta(dtype="<f4"),             # CCCL one-liner
        czarr.LZ4(backend="native"),          # in-house decoder
        czarr.CRC32C(),                       # checksum
    ],
)
```

### Lazy slices for zero-copy handoff

```python
import czarr
import torch

arr = czarr.open_cuda_array("data.zarr")

# No IO yet — just captures the slice spec.
view = arr.lazy[0:8, :, :]

# Materialise via DLPack into a torch tensor on the same device.
tensor = torch.utils.dlpack.from_dlpack(view)
```

### Migrating from existing czarr code (one-line change)

```python
# Before
arr = zarr.open_array(store, mode="r")

# After
arr = czarr.open_cuda_array(store, mode="r")  # OR: czarr.CudaZarrArray.wrap(zarr.open_array(store, mode="r"))

# Everything else is unchanged.
```

## Generics — where the type system actually pays off

PEP 695's `class Name[T: Bound]:` syntax makes parametric types finally
ergonomic in Python.  Used judiciously, czarr's API can carry dtype +
backend + layout through the entire call graph and let the type checker
catch mismatches before they become runtime errors deep inside nvCOMP.
Used unjudiciously, generics turn into a typing puzzle nobody can read.
This section picks the spots where the ROI is real.

### Dtype-parameterised buffer + array

The single highest-value generic is **dtype on the array and buffer**.
The dtype is fixed once the array is opened (it's in the v3 metadata),
so we know it statically — but today every `arr[:]` returns
`cp.ndarray` with no element type, and the user has to remember what
dtype they opened.

```python
# czarr/core/buffer.py

import numpy as np
import cupy as cp
from typing import Any, Self, override
from collections.abc import Buffer


class CzarrGpuBuffer[Dt: np.generic]:
    """1-D byte buffer.  ``Dt`` is the element scalar type for views.

    The underlying storage is always uint8 bytes (we allocate with
    ``VirtualMemoryResource`` and synthesise CAI as ``|u1``).  ``Dt``
    parameterises ``as_array_like()`` / ``as_ndarray_like()`` so a
    typed caller sees the dtype-aware return.
    """

    @classmethod
    def empty(cls, size: int, *, stream: ... = None) -> "CzarrGpuBuffer[np.uint8]":
        ...

    def view[NewDt: np.generic](self, dtype: type[NewDt]) -> "CzarrGpuBuffer[NewDt]":
        """Reinterpret the byte buffer as a different scalar type."""
        ...

    def as_array_like(self) -> "cp.ndarray[Any, np.dtype[Dt]]":
        ...


class CzarrGpuNDBuffer[Dt: np.generic, *Shape]:
    """n-D buffer.  ``Dt`` is the element type; ``*Shape`` is the dim count.

    We use ``*Shape`` (PEP 646 / 3.11 ``TypeVarTuple``) so the array's
    rank is statically known; this catches "wrong-rank slice" bugs in
    the type checker.  Concrete shapes (e.g. ``[3, 256, 256]``) are
    NOT in the type — only the rank — because shape is runtime.
    """

    @classmethod
    def empty(
        cls,
        shape: tuple[*Shape],
        dtype: type[Dt],
        order: Literal["C", "F"] = "C",
    ) -> Self: ...

    def as_ndarray_like(self) -> "cp.ndarray[tuple[*Shape], np.dtype[Dt]]": ...
```

```python
# Usage with full type inference.

buf: CzarrGpuBuffer[np.float32] = CzarrGpuBuffer.empty(1024).view(np.float32)
arr_cp: cp.ndarray[Any, np.dtype[np.float32]] = buf.as_array_like()
# ↑ type checker knows dtype is float32; arithmetic with float64 array errors
```

### Array parameterised on dtype + rank

```python
# czarr/array/_facade.py

class CudaZarrArray[Dt: np.generic, *Shape](zarr.Array):
    """zarr.Array with a typed dtype and statically-known rank.

    Open without rank checking::

        arr = czarr.open_cuda_array("data.zarr")          # CudaZarrArray[Any, ...]

    Open with type annotation for downstream inference::

        arr: czarr.CudaZarrArray[np.float32, int, int, int] = czarr.open_cuda_array("data.zarr")
        slab: cp.ndarray[..., np.dtype[np.float32]] = arr[0, :, :]
    """

    @override
    def __getitem__(
        self, key: int, /
    ) -> "cp.ndarray[tuple[*Shape], np.dtype[Dt]]": ...

    @overload
    @override
    def __getitem__(
        self, key: slice, /
    ) -> "cp.ndarray[tuple[*Shape], np.dtype[Dt]]": ...

    @overload
    @override
    def __getitem__(
        self, key: tuple[int | slice | EllipsisType, ...], /
    ) -> "cp.ndarray[tuple[int, ...], np.dtype[Dt]]": ...
```

```python
# Usage.
arr: czarr.CudaZarrArray[np.float32, int, int, int] = czarr.open_cuda_array("data.zarr")
slab = arr[0:8, :, :]
# slab is typed as cp.ndarray[tuple[int, int, int], np.dtype[np.float32]]
# Editor autocompletes .mean(axis=0) → returns float32 too.
```

For users who can't annotate the dtype explicitly, an inference helper:

```python
def open_cuda_array[Dt: np.generic = Any, *Shape](
    store: zarr.abc.store.Store | str,
    *,
    dtype: type[Dt] = ...,   # caller can pin; otherwise Any
    **kwargs: Unpack[CudaArrayKwargs],
) -> CudaZarrArray[Dt, *Shape]: ...
```

### Lazy slice carries the dtype through DLPack handoff

```python
class _LazySlice[Dt: np.generic, *Shape]:
    """Captures (ranges, dtype, shape) without IO.

    Materialise via the protocol of choice — type checker tracks dtype
    through the conversion.
    """

    def __array__(self, dtype=None, copy=None) -> "np.ndarray[tuple[*Shape], np.dtype[Dt]]": ...

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...

    def __dlpack__(self, *, stream: int | None = None) -> object: ...

    # cupy zero-copy returns the typed array.
    def __cupy_array__(self) -> "cp.ndarray[tuple[*Shape], np.dtype[Dt]]": ...
```

### Buffer prototype generic — replaces today's NamedTuple

zarr's `BufferPrototype = NamedTuple(buffer, nd_buffer)` is untyped on
the buffer dtype.  We can wrap it:

```python
type BufferProto[Dt: np.generic] = tuple[type[CzarrGpuBuffer[Dt]], type[CzarrGpuNDBuffer[Dt, ...]]]

def buffer_prototype_for[Dt: np.generic](
    dtype: type[Dt],
) -> BufferProto[Dt]:
    """Return a typed (buffer, nd_buffer) pair for the given dtype."""
    return (
        type("CzarrGpuBuffer", (CzarrGpuBuffer,), {"__class_getitem__": ...}),
        ...,
    )
```

(In practice we register one untyped pair with zarr's registry; the
generics are for the caller-side type inference.)

### Protocol-based GPU buffer for codec function args

Codecs accept "any device buffer".  We don't want to enumerate concrete
types in every signature; a `Protocol` with generic parameters lets the
type checker validate compatibility without inheritance.

```python
# czarr/types.py

from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class CudaArrayLike[Dt: np.generic](Protocol):
    """Anything that exposes ``__cuda_array_interface__`` v3 over Dt."""

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...


@runtime_checkable
class DLPackable[Dt: np.generic](Protocol):
    """Anything that exposes ``__dlpack__``."""

    def __dlpack__(self, *, stream: int | None = None) -> object: ...

    def __dlpack_device__(self) -> tuple[int, int]: ...


# A codec accepts both protocols transparently.
type DeviceInput[Dt: np.generic] = CudaArrayLike[Dt] | DLPackable[Dt] | CzarrGpuBuffer[Dt]
```

```python
class LZ4(_BackendAware):
    def _decode_impl[Dt: np.uint8](
        self,
        items: list[tuple[DeviceInput[np.uint8], ArraySpec]],
    ) -> list[CzarrGpuBuffer[np.uint8]]: ...
```

Now `_decode_impl` accepts a cupy ndarray, a torch tensor (via DLPack),
or one of our buffers — type-checked.  And the return is statically
typed `CzarrGpuBuffer[np.uint8]`.

### Backend as a type parameter — when it's worth it

The pragmatic shape (`backend: CodecBackend` runtime field) is simpler.
But if we *do* want the type system to enforce backend pairings — e.g.
"a native codec only accepts native-decodable input" — we can encode
it:

```python
type Native = Literal["native"]
type Nvcomp = Literal["nvcomp"]
type CodecBackend = Native | Nvcomp


class Codec[B: CodecBackend](BytesBytesCodec):
    backend: B


class LZ4[B: CodecBackend = Native](Codec[B]):
    @overload
    def __init__(self: "LZ4[Native]", *, backend: Native = ..., **kw): ...
    @overload
    def __init__(self: "LZ4[Nvcomp]", *, backend: Nvcomp = ..., **kw): ...
```

```python
codec_n: LZ4[Native] = czarr.LZ4()                # default
codec_nv: LZ4[Nvcomp] = czarr.LZ4(backend="nvcomp")
codec_w: LZ4[Native] = czarr.LZ4(backend="nvcomp")  # type error
```

My read: **skip this** for v0.1.  It's clever but the maintenance cost
of overload-heavy constructors outweighs the catch.  Keep `backend` as
a plain `Literal[...]` field; type checker still catches typos via
`Literal`.  Revisit if we ever ship a codec where backends accept
different parameter types (e.g. `LZ4["native"]` taking a `block_size`
kwarg that `LZ4["nvcomp"]` doesn't).

### `Selector` as a discriminated union

```python
# czarr/types.py

type ScalarKey = int
type SliceKey = slice
type EllipsisKey = EllipsisType
type SimpleKey = ScalarKey | SliceKey | EllipsisKey
type TupleKey = tuple[SimpleKey, ...]
type BasicKey = SimpleKey | TupleKey   # what our fast path accepts

type BoolMaskKey = np.ndarray | cp.ndarray
type IntArrayKey = np.ndarray | cp.ndarray | list[int]
type AdvancedKey = BoolMaskKey | IntArrayKey

type Selector = BasicKey | AdvancedKey   # the union zarr.Array accepts
```

`_is_basic_indexing(key: Selector) -> TypeGuard[BasicKey]` lets the
type checker narrow on the fast path:

```python
from typing import TypeGuard

def _is_basic_indexing(key: Selector, /) -> TypeGuard[BasicKey]:
    if isinstance(key, int | slice | EllipsisType):
        return True
    if isinstance(key, tuple):
        return all(isinstance(k, int | slice | EllipsisType) for k in key)
    return False


class CudaZarrArray[Dt, *Shape]:
    @override
    def __getitem__(self, key: Selector, /) -> "cp.ndarray[Any, np.dtype[Dt]] | np.ndarray[Any, np.dtype[Dt]]":
        if _is_basic_indexing(key):  # key narrowed to BasicKey here
            return self._fast_path(key)
        return super().__getitem__(key)
```

### Stream pool — generic over stream type

Today's `StreamPool` returns `cuda.core.Stream`.  In the future we may
want `cp.cuda.Stream` for cupy-native paths.  Generic:

```python
class StreamPool[S: cuda.core.Stream | cp.cuda.Stream]:
    def acquire(self) -> S: ...
    def release(self, stream: S) -> None: ...
```

Probably not worth it in v0.1 — one stream type.  Listed for
completeness; revisit when we genuinely have two.

### What we deliberately do NOT make generic

- **Codec config dataclasses.**  Per-codec config dataclasses
  (`Zstd(level: int, ...)`, `LZ4(acceleration: int, ...)`) are
  closed-set with known fields.  Type the fields concretely; don't
  introduce a `Codec[ConfigDict]` parameter — it doesn't pay off in
  type-safety relative to the read overhead.
- **Stores.**  `zarr.abc.store.Store` is itself well-typed; no value
  in wrapping it in another generic layer.
- **Codecs over input/output dtype.**  Codecs operate on bytes
  (`BytesBytesCodec`) or n-d arrays (`ArrayArrayCodec`); the dtype
  is in the spec, not the codec.  Generic codec types would be over-
  engineering.
- **`ArraySpec` generics.**  zarr provides this; reproducing it
  parameterised is churn for no gain.

### Summary of recommended generics

| Type | Params | Why |
|---|---|---|
| `CzarrGpuBuffer[Dt]` | dtype | propagates to `as_array_like` |
| `CzarrGpuNDBuffer[Dt, *Shape]` | dtype + rank | catches wrong-rank slices |
| `CudaZarrArray[Dt, *Shape]` | dtype + rank | propagates to `__getitem__` |
| `_LazySlice[Dt, *Shape]` | dtype + rank | flows through DLPack handoff |
| `CudaArrayLike[Dt]` Protocol | dtype | codec input contract |
| `DLPackable[Dt]` Protocol | dtype | codec input contract |
| `Selector` (PEP 695 union) | — | discriminated union + TypeGuard |
| `BufferProto[Dt]` (alias) | dtype | typed prototype tuple |

What we punt on: backend-as-type-parameter (`Codec[B]`), stream-pool
generics, codec-config-as-type-parameter.  Revisit only if a concrete
need shows up.

## Decisions to confirm before implementing

1. **Default backend for LZ4: `"native"` vs `"nvcomp"`.** Spike says native
   wins 17.8× on A40 kernel-only. Once we wire it end-to-end, default to
   `"native"`? Or stay `"nvcomp"` for v0.1 and flip in v0.2 after a wider
   bench? My vote: ship Phase 1 with both, default `"native"` once the
   v0.1 bench shows H200 parity-or-better.

2. **`auto` backend mode.** Should we expose `backend="auto"` that picks
   per-call based on payload size? Native LZ4 may lose for tiny payloads
   (kernel launch cost vs nvCOMP's amortised path). Adds complexity; my
   vote: skip until profiling shows it's worth.

3. **`CudaZarrArray.wrap` vs `CudaZarrArray(...)` constructor.** zarrs PR
   #147 uses the constructor pattern. The `wrap` classmethod is more
   explicit about "this takes an already-opened array". Both? My vote:
   `wrap` is the documented path; the constructor exists for parity with
   `zarr.Array` but isn't featured.

4. **`use_cuda_array=True` auto-replacement.** Should `configure_gpu` set
   `zarr.open_array` to return `CudaZarrArray` automatically? Two
   sub-options: (a) monkey-patch at module level, (b) set a config flag
   that `czarr.open_cuda_array` reads. (a) is invasive — return-type
   contract differs (cupy vs numpy on fast path). My vote: **no auto-
   replacement**; require explicit `czarr.open_cuda_array` opt-in. Same
   conclusion as the synthesis.

5. **Type aliases — where do they live?** Single `czarr.types` module
   keeps imports clean. The downside is circular-import risk if
   `czarr.types` ever needs to import from concrete codec / array
   modules. My vote: types-only module is fine — only exports `type
   ...` aliases and `Protocol`s.

6. **`@override` discipline.** Adopt strictly — every method overridden
   from zarr / numcodecs ABCs gets `@override`. CI flag (mypy / pyright)
   on misses. Cheap, high signal.

7. **Codec metadata round-trip with `backend` field.** Does Zarr v3
   metadata round-trip `{"name": "lz4", "configuration": {"acceleration":
   1, "backend": "native"}}`? Reader/writer interop: a reader without
   the native backend should ignore the field and fall through to
   nvCOMP. czarr's `from_dict` already filters unknown keys; this means
   the field IS forward-compatible. **But** we should **NOT persist**
   the backend choice in metadata at all — backend is a runtime
   *implementation* choice, not a bitstream attribute. czarr's `to_dict`
   should drop `backend` from the configuration dict.

   *Recommended fix*: mark `backend` as `field(compare=False, metadata={"persist": False})`
   and filter in `to_dict`.

8. **Public preset enumeration.** Three presets: `H200`, `H100`, `A40_COMPAT`.
   Should we expose a `Preset.detect_current()` that picks based on
   `nvidia-smi --query-gpu=name` heuristic? Convenient but error-prone
   on multi-GPU hosts. My vote: keep `Preset.detect_current()` as an
   opt-in helper that emits a warning + a Preset object the user can
   accept or override; do NOT auto-apply.

## What this design does NOT solve

- Per-call nvCOMP overhead (~35-40 ms). That's a backend property, not
  an API one. The API just lets us choose the backend.
- GDS-direct firing on Bruno. Unrelated to API — storage backend
  problem.
- Multi-GPU. Future epic; not in v0.1 scope.
- Encoder rewrites. v0.1 LZ4 encode stays on nvCOMP; only decode is
  native. Documented in the codec docstring.

## Open question for the implementer

We haven't picked a name. Three contenders:

- `CudaZarrArray` — zarr-PR-style; most explicit.
- `CzarrArray` — matches the package name (czarr).
- `Array` (in `czarr.array.Array`) — bare in module-qualified usage:
  `czarr.Array(zarr.open_array(...))`.

My vote: `CudaZarrArray` reads as "Zarr Array with CUDA acceleration",
which is right. `czarr.CudaZarrArray` is verbose, but explicit; users
type it once via `from czarr import CudaZarrArray`. Stays consistent with
`zarr.Array` in the parent namespace.

## References

- `06-numcodecs-exploration.md` — codec ABC, registry, bridge details
- `04-cuda-array-architecture.md` — architectural deep-dive on `_impl`
- `SUMMARY.md` — overarching plan (this doc refines its API surface)
- Python 3.12 What's New: https://docs.python.org/3.12/whatsnew/3.12.html
- PEP 695 (type params): https://peps.python.org/pep-0695/
- PEP 698 (`@override`): https://peps.python.org/pep-0698/
- PEP 692 (`Unpack[TypedDict]`): https://peps.python.org/pep-0692/
- PEP 688 (`__buffer__`): https://peps.python.org/pep-0688/
- zarrs-python PR #147: https://github.com/zarrs/zarrs-python/pull/147
