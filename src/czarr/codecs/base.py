"""Base class + shared types for GPU codecs.

A :class:`Codec` is a Zarr ``BytesBytesCodec`` that wraps an ``nvcomp.Codec``.
Subclasses bind an :class:`_Algorithm` and optionally tweak:

* :attr:`Codec._bitstream_kind` — picks ``NVCOMP_NATIVE`` (default, max perf,
  not interoperable with CPU codecs) or ``RAW`` / ``WITH_UNCOMPRESSED_SIZE``
  (interoperable with the standard format produced by libzstd/liblz4/etc).
* :attr:`Codec._frame_strip_head` / :attr:`Codec._frame_strip_tail` — bytes to
  trim from each compressed chunk before handing to nvCOMP (gzip/zlib wrap
  the deflate payload with header/trailer that nvCOMP doesn't parse).
* :meth:`Codec._wrap_frame` / :meth:`Codec._unwrap_frame` — host-side hook
  for codec-specific framing on encode/decode (CRC32 trailer for gzip,
  Adler-32 trailer for zlib).
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

import cupy as cp
from nvidia import nvcomp
from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer import gpu as gpu_buffer

from czarr._buffer import buffer_to_nvarray, nvarray_to_buffer
from czarr._nvtx import nvtx_range
from czarr.alloc import register_nvcomp_allocator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import JSON


class _Algorithm(StrEnum):
    """nvCOMP algorithm names — private; users pick a Codec subclass instead."""

    LZ4 = "LZ4"
    SNAPPY = "Snappy"
    ZSTD = "Zstd"
    DEFLATE = "Deflate"
    GDEFLATE = "GDeflate"
    BITCOMP = "Bitcomp"
    ANS = "ANS"
    CASCADED = "Cascaded"


class _BitstreamKind(StrEnum):
    """How nvCOMP interprets the compressed bitstream.

    * ``NVCOMP_NATIVE`` — nvCOMP's own chunked format; max parallelism + perf.
      Not consumable by any non-nvCOMP decoder.
    * ``RAW`` — exactly the algorithm's standard bitstream (e.g. RFC 8478
      zstd frame, RFC 1951 raw deflate, snappy block).  Interoperable with
      CPU implementations of the same algorithm.
    * ``WITH_UNCOMPRESSED_SIZE`` — standard bitstream with a 4-byte uncompressed
      size prefix.  Matches numcodecs.LZ4's on-disk format.
    """

    NVCOMP_NATIVE = "NVCOMP_NATIVE"
    RAW = "RAW"
    WITH_UNCOMPRESSED_SIZE = "WITH_UNCOMPRESSED_SIZE"

    def to_nvcomp(self) -> nvcomp.BitstreamKind:
        """Resolve to the nvCOMP ``BitstreamKind`` enum member of the same name."""
        return _BITSTREAM_KIND_MAP[self]


class Checksum(StrEnum):
    """Optional checksum policy for nvCOMP-NATIVE codecs."""

    NO_COMPUTE_NO_VERIFY = "NO_COMPUTE_NO_VERIFY"
    COMPUTE_AND_NO_VERIFY = "COMPUTE_AND_NO_VERIFY"
    NO_COMPUTE_AND_VERIFY_IF_PRESENT = "NO_COMPUTE_AND_VERIFY_IF_PRESENT"
    COMPUTE_AND_VERIFY_IF_PRESENT = "COMPUTE_AND_VERIFY_IF_PRESENT"
    COMPUTE_AND_VERIFY = "COMPUTE_AND_VERIFY"

    def to_nvcomp(self) -> nvcomp.ChecksumPolicy:
        """Resolve to the nvCOMP ``ChecksumPolicy`` enum member of the same name."""
        return _CHECKSUM_MAP[self]


# Static maps validated at import — if NVIDIA renames an enum member we crash
# here at module load, not on first codec creation in some random caller.
_BITSTREAM_KIND_MAP: dict[_BitstreamKind, nvcomp.BitstreamKind] = {
    _BitstreamKind.NVCOMP_NATIVE: nvcomp.BitstreamKind.NVCOMP_NATIVE,
    _BitstreamKind.RAW: nvcomp.BitstreamKind.RAW,
    _BitstreamKind.WITH_UNCOMPRESSED_SIZE: nvcomp.BitstreamKind.WITH_UNCOMPRESSED_SIZE,
}
_CHECKSUM_MAP: dict[Checksum, nvcomp.ChecksumPolicy] = {
    Checksum.NO_COMPUTE_NO_VERIFY: nvcomp.ChecksumPolicy.NO_COMPUTE_NO_VERIFY,
    Checksum.COMPUTE_AND_NO_VERIFY: nvcomp.ChecksumPolicy.COMPUTE_AND_NO_VERIFY,
    Checksum.NO_COMPUTE_AND_VERIFY_IF_PRESENT: nvcomp.ChecksumPolicy.NO_COMPUTE_AND_VERIFY_IF_PRESENT,
    Checksum.COMPUTE_AND_VERIFY_IF_PRESENT: nvcomp.ChecksumPolicy.COMPUTE_AND_VERIFY_IF_PRESENT,
    Checksum.COMPUTE_AND_VERIFY: nvcomp.ChecksumPolicy.COMPUTE_AND_VERIFY,
}


def _is_gpu_prototype(prototype: BufferPrototype) -> bool:
    return issubclass(prototype.buffer, gpu_buffer.Buffer)


@dataclass(frozen=True)
class Codec(BytesBytesCodec):
    """Base class for GPU codecs backed by nvCOMP.

    Concrete subclasses set the ``_algorithm`` ClassVar and the public
    ``codec_name`` (which determines the Zarr metadata ``name`` field — and
    therefore which codec zarr picks when reading the file back).

    Subclasses that target a non-native bitstream (Zstd, LZ4, Gzip, Zlib)
    also set ``_bitstream_kind`` and any framing knobs.
    """

    is_fixed_size: ClassVar[bool] = False
    codec_name: ClassVar[str] = ""
    _algorithm: ClassVar[_Algorithm]
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.NVCOMP_NATIVE
    _frame_strip_head: ClassVar[int] = 0
    _frame_strip_tail: ClassVar[int] = 0

    chunk_size: int = 65536
    checksum_policy: Checksum = Checksum.NO_COMPUTE_NO_VERIFY
    device_id: int | None = None
    # Runtime-only: bind nvCOMP work to a user-owned CUDA stream.  Excluded
    # from equality + serialisation because it is execution context, not
    # codec identity.  Accepts a raw int, ``cuda.core.Stream``,
    # ``cupy.cuda.Stream``, ``rmm.pylibrmm.stream.Stream``, or any object
    # exposing ``__cuda_stream__()``.
    cuda_stream: Any | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Frozen dataclass; bypass __setattr__ to attach the per-instance
        # thread-local cache.  nvCOMP's Codec holds mutable scratch state +
        # is documented thread-local, so we cache one instance per thread.
        object.__setattr__(self, "_thread_local", threading.local())
        register_nvcomp_allocator()

    def _codec_kwargs(self) -> dict[str, Any]:
        """Hook for subclasses to surface algorithm-specific nvCOMP kwargs."""
        return {}

    @staticmethod
    def _resolve_stream(stream: Any) -> int | None:
        """Coerce a stream-like into a ``cudaStream_t`` int.

        Accepted: ``None``, a raw int, or an object exposing the NVIDIA
        ``__cuda_stream__()`` protocol (``cuda.core.Stream``,
        ``cupy.cuda.Stream``, ``rmm.pylibrmm.stream.Stream``, plus anything
        else that implements it).  Everything else raises ``TypeError`` —
        no silent best-effort ``int(stream)`` coercion that may produce a
        bogus handle and crash deep inside cuFile.
        """
        if stream is None:
            return None
        if isinstance(stream, int):
            return stream
        proto = stream.__cuda_stream__ if hasattr(type(stream), "__cuda_stream__") else None
        if proto is None:
            raise TypeError(f"cuda_stream must be None, int, or implement __cuda_stream__; got {type(stream).__name__}")
        version, ptr = proto()
        if version != 0:
            raise ValueError(f"unsupported __cuda_stream__ protocol version: {version}")
        return int(ptr)

    def _create_codec(self) -> nvcomp.Codec:
        kwargs: dict[str, Any] = {
            "algorithm": str(self._algorithm),
            "bitstream_kind": self._bitstream_kind.to_nvcomp(),
            "uncomp_chunk_size": self.chunk_size,
            "checksum_policy": self.checksum_policy.to_nvcomp(),
        }
        if self.device_id is not None:
            kwargs["device_id"] = self.device_id
        stream_handle = self._resolve_stream(self.cuda_stream)
        if stream_handle is not None:
            kwargs["cuda_stream"] = stream_handle
        kwargs.update(self._codec_kwargs())
        return nvcomp.Codec(**kwargs)

    def _get_codec(self) -> nvcomp.Codec:
        try:
            return self._thread_local.codec
        except AttributeError:
            codec = self._create_codec()
            self._thread_local.codec = codec
            return codec

    def get_stream(self) -> int:
        """Return the ``cudaStream_t`` handle nvCOMP runs this codec on.

        Useful for binding downstream user kernels to the same stream as
        the codec output (e.g. ``cupy.cuda.ExternalStream(codec.get_stream())``
        avoids an implicit sync between decode and the next compute).
        """
        try:
            return self._thread_local.stream_handle
        except AttributeError:
            pass
        explicit = self._resolve_stream(self.cuda_stream)
        if explicit is not None:
            self._thread_local.stream_handle = explicit
            return explicit
        probe = cp.empty(64, dtype=cp.uint8)
        encoded = self._get_codec().encode(nvcomp.as_array(probe))
        handle = int(encoded.__cuda_array_interface__["stream"])
        self._thread_local.stream_handle = handle
        return handle

    # ------------------------------------------------------------------
    # Framing hooks — overridden by gzip/zlib to wrap/unwrap host-side.
    # ------------------------------------------------------------------

    def _unwrap_frame(self, compressed: memoryview) -> memoryview:
        """Strip codec-specific framing before handing bytes to nvCOMP."""
        if self._frame_strip_head == 0 and self._frame_strip_tail == 0:
            return compressed
        end = len(compressed) - self._frame_strip_tail if self._frame_strip_tail else len(compressed)
        return compressed[self._frame_strip_head : end]

    def _wrap_frame(self, compressed: bytes, original: memoryview) -> bytes:
        """Add codec-specific framing after nvCOMP encode.  Default no-op."""
        return compressed

    # ------------------------------------------------------------------
    # Batched encode/decode — Zarr passes a list, we call nvCOMP once.
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_decoded_bytes(spec: ArraySpec) -> int:
        item_size = getattr(spec.dtype, "item_size", None) or spec.dtype.to_native_dtype().itemsize
        n = 1
        for dim in spec.shape:
            n *= int(dim)
        return n * int(item_size)

    def _batch_sync(
        self,
        items: list[tuple[Buffer | None, ArraySpec]],
        op: str,
    ) -> list[Buffer | None]:
        with nvtx_range(f"czarr.codec.{op}", n=len(items), algo=str(self._algorithm)):
            non_null_indices: list[int] = []
            specs: list[ArraySpec] = []
            originals: list[Buffer] = []
            with nvtx_range("czarr.codec.wrap_inputs"):
                for i, (chunk, spec) in enumerate(items):
                    if chunk is None:
                        continue
                    non_null_indices.append(i)
                    specs.append(spec)
                    originals.append(chunk)

            out: list[Buffer | None] = [None] * len(items)
            if not originals:
                return out

            codec = self._get_codec()

            if op == "decode":
                # Apply frame strip per-chunk before nvCOMP.  For NVCOMP_NATIVE
                # codecs both hooks are no-ops and this is just a view.
                nv_inputs: list[nvcomp.Array] = []
                for chunk in originals:
                    stripped = self._unwrap_frame(memoryview(chunk.to_bytes()))
                    nv_inputs.append(nvcomp.as_array(bytes(stripped)).cuda())

                with nvtx_range("czarr.codec.alloc_outs"):
                    decode_outs = [cp.empty(self._expected_decoded_bytes(spec), dtype=cp.uint8) for spec in specs]
                with nvtx_range("czarr.codec.nvcomp_decode"):
                    codec.decode(nv_inputs, out=decode_outs)
                with nvtx_range("czarr.codec.wrap_outputs"):
                    for idx, dev, spec in zip(non_null_indices, decode_outs, specs, strict=True):
                        if _is_gpu_prototype(spec.prototype):
                            out[idx] = spec.prototype.buffer.from_array_like(dev)
                        else:
                            out[idx] = spec.prototype.buffer.from_bytes(cp.asnumpy(dev).tobytes())
                return out

            # Encode
            nv_inputs = [buffer_to_nvarray(chunk) for chunk in originals]
            with nvtx_range("czarr.codec.nvcomp_encode"):
                nv_outputs = codec.encode(nv_inputs)
            with nvtx_range("czarr.codec.wrap_outputs"):
                for idx, nv_out, spec, original in zip(non_null_indices, nv_outputs, specs, originals, strict=True):
                    buf = nvarray_to_buffer(nv_out, spec.prototype)
                    if self._frame_strip_head or self._frame_strip_tail:
                        # Codec produced raw bytestream; need to wrap before storing.
                        wrapped = self._wrap_frame(buf.to_bytes(), memoryview(original.to_bytes()))
                        buf = spec.prototype.buffer.from_bytes(wrapped)
                    out[idx] = buf
            return out

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Encode a batch of chunks via a single batched nvCOMP call."""
        items = list(chunks_and_specs)
        return await asyncio.to_thread(self._batch_sync, items, "encode")

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Decode a batch of chunks via a single batched nvCOMP call."""
        items = list(chunks_and_specs)
        return await asyncio.to_thread(self._batch_sync, items, "decode")

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        """Encoded size is data-dependent for compressors; raise to signal unknown."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Metadata serialisation — config fields round-trip in Zarr metadata.
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr metadata."""
        config: dict[str, JSON] = {}
        for f in fields(self):
            if not f.compare:
                continue
            value = getattr(self, f.name)
            if isinstance(value, Checksum):
                value = value.value
            config[f.name] = value
        return {"name": self.codec_name, "configuration": config}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Reconstruct a codec from its ``to_dict`` payload (tolerant of CPU schemas)."""
        raw = data.get("configuration", {})
        if not isinstance(raw, dict):
            raise TypeError(f"codec configuration must be a mapping, got {type(raw).__name__}")
        # Accept CPU-codec config schemas (e.g. zarr's ZstdCodec has
        # {"level", "checksum"}; we ignore them since nvCOMP picks its own).
        valid = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {k: v for k, v in raw.items() if k in valid}
        if "checksum_policy" in filtered and isinstance(filtered["checksum_policy"], str):
            filtered["checksum_policy"] = Checksum(filtered["checksum_policy"])
        return cls(**filtered)
