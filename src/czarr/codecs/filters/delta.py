"""Delta — GPU ArrayArrayCodec for first-differences encoding.

Forward: ``out[i] = arr[i] - arr[i-1]`` (with ``out[0] = arr[0]``).
Inverse: cumulative sum along axis 0.

Compatible with ``numcodecs.Delta`` (same on-disk values).  Useful for
numeric arrays where neighbouring elements correlate — typical for
time-series, monotonic indices, or pre-quantised imaging data.

Two backends:

* ``backend="cupy"`` (default) — ``cp.cumsum`` decode, ``cp.diff`` encode.
  Battle-tested, no extra dependencies.
* ``backend="cccl"`` — :mod:`cuda.compute` ``make_inclusive_scan`` decode.
  Validated 1.28x faster than cupy on H100 / 4 MiB int32 (Phase 2 spike).
  Requires ``cuda-cccl`` + ``numba-cuda``.

Both backends produce bit-identical output to numcodecs and to each
other.  The ``backend`` field is runtime — never persisted in Zarr v3
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
from zarr.abc.codec import ArrayArrayCodec

from czarr.codecs._backend import CodecBackend, resolve_backend_for_filter

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.common import JSON


@dataclass(frozen=True)
class Delta(ArrayArrayCodec):
    """GPU delta filter — numcodecs-compatible ArrayArrayCodec.

    Parameters
    ----------
    dtype:
        Source array dtype (numcodecs metadata field; carried for
        round-trip).  Output of forward delta has the same dtype.
    astype:
        Optional dtype to cast the delta output to (typically a narrower
        signed integer when the values fit).  ``None`` = same as ``dtype``.
    backend:
        Runtime impl choice.  ``"cupy"`` (default) uses cupy primitives;
        ``"cccl"`` uses :mod:`cuda.compute` inclusive_scan.  Not persisted.
    """

    is_fixed_size: ClassVar[bool] = True
    codec_name: ClassVar[str] = "delta"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("cupy", "cccl")
    _default_backend: ClassVar[CodecBackend] = "cupy"

    dtype: DTypeLike = "<f4"
    astype: DTypeLike | None = None
    backend: CodecBackend | None = field(default=None, compare=False, repr=True)

    def __post_init__(self) -> None:
        chosen = resolve_backend_for_filter(
            self.codec_name,
            instance_backend=self.backend,
            supported=self._supported_backends,
            default=self._default_backend,
        )
        object.__setattr__(self, "backend", chosen)

    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        arr = cp.asarray(chunk_data.as_ndarray_like())
        flat = arr.ravel()
        if self.backend == "cccl":
            from czarr.codecs._native.delta import decode_delta_native

            # cuda.compute scan needs the working dtype; cast if astype
            # narrowed the encoded values.  numcodecs preserves source
            # dtype on decode, so we end up at the same shape as cupy.
            working = flat if flat.dtype == cp.dtype(self.dtype) else flat.astype(self.dtype, copy=False)
            out = decode_delta_native(working)
        else:
            out = cp.cumsum(flat, dtype=cp.dtype(self.dtype))
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        # Encode is the same for both backends — cuda.compute has no
        # inverse-of-scan primitive; cupy diff is fine.  Kept in one
        # place since encode isn't a hot path for v0.1.
        arr = cp.asarray(chunk_data.as_ndarray_like())
        flat = arr.ravel().astype(self.dtype, copy=False)
        out = cp.empty_like(flat)
        out[0] = flat[0]
        out[1:] = cp.diff(flat)
        if self.astype is not None:
            out = out.astype(self.astype, copy=False)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        """Encoded size equals input — same element count, same dtype."""
        return input_byte_length

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr v3 metadata.

        ``backend`` is intentionally omitted — bitstream is the only
        persisted identity (same rule as :class:`czarr.LZ4`).
        """
        config: dict[str, JSON] = {"dtype": str(self.dtype)}
        if self.astype is not None:
            config["astype"] = str(self.astype)
        return {"name": self.codec_name, "configuration": config}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Delta:
        """Reconstruct from Zarr v3 metadata: {'name': ..., 'configuration': {...}}.

        Tolerant of writers that leak ``backend`` into the configuration —
        the field is stripped before constructing.
        """
        cfg = dict(data.get("configuration", {k: v for k, v in data.items() if k != "name"}))
        cfg.pop("backend", None)
        return cls(**cfg)
