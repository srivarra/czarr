"""Delta — GPU ArrayArrayCodec for first-differences encoding.

Forward: ``out[i] = arr[i] - arr[i-1]`` (with ``out[0] = arr[0]``).
Inverse: cumulative sum along axis 0.

Compatible with ``numcodecs.Delta`` (same on-disk values).  Useful for
numeric arrays where neighbouring elements correlate — typical for
time-series, monotonic indices, or pre-quantised imaging data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
from zarr.abc.codec import ArrayArrayCodec

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
    """

    is_fixed_size: ClassVar[bool] = True
    codec_name: ClassVar[str] = "delta"

    dtype: DTypeLike = "<f4"
    astype: DTypeLike | None = None

    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        arr = cp.asarray(chunk_data.as_ndarray_like())
        # Inverse delta = cumsum along axis 0.  numcodecs operates on a
        # flat view, so do the same here for byte-equivalence.
        flat = arr.ravel()
        out = cp.cumsum(flat, dtype=cp.dtype(self.dtype))
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
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
        """Serialise codec config for storage in Zarr v3 metadata."""
        config: dict[str, JSON] = {"dtype": str(self.dtype)}
        if self.astype is not None:
            config["astype"] = str(self.astype)
        return {"name": self.codec_name, "configuration": config}
