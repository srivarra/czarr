"""FixedScaleOffset — GPU ArrayArrayCodec for affine quantisation.

Forward (encode): ``q = round((x - offset) * scale).astype(astype)``.
Inverse (decode): ``x = q.astype(dtype) / scale + offset``.

Compatible with ``numcodecs.FixedScaleOffset``.  The typical use is
``dtype=float32, astype=int16`` for lossy compression of bounded-range
floats: store quantised int16s on disk, recover floats on read.
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
class FixedScaleOffset(ArrayArrayCodec):
    """GPU affine-quantisation filter — numcodecs-compatible ArrayArrayCodec.

    Parameters
    ----------
    offset:
        Additive offset (subtracted on encode, added on decode).
    scale:
        Multiplicative scale (multiplied on encode, divided on decode).
    dtype:
        Source / decoded dtype (typically float).
    astype:
        Storage dtype after quantisation (typically integer).  ``None``
        = same as ``dtype`` (no real quantisation, useful only for
        round-trip testing).
    """

    is_fixed_size: ClassVar[bool] = True
    codec_name: ClassVar[str] = "fixedscaleoffset"

    offset: float = 0.0
    scale: float = 1.0
    dtype: DTypeLike = "<f4"
    astype: DTypeLike | None = None

    @property
    def _store_dtype(self) -> DTypeLike:
        return self.astype if self.astype is not None else self.dtype

    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        q = cp.asarray(chunk_data.as_ndarray_like())
        # cast to working dtype, undo the affine transform.
        x = q.astype(self.dtype, copy=False)
        x = x / self.scale + self.offset
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(x.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        x = cp.asarray(chunk_data.as_ndarray_like())
        q = (x - self.offset) * self.scale
        # Round-to-nearest only when storing as integer (the typical lossy case);
        # for float-to-float we leave the scaled values intact.
        if cp.issubdtype(cp.dtype(self._store_dtype), cp.integer):
            q = cp.around(q)
        q = q.astype(self._store_dtype, copy=False)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(q.reshape(chunk_spec.shape))

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        """Encoded size depends on the storage dtype's itemsize."""
        if self.astype is None:
            return input_byte_length
        item = cp.dtype(self._store_dtype).itemsize
        src_item = chunk_spec.dtype.to_native_dtype().itemsize
        return (input_byte_length // src_item) * item

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr v3 metadata."""
        config: dict[str, JSON] = {
            "offset": float(self.offset),
            "scale": float(self.scale),
            "dtype": str(self.dtype),
        }
        if self.astype is not None:
            config["astype"] = str(self.astype)
        return {"name": self.codec_name, "configuration": config}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> FixedScaleOffset:
        """Reconstruct from Zarr v3 metadata: {'name': ..., 'configuration': {...}}."""
        cfg = data.get("configuration", {k: v for k, v in data.items() if k != "name"})
        return cls(**cfg)
