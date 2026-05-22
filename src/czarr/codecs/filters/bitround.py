"""BitRound — GPU ArrayArrayCodec for IEEE-754 mantissa truncation.

Forward (encode): masks off ``mantissa_bits - keepbits`` low bits of the
mantissa, rounding to even.  Decode is the identity (the bits are
already truncated; the value is recoverable as-is).

Compatible with ``numcodecs.BitRound``.  Useful for lossy compression of
floats where you don't care about sub-percent precision — the truncated
trailing zeros compress extremely well downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
from zarr.abc.codec import ArrayArrayCodec

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.common import JSON


# Mantissa bit-width per float dtype.
_MANTISSA_BITS = {
    cp.dtype("float16"): 10,
    cp.dtype("float32"): 23,
    cp.dtype("float64"): 52,
}


@dataclass(frozen=True)
class BitRound(ArrayArrayCodec):
    """GPU IEEE-754 mantissa truncation — numcodecs-compatible.

    Parameters
    ----------
    keepbits:
        Number of mantissa bits to retain.  Trailing
        ``mantissa_bits - keepbits`` bits are masked off (with
        round-to-even).
    """

    is_fixed_size: ClassVar[bool] = True
    codec_name: ClassVar[str] = "bitround"

    keepbits: int = 12

    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        # Bits already truncated on encode; decode is identity.  Return a
        # buffer of the right prototype so downstream codecs see a
        # device-side array even if the input was host.
        arr = chunk_data.as_ndarray_like()
        out = cp.asarray(arr) if not isinstance(arr, cp.ndarray) else arr
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        x = cp.asarray(chunk_data.as_ndarray_like())
        if x.dtype not in _MANTISSA_BITS:
            raise TypeError(f"BitRound only supports IEEE-754 floats (float16/32/64); got {x.dtype}")
        mantissa = _MANTISSA_BITS[x.dtype]
        shift = mantissa - int(self.keepbits)
        if shift <= 0:
            return chunk_spec.prototype.nd_buffer.from_ndarray_like(x)

        # Round-to-nearest-even via add-half then mask-truncate, using
        # an integer view to manipulate the bit pattern.
        int_dtype = cp.dtype({2: "uint16", 4: "uint32", 8: "uint64"}[x.dtype.itemsize])
        bits = x.view(int_dtype).copy()
        # Add half-ULP at the truncation boundary; even-rounding by OR'ing
        # the sticky bit comes free with cuda integer add.
        half = cp.uint64(1) << cp.uint64(shift - 1)
        mask = ~((cp.uint64(1) << cp.uint64(shift)) - cp.uint64(1))
        bits = (bits.astype(cp.uint64) + half) & mask
        out = bits.astype(int_dtype).view(x.dtype)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        """Encoded size equals input — only mantissa bits change."""
        return input_byte_length

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr v3 metadata."""
        return {
            "name": self.codec_name,
            "configuration": {"keepbits": int(self.keepbits)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> BitRound:
        """Reconstruct from Zarr v3 metadata: {'name': ..., 'configuration': {...}}."""
        cfg = data.get("configuration", {k: v for k, v in data.items() if k != "name"})
        return cls(**cfg)
