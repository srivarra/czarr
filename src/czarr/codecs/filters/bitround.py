"""BitRound — GPU ArrayArrayCodec for IEEE-754 mantissa truncation.

Forward (encode): masks off ``mantissa_bits - keepbits`` low bits of the
mantissa, rounding **half-to-even** (banker's rounding) to match
``numcodecs.BitRound``.  Decode is the identity (the bits are already
truncated; the value is recoverable as-is).

Compatible with ``numcodecs.BitRound``.  Useful for lossy compression of
floats where you don't care about sub-percent precision — the truncated
trailing zeros compress extremely well downstream.

No backend dispatch: BitRound is pure integer bit-twiddling, which
cupy's elementwise machinery already fuses well.  cuda.compute would
need a bespoke make_unary_transform closure with no clear performance
upside (no kernel-launch savings vs cupy on small N, no algorithmic
parallelism on large N).
"""

from dataclasses import dataclass
from typing import ClassVar, Self

import cupy as cp
import numpy as np
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import NDBuffer
from zarr.core.common import JSON

from czarr.codecs.base import codec_config

# Mantissa bit-width per float dtype.
_MANTISSA_BITS = {
    cp.dtype("float16"): 10,
    cp.dtype("float32"): 23,
    cp.dtype("float64"): 52,
}

# Integer view dtype per float itemsize — used to manipulate the bit
# pattern in-place.
_INT_VIEW = {2: cp.uint16, 4: cp.uint32, 8: cp.uint64}


@dataclass(frozen=True)
class BitRound(ArrayArrayCodec):
    """GPU IEEE-754 mantissa truncation — numcodecs-compatible.

    Parameters
    ----------
    keepbits:
        Number of mantissa bits to retain.  Trailing
        ``mantissa_bits - keepbits`` bits are masked off using
        round-half-to-even (banker's rounding) — same semantics as
        ``numcodecs.BitRound``.
    """

    is_fixed_size: ClassVar[bool] = True
    codec_name: ClassVar[str] = "bitround"

    keepbits: int = 12

    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        # Bits already truncated on encode; decode is identity.  Return
        # a buffer of the right prototype so downstream codecs see a
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

        out = _bitround_even(x, shift)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        """Encoded size equals input — only mantissa bits change."""
        return input_byte_length

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr v3 metadata."""
        return {
            "name": self.codec_name,
            "configuration": {"keepbits": int(self.keepbits)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Reconstruct from Zarr v3 metadata: {'name': ..., 'configuration': {...}}."""
        return cls(**codec_config(data))


# Fused banker's-rounding kernel — one launch per chunk regardless of
# dtype.  The naive composed-cupy version dispatches ~10 separate
# elementwise kernels (mask + shift + cmp + cmp + and + or + and + add)
# and is 2.3x slower than the (incorrect) legacy add-half code on H100
# 16 MiB f32.  Fusing collapses everything into one device pass.
_BITROUND_EVEN_KERNEL = cp.ElementwiseKernel(
    in_params="T bits, uint64 shift, T half, T low_mask, T lsb_step",
    out_params="T out",
    operation=r"""
    T low = bits & low_mask;
    T lsb = (bits >> shift) & (T)1;
    bool round_up = (low > half) || ((low == half) && (lsb != (T)0));
    T truncated = bits & ~low_mask;
    out = truncated + (round_up ? lsb_step : (T)0);
    """,
    name="czarr_bitround_even",
)


def _bitround_even(x: cp.ndarray, shift: int) -> cp.ndarray:
    """Truncate ``shift`` low mantissa bits of ``x`` with round-half-to-even.

    Standard banker's-rounding bit twiddle, fused into one
    ``ElementwiseKernel`` launch.  For each value the discarded low
    ``shift`` bits are split into the half-bit (position ``shift-1``)
    and the sticky bits (positions ``0..shift-2``); we round up when:

    * sticky != 0 and half-bit is 1  (strictly greater than half), OR
    * sticky == 0, half-bit is 1, and the surviving lsb (bit at
      position ``shift``) is 1  (exactly half + odd lsb → round to even).

    Operates on the integer view of ``x`` so the sign and exponent
    bits ride along unchanged — only mantissa is affected.
    """
    int_dtype = cp.dtype(_INT_VIEW[x.dtype.itemsize])
    bits = x.view(int_dtype)
    one = int_dtype.type(1)
    shift_t = int_dtype.type(shift)
    half = one << (shift_t - one)
    low_mask = (one << shift_t) - one
    lsb_step = one << shift_t
    out_int = cp.empty_like(bits)
    _BITROUND_EVEN_KERNEL(bits, np.uint64(shift), half, low_mask, lsb_step, out_int)
    return out_int.view(x.dtype)
