"""Native FixedScaleOffset filter — cuda.compute unary_transform.

FSO encode: ``q = round((x - offset) * scale).astype(astype)``.
FSO decode: ``x = q.astype(dtype) / scale + offset``.

Both are elementwise unary transforms — :func:`cuda.compute.make_unary_transform`
is the natural fit.  The transform op closes over the Python scalars
``offset`` and ``scale``; numba-cuda JITs that into a typed device
function.

cuda.compute does the dtype dispatch via the input/output array dtypes,
so encode (float→int) and decode (int→float) reuse the same machinery
without separate kernels per direction.
"""

from typing import Any

import cupy as cp
import numpy as np
from numpy.typing import DTypeLike

from czarr.codecs._backends import import_cccl

# (op_kind, in_dtype, out_dtype, scale_bits, offset_bits) → cached transformer.
# We key on the IEEE-754 bit pattern of the scalars so two FSO codecs with
# the same numeric params share one LTO-compiled kernel.
_TRANSFORM_CACHE: dict[Any, Any] = {}


def _make_decoder(
    arr_in: cp.ndarray,
    arr_out: cp.ndarray,
    *,
    scale: float,
    offset: float,
) -> Any:
    """Build (or fetch from cache) a cuda.compute unary transformer for decode."""
    key = (
        "decode",
        arr_in.dtype,
        arr_out.dtype,
        float(scale).hex(),
        float(offset).hex(),
    )
    transformer = _TRANSFORM_CACHE.get(key)
    if transformer is not None:
        return transformer
    cc = import_cccl()
    s = float(scale)
    o = float(offset)

    def _decode_op(q):
        return q / s + o

    transformer = cc.make_unary_transform(d_in=arr_in, d_out=arr_out, op=_decode_op)
    _TRANSFORM_CACHE[key] = (transformer, _decode_op)
    return _TRANSFORM_CACHE[key]


def _make_encoder(
    arr_in: cp.ndarray,
    arr_out: cp.ndarray,
    *,
    scale: float,
    offset: float,
    integer_out: bool,
) -> Any:
    """Build (or fetch from cache) a cuda.compute unary transformer for encode."""
    key = (
        "encode",
        arr_in.dtype,
        arr_out.dtype,
        float(scale).hex(),
        float(offset).hex(),
        integer_out,
    )
    transformer = _TRANSFORM_CACHE.get(key)
    if transformer is not None:
        return transformer
    cc = import_cccl()
    s = float(scale)
    o = float(offset)
    if integer_out:
        # round-to-nearest then cast.  Use the same semantics as cupy.around
        # (banker's rounding) so the bitstream matches the cupy backend.
        def _encode_op(x):
            return round((x - o) * s)
    else:

        def _encode_op(x):
            return (x - o) * s

    transformer = cc.make_unary_transform(d_in=arr_in, d_out=arr_out, op=_encode_op)
    _TRANSFORM_CACHE[key] = (transformer, _encode_op)
    return _TRANSFORM_CACHE[key]


def decode_fso_native(
    encoded: cp.ndarray,
    *,
    dtype: DTypeLike,
    scale: float,
    offset: float,
    out: cp.ndarray | None = None,
) -> cp.ndarray:
    """Inverse FSO via :func:`cuda.compute.make_unary_transform`.

    Parameters
    ----------
    encoded
        Device-resident input (flat 1-D view of the chunk's stored values).
    dtype
        Working dtype to upcast the encoded values to before the affine
        transform — typically a float (numcodecs decodes to ``dtype`` even
        when the storage was integer).
    scale, offset
        Affine parameters.
    out
        Optional pre-allocated output buffer.  Allocated fresh if omitted.
    """
    if encoded.ndim != 1:
        raise ValueError(f"decode_fso_native: ndim={encoded.ndim}, expected 1")
    promoted = encoded if encoded.dtype == cp.dtype(dtype) else encoded.astype(dtype, copy=False)
    if out is None:
        out = cp.empty(promoted.size, dtype=cp.dtype(dtype))
    elif out.dtype != cp.dtype(dtype) or out.size != promoted.size:
        raise ValueError(
            f"decode_fso_native: out dtype/size mismatch "
            f"(got {out.dtype}/{out.size}, want {cp.dtype(dtype)}/{promoted.size})"
        )
    transformer, op = _make_decoder(promoted, out, scale=scale, offset=offset)
    transformer(d_in=promoted, d_out=out, op=op, num_items=promoted.size)
    return out


def encode_fso_native(
    arr: cp.ndarray,
    *,
    astype: DTypeLike,
    scale: float,
    offset: float,
    out: cp.ndarray | None = None,
) -> cp.ndarray:
    """Forward FSO via :func:`cuda.compute.make_unary_transform`.

    Rounds before casting when ``astype`` is an integer dtype, to match
    the lossy-quantisation semantics expected by ``numcodecs.FixedScaleOffset``.
    """
    if arr.ndim != 1:
        raise ValueError(f"encode_fso_native: ndim={arr.ndim}, expected 1")
    store_dtype = cp.dtype(astype)
    integer_out = bool(np.issubdtype(store_dtype, np.integer))
    if out is None:
        out = cp.empty(arr.size, dtype=store_dtype)
    elif out.dtype != store_dtype or out.size != arr.size:
        raise ValueError(
            f"encode_fso_native: out dtype/size mismatch (got {out.dtype}/{out.size}, want {store_dtype}/{arr.size})"
        )
    transformer, op = _make_encoder(arr, out, scale=scale, offset=offset, integer_out=integer_out)
    transformer(d_in=arr, d_out=out, op=op, num_items=arr.size)
    return out
