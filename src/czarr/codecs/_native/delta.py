"""Native Delta filter — cuda.compute scan-based encode/decode.

Delta encode: first-differences, ``out[i] = arr[i] - arr[i-1]``, with
``out[0] = arr[0]``.
Delta decode: cumulative sum.

cuda.compute primitives:

* Decode → :func:`cuda.compute.make_inclusive_scan` with ``a + b``.
  Validated 1.28x faster than ``cupy.cumsum`` on H100 for 4 MiB int32
  (Phase 2 spike at ``.planning/research/cuda-array/spikes/delta_cuda_compute.py``).
* Encode → element-wise ``arr[i] - arr[i-1]``.  cuda.compute has no
  inverse-of-scan; we use ``cupy.diff`` for the bulk + a single
  scalar write for ``out[0]``.  Encode isn't on a hot path for v0.1.

Bit-exact with ``numcodecs.Delta`` for the same dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np

from czarr.codecs._native import import_cccl

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


# (dtype) → cached cuda.compute scanner.  Per-dtype because the scanner
# closes over the operator's typed lambda — different dtypes need
# different LTO-compiled kernels.
_SCANNER_CACHE: dict[Any, Any] = {}


def _scanner_for(arr_in: cp.ndarray, arr_out: cp.ndarray) -> Any:
    """Return a cached ``make_inclusive_scan`` reducer for the dtype.

    cuda.compute does its own internal compile cache keyed on the
    callable identity + arg shapes; we add a Python-level cache so the
    construction doesn't even get probed on every call.
    """
    key = (arr_in.dtype, arr_in.size, arr_out.dtype)
    scanner = _SCANNER_CACHE.get(key)
    if scanner is None:

        def _add(a, b):
            return a + b

        cc = import_cccl()
        scanner = cc.make_inclusive_scan(d_in=arr_in, d_out=arr_out, op=_add)
        _SCANNER_CACHE[key] = scanner
    return scanner


def decode_delta_native(encoded: cp.ndarray, *, out: cp.ndarray | None = None) -> cp.ndarray:
    """Inverse Delta via cuda.compute inclusive scan.

    Parameters
    ----------
    encoded
        Device-resident input.  Any signed integer or float dtype that
        supports ``+``.  Flat 1-D view of the chunk's values.
    out
        Optional pre-allocated output buffer.  Same dtype + shape as
        ``encoded``.  Allocated fresh if omitted.

    Returns
    -------
    cp.ndarray
        Cumulative sum, same dtype + shape as ``encoded``.
    """
    if encoded.ndim != 1:
        raise ValueError(f"decode_delta_native: ndim={encoded.ndim}, expected 1")
    if out is None:
        out = cp.empty_like(encoded)
    elif out.dtype != encoded.dtype or out.size != encoded.size:
        raise ValueError(
            f"decode_delta_native: out dtype/size mismatch "
            f"(got {out.dtype}/{out.size}, want {encoded.dtype}/{encoded.size})"
        )

    scanner = _scanner_for(encoded, out)
    init_value = np.array(0, dtype=encoded.dtype)

    def _add(a, b):
        return a + b

    # Two-step pattern: query temp storage size, allocate, invoke.
    temp_bytes = scanner(
        temp_storage=None,
        d_in=encoded,
        d_out=out,
        num_items=encoded.size,
        op=_add,
        init_value=init_value,
    )
    temp = cp.empty(temp_bytes, dtype=cp.uint8) if temp_bytes else None
    scanner(
        temp_storage=temp,
        d_in=encoded,
        d_out=out,
        num_items=encoded.size,
        op=_add,
        init_value=init_value,
    )
    return out


def encode_delta_native(arr: cp.ndarray, *, dtype: DTypeLike, out: cp.ndarray | None = None) -> cp.ndarray:
    """Forward Delta — ``out[i] = arr[i] - arr[i-1]``, ``out[0] = arr[0]``.

    cuda.compute has no built-in adjacent-difference primitive, so we
    use ``cupy.diff`` + a scalar write.  Acceptable since encode isn't a
    hot path; revisit if a real workload changes that.
    """
    flat = arr.ravel().astype(dtype, copy=False)
    if out is None:
        out = cp.empty_like(flat)
    out[0] = flat[0]
    out[1:] = cp.diff(flat)
    return out
