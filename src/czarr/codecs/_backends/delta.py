"""Native Delta decode — cuda.compute inclusive scan (cumulative sum).

Validated 1.28x faster than ``cupy.cumsum`` on H100 for 4 MiB int32
(Phase 2 spike at ``.planning/research/cuda-array/spikes/delta_cuda_compute.py``).
Decode-only: cuda.compute has no adjacent-difference primitive, so the
Delta filter's encode stays inline (``cupy.diff`` + scalar write).

Bit-exact with ``numcodecs.Delta`` for the same dtype.
"""

from typing import Any

import cupy as cp
import numpy as np

from czarr.codecs._backends import import_cccl

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
