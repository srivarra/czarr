"""Native cuda.compute filter implementations.

Each module exposes a ``decode_<codec>_native`` function (and, where
cuda.compute has a suitable primitive, an encode counterpart) that the
filter codec dispatches to when ``backend="cccl"``.  Output must be
bit-identical to the cupy backend for the same filter.

(Compressors are pure nvCOMP wrappers — czarr no longer ships a
hand-written native compressor.  These modules serve the filters:
Delta and FixedScaleOffset.)
"""

from types import ModuleType
from typing import TYPE_CHECKING


def import_cccl() -> ModuleType:
    """Lazy import of :mod:`cuda.compute`.

    cccl (cuda-cccl + numba-cuda) is an optional dependency — the
    cupy-backed filter paths don't need it.  Deferring the import lets
    these modules load cleanly on hosts without cccl; users that pick
    ``backend="cccl"`` hit a clear ``ModuleNotFoundError`` at decode
    time rather than at import time.
    """
    import cuda.compute as cc

    return cc
