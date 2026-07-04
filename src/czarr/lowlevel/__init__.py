"""czarr.lowlevel — explicit, staged GPU read-path plumbing.

Design: ``.planning/lowlevel-api-design.md``.  Stages:

* :mod:`czarr.lowlevel.plan` — derive-once metadata → coalesced byte ranges
* :mod:`czarr.lowlevel.coalesce` — range fusion primitives
* :mod:`czarr.lowlevel.io` — threaded cuFile reads into device buffers
* :mod:`czarr.lowlevel.decode` — nvCOMP batch decode + scatter into the output

``czarr.core.Array`` composes these; each stage is also callable and
benchable on its own.
"""

from typing import Any

from czarr.lowlevel.coalesce import ByteRange, FusedRead, coalesce_ranges
from czarr.lowlevel.plan import DecodePlan, ReadRequest, ShardSpec, normalize_selection, open_plan, plan_from_metadata

__all__ = [
    "ByteRange",
    "DecodePlan",
    "FusedRead",
    "ReadRequest",
    "ShardSpec",
    "coalesce_ranges",
    "decode",
    "normalize_selection",
    "open_plan",
    "plan_from_metadata",
    "read",
    "read_array",
]

# GPU-touching stages load on first use (planning stays cupy-free).
_LAZY = {
    "read": "czarr.lowlevel.io",
    "decode": "czarr.lowlevel.decode",
    "read_array": "czarr.lowlevel.decode",
}


def __getattr__(name: str) -> Any:
    # Planning stays importable on hosts without CUDA; ``lowlevel.read``
    # / ``decode`` / ``read_array`` pull cupy only when actually used.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
