"""NVTX range markers for nsys / Nsight profiling.

Wraps NVIDIA's ``nvtx`` package (in the tree via ``nsight-python``).  All
czarr ranges live in a dedicated ``czarr`` NVTX domain — their own row in
nsys, filterable from cupy/cuDF ranges in the default domain — and go
through the cached ``Domain`` object, which skips the per-call domain
lookup and interns messages/categories once.

Keep range names static: messages are cached as NVTX registered strings,
so a per-call-unique name grows that registry without bound.  Put numeric
detail in ``payload`` (typed data on the event, queryable in nsys sqlite
exports) and algorithm/stage strings in ``category``.  ``fields`` still
concatenate ``k=v`` into the label for ad-hoc use; prefer payload/category.

Ranges are no-ops when ``CZARR_NVTX=0`` is set (default: enabled), and
NVTX itself is a no-op unless a profiler is attached.

Usage::

    from czarr._nvtx import nvtx_range

    with nvtx_range("czarr.lowlevel.read", payload=len(requests)):
        ...
"""

import os
from contextlib import contextmanager

# All nvtx references live inside the try so the module name is never
# rebound (typing stays environment-independent); ``_DOMAIN is None``
# encodes both "nvtx missing" and "CZARR_NVTX=0".
_DOMAIN = None
try:
    import nvtx

    if os.environ.get("CZARR_NVTX", "1") != "0":
        _DOMAIN = nvtx.get_domain("czarr")
except ImportError:  # pragma: no cover — nvtx rides the cu12/cu13 extras
    pass


@contextmanager
def nvtx_range(
    name: str,
    *,
    color: str = "green",
    category: str | None = None,
    payload: int | float | None = None,
    **fields,
):
    """Range in the czarr NVTX domain; shows up in nsys timelines.

    ``payload`` carries the per-call numeric (chunk count, byte size);
    ``category`` groups ranges within the domain (e.g. the nvCOMP algo).
    Handle-based start/end ranges are process-scope rather than
    thread-local push/pop, so interleaved asyncio tasks on the event-loop
    thread cannot corrupt each other's range stack.
    """
    if _DOMAIN is None:
        yield
        return
    label = name
    if fields:
        label += "|" + "|".join(f"{k}={v}" for k, v in fields.items())
    handle = _DOMAIN.start_range(message=label, color=color, category=category, payload=payload)
    try:
        yield
    finally:
        _DOMAIN.end_range(handle)


def mark(name: str, *, color: str = "blue", payload: int | float | None = None) -> None:
    """Drop a one-shot timeline marker in the czarr domain."""
    if _DOMAIN is not None:
        _DOMAIN.mark(message=name, color=color, payload=payload)
