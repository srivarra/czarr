"""NVTX range markers for nsys / Nsight profiling.

Wraps NVIDIA's ``nvtx`` package (in the tree via ``nsight-python``) as a
context manager.  All czarr ranges live in a dedicated ``czarr`` NVTX
domain, so nsys timelines show them on their own row, filterable from
cupy/cuDF ranges in the default domain.

Ranges are no-ops when ``CZARR_NVTX=0`` is set (default: enabled), and
NVTX itself is a no-op unless a profiler is attached, so the markers
stay in the hot path.

Usage::

    from czarr._nvtx import nvtx_range

    with nvtx_range("decode_batch", n=len(items)):
        ...
"""

import os
from contextlib import contextmanager

try:
    import nvtx
except ImportError:  # pragma: no cover — nvtx rides the cu12/cu13 extras
    nvtx = None  # ty: ignore[invalid-assignment] — optional-module idiom

_ENABLED = os.environ.get("CZARR_NVTX", "1") != "0" and nvtx is not None
_DOMAIN = "czarr"


@contextmanager
def nvtx_range(name: str, *, color: str = "green", **fields):
    """Push an NVTX range that shows up in nsys timelines.

    ``fields`` are concatenated into the range name as ``name|k=v|k=v``
    so the timeline shows useful per-call detail (chunk count, sizes, etc.)
    without needing structured payload support.
    """
    if not _ENABLED:
        yield
        return
    label = name
    if fields:
        label += "|" + "|".join(f"{k}={v}" for k, v in fields.items())
    # start/end ranges are process-scope (not thread-local push/pop), so
    # interleaved asyncio tasks on the event-loop thread can't corrupt
    # each other's range stack.
    handle = nvtx.start_range(message=label, color=color, domain=_DOMAIN)
    try:
        yield
    finally:
        nvtx.end_range(handle)


def mark(name: str) -> None:
    """Drop a one-shot timeline marker."""
    if _ENABLED:
        nvtx.mark(message=name, domain=_DOMAIN)
