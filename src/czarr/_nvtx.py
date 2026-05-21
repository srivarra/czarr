"""NVTX range markers for nsys / Nsight profiling.

Wraps cupy's NVTX bindings as a context manager.  All ranges are no-ops
when ``CZARR_NVTX=0`` is set (default: enabled), which lets us leave the
markers in the hot path with zero overhead in production.

Usage::

    from czarr._nvtx import nvtx_range

    with nvtx_range("decode_batch", n=len(items)):
        ...
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import cupy as cp

_ENABLED = os.environ.get("CZARR_NVTX", "1") != "0"


@contextmanager
def nvtx_range(name: str, **fields):
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
    cp.cuda.nvtx.RangePush(label)
    try:
        yield
    finally:
        cp.cuda.nvtx.RangePop()


def mark(name: str) -> None:
    """Drop a one-shot timeline marker."""
    if _ENABLED:
        cp.cuda.nvtx.Mark(name)
