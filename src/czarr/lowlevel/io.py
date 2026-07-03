"""Batched reads for :class:`~czarr.lowlevel.plan.ReadRequest` lists.

Threadpool of blocking cuFile reads into fresh ``cp.empty`` buffers —
the measured winner on Bruno (async/batch submission 1.8-14x slower;
register-once slabs ~5% slower), and the shape NVIDIA's own guidance
and DALI's reader architecture converge on.  ``max_workers`` is
NIC-concurrency-bound on real GDS (near-line-rate at ~15% host CPU),
not CPU-bound.

Requires a GPU (returns ``cupy`` buffers).  When cuFile itself is
unavailable the reads fall back to host I/O + H2D so the API still
works on compat-hostile nodes.  Store data must live on a real
filesystem — cuFile cannot read tmpfs (/tmp), see the Bruno notes.
"""

from pathlib import Path

import cupy as cp
import numpy as np
from cuda.bindings.cufile import cuFileError

from czarr import cufile
from czarr._nvtx import nvtx_range
from czarr.lowlevel.plan import ReadRequest


def read(requests: list[ReadRequest], *, max_workers: int | None = None) -> list[cp.ndarray]:
    """Read every request window into a fresh device buffer.

    Returns one flat ``uint8`` ``cupy.ndarray`` per request, in input
    order.  Missing files raise ``FileNotFoundError`` (the planner
    already excluded legitimately-absent chunks — an absent file here is
    a race or a corrupt plan, not fill value).  Short reads raise
    ``OSError`` — request windows come from stat/shard-index data, so a
    short read means truncation or corruption, never EOF semantics.

    ``max_workers=None`` lets the threadpool pick
    (``min(32, cpu_count + 4)``); real-GDS reads are NIC-bound, so more
    workers than cores is fine.
    """
    if not requests:
        return []
    buffers = [cp.empty(r.nbytes, dtype=cp.uint8) for r in requests]
    with nvtx_range("czarr.lowlevel.read", n=len(requests)):
        if cufile.is_available():
            try:
                got = cufile.read_into_many(
                    [(r.path, int(b.data.ptr), r.nbytes, r.offset) for r, b in zip(requests, buffers, strict=True)],
                    max_workers=max_workers,
                )
            except cuFileError as e:
                # Normalize driver errors (unsupported FS, truncation, ...)
                # to OSError so callers handle one exception family.
                raise OSError(f"cuFile batched read failed ({len(requests)} requests): {e}") from e
        else:
            got = [_host_read_into(r, b) for r, b in zip(requests, buffers, strict=True)]
    for r, n in zip(requests, got, strict=True):
        if n != r.nbytes:
            raise OSError(f"{r.path}: short read at offset {r.offset}: got {n} of {r.nbytes} bytes")
    return buffers


def _host_read_into(request: ReadRequest, dev: cp.ndarray) -> int:
    """Host-I/O fallback: pread into pageable host memory, then H2D."""
    host = np.empty(request.nbytes, dtype=np.uint8)
    with Path(request.path).open("rb") as f:
        f.seek(request.offset)
        n = f.readinto(memoryview(host))
    dev[:n].set(host[:n])
    return n
