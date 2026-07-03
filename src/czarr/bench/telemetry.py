"""Auto-attached telemetry: GPU identity, GDS state, RMM peak allocation.

Every result row carries the regime it was measured in — without it a number is
unreadable (the H100/H200/A100 GDS matrix and the tmpfs-vs-NVMe lesson both
turn on context).  This module collects that context cheaply.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_NVIDIA_FS = Path("/proc/driver/nvidia-fs")


def gds_active() -> bool:
    """True when the nvidia-fs kernel module is loaded (real GPUDirect Storage).

    Compat-mode cuFile (no nvidia-fs) still "works" but goes through a bounce
    buffer — a different regime that must be recorded.  See the GDS matrix:
    real on H100/H200, compat-only on A100/A40.
    """
    return _NVIDIA_FS.exists()


def gpu_info() -> dict[str, Any]:
    """GPU name + memory via NVML (pynvml ships with nsight-python)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        info = {"gpu": _short_name(name), "gpu_mem_gib": round(mem.total / 2**30, 1)}
        pynvml.nvmlShutdown()
        return info
    except Exception:
        return {"gpu": os.environ.get("CZARR_BENCH_GPU", "unknown"), "gpu_mem_gib": None}


def _short_name(name: str) -> str:
    """`NVIDIA H100 80GB HBM3` -> `H100`; keep the model token."""
    for tok in name.replace("NVIDIA", "").split():
        if tok[0] in "HABL" and any(c.isdigit() for c in tok):
            return tok
    return name.strip()


@contextmanager
def rmm_peak() -> Iterator[dict[str, Any]]:
    """Record peak RMM device allocation over the block (MiB), if RMM is active."""
    out: dict[str, Any] = {}
    try:
        import rmm.statistics as stats

        stats.enable_statistics()  # wraps the current MR with a counting adaptor
        stats.push_statistics()
    except Exception:
        yield out
        return
    try:
        yield out
    finally:
        try:
            rec = stats.pop_statistics()
            if rec is not None:
                out["rmm_peak_mib"] = round(rec.peak_bytes / 2**20, 1)
        except Exception:
            pass
