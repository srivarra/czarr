"""Nsight Systems-friendly slice_compare on a fixture already on disk.

Run via nsys profile with ``--trace=cuda,nvtx``. We push one NVTX range
per timed rep so the timeline is easy to slice; the in-codec /
in-storage ranges from ``czarr._nvtx`` show up automatically. Keep the
rep count low (3 by default) — nsys traces blow up fast.

Usage::

    nsys profile --trace=cuda,nvtx --stats=true \
        --output=profile_%j \
        uv run --extra cu12 python -m bench.buffer.profile_h200
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr

import czarr

SHAPE = (16, 4096, 4096)
CHUNKS = (16, 512, 512)
DTYPE = "float32"


def _ensure_store(path: Path) -> None:
    if path.exists():
        return
    czarr.configure_gpu()
    store = czarr.GPULocalStore(path)
    arr = zarr.create_array(
        store=store,
        shape=SHAPE,
        chunks=CHUNKS,
        dtype=DTYPE,
        compressors=[czarr.Zstd()],
        overwrite=True,
    )
    rng = np.random.default_rng(0)
    arr[:] = cp.asarray(rng.standard_normal(SHAPE).astype("float32"))
    cp.cuda.Stream.null.synchronize()
    zarr.config.reset()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_path = Path(os.environ.get("MYDATA", "/hpc/mydata/" + os.environ["USER"])) / ".czarr_buffer_bench.zarr"
    p.add_argument("--path", type=Path, default=default_path)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    _ensure_store(args.path)
    czarr.configure_gpu()
    arr = zarr.open(czarr.GPULocalStore(args.path, read_only=True), mode="r")

    nbytes = int(np.prod(SHAPE) * 4)

    # Warm: JIT caches + cuFile registrations + nvCOMP scratch.
    for _ in range(args.warmup):
        cp.cuda.nvtx.RangePush("warmup_read")
        _ = arr[:]
        cp.cuda.Stream.null.synchronize()
        cp.cuda.nvtx.RangePop()

    # Timed reps — each one bracketed with a clear NVTX range so the
    # nsys timeline shows where the rep starts/ends.
    samples: list[float] = []
    for i in range(args.reps):
        cp.cuda.nvtx.RangePush(f"rep_{i}")
        t0 = time.perf_counter()
        out = arr[:]
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        cp.cuda.nvtx.RangePop()
        samples.append(dt)
        print(f"rep {i}: {dt * 1e3:.2f} ms  ({(nbytes / (1 << 30)) / dt:.2f} GiB/s)")

    assert isinstance(out, cp.ndarray)
    med = sorted(samples)[len(samples) // 2]
    print(f"\nmedian: {med * 1e3:.2f} ms  ({(nbytes / (1 << 30)) / med:.2f} GiB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
