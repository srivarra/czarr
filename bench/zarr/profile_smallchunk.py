"""cProfile of the small-chunk read path to find where Python time goes.

Reproduces the worst-case workload that showed the H100 small-chunk
regression (0.86x czarr vs default): 1024-ish small chunks of zstd
through the GPU pipeline.  Wrap one ``arr[:]`` in cProfile and dump
the hot functions.

Run:
    uv run --extra cu12 --group test python -m bench.zarr.profile_smallchunk
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import tempfile
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr

import czarr

# Workload calibrated to be small-chunk dominated.
SHAPE = (256, 256, 256)  # 64 MiB raw
CHUNKS = (8, 32, 32)  # 32 KiB chunks  →  8192 chunks total
DTYPE = "float32"


def main() -> None:
    if not os.path.exists("/proc/driver/nvidia-fs"):
        os.environ.setdefault("CUFILE_FORCE_COMPAT_MODE", "true")
    repo = "/hpc/mydata/sricharan.varra/Dev/czarr"
    rng = np.random.default_rng(0)
    data = rng.standard_normal(SHAPE).astype("float32")

    with tempfile.TemporaryDirectory(dir=repo, prefix=".gpustore_profile_", ignore_cleanup_errors=True) as td:
        td_path = Path(td)
        czarr.configure_gpu()
        store = czarr.GPULocalStore(td_path / "store.zarr")
        if not store.gds_available:
            print("cuFile not available — skipping.")
            return
        arr = zarr.create_array(
            store=store,
            shape=SHAPE,
            chunks=CHUNKS,
            dtype=DTYPE,
            compressors=[czarr.Zstd()],
            overwrite=True,
        )
        arr[:] = cp.asarray(data)
        cp.cuda.Stream.null.synchronize()

        # Warm caches and reuse the array handle for the profiled call.
        store_r = czarr.GPULocalStore(td_path / "store.zarr", read_only=True)
        arr_r = zarr.open_array(store=store_r, mode="r")
        for _ in range(2):
            _ = arr_r[:]
            cp.cuda.Stream.null.synchronize()

        # Time first (unprofiled, to compare with profile overhead).
        t0 = time.perf_counter()
        _ = arr_r[:]
        cp.cuda.Stream.null.synchronize()
        dt_clean = time.perf_counter() - t0

        # cProfile single arr[:] call.
        prof = cProfile.Profile()
        prof.enable()
        out = arr_r[:]
        cp.cuda.Stream.null.synchronize()
        prof.disable()
        nbytes = out.nbytes

        # Dump top-50 hot rows by cumulative time + by self time.
        for sort in ("cumulative", "tottime"):
            buf = io.StringIO()
            stats = pstats.Stats(prof, stream=buf).strip_dirs().sort_stats(sort)
            stats.print_stats(40)
            print(f"\n=== sort: {sort} ===")
            print(buf.getvalue())

        print(
            f"\n=== summary ===\n"
            f"workload : shape={SHAPE} chunks={CHUNKS} -> "
            f"{(SHAPE[0] // CHUNKS[0]) * (SHAPE[1] // CHUNKS[1]) * (SHAPE[2] // CHUNKS[2])} chunks\n"
            f"nbytes   : {nbytes / (1 << 20):.1f} MiB\n"
            f"unprof'd : {dt_clean * 1000:.1f} ms  ({nbytes / (1 << 30) / dt_clean:.2f} GiB/s)"
        )


if __name__ == "__main__":
    main()
