"""Compare CzarrPipeline vs zarr's default BatchedCodecPipeline.

Workload: a synthetic v3 zarr array on GPULocalStore (cuFile-backed),
read back via ``arr[:]``.  Times the read wall-clock with each
pipeline implementation.

The two paths differ in two places:

* nvCOMP H2D — default pipeline does
  ``nvcomp.as_array(bytes(chunk.to_bytes())).cuda()`` per chunk, forcing
  GPU -> host -> GPU.  CzarrPipeline's ``CudaBytesBytesCodec._batch_sync``
  detects ``gpu.Buffer`` and passes the cupy ndarray directly.
* Pipeline lookup — same for both today; future phases can swap in
  cross-chunk batching at this layer without touching codec code.

Run::

    uv run --extra cu12 --group test python -m bench.zarr.pipeline_compare
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr

import czarr


def _bench_read(
    arr,
    *,
    reps: int = 5,
    warmup: int = 1,
) -> dict[str, float]:
    # Warm caches and JIT.
    for _ in range(warmup):
        _ = arr[:]
        cp.cuda.Stream.null.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = arr[:]
        cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    nbytes = out.nbytes
    return {
        "nbytes": float(nbytes),
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "samples": samples,
    }


def _gib(n: float) -> float:
    return n / (1 << 30)


def _gibs(nbytes: float, seconds: float) -> float:
    return _gib(nbytes) / seconds


def main() -> None:
    # cuFile in compat mode rejects tmpfs; anchor under repo root.
    if not os.path.exists("/proc/driver/nvidia-fs"):
        os.environ.setdefault("CUFILE_FORCE_COMPAT_MODE", "true")

    repo = "/hpc/mydata/sricharan.varra/Dev/czarr"
    rng = np.random.default_rng(0)

    # Multi-size sweep: same total bytes, different chunk size = different
    # per-chunk Python overhead.  Reveals where the GPU-direct codec path
    # matters (large chunks = host round-trip dominates) vs where Python
    # overhead dominates (small chunks).
    workloads: list[dict] = [
        {
            "name": "many small chunks (1024 x 64KB)",
            "shape": (64, 256, 256),
            "chunks": (8, 32, 32),
        },
        {
            "name": "balanced (64 x 1 MiB)",
            "shape": (256, 256, 256),
            "chunks": (64, 64, 64),
        },
        {
            "name": "few large chunks (8 x 8 MiB)",
            "shape": (512, 256, 256),
            "chunks": (128, 128, 128),
        },
    ]

    all_rows: list[tuple[str, str, dict]] = []

    with tempfile.TemporaryDirectory(dir=repo, prefix=".gpustore_bench_", ignore_cleanup_errors=True) as td:
        td_path = Path(td)
        czarr.configure_gpu()
        probe = czarr.GPULocalStore(td_path / "probe.zarr")
        if not probe.gds_available:
            print("cuFile not available — skipping bench.")
            return

        for wl in workloads:
            shape = wl["shape"]
            chunks = wl["chunks"]
            data = rng.standard_normal(shape).astype("float32")

            store = czarr.GPULocalStore(td_path / f"{wl['name'].replace(' ', '_')}.zarr")
            arr = zarr.create_array(
                store=store,
                shape=shape,
                chunks=chunks,
                dtype="float32",
                compressors=[czarr.Zstd()],
                overwrite=True,
            )
            arr[:] = cp.asarray(data)

            for label, pipeline in [
                ("zarr default", False),
                ("czarr", True),
            ]:
                czarr.configure_gpu(pipeline=pipeline)
                store_r = czarr.GPULocalStore(td_path / f"{wl['name'].replace(' ', '_')}.zarr", read_only=True)
                arr_r = zarr.open_array(store=store_r, mode="r")
                all_rows.append((wl["name"], label, _bench_read(arr_r, reps=5, warmup=2)))

    print(f"\n{'workload':<35}{'pipeline':<18}{'median':>10}{'min':>10}{'GiB/s':>10}\n" + "-" * 83)
    for wl_name, label, r in all_rows:
        print(
            f"{wl_name:<35}{label:<18}"
            f"{r['median_s'] * 1e3:>8.1f} ms"
            f"{r['min_s'] * 1e3:>8.1f} ms"
            f"{_gibs(r['nbytes'], r['median_s']):>9.2f}"
        )

    print("\nspeedup (czarr / default):")
    by_wl: dict[str, dict[str, float]] = {}
    for wl_name, label, r in all_rows:
        by_wl.setdefault(wl_name, {})[label] = r["median_s"]
    for wl_name, modes in by_wl.items():
        if "czarr" in modes and "zarr default" in modes:
            print(f"  {wl_name:<35}{modes['zarr default'] / modes['czarr']:.2f}x")


if __name__ == "__main__":
    main()
