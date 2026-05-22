"""Comprehensive pipeline sweep for the CzarrPipeline refactor.

Designed to run on Bruno's H100 nodes via SLURM.  Sweeps:

* chunk size (small / medium / large / huge)
* compressor (Zstd compat, ANS native)
* pipeline implementation (default zarr BatchedCodecPipeline vs CzarrPipeline)
* storage path (VAST/lustre under $MYDATA vs node-local SSD scratch)

Captures per-cell median + min read wall-clock, throughput, and whether
real GDS is active (presence of /proc/driver/nvidia-fs).  Emits CSV to
stdout for downstream analysis.

Run:
    uv run --extra cu12 --group test python -m bench.zarr.pipeline_sweep \
        --output bench/logs/sweep_<jobid>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import zarr

import czarr


@dataclass(frozen=True)
class Workload:
    name: str
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]


# Total ~1 GiB per workload (float32).  4 chunk sizes cover the regime where
# per-chunk Python overhead vs nvCOMP-dominant vs cuFile-dominant matter
# differently.
WORKLOADS = [
    Workload("small (256KiB chunks)", (512, 512, 512), (16, 64, 64)),
    Workload("medium (4MiB chunks)", (512, 512, 512), (32, 128, 128)),
    Workload("large (16MiB chunks)", (512, 512, 512), (64, 256, 256)),
    Workload("xlarge (128MiB chunks)", (512, 512, 512), (128, 512, 512)),
]


def gpu_name() -> str:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        return "unknown"


def gds_active() -> bool:
    """True if the nvidia_fs kernel module is loaded — real GDS available."""
    return os.path.exists("/proc/driver/nvidia-fs")


@contextmanager
def configured(*, pipeline: bool):
    """Reconfigure czarr's pipeline knob, ensuring clean state on exit."""
    czarr.configure_gpu(pipeline=pipeline)
    try:
        yield
    finally:
        zarr.config.reset()


def _time_reads(arr, *, reps: int, warmup: int) -> tuple[float, float, int]:
    """Time reads of arr[:].  Returns (median_s, min_s, nbytes_of_one_read)."""
    out_nbytes = 0
    for _ in range(warmup):
        out = arr[:]
        cp.cuda.Stream.null.synchronize()
        out_nbytes = out.nbytes

    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = arr[:]
        cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)

    return statistics.median(samples), min(samples), out_nbytes


def _write_array(store_path: Path, wl: Workload, *, compressors: list, data: np.ndarray):
    czarr.configure_gpu(pipeline=True)
    store = czarr.GPULocalStore(store_path)
    arr = zarr.create_array(
        store=store,
        shape=wl.shape,
        chunks=wl.chunks,
        dtype="float32",
        compressors=compressors,
        overwrite=True,
    )
    arr[:] = cp.asarray(data)
    zarr.config.reset()


def _bench_one(store_path: Path, *, pipeline: bool, reps: int, warmup: int) -> dict:
    with configured(pipeline=pipeline):
        store_r = czarr.GPULocalStore(store_path, read_only=True)
        arr_r = zarr.open_array(store=store_r, mode="r")
        median_s, min_s, nbytes = _time_reads(arr_r, reps=reps, warmup=warmup)
    return {"median_s": median_s, "min_s": min_s, "nbytes": nbytes}


def _gibs(nbytes: int, seconds: float) -> float:
    return (nbytes / (1 << 30)) / seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="CSV output path")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--storage",
        choices=("mydata", "local", "both"),
        default="both",
        help="Which storage tier(s) to benchmark.",
    )
    args = parser.parse_args()

    if not os.path.exists("/proc/driver/nvidia-fs"):
        os.environ.setdefault("CUFILE_FORCE_COMPAT_MODE", "true")

    rng = np.random.default_rng(0)

    # Compressor configurations to sweep.
    codec_configs: list[tuple[str, list]] = [
        ("zstd (compat)", [czarr.Zstd()]),
        ("ans (native)", [czarr.ANS()]),
    ]

    # Storage tiers.
    storage_dirs: list[tuple[str, Path]] = []
    if args.storage in ("mydata", "both"):
        storage_dirs.append(("vast/$MYDATA", Path(os.environ.get("MYDATA", "/hpc/mydata/" + os.environ["USER"]))))
    if args.storage in ("local", "both"):
        local = Path(os.environ.get("TMPDIR", "/local/scratch")) / os.environ["USER"]
        local.mkdir(parents=True, exist_ok=True)
        storage_dirs.append(("local/scratch", local))

    rows: list[dict] = []
    gpu = gpu_name()
    gds = gds_active()

    print(f"# gpu={gpu}  gds_active={gds}  reps={args.reps}  warmup={args.warmup}", file=sys.stderr)

    for storage_label, storage_root in storage_dirs:
        with tempfile.TemporaryDirectory(dir=storage_root, prefix=".czarr_sweep_", ignore_cleanup_errors=True) as td:
            td_path = Path(td)
            for wl in WORKLOADS:
                data = rng.standard_normal(wl.shape).astype("float32")

                for codec_label, compressors in codec_configs:
                    arr_path = td_path / f"{wl.name.replace(' ', '_')}__{codec_label.replace(' ', '_')}.zarr"
                    _write_array(arr_path, wl, compressors=compressors, data=data)

                    for pipeline_label, pipeline in (("zarr-default", False), ("czarr", True)):
                        res = _bench_one(arr_path, pipeline=pipeline, reps=args.reps, warmup=args.warmup)
                        row = {
                            "gpu": gpu,
                            "gds_active": gds,
                            "storage": storage_label,
                            "workload": wl.name,
                            "shape": "x".join(map(str, wl.shape)),
                            "chunks": "x".join(map(str, wl.chunks)),
                            "codec": codec_label,
                            "pipeline": pipeline_label,
                            "median_ms": round(res["median_s"] * 1000, 2),
                            "min_ms": round(res["min_s"] * 1000, 2),
                            "throughput_gib_s": round(_gibs(res["nbytes"], res["median_s"]), 2),
                            "nbytes": res["nbytes"],
                        }
                        rows.append(row)
                        print(
                            f"{storage_label:>14} | {wl.name:<25} | {codec_label:<14} | "
                            f"{pipeline_label:<14} | {row['median_ms']:>8.1f} ms | "
                            f"{row['throughput_gib_s']:>5.2f} GiB/s",
                            file=sys.stderr,
                        )

    if rows:
        out = csv.DictWriter(
            sys.stdout if args.output is None else args.output.open("w"),
            fieldnames=list(rows[0].keys()),
        )
        out.writeheader()
        out.writerows(rows)

    # Brief speedup summary on stderr.
    print("\n# speedup (czarr / default):", file=sys.stderr)
    by_key: dict[tuple, dict[str, float]] = {}
    for r in rows:
        key = (r["storage"], r["workload"], r["codec"])
        by_key.setdefault(key, {})[r["pipeline"]] = r["median_ms"]
    for key, modes in by_key.items():
        if "czarr" in modes and "zarr-default" in modes:
            speedup = modes["zarr-default"] / modes["czarr"]
            print(f"#   {key[0]:>14} | {key[1]:<25} | {key[2]:<14}: {speedup:.2f}x", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
