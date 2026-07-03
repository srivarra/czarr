"""Measure whether multi-stream decode beats default-stream decode.

NEGATIVE RESULT — KEPT AS A RECORD.  The multi-stream wiring this bench
drives (``CzarrPipeline.set_decode_multistream`` + per-codec pool-stream
binding) was REVERTED after this measurement: on H100 with real GDS it was
a wash (speedup 0.99-1.02x across every batch_size, both ANS and zstd; see
``bench/logs/multistream_33487575.log``).  czarr is I/O + orchestration
bound, so overlapping GPU decode across streams buys nothing.  To re-run,
re-add the toggle + binding (git history); left here so the experiment
isn't silently repeated.

    batch_size   ans on/off   zstd on/off
       maxsize      1.00x        1.02x
            16      1.00x        1.00x
             8      1.01x        1.00x
             4      0.99x        1.00x
             2      0.99x        1.00x

``CzarrPipeline.set_decode_multistream(True)`` binds each thread-local
nvCOMP codec to a round-robin :class:`StreamPool` stream.  zarr dispatches
decode batches across worker threads (``asyncio.to_thread``); with distinct
streams their GPU decodes can overlap instead of serialising on the default
stream.  A per-thread ``stream.sync()`` after each decode keeps results
correct without killing the cross-thread overlap.

Multi-stream can only help when:

* ``batch_size`` < total chunks, so zarr runs multiple ``read_batch`` calls
  concurrently (otherwise there's one decode, nothing to overlap), and
* GPU decode is a non-trivial slice of wall time.

Hence the comparison sweeps batch_size, OFF vs ON, per compressor.  Each
config opens a *fresh* array so new codec instances re-read the flag (the
stream binding is cached per thread-local codec at creation).

Prior benching says czarr is I/O + orchestration bound, so the honest
expectation is ~no movement — this bench exists to confirm or refute that
before keeping the wiring.

Run::

    uv run --extra cu12 python -m bench.overlap.multistream_h100 \
        --compressor ans --store gpu --rewrite
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from zarr.storage import LocalStore

import czarr
from czarr.pipeline import CzarrPipeline

SHAPE = (16, 4096, 4096)
CHUNKS = (16, 512, 512)
DTYPE = "float32"
TOTAL_GIB = (np.prod(SHAPE) * 4) / (1 << 30)

# 64 chunks total.  Need batch_size < 64 for concurrent batches; maxsize is
# the no-overlap control.
SWEEP_BATCH_SIZES = [sys.maxsize, 16, 8, 4, 2]

_COMPRESSORS = {
    "zstd": lambda: czarr.Zstd(),
    "lz4": lambda: czarr.LZ4(),
    "ans": lambda: czarr.ANS(),
}


def _store(path: Path, kind: str, *, read_only: bool = False):
    if kind == "gpu":
        return czarr.GPULocalStore(path, read_only=read_only)
    return LocalStore(path, read_only=read_only)


def _gibs(nbytes: int, seconds: float) -> float:
    return (nbytes / (1 << 30)) / seconds


def write_store(path: Path, *, compressor: str, store_kind: str) -> None:
    czarr.configure_gpu()
    store = _store(path, store_kind)
    arr = zarr.create_array(
        store=store,
        shape=SHAPE,
        chunks=CHUNKS,
        dtype=DTYPE,
        compressors=[_COMPRESSORS[compressor]()],
        overwrite=True,
    )
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1000, int(np.prod(SHAPE)), dtype="float32").reshape(SHAPE)
    arr[:] = cp.asarray(base + rng.standard_normal(SHAPE).astype("float32") * 5)
    cp.cuda.Stream.null.synchronize()
    zarr.config.reset()


def _run_once(
    path: Path, *, batch_size: int, multistream: bool, reps: int, warmup: int, store_kind: str
) -> list[float]:
    czarr.configure_gpu(batch_size=batch_size)
    CzarrPipeline.set_decode_multistream(multistream)
    try:
        # Fresh open each call so codec instances re-read the multistream
        # flag (stream binding is cached per thread-local codec).
        arr = zarr.open(_store(path, store_kind, read_only=True), mode="r")
        for _ in range(warmup):
            _ = arr[:]
            cp.cuda.Stream.null.synchronize()
        samples: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = arr[:]
            cp.cuda.Stream.null.synchronize()
            samples.append(time.perf_counter() - t0)
        assert isinstance(out, cp.ndarray)
        return samples
    finally:
        CzarrPipeline.set_decode_multistream(False)
        zarr.config.reset()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_path = Path(os.environ.get("MYDATA", "/hpc/mydata/" + os.environ["USER"])) / ".czarr_multistream_bench.zarr"
    p.add_argument("--path", type=Path, default=default_path)
    p.add_argument("--compressor", choices=list(_COMPRESSORS), default="ans")
    p.add_argument("--store", choices=["gpu", "local"], default="gpu")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rewrite", action="store_true")
    args = p.parse_args()

    nbytes = int(np.prod(SHAPE) * 4)
    path = args.path.with_name(f"{args.path.name}.{args.compressor}.{args.store}")

    print(f"compressor={args.compressor}  store={args.store}  slab={SHAPE} chunks={CHUNKS}")
    if args.rewrite or not path.exists():
        print(f"writing store at {path} ...")
        t0 = time.perf_counter()
        write_store(path, compressor=args.compressor, store_kind=args.store)
        dt = time.perf_counter() - t0
        print(f"  wrote {TOTAL_GIB:.2f} GiB in {dt:.2f}s ({TOTAL_GIB / dt:.2f} GiB/s)")
    else:
        print(f"reusing store at {path}")

    print(f"\nbench: {args.reps} timed reps + {args.warmup} warmup")
    print(f"{'batch_size':>12}  {'off ms':>9}  {'on ms':>9}  {'on GiB/s':>9}  {'speedup':>8}")
    print("-" * 56)

    for bs in SWEEP_BATCH_SIZES:
        label = "maxsize" if bs == sys.maxsize else str(bs)
        off = statistics.median(
            _run_once(path, batch_size=bs, multistream=False, reps=args.reps, warmup=args.warmup, store_kind=args.store)
        )
        on = statistics.median(
            _run_once(path, batch_size=bs, multistream=True, reps=args.reps, warmup=args.warmup, store_kind=args.store)
        )
        print(f"{label:>12}  {off * 1e3:>9.2f}  {on * 1e3:>9.2f}  {_gibs(nbytes, on):>9.2f}  {off / on:>7.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
