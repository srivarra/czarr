"""Three-way bench: pure zarr vs zarr+torch vs czarr.

Compares the most common read paths a scientific user takes when getting
zarr data onto a GPU:

1. ``zarr.open_array(LocalStore(path))[:]`` — numpy on host.  Baseline;
   not a GPU read.
2. ``torch.from_numpy(zarr_arr[:]).cuda()`` — the naive PyTorch user
   path.  CPU read + explicit H2D copy + numpy→torch wrap.
3. ``czarr.open_cuda_array(path)[:]`` — cupy on device.  cuFile read
   into GPU memory, nvCOMP/native decode on GPU, no host roundtrip.
4. ``torch.from_dlpack(czarr_arr[:])`` — czarr read + zero-copy DLPack
   handoff to torch.  The GPU-resident result of (3) reinterpreted as
   a torch tensor without a copy.

Workload: 1 GiB Z-slab of float32, (16, 4096, 4096) shape, 16x512x512
chunks, zstd compression.  Matches the slice_compare canonical bench.
Output throughput is reported in GiB/s (raw uncompressed bytes / wall).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch
import zarr
from zarr.storage import LocalStore

import czarr

SHAPE = (16, 4096, 4096)
CHUNKS = (16, 512, 512)
DTYPE = "float32"


def _gibs(nbytes: int, seconds: float) -> float:
    return (nbytes / (1 << 30)) / seconds


def write_store(path: Path) -> None:
    """One-time write of the test store via czarr's GPU encoder."""
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
    data = rng.standard_normal(SHAPE).astype("float32")
    arr[:] = cp.asarray(data)
    cp.cuda.Stream.null.synchronize()
    zarr.config.reset()


def bench_pure_zarr(path: Path, *, reps: int, warmup: int) -> list[float]:
    """Pure zarr on CPU — numpy output, no GPU."""
    zarr.config.reset()
    arr = zarr.open_array(LocalStore(path), mode="r")
    for _ in range(warmup):
        _ = arr[:]
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = arr[:]
        samples.append(time.perf_counter() - t0)
    assert isinstance(out, np.ndarray)
    return samples


def bench_zarr_to_torch(path: Path, *, reps: int, warmup: int, device: torch.device) -> list[float]:
    """Naive PyTorch path — zarr to numpy to torch.cuda."""
    zarr.config.reset()
    arr = zarr.open_array(LocalStore(path), mode="r")
    for _ in range(warmup):
        t = torch.from_numpy(arr[:]).to(device)
        torch.cuda.synchronize()
        del t
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        tensor = torch.from_numpy(arr[:]).to(device)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    assert tensor.is_cuda
    return samples


def bench_czarr_direct(path: Path, *, reps: int, warmup: int) -> list[float]:
    """czarr.open_cuda_array — cupy on device via cuFile + nvCOMP/native."""
    czarr.configure_gpu()
    try:
        arr = czarr.open_cuda_array(path, mode="r")
        for _ in range(warmup):
            _ = arr[:]
            cp.cuda.Stream.null.synchronize()
        samples: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = arr[:]
            cp.cuda.Stream.null.synchronize()
            samples.append(time.perf_counter() - t0)
        assert isinstance(out, cp.ndarray), f"expected cupy.ndarray, got {type(out).__name__}"
        return samples
    finally:
        zarr.config.reset()


def bench_czarr_to_torch(path: Path, *, reps: int, warmup: int, device: torch.device) -> list[float]:
    """czarr direct + DLPack zero-copy to torch.

    The cupy ndarray is the device-resident result of czarr's decode;
    ``torch.from_dlpack`` reinterprets it as a torch tensor without
    moving bytes.
    """
    czarr.configure_gpu()
    try:
        arr = czarr.open_cuda_array(path, mode="r")
        for _ in range(warmup):
            cup = arr[:]
            t = torch.from_dlpack(cup)
            torch.cuda.synchronize()
            del t, cup
        samples: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            cup = arr[:]
            tensor = torch.from_dlpack(cup)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
        assert tensor.is_cuda
        return samples
    finally:
        zarr.config.reset()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_path = Path(os.environ.get("MYDATA", "/hpc/mydata/" + os.environ["USER"])) / ".czarr_three_way_bench.zarr"
    p.add_argument("--path", type=Path, default=default_path)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rewrite", action="store_true")
    args = p.parse_args()

    nbytes = int(np.prod(SHAPE) * 4)
    total_gib = nbytes / (1 << 30)

    if args.rewrite or not args.path.exists():
        print(f"writing store at {args.path} ...")
        t0 = time.perf_counter()
        write_store(args.path)
        dt = time.perf_counter() - t0
        print(f"  wrote {total_gib:.2f} GiB in {dt:.2f}s ({total_gib / dt:.2f} GiB/s)")
    else:
        print(f"reusing store at {args.path}")
    print()

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"bench: {args.reps} timed reps + {args.warmup} warmup, slab = {SHAPE}")
    print()

    pure = bench_pure_zarr(args.path, reps=args.reps, warmup=args.warmup)
    zarr_torch = bench_zarr_to_torch(args.path, reps=args.reps, warmup=args.warmup, device=device)
    czarr_direct = bench_czarr_direct(args.path, reps=args.reps, warmup=args.warmup)
    czarr_torch = bench_czarr_to_torch(args.path, reps=args.reps, warmup=args.warmup, device=device)

    rows = [
        ("zarr (numpy, host)", pure),
        ("zarr → torch.cuda", zarr_torch),
        ("czarr (cupy, device)", czarr_direct),
        ("czarr → torch (DLPack)", czarr_torch),
    ]
    print(f"{'path':<28}{'median':>12}{'min':>10}{'GiB/s':>10}")
    print("-" * 60)
    for label, samples in rows:
        med = statistics.median(samples)
        mn = min(samples)
        print(f"{label:<28}{med * 1e3:>9.2f} ms{mn * 1e3:>9.2f} ms{_gibs(nbytes, med):>10.2f}")

    pure_med = statistics.median(pure)
    zt_med = statistics.median(zarr_torch)
    cd_med = statistics.median(czarr_direct)
    ct_med = statistics.median(czarr_torch)
    print()
    print(f"czarr direct vs pure zarr:       {pure_med / cd_med:>5.2f}× faster")
    print(f"czarr direct vs zarr→torch:      {zt_med / cd_med:>5.2f}× faster")
    print(f"czarr→torch vs zarr→torch:       {zt_med / ct_med:>5.2f}× faster")
    print(f"czarr direct vs czarr→torch:     {ct_med / cd_med:>5.2f}× ratio (DLPack overhead)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
