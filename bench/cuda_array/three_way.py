"""Multi-way bench: pure zarr vs zarr+torch vs kvikio.zarr vs czarr.

Compares the most common read paths a scientific user takes when getting
zarr data onto a GPU:

1. ``zarr.open_array(LocalStore(path))[:]`` — numpy on host.  Baseline;
   not a GPU read.
2. ``torch.from_numpy(zarr_arr[:]).cuda()`` — the naive PyTorch user
   path.  CPU read + explicit H2D copy + numpy→torch wrap.
3. ``zarr.open_array(kvikio.zarr.GDSStore(path))[:]`` — RAPIDS' kvikio
   GDS-backed zarr store.  cuFile reads into device via the same
   primitive czarr's GPULocalStore uses; decode runs through whatever
   codec pipeline zarr picks (CPU decoders by default, GPU if the user
   has configured them).
4. ``czarr.open_cuda_array(path)[:]`` — cupy on device.  cuFile read
   into GPU memory, nvCOMP/native decode on GPU, no host roundtrip.
5. ``torch.from_dlpack(czarr_arr[:])`` — czarr read + zero-copy DLPack
   handoff to torch.

Workload: 1 GiB Z-slab of float32, (16, 4096, 4096) shape, 16x512x512
chunks, zstd compression.  Matches the slice_compare canonical bench.
Output throughput is reported in GiB/s (raw uncompressed bytes / wall).
"""

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


def bench_kvikio_gds(path: Path, *, reps: int, warmup: int) -> list[float] | None:
    """kvikio.zarr.GDSStore with zarr's default codec pipeline.

    GDS reads but CPU-side Zstd decode (the registered numcodecs Zstd).
    Forces D2H to host, decode, H2D back to device — strictly inferior
    on a compressed workload.  Included as the apples-to-apples
    "kvikio alone" baseline.
    """
    try:
        import kvikio.zarr
    except ImportError:
        return None

    zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"})
    try:
        store = kvikio.zarr.GDSStore(str(path))
        arr = zarr.open_array(store, mode="r")
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


def bench_kvikio_gds_with_czarr_codecs(path: Path, *, reps: int, warmup: int) -> list[float] | None:
    """kvikio.zarr.GDSStore + czarr's GPU codec registrations.

    The fair-comparison path: kvikio handles the cuFile-direct read,
    czarr's Zstd (nvCOMP-backed) handles decode on device.  Compared
    against ``czarr.open_cuda_array`` this isolates the value of
    czarr's ``GPULocalStore`` (anything left beyond kvikio's GDS
    primitive) from the value of czarr's codec stack.
    """
    try:
        import kvikio.zarr
    except ImportError:
        return None

    # configure_gpu registers czarr's GPU Zstd / LZ4 / etc. in zarr's
    # codec registry AND sets the GPU buffer prototype.  Result: kvikio
    # does the GDS read, czarr's codecs run on the device-resident bytes.
    czarr.configure_gpu()
    try:
        store = kvikio.zarr.GDSStore(str(path))
        arr = zarr.open_array(store, mode="r")
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
    kvikio_alone = bench_kvikio_gds(args.path, reps=args.reps, warmup=args.warmup)
    kvikio_czarr_codecs = bench_kvikio_gds_with_czarr_codecs(args.path, reps=args.reps, warmup=args.warmup)
    czarr_direct = bench_czarr_direct(args.path, reps=args.reps, warmup=args.warmup)
    czarr_torch = bench_czarr_to_torch(args.path, reps=args.reps, warmup=args.warmup, device=device)

    rows = [
        ("zarr (numpy, host)", pure),
        ("zarr → torch.cuda", zarr_torch),
    ]
    if kvikio_alone is not None:
        rows.append(("kvikio.zarr (CPU decode)", kvikio_alone))
    else:
        print("(kvikio not installed — rows skipped)")
    if kvikio_czarr_codecs is not None:
        rows.append(("kvikio + czarr codecs", kvikio_czarr_codecs))
    rows.extend(
        [
            ("czarr (cupy, device)", czarr_direct),
            ("czarr → torch (DLPack)", czarr_torch),
        ]
    )
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
    if kvikio_alone is not None:
        kv_med = statistics.median(kvikio_alone)
        print(f"czarr direct vs kvikio.zarr alone:        {kv_med / cd_med:>5.2f}× faster")
    if kvikio_czarr_codecs is not None:
        kvc_med = statistics.median(kvikio_czarr_codecs)
        print(
            f"czarr direct vs kvikio + czarr codecs:    {kvc_med / cd_med:>5.2f}× ratio (GPULocalStore vs kvikio.zarr.GDSStore)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
