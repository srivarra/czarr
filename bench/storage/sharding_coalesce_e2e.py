"""End-to-end ``arr[selection]`` bench: stock vs Czarr sharding codec.

Builds a v3 sharded zarr on the chosen filesystem, then times
``arr[selection]`` reads under three configurations:

1. stock :class:`zarr.codecs.ShardingCodec`, host buffers
2. stock :class:`zarr.codecs.ShardingCodec`, GPU buffers (via czarr's
   configure_gpu + LocalStore/GPULocalStore)
3. :class:`czarr.CzarrShardingCodec` with the coalescing override

Selections are chosen so that each touches a *partial* shard (the path
the override actually changes).  Whole-shard selections are also
measured as a sanity check — they go through the unchanged
``_load_full_shard_maybe`` branch and should be ~tied across configs.
"""

import argparse
import shutil
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from zarr.codecs.sharding import ShardingCodec
from zarr.storage import LocalStore

import czarr
from czarr.codecs import CzarrShardingCodec
from czarr.storage import GPULocalStore


def _build_sharded_array(
    root: Path,
    *,
    shape: tuple[int, ...],
    inner_chunk: tuple[int, ...],
    shard: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Write a v3 sharded zarr filled with deterministic random data."""
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(0)
    # ``rng.integers`` only supports integer dtypes; cast through that
    # so the random fill is fast (raw bytes) while the on-disk dtype is
    # whatever the caller asked for.
    if np.issubdtype(dtype, np.integer):
        data = rng.integers(0, 10_000, size=shape, dtype=dtype)
    else:
        data = rng.standard_normal(size=shape).astype(dtype)
    store = LocalStore(str(root))
    # No compressors / filters — uncompressed inner chunks isolate the
    # I/O coalesce signal from the GPU decode path.  zarr's default is
    # zstd at level 3; we explicitly pass () to opt out.  An orthogonal
    # czarr decode bug on GPU partial-shard reads (Buffer.resize on
    # an external buffer) blocks the compressed measurement; track in
    # a follow-up after the coalesce signal is locked.
    arr = zarr.create_array(
        store=store,
        shape=shape,
        chunks=inner_chunk,
        shards=shard,
        dtype=dtype,
        serializer=ShardingCodec(chunk_shape=inner_chunk),
        compressors=(),
        filters=(),
    )
    arr[:] = data
    return data


def _open_with_serializer(root: Path, serializer, *, store_cls):
    """Re-open the sharded zarr with a chosen ShardingCodec + store class."""
    store = store_cls(str(root))
    return zarr.open_array(store=store, mode="r", serializer=serializer)


def _time_selection(arr, selection, *, reps: int, warmup: int) -> tuple[float, float]:
    """Median + min wall time for ``arr[selection]`` over ``reps`` runs."""
    for _ in range(warmup):
        _ = arr[selection]
        if cp.cuda.is_available():
            cp.cuda.Stream.null.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = arr[selection]
        if cp.cuda.is_available():
            cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def _to_numpy(x) -> np.ndarray:
    """Pull array contents to host regardless of whether x is cupy or numpy."""
    if hasattr(x, "get"):
        return cp.asnumpy(x)
    return np.asarray(x)


def _verify_equivalence(stock_arr, czarr_arr, selection) -> None:
    """Both readers must return identical bytes for the same selection.

    ``configure_gpu()`` flips zarr's default buffer prototype to GPU
    globally, so even the LocalStore path returns a cupy array.  Pull
    everything to host before comparing.
    """
    a = _to_numpy(stock_arr[selection])
    b = _to_numpy(czarr_arr[selection])
    if not np.array_equal(a, b):
        n_diff = int((a != b).sum())
        raise AssertionError(f"selection {selection}: {n_diff} elements differ")


def main() -> int:
    """Run the end-to-end coalesce sharding bench."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Where to write the sharded zarr")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    # Fixed geometry: 64x2048x2048 float32 in shards of (32, 2048, 2048)
    # = 2 shards along axis 0, each ~512 MiB.  Inner chunks 32x256x256 =
    # 8 MiB raw → still partial-shard for the test selections, but the
    # whole dataset fits in ~1 GiB host RAM (the SLURM cap is 32 GiB,
    # and parity checks materialise the slice on host + device).
    shape = (64, 2048, 2048)
    inner_chunk = (32, 256, 256)
    shard = (32, 2048, 2048)
    dtype = np.dtype(np.float32)

    print(f"workload: shape={shape}, inner_chunk={inner_chunk}, shard={shard}, dtype={dtype}")
    print(
        f"shards along axis 0: {shape[0] // shard[0]}; total bytes ~ "
        f"{np.prod(shape) * dtype.itemsize / (1 << 30):.1f} GiB"
    )

    data = _build_sharded_array(args.root, shape=shape, inner_chunk=inner_chunk, shard=shard, dtype=dtype)
    print(f"data MD5 of [0, 0, 0:8]: {data[0, 0, :8].tobytes().hex()}")
    print(f"build dir: {args.root}")

    czarr.configure_gpu()

    selections = [
        # Partial shard along axis 1 — coalesce path engaged
        ("partial X (one shard, 4 chunks high)", np.s_[0:32, 0:1024, 0:1024]),
        # Partial shard, sparser slice
        ("partial X sparse", np.s_[0:32, 128:1024:2, 0:512]),
        # Whole shard (one shard, all chunks) — coalesce NOT engaged
        ("whole shard", np.s_[0:32, :, :]),
    ]

    results: list[tuple[str, str, str, float, float, float]] = []

    for sel_name, sel in selections:
        print(f"\n--- selection: {sel_name}  {sel} ---")

        # Stock host
        arr_stock_host = _open_with_serializer(
            args.root,
            ShardingCodec(chunk_shape=inner_chunk),
            store_cls=LocalStore,
        )
        # Stock GPU
        arr_stock_gpu = _open_with_serializer(
            args.root,
            ShardingCodec(chunk_shape=inner_chunk),
            store_cls=GPULocalStore,
        )
        # Czarr GPU coalesced
        arr_czarr_gpu = _open_with_serializer(
            args.root,
            CzarrShardingCodec(chunk_shape=inner_chunk),
            store_cls=GPULocalStore,
        )

        _verify_equivalence(arr_stock_host, arr_stock_gpu, sel)
        _verify_equivalence(arr_stock_host, arr_czarr_gpu, sel)
        print("  parity: OK")

        t_host_med, t_host_min = _time_selection(arr_stock_host, sel, reps=args.reps, warmup=args.warmup)
        t_sgpu_med, t_sgpu_min = _time_selection(arr_stock_gpu, sel, reps=args.reps, warmup=args.warmup)
        t_cgpu_med, t_cgpu_min = _time_selection(arr_czarr_gpu, sel, reps=args.reps, warmup=args.warmup)

        results.append((sel_name, "stock host", "—", t_host_med, t_host_min, 0.0))
        results.append((sel_name, "stock GPU", "—", t_sgpu_med, t_sgpu_min, t_host_med / t_sgpu_med))
        results.append((sel_name, "czarr GPU", "coalesce", t_cgpu_med, t_cgpu_min, t_sgpu_med / t_cgpu_med))

        print(f"{'config':<16}{'median ms':>14}{'min ms':>12}{'speedup vs prev':>20}")
        print("-" * 62)
        print(f"{'stock host':<16}{t_host_med * 1e3:>14.2f}{t_host_min * 1e3:>12.2f}{'baseline':>20}")
        print(f"{'stock GPU':<16}{t_sgpu_med * 1e3:>14.2f}{t_sgpu_min * 1e3:>12.2f}{t_host_med / t_sgpu_med:>17.2f}x")
        print(f"{'czarr GPU':<16}{t_cgpu_med * 1e3:>14.2f}{t_cgpu_min * 1e3:>12.2f}{t_sgpu_med / t_cgpu_med:>17.2f}x")

    print("\n=== summary ===")
    for sel_name, config, mode, med, _mn, gain in results:
        gain_s = "" if gain == 0.0 else f"  {gain:.2f}x over prior"
        print(f"  {sel_name:<36}  {config:<12}  {mode:<10}  {med * 1e3:8.2f} ms{gain_s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
