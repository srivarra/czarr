"""Microbench: GPULocalStore.get_many vs the per-key get() loop.

Phase 1 of the cross-chunk-batching epic.  Confirms that batched cuFile
register/deregister wins on small-chunk workloads where the per-key
``open + register + deregister + close`` cycle was the bottleneck
(measured at 6 s for 2048 chunks on A40, see bench/zarr/profile_smallchunk.py).

Run:
    uv run --extra cu12 --group test python -m bench.storage.get_many_bench
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from zarr.core.buffer import gpu as gpu_buffer

import czarr
from czarr.storage import GPULocalStore

# 2048-chunk workload (same as profile_smallchunk).
SHAPE = (256, 256, 256)
CHUNKS = (8, 32, 32)


async def _collect_keys(store) -> list[str]:
    return [k async for k in store.list_prefix("")]


async def _bench():
    czarr.configure_gpu()
    if not os.path.exists("/proc/driver/nvidia-fs"):
        os.environ.setdefault("CUFILE_FORCE_COMPAT_MODE", "true")

    repo = "/hpc/mydata/sricharan.varra/Dev/czarr"
    with tempfile.TemporaryDirectory(dir=repo, prefix=".gpustore_get_many_", ignore_cleanup_errors=True) as td:
        td_path = Path(td)
        store = GPULocalStore(td_path)
        if not store.gds_available:
            print("cuFile not available — skipping.")
            return

        rng = np.random.default_rng(0)
        data = rng.standard_normal(SHAPE).astype("float32")
        arr = zarr.create_array(
            store=store,
            shape=SHAPE,
            chunks=CHUNKS,
            dtype="float32",
            compressors=[czarr.Zstd()],
            overwrite=True,
        )
        arr[:] = cp.asarray(data)
        cp.cuda.Stream.null.synchronize()

        store_r = GPULocalStore(td_path, read_only=True)
        keys = sorted(await _collect_keys(store_r))
        keys = [k for k in keys if not k.endswith(".json")]
        print(f"workload: shape={SHAPE} chunks={CHUNKS} -> {len(keys)} chunk files")

        proto = gpu_buffer.buffer_prototype

        # Warm cuFile state.
        for _ in range(2):
            _ = await store_r.get(keys[0], prototype=proto)

        # Path A: get() in a loop (mirrors what zarr's concurrent_map does).
        async def loop_get():
            return [await store_r.get(k, prototype=proto) for k in keys]

        # Path B: concurrent_map via asyncio.gather (zarr's actual path).
        async def gather_get():
            return list(await asyncio.gather(*(store_r.get(k, prototype=proto) for k in keys)))

        # Path C: get_many() (new batched path).
        async def many():
            return await store_r.get_many(keys, prototype=proto)

        # Warmup each.
        await loop_get()
        await gather_get()
        await many()

        results = {}
        for name, fn in (("get loop", loop_get), ("gather", gather_get), ("get_many", many)):
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = await fn()
                cp.cuda.Stream.null.synchronize()
                samples.append(time.perf_counter() - t0)
            results[name] = min(samples)

        print(f"\n{'path':<20}{'best':>12}{'GiB/s':>10}")
        total_bytes = sum(os.stat(td_path / k).st_size for k in keys)
        for name, best in results.items():
            print(f"{name:<20}{best * 1000:>9.1f} ms{(total_bytes / (1 << 30)) / best:>9.2f}")

        if "get loop" in results and "get_many" in results:
            speedup = results["get loop"] / results["get_many"]
            print(f"\nspeedup (get_many vs get loop):   {speedup:.2f}x")
        if "gather" in results and "get_many" in results:
            speedup = results["gather"] / results["get_many"]
            print(f"speedup (get_many vs gather):     {speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(_bench())
