# Phase 6 — pipeline benchmark results

Comparison of `zarr.core.codec_pipeline.BatchedCodecPipeline` (stock zarr v3 default) vs `czarr.pipeline.CzarrPipeline` (the GPU-direct decode path landed in Phase 2).

## Workload

Synthetic ``float32`` arrays compressed with ``czarr.Zstd()``, stored via ``GPULocalStore`` + cuFile (compat mode on this host — no real GDS hardware), read back via ``arr[:]``.  All numbers from an A40 + CUDA 13.1, in-process pytest fixture (no SLURM batching), median of 5 reads after 2 warmup reads.

```
workload                           pipeline              median       min     GiB/s
-----------------------------------------------------------------------------------
many small chunks (1024 x 64KB)    zarr default        1087.5 ms   762.7 ms     0.01
many small chunks (1024 x 64KB)    czarr               1268.7 ms  1022.1 ms     0.01
balanced (64 x 1 MiB)              zarr default         209.0 ms   145.6 ms     0.30
balanced (64 x 1 MiB)              czarr                167.5 ms   138.3 ms     0.37
few large chunks (8 x 8 MiB)       zarr default          64.9 ms    62.7 ms     1.93
few large chunks (8 x 8 MiB)       czarr                 63.7 ms    61.1 ms     1.96
```

| workload | speedup |
|---|---|
| many small chunks | **0.86×** (regression) |
| balanced | **1.25×** |
| few large chunks | **1.02×** (parity) |

## Reading

The GPU-direct codec path (skipping ``chunk.to_bytes()`` then ``.cuda()`` round-trip) helps when there's enough bytes per chunk for the host transfer to matter.  In the **balanced** case at 1 MiB/chunk the savings show up at 25%.  At 8 MiB/chunk we're at parity — wall-clock is dominated by the nvCOMP decode and cuFile read itself, the round-trip elimination is a small relative win.

At the small end (64 KiB/chunk × 1024) czarr is **slower** by 14%.  Each chunk's Python-level dispatch overhead is higher in our path; the host round-trip we're avoiding is itself only 64 KiB of PCIe traffic.  The fixed per-chunk costs in the codec base (``isinstance`` check, ``as_array_like()``, ``nvcomp.as_array`` from a cupy view) outweigh the savings.

This is a known limitation of the Phase 2 implementation.  The small-chunk regression should disappear once we add per-chunk batching at the pipeline layer (currently each chunk goes through ``_decode_single`` separately even though we COULD batch them into one nvCOMP call).

## Throughput context

Best throughput in any configuration: **1.96 GiB/s**.  For reference: nvCOMP zstd on A40 alone (no I/O) typically pushes 10-15 GiB/s.  The gap is owed to:

* cuFile **compat mode** (no nvidia-fs on this host) — falls back to host-buffered POSIX reads.  Real GDS would close most of this gap.
* Python overhead in the pipeline per-chunk path.
* nvCOMP scratch buffer allocation on first decode (mostly amortised but still in the timing window).

## Next perf attacks (post-refactor)

1. **Cross-chunk batching at the pipeline layer**.  Today each chunk reaches the codec via ``_decode_single``; nvCOMP could decode all N chunks of a selection in one batched call.  Biggest expected win for small/medium chunks.
2. **Pinned-host staging for CPU-buffer sources**.  Phase 1 substrate is already in place (``PinnedHostPool``); needs wiring into ``CudaBytesBytesCodec._batch_sync`` non-GPU path.
3. **Real GDS hardware**.  cuFile compat-mode read is the wall on a host without nvidia-fs; the architecture is GDS-ready.
4. **GPU CRC32C kernel**.  Currently uses host google_crc32c; small payload but synchronous GPU→host transfer per chunk.
5. **GPU bitshuffle as a standalone filter**.  Reuse the RawKernel from earlier session for the v3 ``bitshuffle`` codec id.

## What's solid

* Architecture: 4 codec ABCs all wired, registry shadows in place, ``configure_gpu`` is the only knob.
* Correctness: 88 tests pass across pipeline / filters / sharding / crc32c / codec / storage.
* Sharding works for free — zarr-stock ``ShardingCodec`` routes through ``CzarrPipeline`` via the global ``codec_pipeline.path``.
* No zarr v2 surface remains.

## Reproduce

```bash
uv run --extra cu12 --group test python -m bench.zarr.pipeline_compare
```

Output above was captured on Bruno GPU node + CUDA 13.1 + CuPy 12 + nvCOMP 5.1 + A40.
