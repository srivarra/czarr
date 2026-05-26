# native LZ4 stream-parallelism probe

Source: `bench/experiments/native_lz4_stream_parallelism.py` (job 33253014,
H200, gpu-h-6). Companion to `nvcomp_stream_parallelism.md` — same question,
asked of the in-house LZ4 kernel from `spikes/lz4_decoder.py` instead of
nvCOMP's closed-source decoder.

Gates design decision #2 in Phase 1 (dex `atyfi07d`): do we add lane
bookkeeping to the native LZ4 backend, or is a single big grid launch
already saturating the GPU?

## Setup

- Blocks per measurement: 1024
- Raw bytes per block: 64 KiB
- Total uncompressed payload per call: 64 MiB
- Reps per measurement: 30
- Compressed total per call: 242,688 bytes (heavily compressible synthetic pattern; spike's "lz4-friendly" fixture)

## Results

Sweep across `lanes ∈ {2, 4, 8}`, three regimes per sweep:

| lanes | regime | median (ms) | min (ms) | GiB/s |
|---:|---|---:|---:|---:|
| 2 | single big launch | 0.362 | 0.359 | **172.57** |
| 2 | serial × N launches | 0.708 | 0.661 | 88.22 |
| 2 | parallel × N streams | 0.409 | 0.377 | 152.83 |
| 4 | single big launch | 0.360 | 0.356 | **173.73** |
| 4 | serial × N launches | 1.379 | 1.224 | 45.31 |
| 4 | parallel × N streams | 0.477 | 0.466 | 130.96 |
| 8 | single big launch | 0.361 | 0.358 | **172.99** |
| 8 | serial × N launches | 2.703 | 2.361 | 23.13 |
| 8 | parallel × N streams | 0.615 | 0.604 | 101.63 |

**Overlap factors** (single ÷ parallel):

| lanes | overlap | ideal | breakeven |
|---:|---:|---:|---:|
| 2 | 0.89× | 2.0× | 1.0× |
| 4 | 0.75× | 4.0× | 1.0× |
| 8 | 0.59× | 8.0× | 1.0× |

## Verdict

**Multi-stream is slower than a single grid launch, and gets monotonically worse with more lanes.** Launch overhead dominates as we split work across streams. The single-launch wall is rock-steady at ~0.36 ms regardless of lane count, which means the H200 SMs are already fully utilised by one grid of 1024 blocks.

Practical conclusions:

1. **Single big grid launch is the right design** for the native LZ4 backend. No lane bookkeeping in the kernel-dispatch layer.
2. **Native LZ4 saturates the H200 at ~173 GiB/s decode throughput.** That's roughly 20× nvCOMP LZ4 on the same hardware. The native backend's win is the kernel itself, not stream parallelism.
3. **Decode is no longer the bottleneck.** 64 MiB of LZ4 finishes in 0.36 ms. For a 1 GiB workload that's roughly 6 ms of decode — far below the 40 ms cuFile read time we measured. The wall time is now I/O-bound.

## Implications for Phase 1 design

This finding mirrors the nvCOMP probe: GPU is saturated by one launch; lanes don't help compute. But the *consequence* is different:

- **nvCOMP path** (Zstd, etc.): single big decode call, fixed ~35-40 ms per-call overhead. Wall time ≈ max(reads, decode) ≈ 47 ms on the H200 slab → ~21 GiB/s ceiling.
- **Native LZ4 path**: single big grid launch, sub-millisecond per-call. Wall time ≈ reads ≈ 40 ms on the same slab → ~25 GiB/s end-to-end, dominated by cuFile reads. With faster storage (local NVMe + actual GDS), reads drop and the wall drops with them — native LZ4 has no fixed-cost floor.

Architecture pivots accordingly:

- **No lanes anywhere.** One stream, one decode call per microbatch.
- **The I/O pipeline does the work.** Stream reads into a bounded queue; the decoder consumes the queue without awaiting all reads first. Removes the `await concurrent_map(reads)` barrier in `BatchedCodecPipeline.read_batch`.
- **Native LZ4 backend unparks and merges into Phase 1.** The spike's kernel is already production-quality for the decode-only case; we wrap it as `czarr.LZ4(backend="native")` and ship it as a v0.1 codec.

## Cross-references

- `nvcomp_stream_parallelism.md` — companion probe; same verdict for nvCOMP.
- `07-api-design.md` — codec backend selection, generic API surface.
- `spikes/lz4_decoder.py` — the LZ4 kernel measured here.
- `SUMMARY.md` — research synthesis (needs revision to reflect this finding).
