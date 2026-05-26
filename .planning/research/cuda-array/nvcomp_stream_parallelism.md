# nvCOMP stream-parallelism probe

Source: ``bench/experiments/nvcomp_stream_parallelism.py``.
Open question #2 from ``SUMMARY.md``. Blocks Phase 1 lanes-architecture
commit (``dex atyfi07d``).

## Setup

- Device: ``NVIDIA H200``
- Lanes: ``8``
- Reps: ``20``
- Raw chunk size: ``16,777,216`` bytes (~16 MiB)
- Encoded sizes (zstd, normal float32): ``[15510522, 15509787, 15509334, 15509702, 15509769, 15510359, 15509692, 15509303]``

## Results

| regime | median (ms) | min (ms) | stdev (ms) |
|---|---:|---:|---:|
| serial (1 codec, 1 stream) | 314.95 | 313.90 | 0.55 |
| per-lane (N codecs, N streams) | 314.79 | 313.28 | 4.33 |
| shared codec (1 codec, N streams) | 314.66 | 313.56 | 0.59 |

**Overlap factors:**

- per-lane: **1.00x** (ideal 8.0x, breakeven 1.0x)
- shared codec: **1.00x**

## Verdict

no parallel decode — lanes architecture DOES NOT win on its own. shared Codec across N streams works equally well; save scratch by sharing.

## Implications for Phase 1

The lanes architecture in ``CudaZarrArray`` runs N microbatches on N CUDA
streams, expecting decode of batch K to overlap reads of batch K+1.
This probe isolates the *decode-side* parallelism question. If the
overlap factor is close to 1.0, the architecture's premise is wrong —
the work serialises at the driver or shared-scratch level no matter how
many streams we use.

Re-read in context with the H200 microbatch-sweep result (linear
slowdown at small batch sizes due to ~35-40 ms nvCOMP per-call
overhead): the lanes architecture only wins if (a) per-call overhead
can run concurrently across streams (this probe) and (b) reads can
actually run concurrently with decode at the storage layer (compat-mode
on VAST limits this).
