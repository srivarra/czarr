"""Delta filter via cuda.compute — Phase 2 filter migration spike.

The Delta filter stores first differences: ``out[i] = arr[i] - arr[i-1]``
(with ``out[0] = arr[0]``).  Inverse is a cumulative sum.  Maps cleanly
to ``cuda.compute.make_inclusive_scan`` with ``op = a + b``.

This spike:
1. Encodes ascending int32 data via numcodecs.Delta (CPU reference).
2. Decodes it three ways:
   a) numcodecs.Delta on CPU — the bit-exact oracle.
   b) cupy.cumsum on GPU — the trivial baseline.
   c) cuda.compute.make_inclusive_scan on GPU — the Phase 2 target.
3. Verifies all three produce the same bytes.
4. Times each.

If (c) matches the oracle and is within an order of magnitude of (b),
the cuda.compute filter migration plan is de-risked.
"""

import time
from collections.abc import Callable

import cuda.compute as cc
import cupy as cp
import numpy as np
from numcodecs import Delta as NumcodecsDelta

N = 1 << 20  # 1M int32 = 4 MiB
DTYPE = np.int32


def _make_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Ascending-with-noise int32 — typical Delta-friendly workload."""
    rng = np.random.default_rng(0)
    raw = np.arange(N, dtype=DTYPE) + rng.integers(-3, 4, size=N, dtype=DTYPE)
    encoded = NumcodecsDelta(dtype=DTYPE.__name__).encode(raw)
    return raw, np.asarray(encoded)


def _time(fn: Callable[[], None], *, reps: int = 10, warmup: int = 3) -> float:
    """Median wall time over ``reps`` calls after ``warmup``."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    """Run the three paths, verify equivalence, print timings."""
    raw, encoded_host = _make_fixture()
    print(f"workload: {N:,} int32 values ({N * 4 / (1 << 20):.1f} MiB)")
    print(f"first 8 raw:     {raw[:8].tolist()}")
    print(f"first 8 encoded: {encoded_host[:8].tolist()}")

    # CPU oracle
    cpu_decoded = NumcodecsDelta(dtype=DTYPE.__name__).decode(encoded_host)
    cpu_decoded = np.asarray(cpu_decoded, dtype=DTYPE)
    assert np.array_equal(cpu_decoded, raw), "CPU oracle disagrees with raw"
    print("CPU oracle decode ✓")

    # Pre-upload encoded to device once.
    encoded_dev = cp.asarray(encoded_host).view(DTYPE)
    cp.cuda.Stream.null.synchronize()

    # Path B: cupy.cumsum
    cumsum_out = cp.empty(N, dtype=DTYPE)

    def run_cupy() -> None:
        cp.cumsum(encoded_dev, dtype=DTYPE, out=cumsum_out)

    run_cupy()
    cp.cuda.Stream.null.synchronize()
    assert np.array_equal(cp.asnumpy(cumsum_out), raw), "cupy.cumsum disagrees"
    print("cupy.cumsum ✓")

    # Path C: cuda.compute inclusive_scan.  Constructor takes op only;
    # init_value is passed at invocation (identity for +: zero).
    scan_out = cp.empty(N, dtype=DTYPE)
    op = lambda a, b: a + b
    init_value = np.array(0, dtype=DTYPE)
    scanner = cc.make_inclusive_scan(d_in=encoded_dev, d_out=scan_out, op=op)
    temp_bytes = scanner(
        temp_storage=None,
        d_in=encoded_dev,
        d_out=scan_out,
        num_items=N,
        op=op,
        init_value=init_value,
    )
    temp = cp.empty(temp_bytes, dtype=cp.uint8) if temp_bytes else None

    def run_cccl() -> None:
        scanner(
            temp_storage=temp,
            d_in=encoded_dev,
            d_out=scan_out,
            num_items=N,
            op=op,
            init_value=init_value,
        )

    run_cccl()
    cp.cuda.Stream.null.synchronize()
    if not np.array_equal(cp.asnumpy(scan_out), raw):
        diff = cp.asnumpy(scan_out) - raw
        bad = int(np.argmax(diff != 0))
        print(f"MISMATCH at index {bad}: got {scan_out[bad]} expected {raw[bad]}")
        return 1
    print("cuda.compute inclusive_scan ✓")

    # Timing
    t_cupy = _time(run_cupy)
    t_cccl = _time(run_cccl)
    gibs_cupy = (N * 4 / (1 << 30)) / t_cupy
    gibs_cccl = (N * 4 / (1 << 30)) / t_cccl

    print()
    print(f"{'backend':<28}{'median ms':>12}{'GiB/s':>10}")
    print("-" * 50)
    print(f"{'cupy.cumsum':<28}{t_cupy * 1e3:>12.4f}{gibs_cupy:>10.2f}")
    print(f"{'cuda.compute scan':<28}{t_cccl * 1e3:>12.4f}{gibs_cccl:>10.2f}")
    print(f"\nratio (cccl / cupy): {t_cccl / t_cupy:.2f}× (>1.0 = cupy faster)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
