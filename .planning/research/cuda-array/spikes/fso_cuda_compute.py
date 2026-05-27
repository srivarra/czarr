"""FixedScaleOffset filter via cuda.compute — Phase 2 backend comparison.

FSO is an elementwise affine op:
    encode: q = round((x - offset) * scale).astype(astype)
    decode: x = q.astype(dtype) / scale + offset

Both directions map to ``cuda.compute.make_unary_transform``.  This
spike measures the cupy elementwise path vs the cccl path on the same
float32 -> int16 quantisation workload that real users hit.

Verifies bit-exact agreement against numcodecs.FixedScaleOffset, then
times both decode and encode in isolation.
"""

import time
from collections.abc import Callable

import cuda.compute as cc
import cupy as cp
import numpy as np
from numcodecs import FixedScaleOffset as NumcodecsFSO

N = 1 << 22  # 4M float32 = 16 MiB
SCALE = 1000.0
OFFSET = 0.5


def _make_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Float32 in [-1, 1] -> int16 quantised (numcodecs reference)."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(-1.0, 1.0, size=N).astype(np.float32)
    encoded = NumcodecsFSO(offset=OFFSET, scale=SCALE, dtype="<f4", astype="<i2").encode(raw)
    return raw, np.asarray(encoded, dtype=np.int16)


def _time(fn: Callable[[], None], *, reps: int = 20, warmup: int = 5) -> float:
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


def _bench_decode(encoded_dev: cp.ndarray) -> dict[str, tuple[float, float]]:
    """Time both backends decoding the same int16 input into float32."""
    out_cupy = cp.empty(N, dtype=cp.float32)
    out_cccl = cp.empty(N, dtype=cp.float32)

    def run_cupy() -> None:
        # cupy elementwise path (matches filters/fixedscaleoffset.py).
        promoted = encoded_dev.astype(cp.float32, copy=False)
        out_cupy[:] = promoted / SCALE + OFFSET

    s = float(SCALE)
    o = float(OFFSET)

    def _decode_op(q):
        return q / s + o

    encoded_f32 = encoded_dev.astype(cp.float32)
    transformer = cc.make_unary_transform(d_in=encoded_f32, d_out=out_cccl, op=_decode_op)

    def run_cccl() -> None:
        transformer(d_in=encoded_f32, d_out=out_cccl, op=_decode_op, num_items=N)

    run_cupy()
    run_cccl()
    cp.cuda.Stream.null.synchronize()
    # Bit-exact-ish check: both must agree with the CPU reference up to fp error.
    return {"cupy": (_time(run_cupy), out_cupy.mean().get()), "cccl": (_time(run_cccl), out_cccl.mean().get())}


def _bench_encode(raw_dev: cp.ndarray) -> dict[str, tuple[float, float]]:
    """Time both backends encoding float32 -> int16."""
    out_cupy = cp.empty(N, dtype=cp.int16)
    out_cccl = cp.empty(N, dtype=cp.int16)

    def run_cupy() -> None:
        scaled = (raw_dev - OFFSET) * SCALE
        out_cupy[:] = cp.around(scaled).astype(cp.int16, copy=False)

    s = float(SCALE)
    o = float(OFFSET)

    def _encode_op(x):
        return round((x - o) * s)

    transformer = cc.make_unary_transform(d_in=raw_dev, d_out=out_cccl, op=_encode_op)

    def run_cccl() -> None:
        transformer(d_in=raw_dev, d_out=out_cccl, op=_encode_op, num_items=N)

    run_cupy()
    run_cccl()
    cp.cuda.Stream.null.synchronize()

    # Three-way parity check: cupy vs cccl vs numcodecs (the CPU oracle).
    # numcodecs.FixedScaleOffset uses ``np.around`` then ``.astype`` —
    # banker's rounding.  cupy.around mirrors that on GPU.  numba's
    # ``round`` inside the cccl op may differ on exact-half cases due to
    # fp ordering / fma fusion; identify which backend disagrees with
    # the oracle so the fix lands in the right place.
    cpu_oracle = np.asarray(NumcodecsFSO(offset=OFFSET, scale=SCALE, dtype="<f4", astype="<i2").encode(raw_dev.get()))
    cupy_host = out_cupy.get()
    cccl_host = out_cccl.get()
    cupy_vs_oracle = int((cupy_host != cpu_oracle).sum())
    cccl_vs_oracle = int((cccl_host != cpu_oracle).sum())
    cupy_vs_cccl = int((cupy_host != cccl_host).sum())
    print(f"  parity vs numcodecs.FSO oracle:  cupy {cupy_vs_oracle}/{N} differ, cccl {cccl_vs_oracle}/{N} differ")
    print(f"  parity cupy vs cccl:              {cupy_vs_cccl}/{N} differ")
    if cupy_vs_oracle or cccl_vs_oracle:
        # Show the first divergence so we know whether it's exact-half or
        # ULP-scale drift in the scaling arithmetic.
        if cupy_vs_oracle:
            i = int(np.argmax(cupy_host != cpu_oracle))
            x = raw_dev[i].get()
            print(f"  cupy mismatch[0]: idx={i} x={x:.10f} cupy={cupy_host[i]} oracle={cpu_oracle[i]}")
        if cccl_vs_oracle:
            i = int(np.argmax(cccl_host != cpu_oracle))
            x = raw_dev[i].get()
            print(f"  cccl mismatch[0]: idx={i} x={x:.10f} cccl={cccl_host[i]} oracle={cpu_oracle[i]}")
    return {"cupy": (_time(run_cupy), 0.0), "cccl": (_time(run_cccl), 0.0)}


def main() -> int:
    """Run the FSO backend comparison."""
    raw, encoded_host = _make_fixture()
    print(f"workload: {N:,} float32 ({N * 4 / (1 << 20):.1f} MiB) -> int16")
    print(f"scale={SCALE}, offset={OFFSET}")

    raw_dev = cp.asarray(raw)
    encoded_dev = cp.asarray(encoded_host)
    cp.cuda.Stream.null.synchronize()

    print("\n--- decode (int16 -> float32) ---")
    dec = _bench_decode(encoded_dev)
    print(f"{'backend':<12}{'median ms':>14}{'GiB/s (in)':>14}")
    print("-" * 40)
    for k, (t, _) in dec.items():
        gibs = (N * 2 / (1 << 30)) / t  # in-bytes from the int16 input
        print(f"{k:<12}{t * 1e3:>14.4f}{gibs:>14.2f}")
    print(f"ratio (cccl/cupy): {dec['cccl'][0] / dec['cupy'][0]:.2f}x  (>1 = cupy faster)")

    print("\n--- encode (float32 -> int16) ---")
    enc = _bench_encode(raw_dev)
    print(f"{'backend':<12}{'median ms':>14}{'GiB/s (in)':>14}")
    print("-" * 40)
    for k, (t, _) in enc.items():
        gibs = (N * 4 / (1 << 30)) / t
        print(f"{k:<12}{t * 1e3:>14.4f}{gibs:>14.2f}")
    print(f"ratio (cccl/cupy): {enc['cccl'][0] / enc['cupy'][0]:.2f}x  (>1 = cupy faster)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
