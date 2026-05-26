"""Quick probe: can we even use cuda.compute from this venv?

The 2026 ``cuda.compute`` module ships in the ``cuda-cccl`` pip
package.  The cluster's CUDA-13 system + CuPy-cu12 install pattern
has bitten us before with libnvrtc.so lookup; verify cuda.compute
plays nicely on a GPU node before we sink time into the byteshuffle
migration.

Smallest meaningful test: ``cuda.compute.reduce_into`` on a 1M-elem
cupy array.  If this runs and matches numpy's reference sum, the
toolchain is usable and we can start the byteshuffle work.
"""

import cupy as cp
import numpy as np
import cuda.compute as cc


def main() -> int:
    rng = np.random.default_rng(42)
    host = rng.standard_normal(1_000_000).astype(np.float32)
    dev = cp.asarray(host)
    out_dev = cp.empty((), dtype=cp.float32)

    # cuda.compute.reduce_into: device-wide sum.  h_init must be a
    # 0-d numpy array; the reducer binds (d_in, d_out, op, h_init) at
    # construction and is invoked with no positional args.
    h_init = np.array(0, dtype=np.float32)
    op = lambda a, b: a + b  # noqa: E731
    reducer = cc.make_reduce_into(d_in=dev, d_out=out_dev, op=op, h_init=h_init)
    # The reducer's __call__ is keyword-only: query temp storage size,
    # alloc, then invoke for real.
    temp_bytes = reducer(temp_storage=None, d_in=dev, d_out=out_dev, num_items=dev.size, op=op, h_init=h_init)
    temp = cp.empty(temp_bytes, dtype=cp.uint8) if temp_bytes else None
    reducer(temp_storage=temp, d_in=dev, d_out=out_dev, num_items=dev.size, op=op, h_init=h_init)

    cp.cuda.Stream.null.synchronize()
    gpu_sum = float(out_dev.get())
    cpu_sum = float(host.sum())

    print(f"gpu_sum = {gpu_sum:.4f}")
    print(f"cpu_sum = {cpu_sum:.4f}")
    print(f"rel err = {abs(gpu_sum - cpu_sum) / max(abs(cpu_sum), 1e-9):.3e}")

    if abs(gpu_sum - cpu_sum) / max(abs(cpu_sum), 1e-9) > 1e-3:
        print("MISMATCH")
        return 1
    print("OK — cuda.compute is usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
