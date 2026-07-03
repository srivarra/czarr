"""Isolate the cuFile large-read failure (dex ve8x3mkv) on H100/GDS.

256 MiB cuFile reads fail nondeterministically (garbage bytes or a libcufile
assertion that aborts the process); 8 MiB reads work. This tests, one variable
at a time, which of {4KiB alignment, cuFileBufRegister, read segmentation}
makes a large read correct. Each strategy MUST run in its own process (an
assert aborts the interpreter), so the sbatch invokes this once per --strategy.

Strategies:
  A  baseline: cp.empty buffer, unregistered, single full read   (reproduce)
  B  +aligned: 4KiB-aligned VMR buffer, unregistered, single read
  C  +registered: aligned + cuFileBufRegister, single read
  D  +segmented: aligned + registered + 16 MiB segments

Run: uv run --extra cu12 python -m bench.cufile_debug --strategy A
"""

import argparse
import os

import cupy as cp

from czarr import cufile as cufile_runtime

_CHUNK = "/hpc/projects/waveorder/tile-stitch/sample_datasets/l0_brightfield_fov.zarr/0/c/0/0/9/0/0"
_SEG = 16 << 20  # max_direct_io_size default


def _make_buf(size: int, aligned: bool):
    if not aligned:
        arr = cp.empty(size, dtype=cp.uint8)
        return arr, int(arr.data.ptr), None
    from cuda.core import Device, VirtualMemoryResource, VirtualMemoryResourceOptions

    d = Device()
    d.set_current()
    mr = VirtualMemoryResource(d, VirtualMemoryResourceOptions(addr_align=4096, gpu_direct_rdma=True))
    buf = mr.allocate(size)  # 4 KiB-aligned, RDMA-tagged
    arr = cp.from_dlpack(buf).view(cp.uint8)
    return arr, int(arr.data.ptr), buf  # return buf to keep it alive


def main() -> int:
    from cuda.bindings import cufile

    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--chunk", default=_CHUNK)
    ap.add_argument("--gen-mib", type=int, default=0, help="if >0, read a freshly-written N-MiB file (size sweep)")
    args = ap.parse_args()

    if args.gen_mib:
        import numpy as np

        args.chunk = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"cufile_gen_{args.gen_mib}.bin")
        np.frombuffer(os.urandom(64), dtype=np.uint8)  # touch
        with open(args.chunk, "wb") as fh:
            fh.write(np.random.default_rng(0).integers(0, 256, args.gen_mib << 20, dtype=np.uint8).tobytes())
            fh.flush()
            os.fsync(fh.fileno())  # GDS reads via O_DIRECT, bypassing page cache; must hit disk
    size = os.path.getsize(args.chunk)
    expect = open(args.chunk, "rb").read(16)
    print(f"strategy={args.strategy}  chunk={size} bytes ({size / 2**20:.0f}MiB)  expect_hdr={expect.hex()}")

    cufile_runtime.ensure_driver_open()
    aligned = args.strategy in ("B", "C", "D")
    registered = args.strategy in ("C", "D")
    segmented = args.strategy == "D"

    arr, ptr, _keep = _make_buf(size, aligned)
    print(
        f"  dev_ptr % 4096 = {ptr % 4096}  (aligned={ptr % 4096 == 0})  registered={registered}  segmented={segmented}"
    )

    if registered:
        cufile_runtime.ensure_buf_registered(ptr, size)

    fd = os.open(args.chunk, os.O_RDONLY)
    try:
        with cufile_runtime.registered_handle(fd) as h:
            if segmented:
                done = 0
                while done < size:
                    n = min(_SEG, size - done)
                    r = cufile.read(h, ptr + done, n, done, 0)
                    if r <= 0:
                        print(f"  segment at {done} returned {r}; stop")
                        break
                    done += r
                got = done
            else:
                got = cufile.read(h, ptr, size, 0, 0)
        cp.cuda.runtime.deviceSynchronize()
        first = bytes(cp.asnumpy(arr[:16]))
        ok = (got == size) and (first == expect)
        print(f"  read returned: {got} (expected {size}, delta={size - got})")
        print(f"  first16 = {first.hex()}  match={first == expect}")
        print(f"  RESULT: {'OK' if ok else 'WRONG'}")
    finally:
        if registered:
            cufile_runtime.deregister_buf(ptr)
        os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
