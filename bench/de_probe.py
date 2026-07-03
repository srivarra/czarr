"""Probe whether this GPU exposes the Blackwell hardware Decompression Engine (DE).

Authoritative capability query: the device attribute
``CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK`` is a bitmask of the
DE-supported algorithms (Deflate / LZ4 / Snappy); it is 0 / UNSUPPORTED on
hardware without the engine.  (The ``CU_MEM_CREATE_USAGE_HW_DECOMPRESS`` usage
flag is only an allocation hint — it succeeds on non-DE hardware too, so it is
NOT a capability gate.)  Datacenter Blackwell (B200/B300/GB200/GB300) reports a
nonzero mask; this checks a given GPU (e.g. RTX PRO 6000 / GB202) directly.

Run: module load cuda/13.1.0_590.44.01 && uv run --extra cu12 python -m bench.de_probe
"""

from __future__ import annotations

import cupy as cp
from cuda.bindings import driver as cuda


def main() -> int:
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(f"GPU: {name}  cc {props['major']}.{props['minor']}")

    cuda.cuInit(0)
    dev = cuda.cuDeviceGet(0)[1]
    attr = cuda.CUdevice_attribute

    mask_res = cuda.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, dev)
    len_res = cuda.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH, dev)
    mask = mask_res[1] if mask_res[0] == cuda.CUresult.CUDA_SUCCESS else None
    max_len = len_res[1] if len_res[0] == cuda.CUresult.CUDA_SUCCESS else None
    print(f"MEM_DECOMPRESS_ALGORITHM_MASK = {mask}")
    print(f"MEM_DECOMPRESS_MAXIMUM_LENGTH = {max_len}")

    if mask is None:
        print("RESULT: attribute query failed — inconclusive")
        return 2

    algos = []
    for a in cuda.CUmemDecompressAlgorithm:
        v = int(a.value)
        if v and (mask & v):
            algos.append(a.name)
    if mask:
        print(f"RESULT: Decompression Engine IS available — algorithms: {algos or hex(mask)}  (max_len={max_len})")
        return 0
    print("RESULT: mask == 0  => Decompression Engine NOT available on this GPU")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
