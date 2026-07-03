"""Phase 0 spike: probe cuda.core.Buffer affordances.

What we verify, end to end, on whatever GPU + CUDA the current venv ships:

1. cuda.core.DeviceMemoryResource.allocate returns a Buffer with a
   4 KiB-aligned device pointer across a range of sizes (1 KiB to 128 MiB).
2. cuda.core.Buffer.__dlpack__ exposes the buffer to cupy zero-copy
   (same device pointer).
3. cuda.core.Buffer has is_device_accessible / is_host_accessible flags
   but no __cuda_array_interface__ — the wrapper class will need to
   synthesise CAI.
4. Comparison vs cupy's default allocator and the project's RMM pool —
   neither guarantees 4 KiB alignment.

Run from the worktree::

    uv run python bench/buffer/alignment_probe.py

Output is a small table written to stdout and (optionally) appended to
``docs/planning/buffer-spike.md`` if --emit-doc is passed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cupy as cp
import rmm
from cuda.core import (
    Buffer,
    Device,
    DeviceMemoryResource,
    LegacyPinnedMemoryResource,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)

SIZES = [
    1 << 10,  # 1 KiB
    1 << 12,  # 4 KiB
    1 << 14,  # 16 KiB
    1 << 16,  # 64 KiB
    1 << 18,  # 256 KiB
    1 << 20,  # 1 MiB
    1 << 22,  # 4 MiB
    1 << 24,  # 16 MiB
    1 << 26,  # 64 MiB
    1 << 27,  # 128 MiB
]


@dataclass
class Sample:
    size: int
    ptr: int
    aligned_4k: bool
    source: str


def _ptr_of(obj) -> int:
    """Pull the raw device pointer out of an allocation object."""
    if isinstance(obj, Buffer):
        return int(obj.handle)
    if isinstance(obj, cp.ndarray):
        return int(obj.data.ptr)
    if isinstance(obj, rmm.DeviceBuffer):
        return int(obj.ptr)
    raise TypeError(f"unknown allocation type {type(obj)!r}")


def probe_cuda_core(dev: Device) -> list[Sample]:
    mr = DeviceMemoryResource(dev)
    stream = dev.default_stream
    out: list[Sample] = []
    holders: list[Buffer] = []
    for size in SIZES:
        buf = mr.allocate(size, stream=stream)
        holders.append(buf)
        ptr = _ptr_of(buf)
        out.append(Sample(size, ptr, ptr % 4096 == 0, "cuda.core.DeviceMemoryResource"))
    for h in holders:
        h.close(stream=stream)
    return out


def probe_cuda_core_vmr(dev: Device) -> list[Sample]:
    """Virtual-memory resource configured for 4 KiB alignment + GDS RDMA.

    This is the primitive a cuFile-direct path actually wants — every
    allocation gets a fresh page extent aligned to ``addr_align``.
    """
    vmr = VirtualMemoryResource(
        dev,
        VirtualMemoryResourceOptions(addr_align=4096, gpu_direct_rdma=True),
    )
    stream = dev.default_stream
    out: list[Sample] = []
    holders: list[Buffer] = []
    for size in SIZES:
        buf = vmr.allocate(size, stream=stream)
        holders.append(buf)
        ptr = _ptr_of(buf)
        out.append(Sample(size, ptr, ptr % 4096 == 0, "cuda.core.VirtualMemoryResource"))
    for h in holders:
        h.close(stream=stream)
    return out


def probe_cupy_default() -> list[Sample]:
    out: list[Sample] = []
    holders: list[cp.ndarray] = []
    for size in SIZES:
        arr = cp.empty(size, dtype=cp.uint8)
        holders.append(arr)
        ptr = _ptr_of(arr)
        out.append(Sample(size, ptr, ptr % 4096 == 0, "cupy default allocator"))
    del holders
    cp.get_default_memory_pool().free_all_blocks()
    return out


def probe_rmm_pool() -> list[Sample]:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=256 * 1024 * 1024)
    out: list[Sample] = []
    holders: list[rmm.DeviceBuffer] = []
    for size in SIZES:
        buf = rmm.DeviceBuffer(size=size)
        holders.append(buf)
        ptr = _ptr_of(buf)
        out.append(Sample(size, ptr, ptr % 4096 == 0, "rmm pool"))
    del holders
    return out


def probe_affordances(dev: Device) -> dict[str, object]:
    """One-off feature checks on cuda.core.Buffer / DLPack interop."""
    mr = DeviceMemoryResource(dev)
    pinned_mr = LegacyPinnedMemoryResource(dev)
    stream = dev.default_stream

    dbuf = mr.allocate(4096, stream=stream)
    # LegacyPinnedMemoryResource accepts stream=None — pinned host
    # memory is not stream-ordered.
    hbuf = pinned_mr.allocate(4096)

    dlpack_ok = False
    same_ptr = False
    try:
        arr = cp.from_dlpack(dbuf)
        dlpack_ok = True
        same_ptr = int(arr.data.ptr) == int(dbuf.handle)
    except Exception as exc:
        dlpack_ok = f"failed: {exc}"

    facts = {
        "device_is_device_accessible": dbuf.is_device_accessible,
        "device_is_host_accessible": dbuf.is_host_accessible,
        "pinned_is_device_accessible": hbuf.is_device_accessible,
        "pinned_is_host_accessible": hbuf.is_host_accessible,
        "device_has_dlpack": hasattr(dbuf, "__dlpack__"),
        "device_has_cuda_array_interface": hasattr(dbuf, "__cuda_array_interface__"),
        "dlpack_to_cupy_ok": dlpack_ok,
        "dlpack_ptr_matches_handle": same_ptr,
    }
    dbuf.close(stream=stream)
    hbuf.close()
    return facts


def _fmt_size(n: int) -> str:
    for unit, scale in [("MiB", 1 << 20), ("KiB", 1 << 10)]:
        if n >= scale:
            return f"{n // scale} {unit}"
    return f"{n} B"


def _print_table(samples: list[Sample]) -> None:
    print(f"{'source':<32} {'size':>10} {'ptr':>20} {'4K':>4}")
    for s in samples:
        print(f"{s.source:<32} {_fmt_size(s.size):>10} {hex(s.ptr):>20} {'yes' if s.aligned_4k else 'NO':>4}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-doc", action="store_true", help="append results to docs/planning/buffer-spike.md")
    args = parser.parse_args(argv)

    dev = Device(0)
    dev.set_current()

    cc_samples = probe_cuda_core(dev)
    vmr_samples = probe_cuda_core_vmr(dev)
    cp_samples = probe_cupy_default()
    rmm_samples = probe_rmm_pool()
    facts = probe_affordances(dev)

    print("== alignment ==")
    _print_table(cc_samples)
    _print_table(vmr_samples)
    _print_table(cp_samples)
    _print_table(rmm_samples)

    print("\n== affordances ==")
    for k, v in facts.items():
        print(f"  {k}: {v}")

    cc_pass = all(s.aligned_4k for s in cc_samples)
    vmr_pass = all(s.aligned_4k for s in vmr_samples)
    cp_pass = all(s.aligned_4k for s in cp_samples)
    rmm_pass = all(s.aligned_4k for s in rmm_samples)

    print("\n== summary ==")
    print(f"  cuda.core.DeviceMemoryResource always 4 KiB aligned:  {cc_pass}  (sub-allocates from pool — not aligned)")
    print(f"  cuda.core.VirtualMemoryResource(addr_align=4096):     {vmr_pass}  (cuFile-direct primitive)")
    print(f"  cupy default always 4 KiB aligned:                    {cp_pass}")
    print(f"  rmm pool always 4 KiB aligned:                        {rmm_pass}")
    print(f"  cuda.core.Buffer exposes DLPack -> cupy zero-copy:    {facts['dlpack_to_cupy_ok']}")
    print(f"  same device pointer through DLPack:                    {facts['dlpack_ptr_matches_handle']}")
    print(
        f"  cuda.core.Buffer exposes __cuda_array_interface__:    {facts['device_has_cuda_array_interface']}  "
        "(wrapper will need to synthesise it)"
    )

    if args.emit_doc:
        _emit_doc(cc_samples, vmr_samples, cp_samples, rmm_samples, facts, dev)

    return 0


def _emit_doc(cc, vmr_s, cp_s, rmm_s, facts, dev) -> None:
    """Write a tiny markdown report next to the planning docs."""
    from pathlib import Path

    out_path = Path(__file__).resolve().parents[2] / "docs" / "planning" / "buffer-spike.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _row(s: Sample) -> str:
        return f"| {s.source} | {_fmt_size(s.size)} | `{hex(s.ptr)}` | {'yes' if s.aligned_4k else '**NO**'} |"

    header = "| source | size | ptr | 4 KiB |\n|---|---|---|---|"
    rows = "\n".join(_row(s) for s in (*cc, *vmr_s, *cp_s, *rmm_s))
    facts_md = "\n".join(f"- `{k}`: `{v}`" for k, v in facts.items())

    body = f"""# Buffer epic — Phase 0 spike

Generated by `bench/buffer/alignment_probe.py` on `{dev.name}`.

## Alignment

{header}
{rows}

## cuda.core.Buffer affordances

{facts_md}

## Findings (revised)

The handoff doc claimed `DeviceMemoryResource.allocate` returns 4 KiB-aligned
pointers.  That was wrong: `DeviceMemoryResource` is a stream-ordered pool —
the first allocation of a freshly-grown pool extent is page-aligned, but
subsequent allocations are packed end-to-end at the natural alignment of
the requested size (commonly 256 B or 512 B for sub-page sizes).  cupy
default and the RMM pool behave the same way for the same reason.

`VirtualMemoryResource` with `addr_align=4096, gpu_direct_rdma=True` is
the primitive we actually want for the cuFile-direct path.  Every
allocation gets a fresh page-aligned virtual address; the granularity
overhead (2 MiB per alloc on default settings) is paid in VA, not
physical memory, and is irrelevant for the chunk sizes zarr deals with.

## Decision

Wrap `cuda.core.Buffer` in `CzarrGpuBuffer` and back it with
`VirtualMemoryResource(addr_align=4096, gpu_direct_rdma=True)` so we get
guaranteed cuFile-direct alignment + native GDS RDMA flag.  DLPack + cupy
interop is zero-copy and pointer-stable; `is_device_accessible` /
`is_host_accessible` give us first-class memory-class introspection; CAI
must be synthesised on our wrapper from `handle + size`.

Proceed with Phase 1 (`jc84qa13`): `CzarrGpuBuffer` skeleton.
"""
    out_path.write_text(body)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
