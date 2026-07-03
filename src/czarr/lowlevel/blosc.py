"""Native nvCOMP batched-zstd blosc decode — ctypes -> libnvcomp.so.

The fast blosc decode path is the nvCOMP *native* batched API
(``nvcompBatchedZstdDecompressAsync``), which is NOT exposed by the nvCOMP
Python binding (only the high-level ``Codec``, which is ~20x slower for many
small streams).  We reach it via ctypes against the ``libnvcomp.so`` shipped
in the ``nvidia-libnvcomp-cu12`` wheel.  Validated 10-15x faster than CPU
blosc + H2D on real data (see ``.planning/research/cuda-array/spikes``).

Pipeline per chunk-batch: parse the blosc1 chunk header -> build the per-block
nvCOMP descriptor arrays on-device (vectorized, no Python loop) -> ONE native
batched zstd decode over all blocks -> a per-chunk bit/byte unshuffle kernel.

Scope (v1): zstd sub-codec, bit/byte/no shuffle, all-codec blocks.  Raw
(memcpy) / RLE-special blocks, non-zstd sub-codecs, and bitshuffle tails raise
``NotImplementedError`` — the stores we target (OME-Zarr f16, zstd+bitshuffle,
power-of-two blocks) hit none of these.

The native API is experimental (subject to change across nvCOMP releases); the
binding is pinned to ``nvidia-libnvcomp-cu12 == 5.2.0.13``.
"""

import ctypes
import struct
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import cupy as cp
import numpy as np

from czarr.kernels.bitunshuffle import bitunshuffle_into

_BLOSC_HEADER = 16
_CODEC_ZSTD = 4  # blosc flags>>5 codec id
# Cap chunks per native decode call: nvCOMP temp scales with block count
# (~6.5 GiB / 65536 blocks).  Sub-batching bounds device scratch regardless
# of how many chunks zarr hands the codec at once.
_MAX_CHUNKS_PER_CALL = 8


class _ZstdDecompressOpts(ctypes.Structure):
    _fields_: ClassVar = [("backend", ctypes.c_int), ("reserved", ctypes.c_char * 60)]  # 64 bytes


@cache
def _lib() -> ctypes.CDLL:
    """Load libnvcomp.so from the installed wheel and bind the native zstd API."""
    import importlib
    from pathlib import Path

    mod = importlib.import_module("nvidia.libnvcomp")  # namespace pkg — import_module binds it
    so = str(Path(mod.__file__).parent / "lib64" / "libnvcomp.so.5")
    lib = ctypes.CDLL(so)
    lib.nvcompBatchedZstdDecompressGetTempSizeAsync.restype = ctypes.c_int
    lib.nvcompBatchedZstdDecompressGetTempSizeAsync.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        _ZstdDecompressOpts,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.nvcompBatchedZstdDecompressAsync.restype = ctypes.c_int
    lib.nvcompBatchedZstdDecompressAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        _ZstdDecompressOpts,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib


@dataclass(frozen=True, slots=True)
class _Layout:
    nbytes: int
    blocksize: int
    nblocks: int
    typesize: int
    shuffle: str  # "bit" | "byte" | "none"
    bstarts: np.ndarray  # int64[nblocks]


def _parse_header(comp: cp.ndarray) -> _Layout:
    """Parse a blosc1 chunk header from a device buffer (small D2H of the header region)."""
    hdr = bytes(cp.asnumpy(comp[:_BLOSC_HEADER]))
    flags, typesize = hdr[2], hdr[3]
    nbytes, blocksize, _cbytes = struct.unpack_from("<iii", hdr, 4)
    codec = flags >> 5
    if codec != _CODEC_ZSTD:
        raise NotImplementedError(f"blosc sub-codec id {codec} unsupported (only zstd=4)")
    if flags & 0x02:
        raise NotImplementedError("memcpyed (uncompressed) blosc chunk not yet supported")
    shuffle = "bit" if flags & 0x04 else "byte" if flags & 0x01 else "none"
    nblocks = -(-nbytes // blocksize)
    bstarts = cp.asnumpy(comp[_BLOSC_HEADER : _BLOSC_HEADER + 4 * nblocks]).view("<i4").astype(np.int64)
    return _Layout(nbytes, blocksize, nblocks, typesize, shuffle, bstarts)


def _decode_subbatch(comps: list[cp.ndarray], stream: int) -> list[cp.ndarray]:
    lib = _lib()
    layouts = [_parse_header(c) for c in comps]
    bs = layouts[0].blocksize
    scratches = [cp.empty(L.nblocks * bs, dtype=cp.uint8) for L in layouts]

    # Vectorized on-device fanout: per chunk, gather the per-block `cb` size
    # prefix at each bstart and build the four nvCOMP descriptor arrays.
    cps, css, dps, dbs = [], [], [], []
    for comp, scr, L in zip(comps, scratches, layouts, strict=True):
        base, sb = int(comp.data.ptr), int(scr.data.ptr)
        bst = cp.asarray(L.bstarts)
        u8 = comp.view(cp.uint8)
        cb = (
            u8[bst].astype(cp.uint32)
            | (u8[bst + 1].astype(cp.uint32) << 8)
            | (u8[bst + 2].astype(cp.uint32) << 16)
            | (u8[bst + 3].astype(cp.uint32) << 24)
        )
        cb_signed = cb.view(cp.int32)
        if not bool(((cb_signed > 0) & (cb_signed < L.blocksize)).all()):
            raise NotImplementedError("raw/RLE blosc blocks (cb<=0 or cb==blocksize) not yet supported")
        cps.append(base + bst.astype(cp.uint64) + 4)
        css.append(cb.astype(cp.uint64))
        dps.append(sb + cp.arange(L.nblocks, dtype=cp.uint64) * bs)
        dbs.append(cp.full(L.nblocks, bs, dtype=cp.uint64))

    comp_ptrs, comp_sizes = cp.concatenate(cps), cp.concatenate(css)
    decomp_ptrs, decomp_buf = cp.concatenate(dps), cp.concatenate(dbs)
    n = int(comp_ptrs.size)
    actual = cp.empty(n, dtype=cp.uint64)
    statuses = cp.empty(n, dtype=cp.int32)
    opts = _ZstdDecompressOpts(0, b"\0" * 60)

    temp_bytes = ctypes.c_size_t(0)
    total = sum(L.nbytes for L in layouts)
    rc = lib.nvcompBatchedZstdDecompressGetTempSizeAsync(n, bs, opts, ctypes.byref(temp_bytes), total)
    if rc != 0:
        raise RuntimeError(f"nvcompBatchedZstdDecompressGetTempSizeAsync rc={rc}")
    d_temp = cp.empty(max(temp_bytes.value, 1), dtype=cp.uint8)
    rc = lib.nvcompBatchedZstdDecompressAsync(
        int(comp_ptrs.data.ptr),
        int(comp_sizes.data.ptr),
        int(decomp_buf.data.ptr),
        int(actual.data.ptr),
        n,
        int(d_temp.data.ptr),
        temp_bytes.value,
        int(decomp_ptrs.data.ptr),
        opts,
        int(statuses.data.ptr),
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"nvcompBatchedZstdDecompressAsync rc={rc}")

    # per-chunk unshuffle -> decoded output
    outs = []
    for scr, L in zip(scratches, layouts, strict=True):
        out = cp.empty(L.nbytes, dtype=cp.uint8)
        if L.shuffle == "bit":
            bitunshuffle_into(scr, out, L.typesize, bs)
        elif L.shuffle == "byte":
            from czarr.kernels.byteshuffle import byteunshuffle

            out = byteunshuffle(scr, L.typesize, bs)[: L.nbytes]
        else:  # none
            out = scr[: L.nbytes].copy()
        outs.append(out)

    if bool((statuses != 0).any().get()):
        raise RuntimeError(f"nvCOMP reported {int((statuses != 0).sum().get())} failed blocks")
    return outs


def decode_blosc_batch(comps: list[cp.ndarray], stream: int = 0) -> list[cp.ndarray]:
    """Decode a batch of blosc(zstd) chunks on the GPU.

    ``comps`` are flat ``uint8`` device buffers of compressed chunk bytes.
    Returns decoded ``uint8`` device buffers (one per input), processing the
    batch in sub-batches of ``_MAX_CHUNKS_PER_CALL`` to bound nvCOMP temp.
    """
    out: list[cp.ndarray] = []
    for i in range(0, len(comps), _MAX_CHUNKS_PER_CALL):
        out.extend(_decode_subbatch(comps[i : i + _MAX_CHUNKS_PER_CALL], stream))
    return out
