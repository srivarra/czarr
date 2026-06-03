"""Bitshuffle inverse — cupy RawKernel (one thread per element).

blosc bitshuffle per block transposes a ``(typesize*8) x elems_per_block``
bit-matrix: each of the ``typesize*8`` bit-planes packs ``elems_per_block``
bits LSB-first into ``elems_per_block/8`` bytes.  The inverse gathers, for
each element ``e``, bit ``e&7`` of byte ``e>>3`` from each plane.

A plain per-element RawKernel (register-resident bit-gather) beat both a
cuTile tile-algebra version (0.23x) and a cccl ``cuda.compute`` transform
(0.72x) in spikes — and unlike cuTile it has no sm_90 compile bug and no
numba/cu12-cu13 toolchain hazard.  So this is the chosen unshuffle path.

Limitation: assumes ``elems_per_block`` is a multiple of 8 (blosc's
non-multiple-of-8 tail handling is not yet reproduced).  Verified for the
common case (e.g. blocksize 32768, typesize 2 -> 16384 elems).
"""

from __future__ import annotations

from functools import cache

import cupy as cp
import numpy as np


@cache
def _make_kernel(typesize: int) -> cp.RawKernel:
    """Build (and cache) a bitunshuffle kernel specialised for one typesize."""
    src = f"""
extern "C" __global__ void bitunshuffle(
    const unsigned char* __restrict__ scratch, unsigned char* __restrict__ out,
    unsigned int blocksize, unsigned int nelems)
{{
    unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nelems) return;
    const unsigned int T = {typesize};
    unsigned int epb = blocksize / T;          // elements per block
    unsigned int bpp = epb >> 3;               // bytes per bit-plane
    const unsigned char* blk = scratch + (size_t)(e / epb) * blocksize;
    unsigned int le = e % epb, eb = le >> 3, ebit = le & 7;
    #pragma unroll
    for (unsigned int bb = 0; bb < T; ++bb) {{
        unsigned int v = 0;
        #pragma unroll
        for (unsigned int bp = 0; bp < 8; ++bp)
            v |= ((blk[(bb * 8 + bp) * bpp + eb] >> ebit) & 1u) << bp;
        out[(size_t)e * T + bb] = (unsigned char)v;
    }}
}}"""
    return cp.RawKernel(src, "bitunshuffle")


def bitunshuffle_into(scratch: cp.ndarray, out: cp.ndarray, typesize: int, blocksize: int) -> None:
    """Invert blosc bitshuffle: block-contiguous ``scratch`` -> element-order ``out``.

    ``scratch`` is the post-zstd (still bit-shuffled) bytes laid out one
    ``blocksize``-byte block after another; ``out`` receives the decoded
    elements in original order.  Both are flat ``uint8`` device arrays.
    """
    epb = blocksize // typesize
    if epb % 8 != 0:
        raise NotImplementedError(f"bitunshuffle: elems_per_block={epb} not a multiple of 8 (tail unsupported)")
    nelems = int(out.size) // typesize
    kernel = _make_kernel(typesize)
    kernel(((nelems + 255) // 256,), (256,), (scratch, out, np.uint32(blocksize), np.uint32(nelems)))
