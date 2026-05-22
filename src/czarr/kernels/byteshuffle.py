"""Byteshuffle inverse — cuTile kernel (one grid-block per shuffle block).

Byteshuffle layout per block: ``typesize`` byte-planes of length
``blocksize / typesize``, concatenated.  Inverse interleaves them back.

i.e. literally a transpose ``(typesize, nelem) -> (nelem, typesize)``.
cuTile's ``ct.transpose`` over a uint8 tile hits near-HBM bandwidth on
A40 (~327 GiB/s observed @ typesize=2, nblocks=81), ~2.5x faster than
the prior cupy ``reshape/transpose/ascontiguousarray`` chain.

Consumed both by the standalone :class:`czarr.codecs.filters.shuffle.Shuffle`
filter (Phase 3) and — when re-added later — by a Blosc1 container
parser for stores written with shuffle=1.
"""

from __future__ import annotations

from functools import cache

import cuda.tile as ct
import cupy as cp


@cache
def _make_byteunshuffle_kernel(typesize: int, nelem: int):
    """Build (and cache) a cuTile kernel specialised for one (typesize, nelem).

    cuTile requires ``shape=`` tuples in :func:`ct.load` to be compile-time
    constants — passing ``ct.Constant[int]`` params through doesn't satisfy
    the constraint in cuda-tile 1.3, so we capture both values in a
    closure and let ``@cache`` keep one specialised kernel per shape.
    """
    TS = typesize
    NE = nelem

    @ct.kernel
    def kernel(src, dst):
        bid = ct.bid(0)
        planes = ct.load(src, index=(bid, 0), shape=(TS, NE))
        ct.store(dst, index=(bid, 0), tile=ct.transpose(planes))

    return kernel


def byteunshuffle_batched(packed: cp.ndarray, typesize: int, blocksize: int) -> cp.ndarray:
    """Invert byteshuffle for a contiguous run of full-size blocks.

    Parameters
    ----------
    packed : cp.ndarray
        Flat ``uint8`` device buffer of ``nblocks * blocksize`` bytes.
    typesize : int
        Element size in bytes (e.g. ``2`` for ``float16``).
    blocksize : int
        Bytes per shuffle block.  Must be a multiple of ``typesize``.
    """
    if packed.size % blocksize != 0:
        raise ValueError(f"byteunshuffle: input {packed.size} not a multiple of blocksize {blocksize}")
    nblocks = packed.size // blocksize
    nelem = blocksize // typesize

    src2d = packed.view(cp.uint8).reshape(nblocks * typesize, nelem)
    out = cp.empty(packed.size, dtype=cp.uint8)
    dst2d = out.reshape(nblocks * nelem, typesize)
    kernel = _make_byteunshuffle_kernel(typesize, nelem)
    ct.launch(cp.cuda.get_current_stream(), (nblocks, 1, 1), kernel, (src2d, dst2d))
    return out


@cache
def _make_byteshuffle_kernel(typesize: int, nelem: int):
    """Forward byteshuffle: ``(nelem, typesize) -> (typesize, nelem)``."""
    TS = typesize
    NE = nelem

    @ct.kernel
    def kernel(src, dst):
        bid = ct.bid(0)
        elems = ct.load(src, index=(bid, 0), shape=(NE, TS))
        ct.store(dst, index=(bid, 0), tile=ct.transpose(elems))

    return kernel


def byteshuffle_batched(raw: cp.ndarray, typesize: int, blocksize: int) -> cp.ndarray:
    """Forward byteshuffle for a contiguous run of full-size blocks."""
    if raw.size % blocksize != 0:
        raise ValueError(f"byteshuffle: input {raw.size} not a multiple of blocksize {blocksize}")
    nblocks = raw.size // blocksize
    nelem = blocksize // typesize

    src2d = raw.view(cp.uint8).reshape(nblocks * nelem, typesize)
    out = cp.empty(raw.size, dtype=cp.uint8)
    dst2d = out.reshape(nblocks * typesize, nelem)
    kernel = _make_byteshuffle_kernel(typesize, nelem)
    ct.launch(cp.cuda.get_current_stream(), (nblocks, 1, 1), kernel, (src2d, dst2d))
    return out
