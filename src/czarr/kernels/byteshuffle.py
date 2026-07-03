"""Byteshuffle forward/inverse — pure cupy reshape/transpose.

Byteshuffle layout per block: ``typesize`` byte-planes of length
``blocksize / typesize``, concatenated; the inverse interleaves them
back.  Both directions are a ``(typesize, nelem)`` uint8 transpose.

Consumed by the :class:`czarr.codecs.filters.shuffle.Shuffle` filter and
the blosc container decoder (``shuffle=1`` chunks).  A cuTile transpose
variant (~2.5x more bandwidth on A40) was deleted — cuda-tile 1.3 fails
to compile on sm_90 and shuffle is not a read-path bottleneck; re-add as
a cupy RawKernel if a filter sweep ever shows it mattering.
"""

import cupy as cp


def byteunshuffle(packed: cp.ndarray, typesize: int, blocksize: int) -> cp.ndarray:
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
    # Each block has typesize planes of nelem bytes; interleave them.
    return cp.ascontiguousarray(packed.reshape(nblocks, typesize, nelem).transpose(0, 2, 1)).ravel()


def byteshuffle(raw: cp.ndarray, typesize: int, blocksize: int) -> cp.ndarray:
    """Forward byteshuffle — inverse of :func:`byteunshuffle`."""
    if raw.size % blocksize != 0:
        raise ValueError(f"byteshuffle: input {raw.size} not a multiple of blocksize {blocksize}")
    nblocks = raw.size // blocksize
    nelem = blocksize // typesize
    return cp.ascontiguousarray(raw.reshape(nblocks, nelem, typesize).transpose(0, 2, 1)).ravel()
