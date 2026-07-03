"""czarr.core — explicit object API over the lowlevel read stages.

:class:`Array` / :class:`AsyncArray` are the zarrista-shaped handles
(design: ``.planning/lowlevel-api-design.md``); the buffer prototype
classes live in :mod:`czarr.core.buffer` (import directly — they pull
cupy/cuda.core, this facade stays host-importable).
"""

from czarr.core.array import Array, AsyncArray, ReadOptions

__all__ = ["Array", "AsyncArray", "ReadOptions"]
