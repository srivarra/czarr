"""Device-memory allocator wiring for czarr.

czarr keeps allocator concerns in one place so the runtime fans out to a
single pool: any code path that ends up calling ``cupy.cuda.alloc`` (most
device allocations in the project) honours whatever allocator is registered
with cupy.

Two functions are exposed:

* :func:`register_nvcomp_allocator` — routes nvCOMP's internal scratch +
  output allocations through cupy's allocator.  Auto-called on first codec
  instantiation; users can call it again with a custom allocator if needed.
* :func:`use_rmm_pool` — opt-in convenience helper that initialises an RMM
  pool and points cupy at it.  After this call, *every* device allocation
  that czarr makes (codec scratch via nvCOMP, cuFile destination buffers
  via cupy) is satisfied from a single RMM pool.

The default behaviour without :func:`use_rmm_pool` is still pool-backed —
cupy ships its own ``MemoryPool`` — so the helper is for users who want to
share a pool across czarr, cuDF, cuML, kvikIO, etc.
"""

import threading

import cupy as cp
import rmm
from nvidia import nvcomp
from rmm.allocators.cupy import rmm_cupy_allocator

__all__ = ["register_nvcomp_allocator", "use_rmm_pool"]


class _CupyAllocBox:
    """Hold a cupy memory pointer with a stable ``ptr`` attribute.

    nvCOMP's :func:`set_device_allocator` expects the allocator callable to
    return an object with a ``ptr`` (integer) attribute; the returned
    object's lifetime owns the underlying buffer (it is freed on garbage
    collection of this box).
    """

    __slots__ = ("_mem", "ptr")

    def __init__(self, nbytes: int) -> None:
        self._mem = cp.cuda.alloc(nbytes)
        self.ptr = int(self._mem.ptr)


def _cupy_alloc(nbytes: int, stream) -> _CupyAllocBox:
    # The ``stream`` argument is honoured implicitly — cupy's allocator (or
    # RMM, if it has been routed through cupy) reads the current stream from
    # cupy's stack.  Passing it explicitly here is an nvCOMP-side concern we
    # don't need to forward to cupy.
    return _CupyAllocBox(nbytes)


_lock = threading.Lock()
_registered = False


def register_nvcomp_allocator(allocator=None) -> None:
    """Hook nvCOMP into cupy's allocator (or a user-supplied callable).

    Idempotent for the default allocator; passing a custom callable replaces
    whatever was registered before.  nvCOMP scratch + output buffers will
    then come from the same pool as the rest of czarr's device allocations.
    """
    global _registered
    if allocator is None:
        if _registered:
            return
        with _lock:
            if _registered:
                return
            nvcomp.set_device_allocator(_cupy_alloc)
            _registered = True
        return
    nvcomp.set_device_allocator(allocator)
    _registered = True


def use_rmm_pool(
    initial_size: int = 1 << 30,
    maximum_size: int | None = None,
) -> None:
    """Initialise an RMM pool and route all czarr device allocations through it.

    After this call, cupy (and therefore the nvCOMP allocator hook installed
    by :func:`register_nvcomp_allocator`) draws from an RMM
    ``PoolMemoryResource``.  This is the right move when czarr is composed
    with other RAPIDS libraries (cuDF, cuML, kvikIO) so they all share one
    pool instead of fragmenting GPU memory across multiple allocators.

    Parameters
    ----------
    initial_size:
        Bytes to pre-allocate up front.  Default 1 GiB.
    maximum_size:
        Maximum pool size in bytes; ``None`` lets the pool grow without bound
        (capped only by available device memory).
    """
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=initial_size,
        maximum_pool_size=maximum_size,
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    register_nvcomp_allocator()
