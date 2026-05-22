"""Round-robin pool of ``cuda.core.Stream`` instances.

Phase 2's :class:`CzarrPipeline` assigns per-chunk decode work to N
concurrent streams.  Round-robin keeps the distribution even without
needing per-stream load tracking; for N small relative to chunks-per-
selection the imbalance is negligible.

Default size is 4 streams.  Tuneable via :func:`czarr.configure_gpu`
(``stream_pool_size=N``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cuda.core import Device, Stream, StreamOptions

if TYPE_CHECKING:
    from collections.abc import Iterator


class StreamPool:
    """N round-robin-assigned non-blocking CUDA streams on one device.

    Parameters
    ----------
    size:
        Number of streams.  Default 4.  Higher = more H2D / decompress
        overlap but more per-stream nvCOMP scratch memory.
    device_id:
        CUDA device ordinal.  ``None`` means current device.
    nonblocking:
        If True (default), streams do not synchronise with the legacy
        default stream — necessary for actual concurrency.
    priority:
        Optional stream priority (lower = higher).  ``None`` = device default.

    Examples
    --------
    >>> pool = StreamPool(size=4)
    >>> stream = pool.acquire()
    >>> # ... launch work on `stream` ...
    >>> pool.sync_all()
    """

    def __init__(
        self,
        size: int = 4,
        *,
        device_id: int | None = None,
        nonblocking: bool = True,
        priority: int | None = None,
    ) -> None:
        if size < 1:
            raise ValueError(f"StreamPool size must be >= 1, got {size}")
        device = Device(device_id) if device_id is not None else Device()
        device.set_current()
        opts = StreamOptions(nonblocking=nonblocking, priority=priority)
        self._device = device
        self._streams: list[Stream] = [device.create_stream(options=opts) for _ in range(size)]
        self._cursor = 0

    @property
    def size(self) -> int:
        """Number of streams in the pool."""
        return len(self._streams)

    @property
    def device(self) -> Device:
        """The CUDA device the pool's streams belong to."""
        return self._device

    def acquire(self) -> Stream:
        """Return the next stream from the round-robin rotation."""
        s = self._streams[self._cursor % len(self._streams)]
        self._cursor += 1
        return s

    def __iter__(self) -> Iterator[Stream]:
        return iter(self._streams)

    def __len__(self) -> int:
        return len(self._streams)

    def sync_all(self) -> None:
        """Block the host until every stream in the pool is idle."""
        for s in self._streams:
            s.sync()

    def close(self) -> None:
        """Destroy all streams.  After this the pool is unusable."""
        for s in self._streams:
            s.close()
        self._streams = []
