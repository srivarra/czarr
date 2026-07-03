"""``czarr.core.Array`` — explicit, zarrista-shaped handle over the lowlevel stages.

The tier-2 object API from the two-tier design: ``Array.open`` parses
metadata once (a :class:`~czarr.lowlevel.plan.DecodePlan` underneath),
metadata is exposed as plain properties, and the ``retrieve_*`` trio
returns ``cupy.ndarray`` directly (czarr decodes fixed-width numerics
only — no DecodedArray union needed).

Selection semantics are lowlevel's: basic indexing only, ndim-preserving
(an int keeps a length-1 axis).  Tier 1 (``CudaZarrArray``) squeezes
axes on top for numpy semantics.

Sync and async duals share one implementation: :class:`AsyncArray`
methods run the sync path in a worker thread (``asyncio.to_thread``),
composing with zarr-python's event loop without new I/O machinery.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self, TypedDict, Unpack

import cupy as cp
import numpy as np

from czarr.lowlevel.plan import DecodePlan, open_plan, plan_from_metadata


class ReadOptions(TypedDict, total=False):
    """Per-call read knobs (zarrista's ``CodecOptions`` pattern — no global state).

    ``max_workers`` — read threadpool size (default: NIC-bound heuristic).
    ``max_fused_bytes`` / ``max_gap_bytes`` — range-coalescing caps.
    ``stream`` — CUDA stream handle; ``None`` = cupy's current stream.
    ``out`` — pre-allocated output array (exact shape + dtype required).
    """

    max_workers: int | None
    max_fused_bytes: int
    max_gap_bytes: int
    stream: int | None
    out: cp.ndarray | None


class Array:
    """A zarr v3 array opened for explicit GPU reads."""

    def __init__(self, plan: DecodePlan) -> None:
        self._plan = plan

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, root: str | Path) -> Self:
        """Open the array at ``root`` — one ``zarr.json`` read, no other I/O."""
        return cls(open_plan(root))

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], root: str | Path) -> Self:
        """Open with caller-supplied metadata (no I/O at all)."""
        return cls(plan_from_metadata(metadata, root))

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    @property
    def plan(self) -> DecodePlan:
        """The underlying :class:`DecodePlan` (shared shard-index cache)."""
        return self._plan

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape in elements."""
        return self._plan.shape

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype."""
        return self._plan.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._plan.ndim

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """Shape of one decode unit (the inner chunk when sharded)."""
        return self._plan.decode_chunk_shape

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Decode-unit grid: number of chunks along each dimension."""
        return self._plan.grid_shape

    @property
    def metadata(self) -> dict[str, Any]:
        """The raw parsed ``zarr.json`` — parse, don't hide."""
        return self._plan.metadata

    @property
    def attrs(self) -> dict[str, Any]:
        """User attributes from the metadata."""
        return self._plan.metadata.get("attributes", {})

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def retrieve_array_subset(self, selection: Any, **opts: Unpack[ReadOptions]) -> cp.ndarray:
        """Read + decode a basic-indexing selection (ndim-preserving)."""
        from czarr import lowlevel

        return lowlevel.read_array(None, selection, plan=self._plan, **opts)

    def retrieve_chunk(self, chunk_coords: Sequence[int], **opts: Unpack[ReadOptions]) -> cp.ndarray:
        """Read + decode the chunk at grid coordinates ``chunk_coords``.

        Missing chunks come back filled with ``fill_value``.  Edge chunks
        are clipped to the array bounds (delta from zarrs, which returns
        the fill-padded full chunk).
        """
        return self.retrieve_array_subset(self._chunk_box(chunk_coords), **opts)

    def retrieve_encoded_chunk(self, chunk_coords: Sequence[int]) -> cp.ndarray | None:
        """Raw pre-decode device bytes of one chunk, or ``None`` if missing.

        The escape hatch for decode-only benching and custom pipelines.
        """
        from czarr import lowlevel

        requests = self._plan.ranges(self._chunk_box(chunk_coords))
        if not requests:
            return None
        (buffer,) = lowlevel.read(requests)
        ((_, intra, length),) = requests[0].members
        return buffer[intra : intra + length]

    def __getitem__(self, selection: Any) -> cp.ndarray:
        """Sugar for :meth:`retrieve_array_subset`."""
        return self.retrieve_array_subset(selection)

    def _chunk_box(self, chunk_coords: Sequence[int]) -> tuple[slice, ...]:
        grid = self._plan.grid_shape
        for axis, (c, g) in enumerate(zip(chunk_coords, grid, strict=True)):
            if not 0 <= c < g:
                raise IndexError(f"axis {axis}: chunk index {c} out of bounds for grid {grid}")
        cs = self._plan.decode_chunk_shape
        return tuple(
            slice(c * s, min((c + 1) * s, dim)) for c, s, dim in zip(chunk_coords, cs, self._plan.shape, strict=True)
        )

    def __repr__(self) -> str:
        kind = "sharded" if self._plan.shard is not None else "plain"
        return f"<czarr.core.Array {self.shape} {self.dtype} chunks={self.chunk_shape} {kind} at {self._plan.root}>"


class AsyncArray:
    """Async dual of :class:`Array` — same reads, awaitable.

    Each call runs the sync read path in a worker thread; cuFile reads
    release the GIL, so concurrent awaits genuinely overlap I/O.
    """

    def __init__(self, plan: DecodePlan) -> None:
        self._array = Array(plan)

    @classmethod
    def open(cls, root: str | Path) -> Self:
        """Open the array at ``root`` (metadata read happens synchronously)."""
        return cls(open_plan(root))

    @property
    def array(self) -> Array:
        """The sync twin (shared plan + shard-index cache)."""
        return self._array

    def __getattr__(self, name: str) -> Any:  # shape/dtype/... delegate
        return getattr(self._array, name)

    async def retrieve_array_subset(self, selection: Any, **opts: Unpack[ReadOptions]) -> cp.ndarray:
        """Awaitable :meth:`Array.retrieve_array_subset`."""
        return await asyncio.to_thread(self._array.retrieve_array_subset, selection, **opts)

    async def retrieve_chunk(self, chunk_coords: Sequence[int], **opts: Unpack[ReadOptions]) -> cp.ndarray:
        """Awaitable :meth:`Array.retrieve_chunk`."""
        return await asyncio.to_thread(self._array.retrieve_chunk, chunk_coords, **opts)

    async def retrieve_encoded_chunk(self, chunk_coords: Sequence[int]) -> cp.ndarray | None:
        """Awaitable :meth:`Array.retrieve_encoded_chunk`."""
        return await asyncio.to_thread(self._array.retrieve_encoded_chunk, chunk_coords)
