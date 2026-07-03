"""Derive-once read planning: zarr.json → :class:`DecodePlan` → coalesced byte ranges.

The planning half of ``czarr.lowlevel`` (design:
``.planning/lowlevel-api-design.md``).  Pure host-side — numpy only, no
cupy/CUDA imports — so plans build on login nodes and in host-only tests.

Selection semantics are ported from zarrista/zarrs: integers, step-1
slices, ``Ellipsis``, and tuples of those; negative indices normalized;
``step != 1``, ``None``/newaxis, boolean, and fancy/array indexing raise.
Reads are ndim-preserving: an integer selects a length-1 range and the
axis is retained (tier 1 squeezes axes on top for numpy semantics).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from czarr.lowlevel.coalesce import ByteRange, coalesce_ranges

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# zarr v3 sharding index sentinel: offset == nbytes == 2**64-1 → missing chunk.
_MISSING = 2**64 - 1


class ReadRequest(NamedTuple):
    """One coalesced byte-range read against a store file.

    ``members`` maps the request back to logical decode units:
    ``(chunk_coords, intra_offset, length)`` per member — slice
    ``buffer[intra_offset : intra_offset + length]`` to recover the
    encoded bytes of the chunk at ``chunk_coords``.
    """

    path: Path
    offset: int
    nbytes: int
    members: tuple[tuple[tuple[int, ...], int, int], ...]


@dataclass(frozen=True)
class ShardSpec:
    """Sharding layout parsed from a ``sharding_indexed`` codec config."""

    inner_chunk_shape: tuple[int, ...]
    codecs: tuple[dict[str, Any], ...]
    index_location: str  # "end" | "start"
    index_codecs: tuple[dict[str, Any], ...]

    @property
    def index_has_checksum(self) -> bool:
        """True when the index carries a crc32c trailer (the zarr default)."""
        return any(c.get("name") == "crc32c" for c in self.index_codecs)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class DecodePlan:
    """Derive-once view of one zarr v3 array's metadata.

    Built by :func:`open_plan` from a single ``zarr.json`` read; reusable
    across reads.  ``_shard_indexes`` caches parsed shard indexes keyed by
    shard coordinate — unbounded, one entry per touched shard (~16 bytes
    per inner chunk); drop the plan to drop the cache.
    """

    root: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    chunk_shape: tuple[int, ...]  # outer grid step (the shard shape when sharded)
    codecs: tuple[dict[str, Any], ...]  # raw v3 codec list of the decode unit
    fill_value: Any
    shard: ShardSpec | None
    key_separator: str
    key_prefix_c: bool  # v3 "default" encoding ("c/0/1") vs "v2" ("0.1")
    metadata: dict[str, Any] = field(repr=False, default_factory=dict)
    _shard_indexes: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # derived geometry
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return len(self.shape)

    @property
    def decode_chunk_shape(self) -> tuple[int, ...]:
        """Shape of one logical decode unit (inner chunk when sharded)."""
        return self.shard.inner_chunk_shape if self.shard is not None else self.chunk_shape

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Decode-unit grid: number of units along each dimension."""
        cs = self.decode_chunk_shape
        return tuple(_ceil_div(s, c) for s, c in zip(self.shape, cs, strict=True))

    @property
    def units_per_shard(self) -> tuple[int, ...]:
        """Inner chunks per shard along each dimension (sharded arrays only)."""
        if self.shard is None:
            raise ValueError("not a sharded array")
        return tuple(s // c for s, c in zip(self.chunk_shape, self.shard.inner_chunk_shape, strict=True))

    def chunk_key(self, chunk_coords: Sequence[int]) -> str:
        """Store key for the outer chunk (file) at ``chunk_coords``."""
        sep = self.key_separator
        if self.key_prefix_c:
            return "c" + sep + sep.join(str(c) for c in chunk_coords) if chunk_coords else "c"
        return sep.join(str(c) for c in chunk_coords)

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    def selected_units(self, selection: Any) -> list[tuple[int, ...]]:
        """Decode-unit coordinates touched by ``selection`` (normalized)."""
        bounds = normalize_selection(selection, self.shape)
        cs = self.decode_chunk_shape
        per_axis = [
            range(start // c, _ceil_div(stop, c)) if stop > start else range(0)
            for (start, stop), c in zip(bounds, cs, strict=True)
        ]
        return [tuple(coords) for coords in product(*per_axis)]

    def ranges(
        self,
        selection: Any,
        *,
        max_fused_bytes: int = 64 << 20,
        max_gap_bytes: int = 0,
        fetch: Callable[[Path, int, int], bytes] | None = None,
    ) -> list[ReadRequest]:
        """Selection → coalesced :class:`ReadRequest` list.

        Missing chunks (absent chunk file, or ``2**64-1`` sentinel index
        entries) are silently omitted — the decode/scatter step fills
        them with ``fill_value`` (units selected but present in no
        request's ``members``).

        ``fetch(path, offset, nbytes) -> bytes`` reads uncached shard
        indexes; the default is plain host I/O (indexes are tiny and read
        once per shard per plan).
        """
        units = self.selected_units(selection)
        if not units:
            return []
        if self.shard is None:
            return self._plain_ranges(units)
        return self._sharded_ranges(units, max_fused_bytes=max_fused_bytes, max_gap_bytes=max_gap_bytes, fetch=fetch)

    def _plain_ranges(self, units: list[tuple[int, ...]]) -> list[ReadRequest]:
        """One whole-file request per existing chunk file.

        Small-chunk unsharded stores are the textbook lots-of-small-files
        anti-pattern (one open+read per chunk); prefer sharded stores —
        this path exists for compatibility, not performance.
        """
        out: list[ReadRequest] = []
        for coords in units:
            path = self.root / self.chunk_key(coords)
            try:
                nbytes = path.stat().st_size
            except FileNotFoundError:
                continue  # fill_value chunk
            out.append(ReadRequest(path, 0, nbytes, ((coords, 0, nbytes),)))
        return out

    def _sharded_ranges(
        self,
        units: list[tuple[int, ...]],
        *,
        max_fused_bytes: int,
        max_gap_bytes: int,
        fetch: Callable[[Path, int, int], bytes] | None,
    ) -> list[ReadRequest]:
        per_shard = self.units_per_shard
        # Group selected units by shard coordinate.
        by_shard: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        for coords in units:
            shard_coords = tuple(u // p for u, p in zip(coords, per_shard, strict=True))
            by_shard.setdefault(shard_coords, []).append(coords)

        out: list[ReadRequest] = []
        for shard_coords, shard_units in sorted(by_shard.items()):
            path = self.root / self.chunk_key(shard_coords)
            index = self._shard_index(shard_coords, path, fetch)
            if index is None:
                continue  # missing shard file: every unit is fill_value
            # Per-unit byte ranges from the index; drop missing entries.
            present: list[tuple[tuple[int, ...], int, int]] = []  # (unit, offset, nbytes)
            for coords in shard_units:
                within = tuple(u % p for u, p in zip(coords, per_shard, strict=True))
                offset, nbytes = (int(x) for x in index[within])
                if offset == _MISSING:
                    continue
                present.append((coords, offset, nbytes))
            if not present:
                continue
            fused = coalesce_ranges(
                [ByteRange(offset=o, length=n) for _, o, n in present],
                max_fused_bytes=max_fused_bytes,
                max_gap_bytes=max_gap_bytes,
            )
            for window in fused:
                if window.length == 0:
                    continue
                members = tuple((present[i][0], intra, ln) for i, intra, ln in window.members)
                out.append(ReadRequest(path, window.offset, window.length, members))
        return out

    # ------------------------------------------------------------------
    # shard index
    # ------------------------------------------------------------------

    def _shard_index(
        self,
        shard_coords: tuple[int, ...],
        path: Path,
        fetch: Callable[[Path, int, int], bytes] | None,
    ) -> np.ndarray | None:
        """Cached ``(offset, nbytes)`` index for one shard; None if the file is absent."""
        cached = self._shard_indexes.get(shard_coords)
        if cached is not None:
            return cached
        assert self.shard is not None
        per_shard = self.units_per_shard
        n_entries = 1
        for p in per_shard:
            n_entries *= p
        index_nbytes = n_entries * 16 + (4 if self.shard.index_has_checksum else 0)
        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            return None
        offset = file_size - index_nbytes if self.shard.index_location == "end" else 0
        raw = (fetch or _read_host)(path, offset, index_nbytes)
        # crc32c trailer (if present) is trusted, not verified — the GPU
        # decode path treats checksum validation as an opt-in cost.
        entries = np.frombuffer(raw[: n_entries * 16], dtype="<u8").reshape(*per_shard, 2)
        self._shard_indexes[shard_coords] = entries
        return entries


def _read_host(path: Path, offset: int, nbytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(nbytes)


# ---------------------------------------------------------------------------
# selection normalization (zarrista/zarrs semantics)
# ---------------------------------------------------------------------------


def normalize_selection(selection: Any, shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Normalize a basic-indexing selection to per-axis ``(start, stop)`` bounds.

    ndim-preserving: an integer axis becomes the length-1 range
    ``(i, i+1)``.  Supported: ``int``, step-1 ``slice``, ``Ellipsis``,
    and tuples thereof (fewer entries than ``ndim`` implies full trailing
    axes).  Rejected: ``bool`` (before int coercion — True is not 1),
    ``step != 1`` and ``None``/newaxis (``NotImplementedError``), arrays
    and anything else (``TypeError``); out-of-bounds ints
    (``IndexError``).
    """
    ndim = len(shape)
    items: list[Any] = list(selection) if isinstance(selection, tuple) else [selection]

    # Expand a single Ellipsis; reject multiples.
    n_ellipsis = sum(1 for it in items if it is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("only one Ellipsis allowed in a selection")
    if n_ellipsis == 1:
        at = items.index(Ellipsis)
        pad = ndim - (len(items) - 1)
        if pad < 0:
            raise IndexError(f"too many indices for array with {ndim} dimensions")
        items[at : at + 1] = [slice(None)] * pad
    if len(items) > ndim:
        raise IndexError(f"too many indices for array with {ndim} dimensions")
    items += [slice(None)] * (ndim - len(items))

    bounds: list[tuple[int, int]] = []
    for axis, (it, dim) in enumerate(zip(items, shape, strict=True)):
        if isinstance(it, bool | np.bool_):
            raise TypeError(f"axis {axis}: boolean indexing is not supported")
        if isinstance(it, int | np.integer):
            i = int(it)
            if i < 0:
                i += dim
            if not 0 <= i < dim:
                raise IndexError(f"axis {axis}: index {int(it)} out of bounds for size {dim}")
            bounds.append((i, i + 1))
        elif isinstance(it, slice):
            if it.step not in (None, 1):
                raise NotImplementedError(f"axis {axis}: step != 1 is not supported")
            start, stop, _ = it.indices(dim)
            bounds.append((start, max(start, stop)))
        elif it is None:
            raise NotImplementedError(f"axis {axis}: None/newaxis is not supported")
        elif isinstance(it, EllipsisType):  # a second Ellipsis snuck through padding
            raise IndexError("only one Ellipsis allowed in a selection")
        else:
            raise TypeError(f"axis {axis}: unsupported selection type {type(it).__name__} (basic indexing only)")
    return tuple(bounds)


# ---------------------------------------------------------------------------
# metadata parse
# ---------------------------------------------------------------------------


def open_plan(root: str | Path) -> DecodePlan:
    """Parse ``{root}/zarr.json`` into a reusable :class:`DecodePlan`.

    One metadata read; no other I/O.  Zarr v3 arrays only.
    """
    root = Path(root)
    metadata = json.loads((root / "zarr.json").read_bytes())
    return plan_from_metadata(metadata, root)


def plan_from_metadata(metadata: dict[str, Any], root: str | Path) -> DecodePlan:
    """Build a :class:`DecodePlan` from already-parsed v3 metadata (no I/O)."""
    root = Path(root)
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
        raise ValueError(f"{root}: expected a zarr v3 array (zarr_format=3, node_type=array)")

    shape = tuple(int(s) for s in metadata["shape"])
    dtype = _parse_dtype(metadata["data_type"])

    grid = metadata["chunk_grid"]
    if grid.get("name") != "regular":
        raise NotImplementedError(f"chunk_grid {grid.get('name')!r} not supported (regular only)")
    chunk_shape = tuple(int(c) for c in grid["configuration"]["chunk_shape"])

    key_enc = metadata.get("chunk_key_encoding", {"name": "default"})
    key_name = key_enc.get("name", "default")
    if key_name not in ("default", "v2"):
        raise NotImplementedError(f"chunk_key_encoding {key_name!r} not supported")
    key_prefix_c = key_name == "default"
    key_separator = key_enc.get("configuration", {}).get("separator", "/" if key_prefix_c else ".")

    codecs = tuple(metadata.get("codecs", ()))
    shard: ShardSpec | None = None
    if codecs and codecs[0].get("name") == "sharding_indexed":
        cfg = codecs[0]["configuration"]
        shard = ShardSpec(
            inner_chunk_shape=tuple(int(c) for c in cfg["chunk_shape"]),
            codecs=tuple(cfg.get("codecs", ())),
            index_location=cfg.get("index_location", "end"),
            index_codecs=tuple(cfg.get("index_codecs", ())),
        )
        for s, c in zip(chunk_shape, shard.inner_chunk_shape, strict=True):
            if s % c != 0:
                raise ValueError(
                    f"shard shape {chunk_shape} not divisible by inner chunk shape {shard.inner_chunk_shape}"
                )
        decode_codecs = shard.codecs
    else:
        decode_codecs = codecs

    return DecodePlan(
        root=root,
        shape=shape,
        dtype=dtype,
        chunk_shape=chunk_shape,
        codecs=decode_codecs,
        fill_value=metadata.get("fill_value", 0),
        shard=shard,
        key_separator=key_separator,
        key_prefix_c=key_prefix_c,
        metadata=metadata,
    )


def _parse_dtype(data_type: Any) -> np.dtype:
    """Zarr v3 ``data_type`` → numpy dtype (fixed-width numerics only)."""
    name = data_type.get("name") if isinstance(data_type, dict) else data_type
    try:
        return np.dtype(name)
    except TypeError:
        raise NotImplementedError(f"data_type {name!r} not supported (fixed-width numerics only)") from None
