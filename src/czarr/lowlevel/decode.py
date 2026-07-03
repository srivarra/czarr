"""Batched GPU decode + scatter for planned reads.

The decode half of ``czarr.lowlevel``: slice per-chunk encoded views out
of fused read buffers, decompress the whole batch in one nvCOMP call
(or the native blosc path), undo filters, and scatter decoded chunks
into the output selection.

Codec support (v1) mirrors the bench targets, not all of zarr:

* compressors — ``zstd`` (nvCOMP batched, RAW bitstream), ``blosc``
  (native ctypes batched path, 15x win), or none
* ``bytes`` serializer (little-endian)
* ``shuffle`` filter (byteshuffle)
* ``crc32c`` — tolerated and skipped (checksum validation is an opt-in
  cost the GPU path does not pay by default)

Anything else raises ``NotImplementedError`` — use tier 1 (zarr-python
+ czarr codecs) for exotic chains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
from nvidia import nvcomp

from czarr._nvtx import nvtx_range
from czarr.alloc import register_nvcomp_allocator
from czarr.kernels.byteshuffle import byteunshuffle

if TYPE_CHECKING:
    from czarr.lowlevel.plan import DecodePlan, ReadRequest

# Codec(algorithm=...) cache — nvCOMP codec construction has real cost
# and the instances are reusable per thread; lowlevel.decode is
# sync-single-caller so a plain module dict suffices.
_CODECS: dict[str, nvcomp.Codec] = {}


def _nvcomp_codec(algorithm: str) -> nvcomp.Codec:
    codec = _CODECS.get(algorithm)
    if codec is None:
        register_nvcomp_allocator()
        codec = nvcomp.Codec(algorithm=algorithm, bitstream_kind=nvcomp.BitstreamKind.RAW)
        _CODECS[algorithm] = codec
    return codec


class _Chain:
    """The decode recipe distilled from a raw v3 codec list."""

    __slots__ = ("compressor", "shuffle_elementsize", "crc32c_trailer")

    def __init__(self, codecs: tuple[dict[str, Any], ...], dtype: np.dtype) -> None:
        self.compressor: str | None = None
        self.shuffle_elementsize: int | None = None
        self.crc32c_trailer = False
        for c in codecs:
            name = c.get("name")
            cfg = c.get("configuration", {})
            if name == "bytes":
                if cfg.get("endian", "little") != "little":
                    raise NotImplementedError("big-endian 'bytes' codec not supported")
            elif name == "zstd":
                self.compressor = "Zstd"
            elif name == "blosc":
                self.compressor = "blosc"
            elif name == "shuffle":
                self.shuffle_elementsize = int(cfg.get("elementsize", dtype.itemsize))
            elif name == "crc32c":
                self.crc32c_trailer = True
            else:
                raise NotImplementedError(
                    f"codec {name!r} not supported by lowlevel.decode — use tier 1 (zarr + czarr codecs)"
                )


def decode(
    plan: DecodePlan,
    requests: list[ReadRequest],
    buffers: list[cp.ndarray],
    selection: Any,
    *,
    out: cp.ndarray | None = None,
    stream: int | None = None,
) -> cp.ndarray:
    """Decode read buffers and scatter into the selected region.

    ``requests``/``buffers`` are the outputs of :meth:`DecodePlan.ranges`
    and :func:`czarr.lowlevel.io.read` for the same ``selection``.
    Selected units present in no request (missing chunks) are filled
    with ``plan.fill_value``.  Returns a ``cupy.ndarray`` of the
    ndim-preserving selection shape (an ``out`` array of that exact
    shape and dtype may be supplied to reuse memory).

    ``stream``: ``None`` uses cupy's current stream (DALI convention);
    the result is synchronized before return.
    """
    from czarr.lowlevel.plan import normalize_selection

    bounds = normalize_selection(selection, plan.shape)
    out_shape = tuple(stop - start for start, stop in bounds)
    if out is not None:
        if out.shape != out_shape or out.dtype != plan.dtype:
            raise ValueError(f"out must be shape {out_shape} dtype {plan.dtype}, got {out.shape} {out.dtype}")
    ctx = cp.cuda.ExternalStream(stream) if stream is not None else cp.cuda.get_current_stream()
    with ctx, nvtx_range("czarr.lowlevel.decode", n=sum(len(r.members) for r in requests)):
        result = _decode_sync(plan, requests, buffers, bounds, out)
    ctx.synchronize()
    return result


def _decode_sync(
    plan: DecodePlan,
    requests: list[ReadRequest],
    buffers: list[cp.ndarray],
    bounds: tuple[tuple[int, int], ...],
    out: cp.ndarray | None,
) -> cp.ndarray:
    chain = _Chain(plan.codecs, plan.dtype)
    chunk_shape = plan.decode_chunk_shape
    chunk_nbytes = int(np.prod(chunk_shape)) * plan.dtype.itemsize

    # Per-unit encoded views out of the fused windows.
    units: list[tuple[int, ...]] = []
    encoded: list[cp.ndarray] = []
    for request, buffer in zip(requests, buffers, strict=True):
        for coords, intra, length in request.members:
            payload_len = length - 4 if chain.crc32c_trailer else length
            units.append(coords)
            encoded.append(buffer[intra : intra + payload_len])

    # Batched decompress → flat uint8 device buffers of chunk_nbytes each.
    if not encoded:
        decoded: list[cp.ndarray] = []
    elif chain.compressor == "blosc":
        from czarr.lowlevel.blosc import decode_blosc_batch

        with nvtx_range("czarr.lowlevel.decode.blosc", n=len(encoded)):
            decoded = decode_blosc_batch(encoded, int(cp.cuda.get_current_stream().ptr))
    elif chain.compressor is not None:
        codec = _nvcomp_codec(chain.compressor)
        outs = [cp.empty(chunk_nbytes, dtype=cp.uint8) for _ in encoded]
        with nvtx_range("czarr.lowlevel.decode.nvcomp", n=len(encoded), algo=chain.compressor):
            codec.decode([nvcomp.as_array(e) for e in encoded], out=outs)
        decoded = outs
    else:
        decoded = list(encoded)

    if chain.shuffle_elementsize is not None:
        with nvtx_range("czarr.lowlevel.decode.unshuffle", n=len(decoded)):
            decoded = [byteunshuffle(d, chain.shuffle_elementsize, int(d.size)) for d in decoded]

    # Scatter into the output selection (ndim-preserving shape).
    out_shape = tuple(stop - start for start, stop in bounds)
    n_selected = len(plan.selected_units(tuple(slice(a, b) for a, b in bounds)))
    if out is None:
        if len(units) < n_selected:
            out = cp.full(out_shape, plan.fill_value, dtype=plan.dtype)
        else:
            out = cp.empty(out_shape, dtype=plan.dtype)
    elif len(units) < n_selected:
        out.fill(plan.fill_value)

    with nvtx_range("czarr.lowlevel.decode.scatter", n=len(units)):
        for coords, flat in zip(units, decoded, strict=True):
            if flat.size != chunk_nbytes:
                raise ValueError(f"chunk {coords}: decoded {flat.size} bytes, expected {chunk_nbytes}")
            chunk = flat.view(plan.dtype).reshape(chunk_shape)
            src, dst = _intersect(coords, chunk_shape, bounds)
            out[dst] = chunk[src]
    return out


def _intersect(
    coords: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    bounds: tuple[tuple[int, int], ...],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Chunk-box ∩ selection → (source slices in chunk, dest slices in out)."""
    src: list[slice] = []
    dst: list[slice] = []
    for c, size, (start, stop) in zip(coords, chunk_shape, bounds, strict=True):
        lo = c * size
        a = max(lo, start)
        b = min(lo + size, stop)
        src.append(slice(a - lo, b - lo))
        dst.append(slice(a - start, b - start))
    return tuple(src), tuple(dst)


def read_array(
    root: Any,
    selection: Any = ...,
    *,
    plan: DecodePlan | None = None,
    max_workers: int | None = None,
    max_fused_bytes: int = 64 << 20,
    max_gap_bytes: int = 0,
    out: cp.ndarray | None = None,
    stream: int | None = None,
) -> cp.ndarray:
    """One-liner composing plan → ranges → read → decode.

    Pass ``plan=`` to reuse a :class:`DecodePlan` (and its shard-index
    cache) across calls; ``root`` is ignored in that case.
    """
    from czarr.lowlevel.io import read
    from czarr.lowlevel.plan import open_plan

    if plan is None:
        plan = open_plan(root)
    requests = plan.ranges(selection, max_fused_bytes=max_fused_bytes, max_gap_bytes=max_gap_bytes)
    buffers = read(requests, max_workers=max_workers)
    return decode(plan, requests, buffers, selection, out=out, stream=stream)
