"""Blosc — GPU decode of blosc(zstd + shuffle) chunks via nvCOMP native API.

Shadows zarr's CPU ``BloscCodec`` (codec id ``"blosc"``) when opted in via
:func:`czarr.configure_gpu`, so existing blosc-compressed OME-Zarr stores
decode on the GPU.  Decode-only: the on-GPU path is the
fanout + native batched zstd + unshuffle pipeline in
:mod:`czarr.lowlevel.blosc` (10-15x over CPU blosc + H2D).

Encoding raises — to produce GPU-decodable output, write ``[Shuffle, Zstd]``
large-chunk instead (matches blosc's ratio within ~2%, no blosc container).
"""

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Self, override

import cupy as cp
import numpy as np
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer
from zarr.core.common import JSON

from czarr.codecs._nvcomp_buffer import device_to_buffer
from czarr.codecs.base import CudaBytesBytesCodec, _Algorithm, _BitstreamKind, codec_config
from czarr.core.buffer import is_gpu_buffer


@dataclass(frozen=True)
class Blosc(CudaBytesBytesCodec):
    """GPU decoder for blosc(zstd)-compressed chunks (decode-only).

    Config mirrors zarr's v3 ``BloscCodec`` for metadata round-trip; the
    authoritative per-chunk layout (blocksize, block offsets, shuffle mode)
    is parsed from each chunk's blosc header at decode time.
    """

    codec_name: ClassVar[str] = "blosc"
    # blosc's sub-codec is zstd; these document that (the decode path uses the
    # native batched API directly, not the base ``_create_codec`` machinery).
    _algorithm: ClassVar[_Algorithm] = _Algorithm.ZSTD
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.RAW

    cname: str = "zstd"
    clevel: int = 5
    shuffle: Any = "shuffle"
    typesize: int = 4
    blocksize: int = 0

    def _decode_sync(self, items: list[tuple[Buffer | None, ArraySpec]]) -> list[Buffer | None]:
        from czarr.lowlevel.blosc import decode_blosc_batch

        out: list[Buffer | None] = [None] * len(items)
        idx, comps, specs = [], [], []
        for i, (chunk, spec) in enumerate(items):
            if chunk is None:
                continue
            if is_gpu_buffer(chunk):
                comp = cp.asarray(chunk.as_array_like()).view(cp.uint8)
            else:
                comp = cp.asarray(np.frombuffer(chunk.to_bytes(), dtype=np.uint8))
            idx.append(i)
            comps.append(comp)
            specs.append(spec)
        if not comps:
            return out
        stream = self._resolve_stream(self.cuda_stream) or 0
        decoded = decode_blosc_batch(comps, stream)
        for i, dev, spec in zip(idx, decoded, specs, strict=True):
            out[i] = device_to_buffer(dev, spec.prototype)
        return out

    @override
    async def decode(self, chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]]) -> Iterable[Buffer | None]:
        """Decode a batch of blosc chunks via one native batched nvCOMP call."""
        items = list(chunks_and_specs)
        return await asyncio.to_thread(self._decode_sync, items)

    @override
    async def encode(self, chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]]) -> Iterable[Buffer | None]:
        """Not supported — write ``[Shuffle, Zstd]`` for GPU-decodable output."""
        raise NotImplementedError("czarr Blosc is decode-only; write [Shuffle, Zstd] large-chunk instead")

    @override
    def to_dict(self) -> dict[str, JSON]:
        """Emit the zarr-v3 BloscCodec schema for round-trip with non-czarr readers."""
        return {
            "name": self.codec_name,
            "configuration": {
                "cname": self.cname,
                "clevel": int(self.clevel),
                "shuffle": self.shuffle,
                "typesize": int(self.typesize),
                "blocksize": int(self.blocksize),
            },
        }

    @classmethod
    @override
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Build from a zarr-v3 BloscCodec metadata dict (tolerant of missing keys)."""
        cfg = codec_config(data)
        return cls(
            cname=str(cfg.get("cname", "zstd")),
            clevel=int(cfg.get("clevel", 5)),
            shuffle=cfg.get("shuffle", "shuffle"),
            typesize=int(cfg.get("typesize", 4)),
            blocksize=int(cfg.get("blocksize", 0)),
        )
