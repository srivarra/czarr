"""Shuffle — GPU byteshuffle BytesBytesCodec.

numcodecs-compatible byteshuffle filter that operates on the encoded
byte buffer.  Each chunk is treated as one big shuffle block of size
``chunk_byte_length`` with element width ``elementsize``; the codec
swaps adjacent ``elementsize`` byte rows so each plane is contiguous.

Two backends:

* ``backend="cupy"`` (default) — cupy ``reshape/transpose/ascontiguousarray``.
  Works on every supported GPU including H100/H200.  This is the
  safe default — cuda-tile 1.3 fails to compile on sm_90 (``tileiras:
  Cannot find option named 'sm_90'``), so the cuTile path crashes
  on Hopper for any shape.
* ``backend="cutile"`` — :mod:`cuda.tile` transpose; ~2.5x faster than
  cupy on A40 (~327 GiB/s @ typesize=2, validated by the Phase 2
  filter bench sweep).  Opt-in for sm_<90 users who need the
  bandwidth; broken on H100/H200 until a cuTile release ships with
  sm_90 support.

cuda.compute has no transpose primitive, so the byte-plane transpose
stays out of the cccl family for now (would need a custom Raw/Program
kernel).  Both backends produce the same on-disk bitstream; the
``backend`` field is runtime — not persisted in Zarr v3 metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
from zarr.abc.codec import BytesBytesCodec

from czarr.codecs._backend import CodecBackend, resolve_backend_for_filter

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON


@dataclass(frozen=True)
class Shuffle(BytesBytesCodec):
    """GPU byteshuffle — numcodecs-compatible BytesBytesCodec.

    Parameters
    ----------
    elementsize:
        Number of bytes per element to shuffle across.  Must divide the
        chunk byte length.
    backend:
        Runtime impl choice.  ``"cupy"`` (default) is Hopper-safe;
        ``"cutile"`` opts in to the cuda.tile transpose kernel for ~2.5x
        more bandwidth on A40 (broken on H100/H200).  Not persisted.
    """

    is_fixed_size: ClassVar[bool] = False
    codec_name: ClassVar[str] = "shuffle"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("cupy", "cutile")
    _default_backend: ClassVar[CodecBackend] = "cupy"

    elementsize: int = 4
    backend: CodecBackend | None = field(default=None, compare=False, repr=True)

    def __post_init__(self) -> None:
        chosen = resolve_backend_for_filter(
            self.codec_name,
            instance_backend=self.backend,
            supported=self._supported_backends,
            default=self._default_backend,
        )
        object.__setattr__(self, "backend", chosen)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        arr = chunk_data.as_array_like()
        cp_arr = cp.asarray(arr).view(cp.uint8) if not isinstance(arr, cp.ndarray) else arr.view(cp.uint8)
        if self.backend == "cupy":
            out = _byteunshuffle_cupy(cp_arr, self.elementsize, cp_arr.size)
        else:
            from czarr.kernels.byteshuffle import byteunshuffle_batched

            out = byteunshuffle_batched(cp_arr, self.elementsize, cp_arr.size)
        return chunk_spec.prototype.buffer.from_array_like(out)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        arr = chunk_data.as_array_like()
        cp_arr = cp.asarray(arr).view(cp.uint8) if not isinstance(arr, cp.ndarray) else arr.view(cp.uint8)
        if self.backend == "cupy":
            out = _byteshuffle_cupy(cp_arr, self.elementsize, cp_arr.size)
        else:
            from czarr.kernels.byteshuffle import byteshuffle_batched

            out = byteshuffle_batched(cp_arr, self.elementsize, cp_arr.size)
        return chunk_spec.prototype.buffer.from_array_like(out)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        """Shuffle is a permutation — same size in and out."""
        return input_byte_length

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr v3 metadata.

        ``backend`` is intentionally omitted — bitstream is the only
        persisted identity.
        """
        return {
            "name": self.codec_name,
            "configuration": {"elementsize": self.elementsize},
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Shuffle:
        """Reconstruct from Zarr v3 metadata: {'name': ..., 'configuration': {...}}.

        Tolerant of writers that leak ``backend`` into the configuration.
        """
        cfg = dict(data.get("configuration", {k: v for k, v in data.items() if k != "name"}))
        cfg.pop("backend", None)
        return cls(**cfg)


def _byteunshuffle_cupy(packed: cp.ndarray, typesize: int, blocksize: int) -> cp.ndarray:
    """Pure cupy byteunshuffle — reshape + transpose + ascontiguousarray.

    Roughly 2.5x slower than the cuda.tile path on A40 but works on
    every GPU we target.  Bit-exact with the cuTile result.
    """
    if packed.size % blocksize != 0:
        raise ValueError(f"byteunshuffle: input {packed.size} not a multiple of blocksize {blocksize}")
    nblocks = packed.size // blocksize
    nelem = blocksize // typesize
    # Each block has typesize planes of nelem bytes; interleave them.
    return cp.ascontiguousarray(packed.reshape(nblocks, typesize, nelem).transpose(0, 2, 1)).ravel()


def _byteshuffle_cupy(raw: cp.ndarray, typesize: int, blocksize: int) -> cp.ndarray:
    """Pure cupy byteshuffle — inverse of :func:`_byteunshuffle_cupy`."""
    if raw.size % blocksize != 0:
        raise ValueError(f"byteshuffle: input {raw.size} not a multiple of blocksize {blocksize}")
    nblocks = raw.size // blocksize
    nelem = blocksize // typesize
    return cp.ascontiguousarray(raw.reshape(nblocks, nelem, typesize).transpose(0, 2, 1)).ravel()
