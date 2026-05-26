"""LZ4 — block format compatible with numcodecs.LZ4.

Backed by either nvCOMP (the legacy path) or a native cuda-python kernel
(the productionised spike from
``.planning/research/cuda-array/spikes/lz4_decoder.py``).  Both backends
produce the same on-disk bitstream — czarr's ``WITH_UNCOMPRESSED_SIZE``
mode, matching ``numcodecs.LZ4``.  The backend choice is a runtime knob;
it does not appear in Zarr v3 metadata.

Encode currently always goes through nvCOMP (see design §4.1, option B);
the native encoder is a future addition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
import numpy as np

from czarr.codecs._backend import CodecBackend, _BackendAware
from czarr.codecs.base import _Algorithm, _BitstreamKind

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON


@dataclass(frozen=True)
class LZ4(_BackendAware):
    """LZ4 block format compatible with ``numcodecs.LZ4``.

    Both backends decode the same ``WITH_UNCOMPRESSED_SIZE`` bitstream
    (4-byte LE uncompressed size prefix + LZ4 block).  The native path
    strips the prefix and runs the spike kernel; the nvCOMP path passes
    the whole buffer to nvCOMP which parses the prefix internally.
    """

    codec_name: ClassVar[str] = "lz4"
    _algorithm: ClassVar[_Algorithm] = _Algorithm.LZ4
    _bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.WITH_UNCOMPRESSED_SIZE
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("native", "nvcomp")
    _native_default: ClassVar[bool] = True

    acceleration: int = 1  # numcodecs metadata field; ignored by both backends

    def to_dict(self) -> dict[str, JSON]:
        """Emit the numcodecs LZ4 schema (no czarr-internal fields).

        The ``backend`` field is intentionally omitted — bitstream is the
        only persisted identity.  Compare against the numcodecs reference
        in ``06-numcodecs-exploration.md``.
        """
        return {
            "name": self.codec_name,
            "configuration": {"acceleration": int(self.acceleration)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> LZ4:
        """Reconstruct from Zarr v3 metadata.

        Tolerant of writers that leak ``backend`` into the configuration:
        we strip the field before constructing.  Runtime backend comes
        from the override map or class default.
        """
        cfg = dict(data.get("configuration", {}))
        cfg.pop("backend", None)
        return cls(**cfg)

    # ----- native decode --------------------------------------------------

    def _decode_native(self, items, op):
        """Decode a batch of LZ4 blocks via the native cuda-python kernel.

        Strips the 4-byte LE uncompressed-size prefix per chunk (the
        ``WITH_UNCOMPRESSED_SIZE`` framing), then dispatches to
        :func:`czarr.codecs._native.lz4.decode_lz4_native`.

        Builds chunk outputs as :class:`zarr.core.buffer.Buffer` instances
        of the spec's prototype, matching the contract used by the nvCOMP
        path in :meth:`CudaBytesBytesCodec._batch_sync`.
        """
        # ``op`` is always "decode" here — _BackendAware only dispatches
        # decode to native; encode falls back to nvCOMP via super().
        assert op == "decode"
        from czarr.codecs._native.lz4 import decode_lz4_native

        valid_items: list[tuple[Buffer, ArraySpec]] = []
        out: list[Buffer | None] = []
        non_null_indices: list[int] = []
        for i, (chunk, spec) in enumerate(items):
            if chunk is None:
                out.append(None)
                continue
            non_null_indices.append(i)
            valid_items.append((chunk, spec))
            out.append(None)  # filled in below

        if not valid_items:
            return out

        # Strip the 4-byte LE uncompressed-size prefix from each chunk
        # *without* reading it back to host: the spec already carries the
        # uncompressed size (spec.shape[0] bytes for our flat-bytes spec).
        # Pass views into ``decode_lz4_native``; its one-shot concatenate
        # handles the device-side gather in a single launch instead of
        # 1024 per-chunk .copy() launches.
        cp_inputs: list[cp.ndarray] = []
        sizes: list[int] = []
        for chunk, spec in valid_items:
            cp_arr = _buffer_to_uint8_view(chunk)
            if cp_arr.size < 4:
                raise ValueError(f"LZ4 native decode: chunk too small for size prefix ({cp_arr.size} bytes)")
            cp_inputs.append(cp_arr[4:])  # view, not copy
            sizes.append(int(spec.shape[0]))

        decoded = decode_lz4_native(cp_inputs, sizes)
        for idx, dec, (_chunk, spec) in zip(non_null_indices, decoded, valid_items, strict=True):
            out[idx] = spec.prototype.buffer.from_array_like(dec)
        return out


def _buffer_to_uint8_view(buf: Buffer) -> cp.ndarray:
    """Coerce a Zarr v3 ``Buffer`` to a 1-D uint8 ``cupy.ndarray``.

    Two cases mirroring :meth:`CudaBytesBytesCodec._batch_sync`:

    * GPU prototype — ``as_array_like`` returns a cupy array already on
      device; view as uint8 and return.
    * Host prototype — ``to_bytes`` lands a host bytes; upload via
      ``cupy.asarray`` (small per-call alloc, acceptable for v0.1).
    """
    backing = buf.as_array_like()
    if isinstance(backing, cp.ndarray):
        return backing.view(cp.uint8)
    return cp.asarray(np.frombuffer(buf.to_bytes(), dtype=np.uint8))
