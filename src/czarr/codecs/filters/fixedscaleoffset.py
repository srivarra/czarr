"""FixedScaleOffset — GPU ArrayArrayCodec for affine quantisation.

Forward (encode): ``q = round((x - offset) * scale).astype(astype)``.
Inverse (decode): ``x = q.astype(dtype) / scale + offset``.

Compatible with ``numcodecs.FixedScaleOffset``.  The typical use is
``dtype=float32, astype=int16`` for lossy compression of bounded-range
floats: store quantised int16s on disk, recover floats on read.

Two backends:

* ``backend="cupy"`` (default) — cupy elementwise expression.  Fuses
  into a single kernel via cupy's elementwise machinery.
* ``backend="cccl"`` — :func:`cuda.compute.make_unary_transform`.  Same
  cuda.compute toolchain as Delta; numba-cuda JITs the affine op into
  a typed device function and caches the LTO-compiled kernel by scale
  / offset bit-pattern.

The ``"cupy"`` backend produces bit-identical output to numcodecs
(cupy.around mirrors np.around).  The ``"cccl"`` backend may differ
from numcodecs by ~ULP on a handful of exact-half values per chunk
(~0.003% on the 16 MiB f32 spike) — numba's CUDA codegen may fuse
the affine multiply-add into an fma, putting borderline values on
the other side of the round-half-to-even boundary.  Use ``"cupy"``
when bit-exact round-trip with CPU numcodecs matters; use ``"cccl"``
when throughput matters more than ULP-scale rounding drift on
quantisation boundaries.

``backend`` is runtime and not persisted in Zarr v3 metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
from zarr.abc.codec import ArrayArrayCodec

from czarr.codecs._backend import CodecBackend, resolve_backend_for_filter

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.common import JSON


@dataclass(frozen=True)
class FixedScaleOffset(ArrayArrayCodec):
    """GPU affine-quantisation filter — numcodecs-compatible ArrayArrayCodec.

    Parameters
    ----------
    offset:
        Additive offset (subtracted on encode, added on decode).
    scale:
        Multiplicative scale (multiplied on encode, divided on decode).
    dtype:
        Source / decoded dtype (typically float).
    astype:
        Storage dtype after quantisation (typically integer).  ``None``
        = same as ``dtype`` (no real quantisation, useful only for
        round-trip testing).
    backend:
        Runtime impl choice.  ``"cupy"`` (default) uses cupy elementwise
        expressions; ``"cccl"`` uses cuda.compute.  Not persisted.
    """

    is_fixed_size: ClassVar[bool] = True
    codec_name: ClassVar[str] = "fixedscaleoffset"
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("cupy", "cccl")
    _default_backend: ClassVar[CodecBackend] = "cupy"

    offset: float = 0.0
    scale: float = 1.0
    dtype: DTypeLike = "<f4"
    astype: DTypeLike | None = None
    backend: CodecBackend | None = field(default=None, compare=False, repr=True)

    def __post_init__(self) -> None:
        chosen = resolve_backend_for_filter(
            self.codec_name,
            instance_backend=self.backend,
            supported=self._supported_backends,
            default=self._default_backend,
        )
        object.__setattr__(self, "backend", chosen)

    @property
    def _store_dtype(self) -> DTypeLike:
        return self.astype if self.astype is not None else self.dtype

    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        q = cp.asarray(chunk_data.as_ndarray_like())
        flat = q.ravel()
        if self.backend == "cccl":
            from czarr.codecs._native.fixedscaleoffset import decode_fso_native

            x_flat = decode_fso_native(flat, dtype=self.dtype, scale=self.scale, offset=self.offset)
        else:
            x_flat = flat.astype(self.dtype, copy=False) / self.scale + self.offset
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(x_flat.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        x = cp.asarray(chunk_data.as_ndarray_like())
        flat = x.ravel()
        if self.backend == "cccl":
            from czarr.codecs._native.fixedscaleoffset import encode_fso_native

            q_flat = encode_fso_native(flat, astype=self._store_dtype, scale=self.scale, offset=self.offset)
        else:
            q_flat = (flat - self.offset) * self.scale
            if cp.issubdtype(cp.dtype(self._store_dtype), cp.integer):
                q_flat = cp.around(q_flat)
            q_flat = q_flat.astype(self._store_dtype, copy=False)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(q_flat.reshape(chunk_spec.shape))

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        """Encoded size depends on the storage dtype's itemsize."""
        if self.astype is None:
            return input_byte_length
        item = cp.dtype(self._store_dtype).itemsize
        src_item = chunk_spec.dtype.to_native_dtype().itemsize
        return (input_byte_length // src_item) * item

    def to_dict(self) -> dict[str, JSON]:
        """Serialise codec config for storage in Zarr v3 metadata.

        ``backend`` is intentionally omitted — bitstream is the only
        persisted identity.
        """
        config: dict[str, JSON] = {
            "offset": float(self.offset),
            "scale": float(self.scale),
            "dtype": str(self.dtype),
        }
        if self.astype is not None:
            config["astype"] = str(self.astype)
        return {"name": self.codec_name, "configuration": config}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> FixedScaleOffset:
        """Reconstruct from Zarr v3 metadata: {'name': ..., 'configuration': {...}}.

        Tolerant of writers that leak ``backend`` into the configuration.
        """
        cfg = dict(data.get("configuration", {k: v for k, v in data.items() if k != "name"}))
        cfg.pop("backend", None)
        return cls(**cfg)
