"""Codec backend dispatch and process-global override registry.

A codec can route its encode/decode through one of several
implementations — for compressors that's nvCOMP vs a native
cuda-python kernel; for filters that's a cupy-primitive impl vs a
cuda.compute (CCCL) impl.  The choice is a runtime knob — both
backends produce the same on-disk bitstream — so metadata never
carries the backend choice.

The design is in ``docs/planning/research/cuda-array/08-two-tier-codecs.md``.
This module is the runtime substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from czarr.codecs.base import CudaBytesBytesCodec

if TYPE_CHECKING:
    from collections.abc import Mapping

# CodecBackend is intentionally widened to ``str`` — each codec class
# pins the valid set via its ``_supported_backends`` tuple.  Compressors
# use ``"native"`` / ``"nvcomp"``; filters use ``"cupy"`` / ``"cccl"``.
type CodecBackend = str


# Process-global override map.  ``configure_gpu(codec_backend_overrides=...)``
# writes here; ``resolve_default_backend`` reads it.  Per-instance
# ``backend=`` kwargs win over this map.
_BACKEND_OVERRIDES: dict[str, CodecBackend] = {}


def set_backend_overrides(overrides: Mapping[str, CodecBackend]) -> None:
    """Replace the process-global codec-backend override map.

    Called from :func:`czarr.configure_gpu`.  ``{}`` clears the table.
    """
    _BACKEND_OVERRIDES.clear()
    _BACKEND_OVERRIDES.update(overrides)


def get_backend_overrides() -> Mapping[str, CodecBackend]:
    """Read-only view of the current overrides.  Test helper."""
    return dict(_BACKEND_OVERRIDES)


def resolve_default_backend(
    codec_name: str,
    *,
    supported: tuple[CodecBackend, ...],
    default: CodecBackend,
) -> CodecBackend:
    """Resolve the default backend for a codec at construction time.

    Precedence:

    1. Process-global override (set via :func:`czarr.configure_gpu`)
    2. Class default supplied as ``default``

    Raises :class:`ValueError` when the override picks a backend the
    codec does not support — silent fallback hides user mistakes.
    """
    override = _BACKEND_OVERRIDES.get(codec_name)
    if override is not None:
        if override not in supported:
            allowed = ", ".join(supported)
            raise ValueError(f"codec_backend_overrides[{codec_name!r}]={override!r}; codec only supports ({allowed!r})")
        return override
    if default not in supported:
        raise ValueError(f"codec={codec_name!r} class default {default!r} not in supported ({', '.join(supported)!r})")
    return default


@dataclass(frozen=True)
class _BackendAware(CudaBytesBytesCodec):
    """``BytesBytesCodec`` that may route through nvCOMP or a native kernel.

    Subclasses declare ``_supported_backends`` (a tuple of allowed
    backends; default is nvCOMP-only) and ``_default_backend`` (the
    class default when no per-instance kwarg / override is present).
    They override ``_decode_native`` to provide the native body; the
    nvCOMP body inherits from :class:`CudaBytesBytesCodec`.

    The ``backend`` field is the runtime choice — never persisted, never
    contributes to equality.  Per-instance ``backend=`` wins over the
    process-global override which wins over the class default.
    """

    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("nvcomp",)
    _default_backend: ClassVar[CodecBackend] = "nvcomp"

    backend: CodecBackend | None = field(default=None, compare=False, repr=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        chosen = self.backend or resolve_default_backend(
            self.codec_name,
            supported=self._supported_backends,
            default=self._default_backend,
        )
        if chosen not in self._supported_backends:
            allowed = ", ".join(self._supported_backends)
            raise ValueError(f"codec={self.codec_name!r} backend={chosen!r} unsupported; allowed: ({allowed!r})")
        # frozen dataclass — bypass __setattr__
        object.__setattr__(self, "backend", chosen)

    # Subclasses with a native implementation override this.  Encode falls
    # back to nvCOMP unconditionally for v0.1 (see design §4.1, Option B).
    def _decode_native(self, items, op):
        raise NotImplementedError(
            f"codec={self.codec_name!r} declares native support but does not implement _decode_native"
        )

    # The dispatcher.  Overrides :meth:`CudaBytesBytesCodec._batch_sync`;
    # delegates back to the parent for the nvCOMP path so today's
    # thread-local Codec cache + framing logic is reused unchanged.
    def _batch_sync(self, items, op):
        if op == "decode" and self.backend == "native":
            return self._decode_native(items, op)
        return super()._batch_sync(items, op)


def resolve_backend_for_filter(
    codec_name: str,
    *,
    instance_backend: CodecBackend | None,
    supported: tuple[CodecBackend, ...],
    default: CodecBackend,
) -> CodecBackend:
    """Convenience for filter codecs that aren't ``_BackendAware`` subclasses.

    Same precedence stack as :func:`resolve_default_backend` but built
    around per-instance ``backend=`` kwarg semantics.  Filter codecs
    (Shuffle / Delta / FixedScaleOffset / BitRound) inherit
    :class:`zarr.abc.codec.ArrayArrayCodec` directly and can't use the
    ``_BackendAware`` mixin — call this from their ``__post_init__``
    instead.
    """
    if instance_backend is not None:
        if instance_backend not in supported:
            allowed = ", ".join(supported)
            raise ValueError(f"codec={codec_name!r} backend={instance_backend!r} unsupported; allowed: ({allowed!r})")
        return instance_backend
    return resolve_default_backend(codec_name, supported=supported, default=default)
