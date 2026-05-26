"""Codec backend dispatch and process-global override registry.

The ``_BackendAware`` mixin lets one codec class dispatch to either
nvCOMP or a native cuda-python kernel.

A codec may be backed by NVIDIA's nvCOMP (closed-source, broad coverage)
or by a native cuda-python kernel that czarr owns.  The choice is a
runtime knob — both backends produce the same on-disk bitstream — so
metadata never carries the backend choice (see
:meth:`czarr.codecs.compressors.lz4.LZ4.to_dict` for the round-trip
rules).

The design is in ``docs/planning/research/cuda-array/08-two-tier-codecs.md``.
This module is the runtime substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from czarr.codecs.base import CudaBytesBytesCodec

if TYPE_CHECKING:
    from collections.abc import Mapping

type CodecBackend = Literal["native", "nvcomp"]


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
    native_preferred: bool,
) -> CodecBackend:
    """Resolve the default backend for a codec at construction time.

    Precedence:

    1. Process-global override (set via :func:`czarr.configure_gpu`)
    2. Class default — ``"native"`` when the codec opted in via
       ``_native_default = True`` and ``"native"`` is in its supported set
    3. ``"nvcomp"`` fallback

    Raises :class:`ValueError` when the override picks a backend the
    codec does not support — silent fallback hides user mistakes.
    """
    override = _BACKEND_OVERRIDES.get(codec_name)
    if override is not None:
        if override not in supported:
            allowed = ", ".join(supported)
            raise ValueError(f"codec_backend_overrides[{codec_name!r}]={override!r}; codec only supports ({allowed!r})")
        return override
    if native_preferred and "native" in supported:
        return "native"
    return "nvcomp"


@dataclass(frozen=True)
class _BackendAware(CudaBytesBytesCodec):
    """``BytesBytesCodec`` that may route through nvCOMP or a native kernel.

    Subclasses declare ``_supported_backends`` (a tuple of allowed
    backends; default is nvCOMP-only) and ``_native_default`` (whether
    to pick ``"native"`` when the user has not explicitly chosen).  They
    override ``_decode_native`` / ``_encode_native`` to provide the
    native body; the nvCOMP body inherits from :class:`CudaBytesBytesCodec`
    and runs through :meth:`_batch_sync`.

    The ``backend`` field is the runtime choice — never persisted, never
    contributes to equality.  Per-instance ``backend=`` wins over the
    process-global override which wins over the class default.
    """

    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("nvcomp",)
    _native_default: ClassVar[bool] = False

    backend: CodecBackend | None = field(default=None, compare=False, repr=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        chosen = self.backend or resolve_default_backend(
            self.codec_name,
            supported=self._supported_backends,
            native_preferred=self._native_default,
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
