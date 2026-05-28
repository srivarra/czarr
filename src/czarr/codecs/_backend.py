"""Codec backend dispatch and process-global override registry.

A filter codec can route its array transform through one of several
implementations — e.g. a cupy-primitive impl vs a cuda.compute (CCCL)
impl, or cuTile vs cupy for Shuffle.  The choice is a runtime knob —
all backends produce the same on-disk bitstream — so metadata never
carries the backend choice.

Compressors are pure nvCOMP wrappers (no backend switch): czarr dropped
its hand-written native codecs after profiling showed the codec is
rarely the read-path bottleneck.  This module now serves the filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# CodecBackend is intentionally widened to ``str`` — each codec class
# pins the valid set via its ``_supported_backends`` tuple.  Filters use
# ``"cupy"`` / ``"cccl"`` / ``"cutile"``.
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


def resolve_backend_for_filter(
    codec_name: str,
    *,
    instance_backend: CodecBackend | None,
    supported: tuple[CodecBackend, ...],
    default: CodecBackend,
) -> CodecBackend:
    """Resolve a filter codec's backend at construction time.

    Same precedence stack as :func:`resolve_default_backend` but built
    around per-instance ``backend=`` kwarg semantics.  Filter codecs
    (Shuffle / Delta / FixedScaleOffset) call this from their
    ``__post_init__``: per-instance ``backend=`` wins over the
    process-global override, which wins over the class default.
    """
    if instance_backend is not None:
        if instance_backend not in supported:
            allowed = ", ".join(supported)
            raise ValueError(f"codec={codec_name!r} backend={instance_backend!r} unsupported; allowed: ({allowed!r})")
        return instance_backend
    return resolve_default_backend(codec_name, supported=supported, default=default)
