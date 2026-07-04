"""Shared dataclass base for backend-switchable filter codecs.

Delta / FixedScaleOffset / Shuffle carry an instance-level ``backend``
knob resolved at construction time (instance kwarg > process-global
override > class default).  The field is ``kw_only`` so it never shifts
a subclass's positional ``__init__`` signature, and ``compare=False``
keeps backend choice out of equality — all backends produce the same
bitstream.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from czarr.codecs._backend import CodecBackend, resolve_backend_for_filter


@dataclass(frozen=True)
class BackendFilter:
    """Mixin for filter codecs with a runtime-selectable backend.

    Inherit alongside the zarr ABC: ``class Delta(BackendFilter,
    ArrayArrayCodec)`` / ``class Shuffle(BackendFilter, BytesBytesCodec)``.
    """

    codec_name: ClassVar[str] = ""
    _supported_backends: ClassVar[tuple[CodecBackend, ...]] = ("cupy",)
    _default_backend: ClassVar[CodecBackend] = "cupy"

    backend: CodecBackend | None = field(default=None, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        chosen = resolve_backend_for_filter(
            self.codec_name,
            instance_backend=self.backend,
            supported=self._supported_backends,
            default=self._default_backend,
        )
        object.__setattr__(self, "backend", chosen)
