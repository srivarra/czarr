"""Two-tier LZ4 codec tests.

Covers the backend-aware machinery in :mod:`czarr.codecs._backend` and the
LZ4 dispatch in :mod:`czarr.codecs.compressors.lz4`.  Skips the actual
GPU kernel invocation tests when cupy JIT can't load on the host (CUDA-13
on a CuPy-cu12 install — the cluster-known nvrtc lookup issue); the
metadata + dispatcher tests run unconditionally.
"""

from __future__ import annotations

import importlib

import pytest

import czarr
from czarr.codecs._backend import (
    _BACKEND_OVERRIDES,
    resolve_default_backend,
    set_backend_overrides,
)


@pytest.fixture(autouse=True)
def _reset_overrides() -> None:
    """Tests touch the global override map; reset before each."""
    yield
    set_backend_overrides({})


# ---------------------------------------------------------------------------
# resolve_default_backend
# ---------------------------------------------------------------------------


class TestResolveDefaultBackend:
    """The precedence stack: override > class default."""

    def test_class_default_returned(self) -> None:
        assert resolve_default_backend("lz4", supported=("native", "nvcomp"), default="native") == "native"

    def test_nvcomp_default(self) -> None:
        assert resolve_default_backend("zstd", supported=("nvcomp",), default="nvcomp") == "nvcomp"

    def test_default_not_supported_raises(self) -> None:
        with pytest.raises(ValueError, match="class default"):
            resolve_default_backend("zstd", supported=("nvcomp",), default="native")

    def test_override_wins(self) -> None:
        set_backend_overrides({"lz4": "nvcomp"})
        assert resolve_default_backend("lz4", supported=("native", "nvcomp"), default="native") == "nvcomp"

    def test_override_unsupported_raises(self) -> None:
        set_backend_overrides({"zstd": "native"})
        with pytest.raises(ValueError, match="codec_backend_overrides.*zstd.*native"):
            resolve_default_backend("zstd", supported=("nvcomp",), default="nvcomp")


# ---------------------------------------------------------------------------
# LZ4 constructor — backend resolution + validation
# ---------------------------------------------------------------------------


class TestLZ4Backend:
    """Per-instance backend selection on ``czarr.LZ4``."""

    def test_default_picks_native(self) -> None:
        assert czarr.LZ4().backend == "native"

    def test_explicit_nvcomp(self) -> None:
        assert czarr.LZ4(backend="nvcomp").backend == "nvcomp"

    def test_explicit_native(self) -> None:
        assert czarr.LZ4(backend="native").backend == "native"

    def test_unsupported_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend='nope' unsupported"):
            czarr.LZ4(backend="nope")  # type: ignore[arg-type]

    def test_override_takes_effect_at_construction(self) -> None:
        set_backend_overrides({"lz4": "nvcomp"})
        assert czarr.LZ4().backend == "nvcomp"

    def test_per_instance_kwarg_beats_override(self) -> None:
        set_backend_overrides({"lz4": "nvcomp"})
        assert czarr.LZ4(backend="native").backend == "native"


# ---------------------------------------------------------------------------
# Metadata round-trip — backend NEVER persisted
# ---------------------------------------------------------------------------


class TestMetadataRoundTrip:
    """``to_dict`` strips ``backend``; ``from_dict`` tolerates leaks."""

    def test_to_dict_omits_backend(self) -> None:
        codec = czarr.LZ4(backend="native", acceleration=2)
        d = codec.to_dict()
        assert d == {"name": "lz4", "configuration": {"acceleration": 2}}
        assert "backend" not in d["configuration"]

    def test_to_dict_identical_across_backends(self) -> None:
        a = czarr.LZ4(backend="native").to_dict()
        b = czarr.LZ4(backend="nvcomp").to_dict()
        assert a == b

    def test_from_dict_tolerates_backend_leak(self) -> None:
        """A buggy writer might leak ``backend`` into configuration."""
        codec = czarr.LZ4.from_dict(
            {
                "name": "lz4",
                "configuration": {"acceleration": 1, "backend": "nvcomp"},
            }
        )
        assert codec.acceleration == 1
        # Backend resolves via overrides + class default, NOT the leaked field.
        assert codec.backend == "native"

    def test_from_dict_resolves_to_native_default(self) -> None:
        codec = czarr.LZ4.from_dict(
            {
                "name": "lz4",
                "configuration": {"acceleration": 1},
            }
        )
        assert codec.backend == "native"

    def test_from_dict_respects_override(self) -> None:
        set_backend_overrides({"lz4": "nvcomp"})
        codec = czarr.LZ4.from_dict(
            {
                "name": "lz4",
                "configuration": {"acceleration": 1},
            }
        )
        assert codec.backend == "nvcomp"

    def test_equality_ignores_backend(self) -> None:
        """A native-LZ4 and an nvCOMP-LZ4 codec are equal; they encode the same bytes."""
        a = czarr.LZ4(backend="native", acceleration=1)
        b = czarr.LZ4(backend="nvcomp", acceleration=1)
        assert a == b


# ---------------------------------------------------------------------------
# configure_gpu integration
# ---------------------------------------------------------------------------


class TestConfigureGpuOverrides:
    """``configure_gpu(codec_backend_overrides=...)`` writes the global map."""

    def test_passing_overrides_writes_map(self, monkeypatch) -> None:
        # Stub out the GPU-state-mutating bits so the test is hermetic.
        from czarr import alloc, storage
        from czarr import pipeline as pipe_mod

        monkeypatch.setattr(alloc, "register_nvcomp_allocator", lambda: None)
        monkeypatch.setattr(alloc, "use_rmm_pool", lambda **kw: None)
        monkeypatch.setattr(storage.cufile_runtime, "is_available", lambda: False)
        monkeypatch.setattr(
            pipe_mod.CzarrPipeline,
            "configure",
            classmethod(lambda cls, **kw: None),
        )
        import zarr

        monkeypatch.setattr(zarr.config, "set", lambda *a, **kw: None)

        czarr.configure_gpu(codec_backend_overrides={"lz4": "nvcomp"})
        assert _BACKEND_OVERRIDES == {"lz4": "nvcomp"}

    def test_none_clears_map(self, monkeypatch) -> None:
        from czarr import alloc, storage
        from czarr import pipeline as pipe_mod

        monkeypatch.setattr(alloc, "register_nvcomp_allocator", lambda: None)
        monkeypatch.setattr(alloc, "use_rmm_pool", lambda **kw: None)
        monkeypatch.setattr(storage.cufile_runtime, "is_available", lambda: False)
        monkeypatch.setattr(
            pipe_mod.CzarrPipeline,
            "configure",
            classmethod(lambda cls, **kw: None),
        )
        import zarr

        monkeypatch.setattr(zarr.config, "set", lambda *a, **kw: None)

        set_backend_overrides({"lz4": "nvcomp"})
        czarr.configure_gpu()  # codec_backend_overrides=None
        assert _BACKEND_OVERRIDES == {}


# ---------------------------------------------------------------------------
# Native kernel — module loads, decoder function callable
# ---------------------------------------------------------------------------


class TestNativeKernelLoads:
    """Smoke test: the native module imports + the kernel source compiles.

    Skipped when cupy JIT can't load on the host (the cluster-known
    CUDA-13 vs CuPy-cu12 nvrtc lookup issue).  Real kernel-execution
    coverage lives in ``bench/cuda_array/`` on a GPU node.
    """

    def test_module_imports(self) -> None:
        mod = importlib.import_module("czarr.codecs._native.lz4")
        assert hasattr(mod, "decode_lz4_native")
        assert hasattr(mod, "CodecDecodeError")

    def test_empty_input_returns_empty(self) -> None:
        from czarr.codecs._native.lz4 import decode_lz4_native

        # Zero-block batch — no kernel launch, no JIT triggered.
        assert decode_lz4_native([], []) == []

    def test_mismatched_lengths_raise(self) -> None:
        import cupy as cp

        from czarr.codecs._native.lz4 import decode_lz4_native

        with pytest.raises(ValueError, match="length"):
            decode_lz4_native([cp.empty(0, dtype=cp.uint8)], [10, 20])
