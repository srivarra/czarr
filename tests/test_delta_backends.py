"""Two-tier Delta filter tests.

Parallels :mod:`tests.test_lz4_backends` — covers backend resolution,
metadata round-trip, and configure_gpu integration for the Delta filter.

Delta backends are ``"cupy"`` (default, cupy.cumsum / cupy.diff) and
``"cccl"`` (cuda.compute inclusive_scan).  Backend choice does not enter
Zarr v3 metadata; both produce bit-identical bytes.
"""

import importlib

import pytest

import czarr
from czarr.codecs._backend import (
    _BACKEND_OVERRIDES,
    set_backend_overrides,
)


@pytest.fixture(autouse=True)
def _reset_overrides() -> None:
    """Tests touch the global override map; reset before each."""
    yield
    set_backend_overrides({})


# ---------------------------------------------------------------------------
# Delta constructor — backend resolution + validation
# ---------------------------------------------------------------------------


class TestDeltaBackend:
    """Per-instance backend selection on ``czarr.Delta``."""

    def test_default_picks_cupy(self) -> None:
        assert czarr.Delta(dtype="<i4").backend == "cupy"

    def test_explicit_cccl(self) -> None:
        assert czarr.Delta(dtype="<i4", backend="cccl").backend == "cccl"

    def test_explicit_cupy(self) -> None:
        assert czarr.Delta(dtype="<i4", backend="cupy").backend == "cupy"

    def test_unsupported_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend='nope' unsupported"):
            czarr.Delta(dtype="<i4", backend="nope")  # type: ignore[arg-type]

    def test_nvcomp_not_supported_raises(self) -> None:
        """Compressor-tier backend names must not leak into filters."""
        with pytest.raises(ValueError, match="backend='nvcomp' unsupported"):
            czarr.Delta(dtype="<i4", backend="nvcomp")  # type: ignore[arg-type]

    def test_override_takes_effect_at_construction(self) -> None:
        set_backend_overrides({"delta": "cccl"})
        assert czarr.Delta(dtype="<i4").backend == "cccl"

    def test_per_instance_kwarg_beats_override(self) -> None:
        set_backend_overrides({"delta": "cccl"})
        assert czarr.Delta(dtype="<i4", backend="cupy").backend == "cupy"

    def test_override_unsupported_raises(self) -> None:
        set_backend_overrides({"delta": "native"})
        with pytest.raises(ValueError, match=r"codec_backend_overrides.*delta.*native"):
            czarr.Delta(dtype="<i4")


# ---------------------------------------------------------------------------
# Metadata round-trip — backend NEVER persisted
# ---------------------------------------------------------------------------


class TestMetadataRoundTrip:
    """``to_dict`` strips ``backend``; ``from_dict`` tolerates leaks."""

    def test_to_dict_omits_backend(self) -> None:
        codec = czarr.Delta(dtype="<i4", backend="cccl")
        d = codec.to_dict()
        assert d == {"name": "delta", "configuration": {"dtype": "<i4"}}
        assert "backend" not in d["configuration"]

    def test_to_dict_includes_astype(self) -> None:
        codec = czarr.Delta(dtype="<i4", astype="<i2")
        d = codec.to_dict()
        assert d == {"name": "delta", "configuration": {"dtype": "<i4", "astype": "<i2"}}

    def test_to_dict_identical_across_backends(self) -> None:
        a = czarr.Delta(dtype="<i4", backend="cupy").to_dict()
        b = czarr.Delta(dtype="<i4", backend="cccl").to_dict()
        assert a == b

    def test_from_dict_tolerates_backend_leak(self) -> None:
        """A buggy writer might leak ``backend`` into configuration."""
        codec = czarr.Delta.from_dict(
            {
                "name": "delta",
                "configuration": {"dtype": "<i4", "backend": "cccl"},
            }
        )
        assert str(codec.dtype) == "<i4"
        # Backend resolves via overrides + class default, NOT the leaked field.
        assert codec.backend == "cupy"

    def test_from_dict_resolves_to_cupy_default(self) -> None:
        codec = czarr.Delta.from_dict(
            {
                "name": "delta",
                "configuration": {"dtype": "<i4"},
            }
        )
        assert codec.backend == "cupy"

    def test_from_dict_respects_override(self) -> None:
        set_backend_overrides({"delta": "cccl"})
        codec = czarr.Delta.from_dict(
            {
                "name": "delta",
                "configuration": {"dtype": "<i4"},
            }
        )
        assert codec.backend == "cccl"

    def test_equality_ignores_backend(self) -> None:
        """A cupy-Delta and a cccl-Delta codec are equal; they encode the same bytes."""
        a = czarr.Delta(dtype="<i4", backend="cupy")
        b = czarr.Delta(dtype="<i4", backend="cccl")
        assert a == b


# ---------------------------------------------------------------------------
# configure_gpu integration
# ---------------------------------------------------------------------------


class TestConfigureGpuOverrides:
    """``configure_gpu(codec_backend_overrides=...)`` writes the global map.

    Mirrors the LZ4 test — verifies the filter-tier name ``cccl`` flows
    through the same override channel as the compressor-tier names.
    """

    def test_passing_filter_override_writes_map(self, monkeypatch) -> None:
        from czarr import alloc

        monkeypatch.setattr(alloc, "register_nvcomp_allocator", lambda: None)
        monkeypatch.setattr(alloc, "use_rmm_pool", lambda **kw: None)
        import zarr

        monkeypatch.setattr(zarr.config, "set", lambda *a, **kw: None)

        czarr.configure_gpu(codec_backend_overrides={"delta": "cccl"})
        assert _BACKEND_OVERRIDES == {"delta": "cccl"}


# ---------------------------------------------------------------------------
# Native module — imports
# ---------------------------------------------------------------------------


class TestNativeModuleLoads:
    """Smoke test: ``_backends.delta`` imports + exposes the expected API.

    Real kernel-execution coverage lives in
    ``.planning/research/cuda-array/spikes/delta_cuda_compute.py`` and
    in ``bench/cuda_array/`` on a GPU node.
    """

    def test_module_imports(self) -> None:
        mod = importlib.import_module("czarr.codecs._backends.delta")
        assert hasattr(mod, "decode_delta_native")
