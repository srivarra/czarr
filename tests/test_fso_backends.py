"""Two-tier FixedScaleOffset filter tests.

Parallels the Delta and Shuffle backend tests.  FSO supports both
``"cupy"`` (default elementwise) and ``"cccl"``
(cuda.compute make_unary_transform).
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


class TestFsoBackend:
    """Per-instance backend selection on ``czarr.FixedScaleOffset``."""

    def test_default_picks_cupy(self) -> None:
        assert czarr.FixedScaleOffset(scale=10.0).backend == "cupy"

    def test_explicit_cccl(self) -> None:
        assert czarr.FixedScaleOffset(scale=10.0, backend="cccl").backend == "cccl"

    def test_explicit_cupy(self) -> None:
        assert czarr.FixedScaleOffset(scale=10.0, backend="cupy").backend == "cupy"

    def test_unsupported_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend='nope' unsupported"):
            czarr.FixedScaleOffset(scale=10.0, backend="nope")  # type: ignore[arg-type]

    def test_nvcomp_not_supported_raises(self) -> None:
        with pytest.raises(ValueError, match="backend='nvcomp' unsupported"):
            czarr.FixedScaleOffset(scale=10.0, backend="nvcomp")  # type: ignore[arg-type]

    def test_override_takes_effect_at_construction(self) -> None:
        set_backend_overrides({"fixedscaleoffset": "cccl"})
        assert czarr.FixedScaleOffset(scale=10.0).backend == "cccl"

    def test_per_instance_kwarg_beats_override(self) -> None:
        set_backend_overrides({"fixedscaleoffset": "cccl"})
        assert czarr.FixedScaleOffset(scale=10.0, backend="cupy").backend == "cupy"


class TestMetadataRoundTrip:
    """``to_dict`` strips ``backend``; ``from_dict`` tolerates leaks."""

    def test_to_dict_omits_backend(self) -> None:
        codec = czarr.FixedScaleOffset(scale=10.0, offset=1.0, backend="cccl")
        d = codec.to_dict()
        assert d == {
            "name": "fixedscaleoffset",
            "configuration": {"offset": 1.0, "scale": 10.0, "dtype": "<f4"},
        }
        assert "backend" not in d["configuration"]

    def test_to_dict_includes_astype(self) -> None:
        codec = czarr.FixedScaleOffset(scale=10.0, dtype="<f4", astype="<i2")
        d = codec.to_dict()
        assert d["configuration"]["astype"] == "<i2"

    def test_to_dict_identical_across_backends(self) -> None:
        a = czarr.FixedScaleOffset(scale=10.0, backend="cupy").to_dict()
        b = czarr.FixedScaleOffset(scale=10.0, backend="cccl").to_dict()
        assert a == b

    def test_from_dict_tolerates_backend_leak(self) -> None:
        codec = czarr.FixedScaleOffset.from_dict(
            {
                "name": "fixedscaleoffset",
                "configuration": {"offset": 0.0, "scale": 10.0, "dtype": "<f4", "backend": "cccl"},
            }
        )
        assert codec.scale == 10.0
        assert codec.backend == "cupy"

    def test_from_dict_resolves_to_cupy_default(self) -> None:
        codec = czarr.FixedScaleOffset.from_dict(
            {
                "name": "fixedscaleoffset",
                "configuration": {"offset": 0.0, "scale": 10.0, "dtype": "<f4"},
            }
        )
        assert codec.backend == "cupy"

    def test_from_dict_respects_override(self) -> None:
        set_backend_overrides({"fixedscaleoffset": "cccl"})
        codec = czarr.FixedScaleOffset.from_dict(
            {
                "name": "fixedscaleoffset",
                "configuration": {"offset": 0.0, "scale": 10.0, "dtype": "<f4"},
            }
        )
        assert codec.backend == "cccl"

    def test_equality_ignores_backend(self) -> None:
        a = czarr.FixedScaleOffset(scale=10.0, backend="cupy")
        b = czarr.FixedScaleOffset(scale=10.0, backend="cccl")
        assert a == b


class TestConfigureGpuOverrides:
    """``configure_gpu(codec_backend_overrides=...)`` writes the global map."""

    def test_passing_filter_override_writes_map(self, monkeypatch) -> None:
        from czarr import alloc

        monkeypatch.setattr(alloc, "register_nvcomp_allocator", lambda: None)
        monkeypatch.setattr(alloc, "use_rmm_pool", lambda **kw: None)
        import zarr

        monkeypatch.setattr(zarr.config, "set", lambda *a, **kw: None)

        czarr.configure_gpu(codec_backend_overrides={"fixedscaleoffset": "cccl"})
        assert _BACKEND_OVERRIDES == {"fixedscaleoffset": "cccl"}


class TestNativeModuleLoads:
    """Smoke test: ``_backends.fixedscaleoffset`` imports + exposes the API.

    Real kernel-execution coverage lives on a GPU node with cccl.
    """

    def test_module_imports(self) -> None:
        mod = importlib.import_module("czarr.codecs._backends.fixedscaleoffset")
        assert hasattr(mod, "decode_fso_native")
        assert hasattr(mod, "encode_fso_native")
