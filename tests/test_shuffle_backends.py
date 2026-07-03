"""Shuffle filter backend tests.

Parallels :mod:`tests.test_lz4_backends` and :mod:`tests.test_delta_backends`.

Shuffle has a single ``"cupy"`` backend (reshape/transpose).  The cuTile
variant was deleted (sm_90 compile bug, not a read-path bottleneck); the
backend-resolution machinery stays so an override or kwarg naming a
removed/unknown backend fails loudly instead of silently falling back.
"""

from __future__ import annotations

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
# Shuffle constructor — backend resolution + validation
# ---------------------------------------------------------------------------


class TestShuffleBackend:
    """Per-instance backend selection on ``czarr.Shuffle``."""

    def test_default_picks_cupy(self) -> None:
        assert czarr.Shuffle(elementsize=4).backend == "cupy"

    def test_explicit_cupy(self) -> None:
        assert czarr.Shuffle(elementsize=4, backend="cupy").backend == "cupy"

    def test_unsupported_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend='nope' unsupported"):
            czarr.Shuffle(elementsize=4, backend="nope")  # type: ignore[arg-type]

    def test_removed_cutile_backend_raises(self) -> None:
        """The deleted cuTile backend must fail loudly, not silently fall back."""
        with pytest.raises(ValueError, match="backend='cutile' unsupported"):
            czarr.Shuffle(elementsize=4, backend="cutile")  # type: ignore[arg-type]

    def test_cccl_not_supported_raises(self) -> None:
        """cuda.compute isn't a Shuffle backend — make it explicit."""
        with pytest.raises(ValueError, match="backend='cccl' unsupported"):
            czarr.Shuffle(elementsize=4, backend="cccl")  # type: ignore[arg-type]

    def test_override_naming_removed_backend_raises(self) -> None:
        set_backend_overrides({"shuffle": "cutile"})
        with pytest.raises(ValueError, match="codec_backend_overrides"):
            czarr.Shuffle(elementsize=4)


# ---------------------------------------------------------------------------
# Metadata round-trip — backend NEVER persisted
# ---------------------------------------------------------------------------


class TestMetadataRoundTrip:
    """``to_dict`` strips ``backend``; ``from_dict`` tolerates leaks."""

    def test_to_dict_omits_backend(self) -> None:
        codec = czarr.Shuffle(elementsize=4, backend="cupy")
        d = codec.to_dict()
        assert d == {"name": "shuffle", "configuration": {"elementsize": 4}}
        assert "backend" not in d["configuration"]

    def test_from_dict_tolerates_backend_leak(self) -> None:
        """A buggy writer might leak ``backend`` into configuration."""
        codec = czarr.Shuffle.from_dict(
            {
                "name": "shuffle",
                "configuration": {"elementsize": 4, "backend": "cutile"},
            }
        )
        assert codec.elementsize == 4
        # Backend resolves via overrides + class default, NOT the leaked field.
        assert codec.backend == "cupy"

    def test_equality_ignores_backend(self) -> None:
        """Explicit and defaulted backend kwargs compare equal."""
        a = czarr.Shuffle(elementsize=4, backend="cupy")
        b = czarr.Shuffle(elementsize=4)
        assert a == b


# ---------------------------------------------------------------------------
# Pure-cupy byteshuffle round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("typesize", [2, 4, 8])
def test_cupy_byteshuffle_round_trip(typesize: int) -> None:
    cp = pytest.importorskip("cupy")
    from czarr.codecs.filters.shuffle import _byteshuffle_cupy, _byteunshuffle_cupy

    nblocks = 4
    nelem = 64
    blocksize = nelem * typesize
    raw = cp.arange(nblocks * blocksize, dtype=cp.uint8)

    shuffled = _byteshuffle_cupy(raw, typesize, blocksize)
    unshuffled = _byteunshuffle_cupy(shuffled, typesize, blocksize)
    assert bool((unshuffled == raw).all())


# ---------------------------------------------------------------------------
# configure_gpu integration
# ---------------------------------------------------------------------------


class TestConfigureGpuOverrides:
    """``configure_gpu(codec_backend_overrides=...)`` writes the global map."""

    def test_passing_shuffle_override_writes_map(self, monkeypatch) -> None:
        from czarr import alloc

        monkeypatch.setattr(alloc, "register_nvcomp_allocator", lambda: None)
        monkeypatch.setattr(alloc, "use_rmm_pool", lambda **kw: None)
        import zarr

        monkeypatch.setattr(zarr.config, "set", lambda *a, **kw: None)

        czarr.configure_gpu(codec_backend_overrides={"shuffle": "cupy"})
        assert _BACKEND_OVERRIDES == {"shuffle": "cupy"}
