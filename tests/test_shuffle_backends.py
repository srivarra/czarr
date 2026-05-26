"""Two-tier Shuffle filter tests.

Parallels :mod:`tests.test_lz4_backends` and :mod:`tests.test_delta_backends`.

Shuffle backends are ``"cutile"`` (default, cuda.tile transpose) and
``"cupy"`` (fallback, reshape/transpose).  cuda.compute is intentionally
not a Shuffle backend — byte-plane transpose has no scan/reduce/transform
analogue in cccl; that would need a custom Raw/Program kernel.
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

    def test_default_picks_cutile(self) -> None:
        assert czarr.Shuffle(elementsize=4).backend == "cutile"

    def test_explicit_cupy(self) -> None:
        assert czarr.Shuffle(elementsize=4, backend="cupy").backend == "cupy"

    def test_explicit_cutile(self) -> None:
        assert czarr.Shuffle(elementsize=4, backend="cutile").backend == "cutile"

    def test_unsupported_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend='nope' unsupported"):
            czarr.Shuffle(elementsize=4, backend="nope")  # type: ignore[arg-type]

    def test_cccl_not_supported_raises(self) -> None:
        """cuda.compute isn't a Shuffle backend — make it explicit."""
        with pytest.raises(ValueError, match="backend='cccl' unsupported"):
            czarr.Shuffle(elementsize=4, backend="cccl")  # type: ignore[arg-type]

    def test_override_takes_effect_at_construction(self) -> None:
        set_backend_overrides({"shuffle": "cupy"})
        assert czarr.Shuffle(elementsize=4).backend == "cupy"

    def test_per_instance_kwarg_beats_override(self) -> None:
        set_backend_overrides({"shuffle": "cupy"})
        assert czarr.Shuffle(elementsize=4, backend="cutile").backend == "cutile"


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

    def test_to_dict_identical_across_backends(self) -> None:
        a = czarr.Shuffle(elementsize=4, backend="cutile").to_dict()
        b = czarr.Shuffle(elementsize=4, backend="cupy").to_dict()
        assert a == b

    def test_from_dict_tolerates_backend_leak(self) -> None:
        """A buggy writer might leak ``backend`` into configuration."""
        codec = czarr.Shuffle.from_dict(
            {
                "name": "shuffle",
                "configuration": {"elementsize": 4, "backend": "cupy"},
            }
        )
        assert codec.elementsize == 4
        # Backend resolves via overrides + class default, NOT the leaked field.
        assert codec.backend == "cutile"

    def test_from_dict_respects_override(self) -> None:
        set_backend_overrides({"shuffle": "cupy"})
        codec = czarr.Shuffle.from_dict(
            {
                "name": "shuffle",
                "configuration": {"elementsize": 4},
            }
        )
        assert codec.backend == "cupy"

    def test_equality_ignores_backend(self) -> None:
        """Two Shuffle codecs with different backends are equal."""
        a = czarr.Shuffle(elementsize=4, backend="cutile")
        b = czarr.Shuffle(elementsize=4, backend="cupy")
        assert a == b


# ---------------------------------------------------------------------------
# Pure-cupy byteshuffle: bit-exact with cuTile path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("typesize", [2, 4, 8])
def test_cupy_byteshuffle_round_trip(typesize: int) -> None:
    """Encode/decode round-trip + parity against cuTile (if available)."""
    cp = pytest.importorskip("cupy")
    from czarr.codecs.filters.shuffle import _byteshuffle_cupy, _byteunshuffle_cupy

    nblocks = 4
    nelem = 64
    blocksize = nelem * typesize
    raw = cp.arange(nblocks * blocksize, dtype=cp.uint8)

    shuffled = _byteshuffle_cupy(raw, typesize, blocksize)
    unshuffled = _byteunshuffle_cupy(shuffled, typesize, blocksize)
    assert bool((unshuffled == raw).all())

    # Parity against the cuTile path — same input, same output.
    try:
        from czarr.kernels.byteshuffle import byteshuffle_batched, byteunshuffle_batched
    except ImportError:
        return
    try:
        cutile_shuffled = byteshuffle_batched(raw, typesize, blocksize)
    except Exception:  # noqa: BLE001
        # cuTile may fail to compile on hosts without nvrtc; skip the
        # parity check, the round-trip above still validates correctness.
        return
    assert bool((cutile_shuffled == shuffled).all())
    cutile_unshuffled = byteunshuffle_batched(shuffled, typesize, blocksize)
    assert bool((cutile_unshuffled == unshuffled).all())


# ---------------------------------------------------------------------------
# configure_gpu integration
# ---------------------------------------------------------------------------


class TestConfigureGpuOverrides:
    """``configure_gpu(codec_backend_overrides=...)`` writes the global map."""

    def test_passing_shuffle_override_writes_map(self, monkeypatch) -> None:
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

        czarr.configure_gpu(codec_backend_overrides={"shuffle": "cupy"})
        assert _BACKEND_OVERRIDES == {"shuffle": "cupy"}
