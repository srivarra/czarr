"""Tests for :mod:`czarr.core.slab`.

The free-list algorithm (``_round_up`` / ``_take`` / ``_give_back``) is
pure and runs without a GPU.  The allocation path needs a CUDA device +
``cuda.core`` VMR, so those tests skip when no device is present.
"""

from __future__ import annotations

import pytest

from czarr.core.slab import CuFileSlabPool, _round_up, _Slab

# ---------------------------------------------------------------------------
# Pure free-list algorithm — no GPU
# ---------------------------------------------------------------------------


class TestRoundUp:
    def test_already_aligned(self) -> None:
        assert _round_up(4096) == 4096
        assert _round_up(8192) == 8192

    def test_rounds_up(self) -> None:
        assert _round_up(1) == 4096
        assert _round_up(4097) == 8192
        assert _round_up(4095) == 4096

    def test_zero(self) -> None:
        assert _round_up(0) == 0


def _slab(size: int, free: list[tuple[int, int]]) -> _Slab:
    """Construct a _Slab with no real device memory for free-list tests."""
    return _Slab(cuda_buffer=None, view=None, base_ptr=0, size=size, registered=False, free=list(free))


class TestTake:
    def test_exact_fit_removes_region(self) -> None:
        slab = _slab(8192, [(0, 8192)])
        off = CuFileSlabPool._take(slab, 8192)
        assert off == 0
        assert slab.free == []

    def test_partial_fit_shrinks_region(self) -> None:
        slab = _slab(8192, [(0, 8192)])
        off = CuFileSlabPool._take(slab, 4096)
        assert off == 0
        assert slab.free == [(4096, 4096)]

    def test_first_fit_picks_first_large_enough(self) -> None:
        slab = _slab(12288, [(0, 4096), (4096, 8192)])
        # Need 8192 — first region too small, second fits exactly.
        off = CuFileSlabPool._take(slab, 8192)
        assert off == 4096
        assert slab.free == [(0, 4096)]

    def test_no_fit_returns_none(self) -> None:
        slab = _slab(4096, [(0, 4096)])
        assert CuFileSlabPool._take(slab, 8192) is None
        assert slab.free == [(0, 4096)]  # untouched


class TestGiveBack:
    def test_coalesce_with_right(self) -> None:
        slab = _slab(8192, [(4096, 4096)])
        CuFileSlabPool._give_back(slab, 0, 4096)
        assert slab.free == [(0, 8192)]

    def test_coalesce_with_left(self) -> None:
        slab = _slab(8192, [(0, 4096)])
        CuFileSlabPool._give_back(slab, 4096, 4096)
        assert slab.free == [(0, 8192)]

    def test_coalesce_both_sides(self) -> None:
        slab = _slab(12288, [(0, 4096), (8192, 4096)])
        CuFileSlabPool._give_back(slab, 4096, 4096)
        assert slab.free == [(0, 12288)]

    def test_no_coalesce_when_gap(self) -> None:
        slab = _slab(16384, [(0, 4096)])
        CuFileSlabPool._give_back(slab, 8192, 4096)
        assert slab.free == [(0, 4096), (8192, 4096)]

    def test_insert_keeps_offset_order(self) -> None:
        slab = _slab(16384, [(12288, 4096)])
        CuFileSlabPool._give_back(slab, 4096, 4096)
        assert slab.free == [(4096, 4096), (12288, 4096)]


def test_alloc_free_cycle_restores_full_slab() -> None:
    """Carve a slab into N regions, free all, expect one whole-slab region."""
    slab = _slab(16384, [(0, 16384)])
    offs = [CuFileSlabPool._take(slab, 4096) for _ in range(4)]
    assert offs == [0, 4096, 8192, 12288]
    assert slab.free == []
    # Free in scrambled order — coalescing must still rebuild the slab.
    for off in (8192, 0, 12288, 4096):
        CuFileSlabPool._give_back(slab, off, 4096)
    assert slab.free == [(0, 16384)]


# ---------------------------------------------------------------------------
# Allocation path — needs a CUDA device + cuda.core VMR
# ---------------------------------------------------------------------------


@pytest.fixture
def _has_device():
    """Skip if no CUDA device / cuda.core VMR available on this host."""
    try:
        from cuda.core import Device

        Device().set_current()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no CUDA device for slab allocation: {exc}")


@pytest.mark.usefixtures("_has_device")
class TestAllocation:
    def test_allocate_is_4k_aligned(self) -> None:
        pool = CuFileSlabPool(slab_bytes=1 << 20, register=False)
        try:
            a = pool.allocate(100)
            assert a.device_ptr % 4096 == 0
            assert a.size == 100
            assert a.array.nbytes == 100
        finally:
            pool.close()

    def test_reuse_after_free(self) -> None:
        pool = CuFileSlabPool(slab_bytes=1 << 20, register=False)
        try:
            a = pool.allocate(4096)
            ptr = a.device_ptr
            del a  # returns region to free-list
            import gc

            gc.collect()
            b = pool.allocate(4096)
            assert b.device_ptr == ptr  # same region reused
        finally:
            pool.close()

    def test_growth_when_slab_full(self) -> None:
        pool = CuFileSlabPool(slab_bytes=8192, register=False)
        try:
            keep = [pool.allocate(4096) for _ in range(4)]  # 16 KiB > 8 KiB slab
            assert pool.n_slabs >= 2
            assert len({a.device_ptr for a in keep}) == 4  # all distinct
        finally:
            pool.close()

    def test_rejects_nonpositive(self) -> None:
        pool = CuFileSlabPool(register=False)
        try:
            with pytest.raises(ValueError, match="positive"):
                pool.allocate(0)
        finally:
            pool.close()
