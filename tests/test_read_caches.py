"""Persistent read caches: cuFile handle LRU + process-wide DecodePlan LRU.

Correctness contract under test: caches never serve stale data — both
revalidate a stat signature (inode, mtime, size) per acquire, so
rewritten files reparse/reopen.
"""

import os

import cupy as cp
import numpy as np
import pytest
import zarr

from czarr import cufile
from czarr.core import Array
from czarr.lowlevel import plan as plan_mod
from czarr.lowlevel.plan import clear_plan_cache, open_plan


@pytest.fixture(autouse=True)
def _fresh_caches():
    cufile.clear_handle_cache()
    clear_plan_cache()
    yield
    cufile.clear_handle_cache()
    clear_plan_cache()


def _read_bytes_via_cufile(path, size: int) -> bytes:
    dev = cp.empty(size, dtype=cp.uint8)
    got = cufile.read_into(path, int(dev.data.ptr), size, 0)
    assert got == size
    return bytes(cp.asnumpy(dev))


class TestHandleCache:
    """Registered cuFile handles are reused, revalidated, and capped."""

    @pytest.fixture(autouse=True)
    def _needs_cufile(self):
        if not cufile.is_available():
            pytest.skip("cuFile unavailable on this host")

    def test_repeat_read_hits_cache(self, gpustore_tmpdir) -> None:
        p = gpustore_tmpdir / "a.bin"
        payload = os.urandom(4096)
        p.write_bytes(payload)
        assert _read_bytes_via_cufile(p, 4096) == payload
        assert cufile.handle_cache_len() == 1
        assert _read_bytes_via_cufile(p, 4096) == payload
        assert cufile.handle_cache_len() == 1

    def test_rewritten_file_is_not_served_stale(self, gpustore_tmpdir) -> None:
        p = gpustore_tmpdir / "b.bin"
        old = bytes(range(256)) * 16
        p.write_bytes(old)
        assert _read_bytes_via_cufile(p, len(old)) == old

        new = os.urandom(len(old))
        tmp = gpustore_tmpdir / "b.bin.tmp"
        tmp.write_bytes(new)
        tmp.replace(p)  # new inode — the strictest replacement case

        assert _read_bytes_via_cufile(p, len(new)) == new

    def test_cap_evicts_lru(self, gpustore_tmpdir, monkeypatch) -> None:
        monkeypatch.setattr(cufile, "_HANDLE_CACHE_CAP", 2)
        for name in ("c0.bin", "c1.bin", "c2.bin"):
            p = gpustore_tmpdir / name
            p.write_bytes(b"\x01" * 512)
            _read_bytes_via_cufile(p, 512)
        assert cufile.handle_cache_len() == 2

    def test_clear(self, gpustore_tmpdir) -> None:
        p = gpustore_tmpdir / "d.bin"
        p.write_bytes(b"\x02" * 512)
        _read_bytes_via_cufile(p, 512)
        cufile.clear_handle_cache()
        assert cufile.handle_cache_len() == 0


@pytest.fixture
def store_root(tmp_path):
    """A small zstd zarr v3 store (plan building is host-only)."""
    root = tmp_path / "plan.zarr"
    arr = zarr.create_array(store=str(root), shape=(8, 8), chunks=(4, 4), dtype="float32")
    arr[:] = np.arange(64, dtype="float32").reshape(8, 8)
    return root


class TestPlanCache:
    """DecodePlans are shared per root and reparsed when zarr.json changes."""

    def test_repeat_open_shares_plan(self, store_root) -> None:
        assert open_plan(store_root) is open_plan(store_root)
        assert Array.open(store_root).plan is Array.open(store_root).plan

    def test_uncached_is_private(self, store_root) -> None:
        assert open_plan(store_root, cached=False) is not open_plan(store_root, cached=False)
        assert Array.open(store_root, cached=False).plan is not open_plan(store_root)

    def test_metadata_change_reparses(self, store_root) -> None:
        first = open_plan(store_root)
        zj = store_root / "zarr.json"
        os.utime(zj, ns=(0, 0))  # deterministic mtime bump
        assert open_plan(store_root) is not first

    def test_cap_evicts_lru(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(plan_mod, "_PLAN_CACHE_CAP", 2)
        roots = []
        for i in range(3):
            root = tmp_path / f"p{i}.zarr"
            zarr.create_array(store=str(root), shape=(4,), chunks=(2,), dtype="int8")
            roots.append(root)
            open_plan(root)
        assert len(plan_mod._plan_cache) == 2
