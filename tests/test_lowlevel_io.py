"""Tests for czarr.lowlevel.io — batched request reads (GPU required).

Fixture stores live on real disk (``gpustore_tmpdir``) — cuFile cannot
read tmpfs, so ``tmp_path`` (usually /tmp) is off-limits for the read
tests.
"""

import numpy as np
import pytest
import zarr

cp = pytest.importorskip("cupy")

from czarr import lowlevel  # noqa: E402
from czarr.lowlevel.plan import ReadRequest  # noqa: E402


@pytest.fixture
def sharded_store(gpustore_tmpdir):
    root = gpustore_tmpdir / "io.zarr"
    arr = zarr.create_array(
        store=str(root),
        shape=(16, 12),
        chunks=(4, 4),
        shards=(8, 8),
        dtype="uint16",
        fill_value=0,
        zarr_format=3,
    )
    data = (np.arange(16 * 12) % 4999).astype("uint16").reshape(16, 12)
    arr[:] = data
    return root, data


def test_read_matches_host_bytes(sharded_store) -> None:
    root, _ = sharded_store
    plan = lowlevel.open_plan(root)
    reqs = plan.ranges(np.s_[:8, :])
    assert reqs
    bufs = lowlevel.read(reqs)
    assert len(bufs) == len(reqs)
    for r, b in zip(reqs, bufs, strict=True):
        assert isinstance(b, cp.ndarray)
        assert b.dtype == cp.uint8
        assert b.size == r.nbytes
        expected = r.path.read_bytes()[r.offset : r.offset + r.nbytes]
        assert bytes(cp.asnumpy(b).tobytes()) == expected


def test_member_slices_are_the_encoded_chunks(sharded_store) -> None:
    """Slicing a fused device buffer at member offsets yields exact chunk bytes."""
    root, _ = sharded_store
    plan = lowlevel.open_plan(root)
    reqs = plan.ranges(np.s_[:])
    bufs = lowlevel.read(reqs)
    for r, b in zip(reqs, bufs, strict=True):
        raw = r.path.read_bytes()
        index = plan._shard_index(
            tuple(c // p for c, p in zip(r.members[0][0], plan.units_per_shard, strict=True)), r.path, None
        )
        for coords, intra, length in r.members:
            within = tuple(c % p for c, p in zip(coords, plan.units_per_shard, strict=True))
            abs_offset, abs_len = (int(x) for x in index[within])
            assert abs_len == length
            got = bytes(cp.asnumpy(b[intra : intra + length]).tobytes())
            assert got == raw[abs_offset : abs_offset + length]


def test_empty_request_list() -> None:
    assert lowlevel.read([]) == []


def test_missing_file_raises(gpustore_tmpdir) -> None:
    req = ReadRequest(gpustore_tmpdir / "nope.bin", 0, 16, (((0,), 0, 16),))
    with pytest.raises(FileNotFoundError):
        lowlevel.read([req])


def test_short_read_raises(gpustore_tmpdir) -> None:
    """Requesting past EOF must error (OSError), never silently truncate."""
    p = gpustore_tmpdir / "short.bin"
    p.write_bytes(b"x" * 8)
    req = ReadRequest(p, 0, 64, (((0,), 0, 64),))
    with pytest.raises(OSError, match="read"):
        lowlevel.read([req])
