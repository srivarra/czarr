"""Shared pytest fixtures for the czarr test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import zarr

# cuFile in compat mode rejects tmpfs/ramfs; force compat path on hosts
# without nvidia_fs so cuFile-dependent tests still run on tmpfs-free
# real-disk paths.  When nvidia_fs IS loaded we want real GDS — the
# async path requires it and silently corrupts under forced compat.
if not os.path.exists("/proc/driver/nvidia-fs"):
    os.environ.setdefault("CUFILE_FORCE_COMPAT_MODE", "true")

# pytest's default tmp_path is on /tmp (tmpfs on many hosts), which
# cuFile cannot use even in compat mode (udev attrs are unavailable for
# tmpfs).  Anchor cuFile-using tests under the repo root, which lives
# on real disk / lustre.
_LUSTRE_TMP_PARENT = "/hpc/mydata/sricharan.varra/Dev/czarr"


@pytest.fixture(autouse=True)
def _reset_zarr_config():
    """Reset zarr.config between tests so configure_gpu state doesn't leak.

    Several tests mutate global zarr config (buffer prototypes, codec
    pipeline path).  Without a reset, the next test that depends on the
    default CPU prototype fails when handed a gpu.NDBuffer.
    """
    yield
    zarr.config.reset()


@pytest.fixture
def gpustore_tmpdir() -> Path:
    """Real-disk tempdir suitable for ``GPULocalStore`` + cuFile reads/writes.

    Uses ``ignore_cleanup_errors=True`` because NFS metadata caching can
    leave stale ``.nfs*`` sentinel files briefly after handle close,
    making rmdir() race-fail.
    """
    with tempfile.TemporaryDirectory(
        dir=_LUSTRE_TMP_PARENT,
        prefix=".gpustore_test_",
        ignore_cleanup_errors=True,
    ) as td:
        yield Path(td)
