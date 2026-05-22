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


@pytest.fixture(scope="session", autouse=True)
def _rmm_teardown():
    """Reset RMM to a vanilla CUDA MR after the session.

    Some tests reinitialise RMM to a pool MR (``use_rmm_pool``).  At
    process exit, the pool's teardown can race with cuda.core +
    nvCOMP's own teardown and segfault.  Switching back to the simple
    CudaMemoryResource at session end gives a deterministic shutdown
    path that matches the pristine pre-test state.
    """
    yield
    try:
        import rmm

        rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
    except (ImportError, RuntimeError):
        # Best-effort cleanup; RMM may be absent or already torn down,
        # in which case let the interpreter handle the rest.
        pass


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
