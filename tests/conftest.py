"""Shared pytest fixtures for the czarr test suite."""

import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
import zarr

# cuFile in compat mode rejects tmpfs/ramfs; force compat path on hosts
# without nvidia_fs so cuFile-dependent tests still run on tmpfs-free
# real-disk paths.  When nvidia_fs IS loaded we want real GDS — the
# async path requires it and silently corrupts under forced compat.
if not Path("/proc/driver/nvidia-fs").exists():
    os.environ.setdefault("CUFILE_FORCE_COMPAT_MODE", "true")

# pytest's default tmp_path is on /tmp (tmpfs on many hosts), which
# cuFile cannot use even in compat mode (udev attrs are unavailable for
# tmpfs).  Anchor cuFile-using tests in a dedicated real-FS dir outside
# the repo tree, swept of stale leftovers once per session.  Container CI
# (Modal) points CZARR_TEST_TMP at container-local disk instead.
_GPUSTORE_TMP_PARENT = Path(
    os.environ.get("CZARR_TEST_TMP") or f"/hpc/mydata/{os.environ.get('USER', 'nobody')}/.czarr-test-tmp"
)


@pytest.fixture(scope="session", autouse=True)
def _sweep_stale_gpustore_tmp():
    """Remove run dirs older than a day that NFS cleanup races left behind."""
    if _GPUSTORE_TMP_PARENT.is_dir():
        cutoff = time.time() - 86_400
        for d in _GPUSTORE_TMP_PARENT.iterdir():
            try:
                if d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
    return


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

    ``ignore_cleanup_errors=True`` because NFS metadata caching can leave
    ``.nfs*`` sentinels briefly after handle close; anything that still
    survives is reaped by the session sweep on a later run.
    """
    _GPUSTORE_TMP_PARENT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=_GPUSTORE_TMP_PARENT,
        prefix="run_",
        ignore_cleanup_errors=True,
    ) as td:
        yield Path(td)
        # The process-wide cuFile handle cache deliberately keeps fds open;
        # on NFS an open fd on a deleted file becomes a .nfs* sentinel that
        # defeats rmtree.  Release the handles before cleanup runs.
        from czarr import cufile

        cufile.clear_handle_cache()
