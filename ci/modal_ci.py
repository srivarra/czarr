"""Run the czarr test suite on Modal GPUs.

Usage (GPU type and CUDA lane are both parametrized)::

    modal run ci/modal_ci.py --gpu T4 --extra cu12
    modal run ci/modal_ci.py --gpu B200 --extra cu13

Dependencies are baked into one image per CUDA extra straight from uv.lock
via Image.uv_sync (cached until the lock changes); the working tree is
mounted at run time, so iterating on source costs an upload, not an image
rebuild.  cuFile runs in compat mode inside containers (no nvidia_fs) —
same semantics as the A40 dev baseline.  See issue #14 for the SM/CUDA
matrix this feeds.
"""

import os
import tomllib
from pathlib import Path

import modal

# uv_sync sanity-checks pyproject.toml with the abandoned `toml` package,
# which IndexErrors on PEP 735 `{ include-group = ... }` entries.  Reroute
# its parsing to stdlib tomllib (accepting both the path and file-object
# call forms toml.load supports).  Client-side only: the check runs where
# the CLI runs, and the container image has no toml to patch (this module
# is re-imported there to hydrate the classes).  Drop when modal-labs
# switches parsers.


def _tomllib_load(f) -> dict:
    if isinstance(f, str | os.PathLike):
        return tomllib.loads(Path(f).read_text())
    return tomllib.loads(f.read())


try:
    import toml as _toml

    _toml.load = _tomllib_load
    _toml.loads = tomllib.loads
except ModuleNotFoundError:
    pass

REPO_ROOT = Path(__file__).parent.parent
REMOTE_ROOT = "/root/czarr"

GPUS = ("T4", "L4", "A10", "A100-40GB", "H100!", "B200", "RTX-PRO-6000")

# Runtime mount of the working tree — everything pytest needs, nothing else.
_MOUNT_IGNORE = [
    ".git",
    "**/.venv*",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    ".pixi",
    ".planning",
    ".dex",
    "bench",
    "docs",
    "site",
    ".czarr-test-tmp",
]

app = modal.App("czarr-gpu-tests")


def _image(extra: str) -> modal.Image:
    """Debian slim + the locked czarr environment for one CUDA extra.

    No CUDA base image: the extras are all-wheels (cupy, nvrtc, runtime,
    nvcomp, cufile) and the driver comes from Modal's host.  uv_sync skips
    the project itself (hatch-vcs has no .git at build time); czarr installs
    at run time under a pretend version.  hatchling/hatch-vcs are pre-baked
    so that install can skip build isolation, and uv provides the installer.
    """
    return (
        modal.Image.debian_slim(python_version="3.13")
        .uv_sync(str(REPO_ROOT), groups=["test"], extras=[extra], extra_options="--no-default-groups")
        .uv_pip_install("hatchling", "hatch-vcs", "uv")
        .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, ignore=_MOUNT_IGNORE)
    )


def _probe() -> None:
    """Print which hardware path this lane actually exercised."""
    import subprocess

    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv"],
        check=False,
    )
    try:
        from cuda.bindings import driver

        driver.cuInit(0)
        attr = getattr(driver.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK", None)
        if attr is None:
            print("decompression engine: attribute not in these bindings")
        else:
            _, mask = driver.cuDeviceGetAttribute(attr, 0)
            print(f"decompression engine mask: {mask} ({'present' if mask else 'absent'})")
    except Exception as exc:  # noqa: BLE001 — diagnostics must not fail the lane
        print(f"probe skipped: {exc}")


def _run_suite() -> None:
    """Install czarr into the lane's environment and run pytest."""
    import os
    import subprocess
    import sys

    # hatch-vcs calls setuptools-scm without a dist name, so only the
    # generic (unsuffixed) pretend-version variable is honoured.
    env = os.environ | {"SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--no-build-isolation",
            ".",
        ],
        check=True,
        cwd=REMOTE_ROOT,
        env=env,
    )
    _probe()
    test_tmp = "/root/czarr-test-tmp"
    Path(test_tmp).mkdir(exist_ok=True)
    # Unbuffered so the dot-progress line streams instead of arriving in
    # 72-char chunks.  faulthandler dumps the stack of any test stuck >120s
    # and aborts, instead of hanging mutely until Modal's function timeout
    # (pytest 9 dropped the --faulthandler-timeout flag; ini options only).
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-o",
            "faulthandler_timeout=120",
            "-o",
            "faulthandler_exit_on_timeout=true",
        ],
        check=True,
        cwd=REMOTE_ROOT,
        env=env
        | {
            "CZARR_TEST_TMP": test_tmp,
            "PYTHONUNBUFFERED": "1",
            # cuFileDriverOpen never returns under Modal's gVisor sandbox
            # (hangs in C, no error).  Disable cuFile: gated tests skip and
            # reads take the host-I/O + H2D fallback.  GDS coverage stays
            # on Bruno.
            "CZARR_CUFILE": "0",
            # CUDA context destructors segfault at interpreter exit under
            # gVisor; exit on pytest's verdict before they run.
            "CZARR_TEST_HARD_EXIT": "1",
        },
    )


@app.cls(image=_image("cu12"), gpu="T4", timeout=900)
class RunnerCu12:
    """Test runner for the cu12 lane; GPU type overridden via with_options."""

    @modal.method()
    def run(self) -> None:
        """Run the suite."""
        _run_suite()


@app.cls(image=_image("cu13"), gpu="T4", timeout=900)
class RunnerCu13:
    """Test runner for the cu13 lane; GPU type overridden via with_options."""

    @modal.method()
    def run(self) -> None:
        """Run the suite."""
        _run_suite()


@app.local_entrypoint()
def main(gpu: str = "T4", extra: str = "cu12") -> None:
    """Run the suite on one (gpu, extra) lane."""
    if gpu not in GPUS:
        msg = f"unknown gpu {gpu!r}; expected one of {GPUS}"
        raise ValueError(msg)
    runner = {"cu12": RunnerCu12, "cu13": RunnerCu13}[extra]
    runner.with_options(gpu=gpu)().run.remote()
