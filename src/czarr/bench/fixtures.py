"""Shared store fixtures — the write-blosc / open-store dance every bench repeats.

A fixture writes a blosc ``[bitshuffle, zstd]`` store at a requested chunk size
onto a real block FS (never tmpfs — GDS can't DMA from it), computes a CPU
reference for the correctness gate *before* ``configure_gpu`` shadows the codec,
and opens the store as either a ``GPULocalStore`` (cuFile/GPU decode) or a plain
``LocalStore`` (CPU baseline).  Records the compressed size/ratio as regime.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def real_fs_tmpdir() -> Path:
    """A writable directory on a real block FS, never tmpfs.

    GDS/cuFile reads return zeros from tmpfs (``/tmp``).  Trust ``$TMPDIR`` only
    after confirming it is not tmpfs; otherwise fall back to the repo's bench
    area.  The sbatch already forces this — belt and suspenders for local runs.
    """
    user = os.environ.get("USER", "")
    candidates = [os.environ.get("TMPDIR"), f"/local/scratch/{user}", f"/hpc/mydata/{user}/.czarr-bench-fixtures"]
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
        if p.is_dir() and os.access(p, os.W_OK) and not _is_tmpfs(p):
            return p
    # Last resort: a temp dir on whatever FS mkdtemp picks (never the repo tree).
    import tempfile

    return Path(tempfile.mkdtemp(prefix="czarr-bench-"))


def _is_tmpfs(path: Path) -> bool:
    try:
        import subprocess

        fstype = subprocess.check_output(["df", "-T", "--output=fstype", str(path)], stderr=subprocess.DEVNULL)
        return b"tmpfs" in fstype
    except Exception:
        return False


def _write_blosc_store(chunk_mib: int, n_chunks: int, dtype: str) -> tuple[Path, tuple[int, ...], int]:
    """Write an N-chunk blosc [bitshuffle, zstd] store; return (path, shape, itemsize).

    Chunks are ``(cz, 1024, 1024)`` — a power-of-two-byte layout matching the
    validated waveorder geometry, so the chunk size is an exact multiple of the
    blosc blocksize (no bitshuffle tail; that case is unhandled in v1, dex
    4au95yu0).  ``chunk_mib`` must yield an integer ``cz`` (multiple of 2 for f16).
    """
    import zarr
    from zarr.codecs import BloscCodec, BloscShuffle
    from zarr.storage import LocalStore

    itemsize = np.dtype(dtype).itemsize
    plane = 1024
    cz = max(1, (chunk_mib << 20) // (plane * plane * itemsize))  # depth so chunk == chunk_mib
    shape = (cz * n_chunks, plane, plane)
    chunks = (cz, plane, plane)
    path = real_fs_tmpdir() / f"blosc_{chunk_mib}mib_{n_chunks}c.zarr"

    arr = zarr.create_array(
        store=LocalStore(path),
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressors=[
            BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.bitshuffle, typesize=itemsize, blocksize=0)
        ],
        overwrite=True,
    )
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, int(np.prod(shape)), dtype="float32").reshape(shape)
    arr[:] = (base + rng.standard_normal(shape).astype("float32") * 0.01).astype(dtype)
    return path, shape, itemsize


@dataclass(slots=True)
class StoreFixture:
    """An opened store plus the CPU reference and regime a bench needs."""

    store: Any  # GPULocalStore | LocalStore
    ref: np.ndarray  # CPU-decoded reference for the correctness gate
    regime: dict[str, Any]
    path: Path


class Fixtures:
    """Factory handed to benches via ``ctx.fixture``."""

    def blosc_store(
        self,
        chunk_mib: int,
        store: str = "gpu",
        *,
        n_chunks: int = 8,
        dtype: str = "float16",
    ) -> StoreFixture:
        """Write an N-chunk blosc store at ``chunk_mib`` and open it.

        ``store="gpu"`` configures czarr's GPU codecs + returns a GPULocalStore;
        ``store="local"`` returns a plain LocalStore (the CPU baseline path).
        The CPU reference is always decoded before ``configure_gpu`` runs.
        """
        import zarr
        from zarr.storage import LocalStore

        path, shape, itemsize = _write_blosc_store(chunk_mib, n_chunks, dtype)

        # CPU reference + compressed footprint — both BEFORE configure_gpu.
        ref = np.asarray(zarr.open_array(store=LocalStore(path), mode="r")[:])
        on_disk = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        nbytes = int(np.prod(shape)) * itemsize
        regime = {
            "chunk_mib": chunk_mib,
            "store": store,
            "compressed_mib": round(on_disk / 2**20, 2),
            "compressed_ratio": round(nbytes / on_disk, 2) if on_disk else None,
        }

        if store == "gpu":
            import czarr

            czarr.configure_gpu()
            opened = czarr.GPULocalStore(path, read_only=True)
            regime["gds_available"] = bool(getattr(opened, "gds_available", False))
        else:
            opened = LocalStore(path, read_only=True)
        return StoreFixture(store=opened, ref=ref, regime=regime, path=path)
