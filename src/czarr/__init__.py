"""czarr — GPU codecs for Zarr 3.

Quick start
-----------

::

    import czarr

    czarr.configure_gpu()  # batched pipeline + GPU buffers + RMM (optional)

    store = czarr.GPULocalStore("data.zarr")
    arr = zarr.create_array(
        store=store,
        shape=...,
        chunks=...,
        dtype="float32",
        compressors=[czarr.ANS()],  # nvCOMP-native, max perf
    )
    arr[:] = cp_data

    out = arr[:]  # GPU decode, cupy.ndarray

Existing CPU-written zstd/lz4/gzip/zlib zarr files decode on the GPU
transparently after :func:`configure_gpu`.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import version
from typing import Any

if platform.system() != "Linux":
    raise RuntimeError(f"czarr only supports Linux, not {platform.system()}")

import zarr

from czarr.alloc import register_nvcomp_allocator, use_rmm_pool
from czarr.codecs import (
    ANS,
    LZ4,
    Bitcomp,
    Cascaded,
    Checksum,
    CudaBytesBytesCodec,
    Deflate,
    GDeflate,
    Gzip,
    Snappy,
    Zlib,
    Zstd,
)
from czarr.storage import GPULocalStore, cufile_runtime


def configure_gpu(
    *,
    batch_size: int | None = None,
    async_concurrency: int = 32,
    rmm_pool_gb: float | None = None,
    cufile_poll_mode: bool = False,
    cufile_poll_threshold_kb: int = 4,
) -> None:
    """One-call setup for GPU-codec workloads.

    Configures zarr's runtime so that:

    * Every ``arr[:]`` (and any other selection) decodes the whole chunk
      batch in a single nvCOMP call — sets ``codec_pipeline.batch_size``
      to ``sys.maxsize`` on zarr's default ``BatchedCodecPipeline`` so one
      ``CudaBytesBytesCodec.decode([all])`` runs per read.
    * Buffers default to GPU prototypes — :class:`zarr.core.buffer.gpu.Buffer`
      and ``gpu.NDBuffer``.  No more wrapping every call in
      ``zarr.config.set({"buffer": ...})``.
    * Compat codecs (Zstd, LZ4, Gzip, Zlib) win the registry lookup over
      their CPU equivalents — existing CPU-written zarr stores decode on
      the GPU transparently.

    Parameters
    ----------
    batch_size:
        Pipeline batch size.  ``None`` (default) means "all chunks in one
        decode call" — best perf for most cases.  Lower it (e.g. ``8``) only
        when nvCOMP scratch memory matters (e.g. very large Zstd batches).
    async_concurrency:
        Parallel ``store.get`` calls inside a batch.  Default 32.
    rmm_pool_gb:
        If set, initialise an RMM pool of this size (in GiB) and route all
        device allocations (cupy + nvCOMP) through it.  Use when sharing a
        process with cuDF / cuML / kvikIO.
    cufile_poll_mode:
        If True, switch cuFile from IRQ-driven to spin-polling completion
        for I/Os up to ``cufile_poll_threshold_kb``.  Lower latency on
        small reads at the cost of CPU.  Default False — fine for our
        typical multi-MiB chunks.
    cufile_poll_threshold_kb:
        Max I/O size (KiB) that uses polling when ``cufile_poll_mode=True``.
        Larger I/Os fall back to IRQ-driven completion regardless.
    """
    if rmm_pool_gb is not None:
        use_rmm_pool(initial_size=int(rmm_pool_gb * (1 << 30)))
    register_nvcomp_allocator()
    if cufile_poll_mode and cufile_runtime.is_available():
        cufile_runtime.set_poll_mode(True, cufile_poll_threshold_kb)

    settings: dict[str, Any] = {
        "codec_pipeline.batch_size": batch_size if batch_size is not None else sys.maxsize,
        "async.concurrency": async_concurrency,
        "buffer": "zarr.core.buffer.gpu.Buffer",
        "ndbuffer": "zarr.core.buffer.gpu.NDBuffer",
    }
    # Resolve registry conflicts where czarr and zarr both registered a
    # codec under the same id (zstd, gzip).  Without this zarr emits a
    # ZarrUserWarning on every read.
    for cls in (Zstd, LZ4, Gzip, Zlib):
        settings[f"codecs.{cls.codec_name}"] = f"{cls.__module__}.{cls.__qualname__}"

    zarr.config.set(settings)


__all__ = [
    "ANS",
    "Bitcomp",
    "Cascaded",
    "Checksum",
    "CudaBytesBytesCodec",
    "Deflate",
    "GDeflate",
    "GPULocalStore",
    "Gzip",
    "LZ4",
    "Snappy",
    "Zlib",
    "Zstd",
    "configure_gpu",
    "register_nvcomp_allocator",
    "use_rmm_pool",
]

__version__ = version("czarr")
