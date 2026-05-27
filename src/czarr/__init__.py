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
from czarr.array import CudaZarrArray, create_cuda_array, open_cuda_array
from czarr.codecs import (
    ANS,
    LZ4,
    Bitcomp,
    BitRound,
    Cascaded,
    Checksum,
    CudaBytesBytesCodec,
    Deflate,
    Delta,
    FixedScaleOffset,
    GDeflate,
    Gzip,
    Shuffle,
    Snappy,
    Zlib,
    Zstd,
)

# Internal — registered against "sharding_indexed" so zarr's registry
# resolves all v3 sharded metadata to the coalescing variant; not part
# of the public API.
from czarr.codecs.sharding import CzarrShardingCodec as _CzarrShardingCodec
from czarr.storage import GPULocalStore, cufile_runtime


def configure_gpu(
    *,
    batch_size: int | None = None,
    decode_batch_size: int | None = None,
    async_concurrency: int = 32,
    rmm_pool_gb: float | None = None,
    cufile_poll_mode: bool = False,
    cufile_poll_threshold_kb: int = 4,
    stream_pool_size: int = 4,
    pinned_prealloc: Any = None,
    pipeline: bool = True,
    codec_backend_overrides: dict[str, str] | None = None,
) -> None:
    """One-call setup for GPU-codec workloads.

    Configures zarr's runtime so that:

    * The :class:`czarr.pipeline.CzarrPipeline` becomes the global
      ``codec_pipeline`` (unless ``pipeline=False``).  Codecs route through
      it and the shared :class:`StreamPool` + :class:`PinnedHostPool`
      substrate.
    * Every ``arr[:]`` (and any other selection) decodes the whole chunk
      batch in a single nvCOMP call — sets ``codec_pipeline.batch_size``
      to ``sys.maxsize`` so one ``CudaBytesBytesCodec.decode([all])``
      runs per read.
    * Buffers default to GPU prototypes — :class:`zarr.core.buffer.gpu.Buffer`
      and ``gpu.NDBuffer``.
    * Compat codecs (Zstd, LZ4, Gzip, Zlib) win the registry lookup over
      their CPU equivalents — existing CPU-written zarr v3 stores decode
      on the GPU transparently.

    Parameters
    ----------
    batch_size:
        Explicit override for ``codec_pipeline.batch_size``.  When set,
        wins over ``decode_batch_size``.  Use ``sys.maxsize`` to force
        "all chunks in one decode call" (max nvCOMP batching, no
        read/decode overlap).
    decode_batch_size:
        Micro-batch size for the read/decode pipeline.  ``None`` (default)
        sends every chunk through a single nvCOMP call — best on our
        H200 bench because nvCOMP has ~35 ms per-call overhead and a
        per-thread codec warmup that smaller batches keep paying.
        Set to a finite value (8, 16, 32) only if profiling shows you
        can amortise that cost; see ``bench/overlap/sweep_h200`` for
        evidence that the naive microbatch knob alone regresses 4-15×.
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
    stream_pool_size:
        Number of CUDA streams in the shared :class:`StreamPool`.
        Default 4.
    pinned_prealloc:
        Optional iterable of ``(size, count)`` tuples for the shared
        :class:`PinnedHostPool` pre-allocation hint.
    pipeline:
        If True (default), register :class:`CzarrPipeline` as zarr's
        default ``codec_pipeline``.  Set False to opt into per-array
        pipeline registration only.
    codec_backend_overrides:
        Per-codec backend pin, e.g. ``{"lz4": "nvcomp"}`` to force the
        nvCOMP path for LZ4.  Overrides the per-codec default but loses
        to a per-instance ``backend=`` kwarg.  ``None`` clears any prior
        override map; pass ``{}`` to keep it empty without changing
        other settings.
    """
    from czarr.codecs._backend import set_backend_overrides
    from czarr.pipeline import CzarrPipeline

    if rmm_pool_gb is not None:
        use_rmm_pool(initial_size=int(rmm_pool_gb * (1 << 30)))
    register_nvcomp_allocator()
    set_backend_overrides(codec_backend_overrides or {})
    if cufile_poll_mode and cufile_runtime.is_available():
        cufile_runtime.set_poll_mode(True, cufile_poll_threshold_kb)

    CzarrPipeline.configure(stream_pool_size=stream_pool_size, pinned_prealloc=pinned_prealloc)

    if batch_size is not None:
        effective_batch_size = batch_size
    elif decode_batch_size is not None:
        effective_batch_size = decode_batch_size
    else:
        effective_batch_size = sys.maxsize
    settings: dict[str, Any] = {
        "codec_pipeline.batch_size": effective_batch_size,
        "async.concurrency": async_concurrency,
        "buffer": "zarr.core.buffer.gpu.Buffer",
        "ndbuffer": "zarr.core.buffer.gpu.NDBuffer",
    }
    # Explicit set in both branches so toggling pipeline=False at runtime
    # actually reverts to zarr's default BatchedCodecPipeline (otherwise the
    # last codec_pipeline.path setting sticks across configure_gpu calls).
    if pipeline:
        settings["codec_pipeline.path"] = f"{CzarrPipeline.__module__}.{CzarrPipeline.__qualname__}"
    else:
        settings["codec_pipeline.path"] = "zarr.core.codec_pipeline.BatchedCodecPipeline"
    # Resolve registry conflicts where czarr and zarr both registered a
    # codec under the same id (zstd, gzip, shuffle, bitround,
    # sharding_indexed, ...).  Without this zarr emits a ZarrUserWarning
    # on every read.  CzarrShardingCodec is selected here so existing
    # sharded v3 stores transparently get the coalescing partial-shard
    # decode path.
    for cls in (Zstd, LZ4, Gzip, Zlib, Shuffle, Delta, FixedScaleOffset, BitRound, _CzarrShardingCodec):
        settings[f"codecs.{cls.codec_name}"] = f"{cls.__module__}.{cls.__qualname__}"

    zarr.config.set(settings)


__all__ = [
    "ANS",
    "Bitcomp",
    "BitRound",
    "Cascaded",
    "Checksum",
    "CudaBytesBytesCodec",
    "CudaZarrArray",
    "Delta",
    "Deflate",
    "FixedScaleOffset",
    "GDeflate",
    "GPULocalStore",
    "Gzip",
    "LZ4",
    "Shuffle",
    "Snappy",
    "Zlib",
    "Zstd",
    "configure_gpu",
    "create_cuda_array",
    "open_cuda_array",
    "register_nvcomp_allocator",
    "use_rmm_pool",
]

__version__ = version("czarr")
