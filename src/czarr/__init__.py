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
    Blosc,
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
from czarr.storage import GPULocalStore


class _GpuConfigToken:
    """Handle returned by :func:`configure_gpu` — also a context manager.

    ``configure_gpu`` applies its settings immediately, so the imperative call
    (``czarr.configure_gpu()``) behaves exactly as before and the return value
    can be ignored.  Used in a ``with`` block it additionally restores, on exit,
    the prior **zarr.config** (codec-registry shadowing, GPU buffer prototypes,
    pipeline path, batch_size) and the **codec-backend override map**::

        with czarr.configure_gpu():
            out = arr[:]  # GPU decode path active here
        # zarr.config + backend overrides restored to their prior values

    Not reverted: process-global one-time setup (RMM pool, nvCOMP
    allocator).  Those install once and have no clean teardown — a
    second ``configure_gpu`` reconfigures them in place.
    """

    def __init__(self, config_token: Any, prev_overrides: dict[str, str]) -> None:
        self._config_token = config_token
        self._prev_overrides = prev_overrides

    def __enter__(self) -> _GpuConfigToken:
        return self

    def __exit__(self, *exc: object) -> bool:
        from czarr.codecs._backend import set_backend_overrides

        self._config_token.__exit__(None, None, None)  # revert the zarr.config keys we set
        set_backend_overrides(self._prev_overrides)
        return False


def configure_gpu(
    *,
    batch_size: int | None = None,
    async_concurrency: int = 32,
    rmm_pool_gb: float | None = None,
    pipeline: bool = True,
    codec_backend_overrides: dict[str, str] | None = None,
) -> _GpuConfigToken:
    """One-call setup for GPU-codec workloads.

    Configures zarr's runtime so that:

    * The :class:`czarr.pipeline.CzarrPipeline` becomes the global
      ``codec_pipeline`` (unless ``pipeline=False``).
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
        Micro-batch size for the read/decode pipeline
        (``codec_pipeline.batch_size``).  ``None`` (default) sends every
        chunk through a single nvCOMP call — best on our H200 bench
        because nvCOMP has ~35 ms per-call overhead and a per-thread
        codec warmup that smaller batches keep paying.  Set to a finite
        value (8, 16, 32) only if profiling shows you can amortise that
        cost; see ``bench/overlap/sweep_h200`` for evidence that the
        naive microbatch knob alone regresses 4-15×.
    async_concurrency:
        Parallel ``store.get`` calls inside a batch.  Default 32.
    rmm_pool_gb:
        If set, initialise an RMM pool of this size (in GiB) and route all
        device allocations (cupy + nvCOMP) through it.  Use when sharing a
        process with cuDF / cuML / kvikIO.
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

    Returns
    -------
    _GpuConfigToken
        Applied immediately; the return value can be ignored for the usual
        process-wide setup.  Used as a context manager
        (``with configure_gpu(): ...``) it restores the prior zarr.config and
        codec-backend overrides on block exit — handy for scoping the GPU
        decode path or alternating it with the CPU path in one process.
    """
    from czarr.codecs._backend import get_backend_overrides, set_backend_overrides
    from czarr.pipeline import CzarrPipeline

    prev_overrides = get_backend_overrides()  # snapshot for the reversible token

    if rmm_pool_gb is not None:
        use_rmm_pool(initial_size=int(rmm_pool_gb * (1 << 30)))
    register_nvcomp_allocator()
    set_backend_overrides(codec_backend_overrides or {})

    effective_batch_size = batch_size if batch_size is not None else sys.maxsize
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
    for cls in (Zstd, LZ4, Gzip, Zlib, Blosc, Shuffle, Delta, FixedScaleOffset, BitRound, _CzarrShardingCodec):
        settings[f"codecs.{cls.codec_name}"] = f"{cls.__module__}.{cls.__qualname__}"

    # ``zarr.config.set`` applies immediately AND returns a revertible token;
    # hold it so ``with configure_gpu(): ...`` can restore the prior config.
    config_token = zarr.config.set(settings)
    return _GpuConfigToken(config_token, prev_overrides)


__all__ = [
    "ANS",
    "Bitcomp",
    "BitRound",
    "Blosc",
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
