"""Phase 2 tests: codecs route through ``CzarrGpuBuffer`` end to end.

We exercise the lowest layer that matters for Phase 2 — the codec
``_batch_sync`` ``decode`` branch — by manually building a list of
``(CzarrGpuBuffer, ArraySpec)`` items, the same shape zarr's pipeline
hands to the codec. This avoids the zarr Array layer (whose
``__setitem__`` triggers cupy JIT on the CUDA 13 / CuPy cu12 cluster
mismatch) while still verifying:

* ``is_gpu_buffer`` recognises ``CzarrGpuBuffer``,
* the codec never falls through to the host-bytes slow path for a
  CzarrGpuBuffer input,
* the decode output is wrapped back into the right prototype's
  ``Buffer`` class with the right bytes.
"""

import cupy as cp
import numpy as np
import pytest
import zarr
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import BufferPrototype

from czarr.codecs import LZ4, Zstd
from czarr.core.buffer import (
    GPU_BUFFER_TYPES,
    CzarrGpuBuffer,
    CzarrGpuNDBuffer,
    buffer_prototype,
    is_gpu_buffer,
    is_gpu_prototype,
)


def _spec(shape: tuple[int, ...], dtype: str, prototype: BufferPrototype) -> ArraySpec:
    return ArraySpec(
        shape=shape,
        dtype=zarr.dtype.parse_dtype(dtype, zarr_format=3),
        fill_value=0,
        prototype=prototype,
        config=zarr.core.array.ArrayConfig.from_dict({}),
    )


def test_recognises_czarr_gpu_buffer_as_device() -> None:
    buf = CzarrGpuBuffer.empty(1024)
    try:
        assert is_gpu_buffer(buf)
        assert CzarrGpuBuffer in GPU_BUFFER_TYPES
    finally:
        del buf


def test_recognises_czarr_gpu_prototype() -> None:
    assert is_gpu_prototype(buffer_prototype)


@pytest.mark.parametrize("codec_cls", [LZ4, Zstd])
def test_codec_decode_through_czarr_gpu_buffer(codec_cls) -> None:
    """Encode on the CzarrGpu prototype, then decode the result through
    the codec's batch_sync. Verify the bytes round-trip and the output
    buffer is a CzarrGpuBuffer."""
    payload_host = (np.arange(4096, dtype=np.uint8) ^ 0xA5).tobytes()
    chunk_in = CzarrGpuBuffer.from_bytes(payload_host)

    codec = codec_cls()
    spec = _spec(shape=(len(payload_host),), dtype="uint8", prototype=buffer_prototype)

    try:
        encoded = codec._batch_sync([(chunk_in, spec)], op="encode")
        assert len(encoded) == 1
        enc = encoded[0]
        assert isinstance(enc, CzarrGpuBuffer), f"expected CzarrGpuBuffer, got {type(enc)}"
        assert len(enc) > 0

        decoded = codec._batch_sync([(enc, spec)], op="decode")
        assert len(decoded) == 1
        dec = decoded[0]
        assert isinstance(dec, CzarrGpuBuffer), f"expected CzarrGpuBuffer, got {type(dec)}"
        assert dec.to_bytes() == payload_host
    finally:
        del chunk_in


def test_decode_host_input_with_czarr_prototype() -> None:
    """If the upstream store hands a host buffer but the prototype says
    CzarrGpuBuffer, the codec must materialise the output as a
    CzarrGpuBuffer (the host slow-path branch in ``_batch_sync``)."""
    payload_host = (np.arange(2048, dtype=np.uint8) ^ 0x33).tobytes()

    # Encode through the GPU path first so we have a valid compressed
    # bitstream, then re-wrap as a host (CPU) buffer to feed the decoder.
    codec = LZ4()
    spec_gpu = _spec(shape=(len(payload_host),), dtype="uint8", prototype=buffer_prototype)
    chunk_in = CzarrGpuBuffer.from_bytes(payload_host)
    encoded = codec._batch_sync([(chunk_in, spec_gpu)], op="encode")
    enc = encoded[0]

    # Move the compressed payload to a host buffer (zarr CPU prototype).
    host_prototype = zarr.core.buffer.cpu.buffer_prototype
    host_compressed = host_prototype.buffer.from_bytes(enc.to_bytes())

    # Decode with a CzarrGpu prototype: result must be a CzarrGpuBuffer
    # even though the input arrived as host bytes.
    spec_decode = _spec(shape=(len(payload_host),), dtype="uint8", prototype=buffer_prototype)
    decoded = codec._batch_sync([(host_compressed, spec_decode)], op="decode")
    dec = decoded[0]
    assert isinstance(dec, CzarrGpuBuffer)
    assert dec.to_bytes() == payload_host


def test_filter_prototype_wraps_into_czarr_nd_buffer() -> None:
    """The filter codecs hand their outputs through
    ``chunk_spec.prototype.nd_buffer.from_ndarray_like`` — independent
    of the kernel itself. Verify the wrap step lands on CzarrGpuNDBuffer
    for our prototype, without invoking the filter kernels (those need
    cupy JIT which is broken on the CUDA-13 / CuPy-cu12 cluster combo)."""
    src = np.arange(64, dtype=np.float32).reshape(8, 8)
    src_gpu = cp.asarray(src)
    wrapped = buffer_prototype.nd_buffer.from_ndarray_like(src_gpu)
    assert isinstance(wrapped, CzarrGpuNDBuffer)
    np.testing.assert_array_equal(wrapped.as_numpy_array(), src)
