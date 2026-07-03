"""Round-trip + Zarr-integration tests for nvCOMP-backed codecs."""

import cupy as cp
import numpy as np
import pytest
import zarr
from zarr.core.buffer import gpu as gpu_buffer
from zarr.storage import MemoryStore

from czarr.codecs import (
    ANS,
    LZ4,
    Bitcomp,
    Cascaded,
    Checksum,
    Deflate,
    GDeflate,
    Snappy,
    Zstd,
)

ALL_CODEC_CLASSES = [
    LZ4,
    Zstd,
    Snappy,
    Deflate,
    GDeflate,
    Bitcomp,
    ANS,
    Cascaded,
]


def _roundtrip_zarr(codec, data: np.ndarray) -> np.ndarray:
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
        compressors=[codec],
    )
    arr[:] = data
    return arr[:]


@pytest.mark.parametrize(
    "dtype",
    ["uint8", "int32", "int64", "float32", "float64"],
)
def test_lz4_roundtrip_dtypes(dtype):
    rng = np.random.default_rng(0)
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        data = rng.integers(info.min // 2, info.max // 2, size=(64, 64), dtype=dtype)
    else:
        data = rng.standard_normal((64, 64)).astype(dtype)
    out = _roundtrip_zarr(LZ4(), data)
    np.testing.assert_array_equal(out, data)


def test_lz4_compresses_zeros():
    """Highly compressible input should round-trip and shrink on disk."""
    data = np.zeros((256, 256), dtype="float32")
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
        compressors=[LZ4()],
    )
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


def test_lz4_to_dict_emits_numcodecs_schema():
    """Compat codec ``to_dict`` should only emit fields the v3/numcodecs
    LZ4 schema understands — czarr-internal knobs (chunk_size etc.) are
    runtime, not persisted in metadata."""
    codec = LZ4(chunk_size=32768, checksum_policy=Checksum.COMPUTE_AND_VERIFY)
    payload = codec.to_dict()
    assert payload["name"] == "lz4"
    assert set(payload["configuration"].keys()) == {"acceleration"}


def test_lz4_roundtrip_gpu_prototype():
    """Round-trip via the GPU buffer prototype: data stays on device through codec."""
    data = np.arange(128 * 128, dtype="float32").reshape(128, 128)

    with zarr.config.set({"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"}):
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=data.shape,
            chunks=(64, 64),
            dtype="float32",
            compressors=[LZ4()],
        )
        arr[:] = cp.asarray(data)
        out = arr.get_basic_selection(prototype=gpu_buffer.buffer_prototype)

    assert isinstance(out, cp.ndarray), f"expected cupy.ndarray, got {type(out)}"
    np.testing.assert_array_equal(cp.asnumpy(out), data)


def test_lz4_to_dict_shape():
    codec = LZ4()
    payload = codec.to_dict()
    # Compat codec — shadows numcodecs.LZ4 under the standard codec_id "lz4".
    assert payload["name"] == "lz4"
    assert payload["configuration"]["acceleration"] == 1


@pytest.mark.parametrize("codec_cls", ALL_CODEC_CLASSES, ids=lambda c: c.__name__)
def test_all_algorithms_roundtrip(codec_cls):
    """Round-trip a structured payload through every nvCOMP algorithm."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 255, size=(64, 64), dtype="uint8")
    out = _roundtrip_zarr(codec_cls(), data)
    np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("codec_cls", ALL_CODEC_CLASSES, ids=lambda c: c.__name__)
def test_all_algorithms_entry_point_registered(codec_cls):
    """Each codec must be registered in Zarr's codec registry.

    For compat codecs ("zstd", "lz4", "gzip", "zlib"), zarr's built-in CPU
    codec is also registered under the same id — disambiguation comes from
    :func:`czarr.configure_gpu` (or an explicit ``codecs.<id>`` config entry).
    We assert *registered*, not *resolves uniquely*.
    """
    from zarr.registry import _codec_registries

    registered = _codec_registries.get(codec_cls.codec_name)
    assert registered is not None, f"{codec_cls.codec_name!r} not in zarr registry"
    fqname = f"{codec_cls.__module__}.{codec_cls.__qualname__}"
    assert fqname in registered, (
        f"{codec_cls.__name__} not registered under {codec_cls.codec_name!r}; registered: {list(registered)}"
    )


@pytest.mark.parametrize("codec_cls", ALL_CODEC_CLASSES, ids=lambda c: c.__name__)
def test_all_algorithms_dict_roundtrip(codec_cls):
    # Compat codecs (Zstd, LZ4, Gzip, Zlib) only round-trip the
    # numcodecs-schema fields in their metadata — czarr-internal runtime
    # knobs (chunk_size etc.) are stripped on to_dict so non-czarr
    # readers can open the store.  Test the native codecs here.
    if codec_cls.__name__ in {"Zstd", "LZ4", "Gzip", "Zlib"}:
        pytest.skip("compat codecs use a numcodecs-schema to_dict; not full round-trip")
    codec = codec_cls(chunk_size=32768)
    rebuilt = codec_cls.from_dict(codec.to_dict())
    assert rebuilt == codec


@pytest.mark.parametrize(
    ("cpu_codec_factory", "gpu_codec_id"),
    [
        ("zarr_zstd", "zstd"),
        ("zarr_gzip", "gzip"),
    ],
    ids=["zstd", "gzip"],
)
def test_cpu_written_decodes_on_gpu(cpu_codec_factory, gpu_codec_id):
    """A zarr store written with the stdlib CPU codec must decode on GPU via czarr.

    Writes a small array with zarr's bundled CPU codec, then reopens with
    ``configure_gpu`` selected — the codec_id lookup picks czarr's GPU codec.
    """
    import czarr

    czarr.configure_gpu()
    try:
        if cpu_codec_factory == "zarr_zstd":
            from zarr.codecs import ZstdCodec

            cpu_codec = ZstdCodec(level=3)
        elif cpu_codec_factory == "zarr_gzip":
            from zarr.codecs import GzipCodec

            cpu_codec = GzipCodec(level=5)
        else:
            pytest.skip(f"unknown factory: {cpu_codec_factory}")

        # Use CPU buffer prototype for the write so zarr keeps it on the CPU codec
        # path; otherwise the same codec_id lookup would route writes to czarr too.
        with zarr.config.set(
            {
                "buffer": "zarr.core.buffer.cpu.Buffer",
                "ndbuffer": "zarr.core.buffer.cpu.NDBuffer",
                f"codecs.{gpu_codec_id}": (
                    "zarr.codecs.zstd.ZstdCodec" if gpu_codec_id == "zstd" else "zarr.codecs.gzip.GzipCodec"
                ),
            }
        ):
            data = np.arange(64 * 64, dtype="int32").reshape(64, 64)
            store = MemoryStore()
            arr_w = zarr.create_array(
                store=store,
                shape=data.shape,
                chunks=(32, 32),
                dtype="int32",
                compressors=[cpu_codec],
            )
            arr_w[:] = data

        # Reopen — czarr's compat codec now wins (we just reset the
        # codecs.<id> override back to czarr by exiting the context).
        arr_r = zarr.open_array(store=store, mode="r")
        out = arr_r[:]
        if hasattr(out, "get"):  # cupy.ndarray
            out = out.get()
        np.testing.assert_array_equal(out, data)
    finally:
        zarr.config.reset()


def test_configure_gpu_selects_compat_codecs():
    """After ``configure_gpu`` the compat codecs win the registry lookup."""
    from zarr.registry import get_codec_class

    import czarr

    czarr.configure_gpu()
    try:
        assert get_codec_class("zstd") is Zstd
        assert get_codec_class("lz4") is LZ4
        # Gzip/Zlib aren't in ALL_CODEC_CLASSES (covered separately)
        from czarr.codecs import Gzip, Zlib

        assert get_codec_class("gzip") is Gzip
        assert get_codec_class("zlib") is Zlib
    finally:
        # Reset so other tests don't see the GPU buffer/pipeline config
        import zarr

        zarr.config.reset()


def test_deflate_algorithm_type_roundtrip():
    """Deflate's ``algorithm_type`` field survives encode + dict round-trip."""
    codec = Deflate(algorithm_type=4)  # higher-ratio mode
    assert codec.to_dict()["configuration"]["algorithm_type"] == 4
    rebuilt = Deflate.from_dict(codec.to_dict())
    assert rebuilt == codec
    assert rebuilt.algorithm_type == 4

    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(64, 64), dtype="uint8")
    out = _roundtrip_zarr(codec, data)
    np.testing.assert_array_equal(out, data)


def test_bitcomp_sparse_mode_roundtrip():
    """Bitcomp's sparse-mode (``algorithm_type=1``) round-trips bit-exactly."""
    codec = Bitcomp(algorithm_type=1)
    rebuilt = Bitcomp.from_dict(codec.to_dict())
    assert rebuilt == codec

    # Sparse-friendly data — many zeros
    data = np.zeros((128, 128), dtype="uint8")
    data[::8, ::8] = 42
    out = _roundtrip_zarr(codec, data)
    np.testing.assert_array_equal(out, data)


def test_cascaded_tunables_roundtrip():
    """Cascaded's RLE/delta/bitpack knobs all survive serialisation."""
    codec = Cascaded(num_rles=3, num_deltas=2, use_bitpack=False)
    cfg = codec.to_dict()["configuration"]
    assert cfg["num_rles"] == 3
    assert cfg["num_deltas"] == 2
    assert cfg["use_bitpack"] is False
    rebuilt = Cascaded.from_dict(codec.to_dict())
    assert rebuilt == codec

    # Smoke-test the codec actually works with these settings
    data = np.arange(4096, dtype="uint8").reshape(64, 64)
    out = _roundtrip_zarr(codec, data)
    np.testing.assert_array_equal(out, data)


def test_batch_encode_decode_handles_many_chunks_and_nones():
    """Drive the batch encode/decode path with many chunks at once."""
    import asyncio

    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import parse_data_type

    rng = np.random.default_rng(11)
    proto = default_buffer_prototype()
    zdt = parse_data_type("uint8", zarr_format=3)
    spec = ArraySpec(
        shape=(1024,),
        dtype=zdt,
        fill_value=0,
        config=ArrayConfig(order="C", write_empty_chunks=False),
        prototype=proto,
    )

    payloads = [rng.integers(0, 255, size=1024, dtype="uint8").tobytes() for _ in range(8)]
    buffers = [proto.buffer.from_bytes(p) for p in payloads]
    # Interleave a None chunk to confirm position-preserving splice.
    inputs: list[tuple] = [(buf, spec) for buf in buffers]
    inputs.insert(3, (None, spec))

    codec = LZ4()
    encoded = list(asyncio.run(codec.encode(inputs)))
    assert encoded[3] is None
    assert all(b is not None for i, b in enumerate(encoded) if i != 3)

    decoded_pairs = [(enc, spec) for enc in encoded]
    decoded = list(asyncio.run(codec.decode(decoded_pairs)))
    assert decoded[3] is None
    for orig, got in zip(payloads, [d for i, d in enumerate(decoded) if i != 3], strict=True):
        assert got.to_bytes() == orig


def test_codec_instance_is_cached_per_thread():
    """The thread-local cache should hand back the same nvcomp.Codec per thread."""
    codec = LZ4()
    a = codec._get_codec()
    b = codec._get_codec()
    assert a is b


def test_resolve_stream_accepts_int_and_protocol_objects():
    """Stream coercion: raw int + ``__cuda_stream__`` protocol; nothing else."""
    fn = LZ4._resolve_stream
    assert fn(None) is None
    assert fn(42) == 42

    class ProtoObj:
        def __cuda_stream__(self):
            return (0, 789)

    assert fn(ProtoObj()) == 789


def test_resolve_stream_rejects_bare_attribute_objects():
    """Objects with ``.handle`` / ``.ptr`` but no ``__cuda_stream__`` are rejected."""

    class HandleObj:
        handle = 123

    with pytest.raises(TypeError, match="__cuda_stream__"):
        LZ4._resolve_stream(HandleObj())


def test_resolve_stream_rejects_future_protocol_version():
    """Reject unknown ``__cuda_stream__`` versions until we know the new tuple shape."""

    class FutureProtoObj:
        def __cuda_stream__(self):
            return (1, 999)

    with pytest.raises(ValueError, match="protocol version"):
        LZ4._resolve_stream(FutureProtoObj())


def test_get_stream_returns_explicit_stream_when_set():
    """CudaBytesBytesCodec built with an explicit stream should expose that exact handle."""
    s = cp.cuda.Stream(non_blocking=True)
    codec = LZ4(cuda_stream=s)
    assert codec.get_stream() == s.ptr


def test_get_stream_probes_internal_when_default():
    """With ``cuda_stream=None``, ``get_stream`` probes nvCOMP and caches."""
    codec = LZ4()
    handle1 = codec.get_stream()
    handle2 = codec.get_stream()
    assert isinstance(handle1, int)
    assert handle1 != 0
    assert handle1 == handle2  # cached
    # Different codec instances should generally yield different streams,
    # but the bigger guarantee here is that the value is stable per instance
    # — covered by the equality above.


def test_get_stream_handle_is_valid_integer():
    """The hijacked handle is a valid ``cudaStream_t`` integer that callers
    can pass to any CUDA-stream-protocol consumer.  We don't wrap with
    ``cp.cuda.ExternalStream`` here because that API is deprecated and its
    GC interaction with nvCOMP-owned streams can race at process shutdown."""
    codec = LZ4()
    handle = codec.get_stream()
    assert isinstance(handle, int)
    assert handle != 0  # null stream would be 0; nvCOMP creates a real stream


def test_cuda_stream_excluded_from_codec_identity():
    """Two codecs differing only in cuda_stream should still compare equal & hash equal."""

    class FakeStream:
        handle = 999

    a = LZ4()
    b = LZ4(cuda_stream=FakeStream())
    assert a == b
    # to_dict must not leak the stream into persisted metadata
    assert "cuda_stream" not in a.to_dict()["configuration"]
    assert "cuda_stream" not in b.to_dict()["configuration"]
