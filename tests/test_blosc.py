"""GPU blosc decode (czarr.codecs.compressors.Blosc) — native nvCOMP batched path."""

import struct
import tempfile
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
import zarr
from numcodecs import Blosc as NumcodecsBlosc
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import LocalStore

import czarr

_REAL_CHUNK = "/hpc/projects/waveorder/tile-stitch/sample_datasets/l0_brightfield_fov.zarr/0/c/0/0/9/0/0"


def _to_np(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


@pytest.mark.parametrize(
    ("shuffle", "dtype", "typesize"),
    [
        (BloscShuffle.bitshuffle, "float16", 2),
        (BloscShuffle.shuffle, "float16", 2),
        (BloscShuffle.noshuffle, "float16", 2),
        (BloscShuffle.bitshuffle, "uint32", 4),
    ],
)
def test_blosc_gpu_decode_matches_cpu(shuffle, dtype, typesize):
    """A CPU blosc(zstd)-written store decodes bit-exact on the GPU via czarr."""
    rng = np.random.default_rng(0)
    src = rng.integers(0, 1000, size=(4, 256, 256)).astype(dtype)  # 131072 B/chunk, 4x 32KiB blocks
    d = tempfile.mkdtemp()

    # write with plain zarr CPU blosc (configure_gpu NOT yet active)
    arr = zarr.create_array(
        store=LocalStore(d),
        shape=src.shape,
        chunks=(1, 256, 256),
        dtype=dtype,
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle=shuffle, typesize=typesize, blocksize=32768)],
        overwrite=True,
    )
    arr[:] = src
    cpu = _to_np(zarr.open_array(store=LocalStore(d), mode="r")[:])

    # read with czarr GPU decode
    czarr.configure_gpu()
    out = zarr.open_array(store=LocalStore(d), mode="r")[:]
    assert isinstance(out, cp.ndarray), "expected a cupy array under configure_gpu"
    np.testing.assert_array_equal(_to_np(out), cpu)
    np.testing.assert_array_equal(_to_np(out), src)


def test_blosc_encode_rejected():
    """Encode is unsupported — write [Shuffle, Zstd] for GPU-decodable output."""
    import asyncio

    with pytest.raises(NotImplementedError, match="decode-only"):
        asyncio.run(czarr.Blosc().encode([]))


@pytest.mark.skipif(not Path(_REAL_CHUNK).exists(), reason="waveorder dataset not present")
def test_engine_real_chunk_bit_exact():
    """The native engine decodes a real 256 MiB / 8192-block chunk bit-exact."""
    from czarr.lowlevel.blosc import decode_blosc_batch

    buf = Path(_REAL_CHUNK).read_bytes()
    nbytes, _blocksize, _ = struct.unpack_from("<iii", buf, 4)
    comp = cp.asarray(np.frombuffer(buf, dtype=np.uint8))
    (decoded,) = decode_blosc_batch([comp])
    truth = np.frombuffer(NumcodecsBlosc().decode(buf), dtype=np.uint8)
    assert decoded.size == nbytes
    np.testing.assert_array_equal(cp.asnumpy(decoded), truth)


@pytest.mark.skipif(not Path(_REAL_CHUNK).exists(), reason="waveorder dataset not present")
def test_engine_multichunk_batch():
    """Cross-chunk batched decode stays bit-exact across several real chunks."""
    from czarr.lowlevel.blosc import decode_blosc_batch

    files = sorted(Path(_REAL_CHUNK).parent.glob("*"))[:4]
    bufs = [f.read_bytes() for f in files]
    comps = [cp.asarray(np.frombuffer(b, dtype=np.uint8)) for b in bufs]
    decoded = decode_blosc_batch(comps)
    for dev, buf in zip(decoded, bufs, strict=True):
        truth = np.frombuffer(NumcodecsBlosc().decode(buf), dtype=np.uint8)
        np.testing.assert_array_equal(cp.asnumpy(dev), truth)
