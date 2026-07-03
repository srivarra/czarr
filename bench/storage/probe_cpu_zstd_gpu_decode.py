"""Probe: can nvCOMP decode zarr chunks written by numcodecs.Zstd (CPU)?

Step 1: write a zarr array on CPU with numcodecs.Zstd compressor.
Step 2: read raw on-disk chunk bytes, push to GPU.
Step 3: decode with nvCOMP Zstd.
Step 4: compare to source.

Also: try opening the same store through zarr with czarr's Zstd
registered under "zstd" to see if the swap works transparently.
"""

import shutil
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from nvidia import nvcomp
from zarr.codecs import ZstdCodec


def main() -> int:
    base = Path("/hpc/mydata/sricharan.varra/czarr_cpu_zstd_probe")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    path = base / "cpu_written.zarr"

    rng = np.random.default_rng(0)
    src = rng.integers(0, 64, (16, 1, 8, 256, 256), dtype=np.int32).astype(np.float32)

    print("=== STEP 1: write zarr with numcodecs.Zstd (pure CPU path) ===")
    store = zarr.storage.LocalStore(path)
    arr = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(1, 1, 8, 256, 256),  # 2 MiB per chunk
        dtype="float32",
        compressors=[ZstdCodec(level=3)],
    )
    arr[:] = src
    print(f"wrote {src.nbytes / 1e6:.1f} MB source to {path}")
    print(f"compressors meta: {arr.metadata.codecs}")

    print("\n=== STEP 2: inspect on-disk chunks ===")
    chunks_root = path / "c"
    chunk_files = sorted(chunks_root.rglob("*"))
    chunk_files = [p for p in chunk_files if p.is_file()]
    print(
        f"found {len(chunk_files)} chunk files; total on-disk {sum(p.stat().st_size for p in chunk_files) / 1e6:.2f} MB"
    )
    print(f"first chunk: {chunk_files[0]} ({chunk_files[0].stat().st_size} bytes)")
    head = chunk_files[0].read_bytes()[:8]
    print(f"first 8 bytes hex: {head.hex()}  (zstd magic = 28 b5 2f fd)")

    print("\n=== STEP 3: decode one chunk on GPU via nvCOMP Zstd (RAW bitstream) ===")
    nv_codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW)
    chunk0_bytes = chunk_files[0].read_bytes()
    chunk0_host = np.frombuffer(chunk0_bytes, dtype=np.uint8)
    nv_in = nvcomp.as_array(chunk0_host).cuda()  # H2D
    nv_out = nv_codec.decode([nv_in])[0]
    decoded_gpu = cp.asarray(nv_out).view(cp.uint8)
    print(f"nvCOMP returned {decoded_gpu.nbytes} decompressed bytes")

    expected_chunk0 = src[0:1, 0:1, :, :, :].tobytes()
    decoded_host = decoded_gpu.get().tobytes()
    match = decoded_host == expected_chunk0
    print(f"chunk 0 content match: {match}  (expected {len(expected_chunk0)} bytes, got {len(decoded_host)})")
    if not match:
        print(f"  first 16 bytes expected: {expected_chunk0[:16].hex()}")
        print(f"  first 16 bytes actual:   {decoded_host[:16].hex()}")

    print("\n=== STEP 4: batch decode all chunks on GPU ===")
    all_compressed = [np.frombuffer(p.read_bytes(), dtype=np.uint8) for p in chunk_files]
    nv_inputs = [nvcomp.as_array(b).cuda() for b in all_compressed]
    decoded_arrays = nv_codec.decode(nv_inputs)
    print(f"batched: decoded {len(decoded_arrays)} chunks")
    all_match = True
    for path_i, nv_out_i in zip(chunk_files, decoded_arrays, strict=False):
        chunk_idx = path_i.relative_to(chunks_root).parts
        out_bytes = cp.asarray(nv_out_i).view(cp.uint8).get().tobytes()
        try:
            t = int(chunk_idx[0])
            c = int(chunk_idx[1])
            z = int(chunk_idx[2])
            expected = src[t : t + 1, c : c + 1, :, :, :].tobytes()
        except Exception:
            continue
        if out_bytes != expected:
            print(f"  MISMATCH at chunk {chunk_idx}")
            all_match = False
    print(f"batch decode all match: {all_match}")

    print("\n=== STEP 5: try opening through zarr with czarr's Zstd ===")
    print("  zarr's metadata says codec id 'zstd' (numcodecs name).")
    print("  czarr registers as 'czarr.nvcomp_zstd' — different name.")
    print("  zarr will NOT auto-swap.  Confirmed by trying reopen:")
    try:
        arr_ro = zarr.open_array(store=zarr.storage.LocalStore(path), mode="r")
        out = arr_ro[:]
        print(f"  zarr.open + arr[:] returned {type(out).__name__} (CPU path — numcodecs decoded)")
        print(f"  array_equal(out, src): {np.array_equal(out, src)}")
    except Exception as e:
        print(f"  raised: {type(e).__name__}: {e}")

    return 0 if match and all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
