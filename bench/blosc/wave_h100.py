"""Wave-pipeline experiment: where does e2e blosc-read time go, and does a
register-once batched read ("wave") beat the per-chunk path? — pure Python.

Measures, at a configurable chunk size (default 256 MiB to match waveorder):
  - decode-only : native batched decode on pre-read buffers
  - czarr e2e   : full zarr pipeline read (arr[:])
  - wave e2e    : read ALL chunks into one register-once buffer via
                  read_into_many, then one native batched decode
vs CPU (numcodecs blosc + H2D). Settles the read-vs-decode fork and whether
the wave (register-once batched read) is worth a CzarrPipeline. No C++.

Run: module load cuda && uv run --extra cu12 python -m bench.blosc.wave_h100 [--chunk-z 128]
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import time
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import LocalStore

import czarr
from czarr.codecs._native.blosc_nvcomp import decode_blosc_batch
from czarr.storage import cufile_runtime

REPS, WARMUP = 5, 2


def _to_np(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def _med_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
        cp.cuda.runtime.deviceSynchronize()
    ts = []
    for _ in range(REPS):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.runtime.deviceSynchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-z", type=int, default=128, help="Z per chunk; 128 -> 256 MiB f16 chunk")
    p.add_argument("--nz", type=int, default=512, help="total Z (nz/chunk_z chunks)")
    args = p.parse_args()
    shape = (args.nz, 1024, 1024)
    chunks = (args.chunk_z, 1024, 1024)
    nbytes = int(np.prod(shape)) * 2
    nchunks = -(-shape[0] // chunks[0])
    chunk_mib = chunks[0] * 1024 * 1024 * 2 / 2**20
    print(f"shape={shape} chunks={chunks}  {nbytes / 2**20:.0f}MiB / {nchunks} chunks of {chunk_mib:.0f}MiB")

    base = os.environ.get("TMPDIR", "/tmp")
    path = Path(base) / f"wave_cz{args.chunk_z}.zarr"
    arr = zarr.create_array(
        store=LocalStore(path),
        shape=shape,
        chunks=chunks,
        dtype="float16",
        compressors=[BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.bitshuffle, typesize=2, blocksize=32768)],
        overwrite=True,
    )
    rng = np.random.default_rng(0)
    bb = np.linspace(0, 1, int(np.prod(shape)), dtype="float32").reshape(shape)
    arr[:] = (bb + rng.standard_normal(shape).astype("float32") * 0.01).astype("float16")
    del arr

    # CPU ref (before configure_gpu)
    ref = _to_np(zarr.open_array(store=LocalStore(path), mode="r")[:])

    def cpu():
        h = np.asarray(zarr.open_array(store=LocalStore(path), mode="r")[:])
        return cp.asarray(h)

    cpu_ms = _med_ms(cpu)

    czarr.configure_gpu()
    store = czarr.GPULocalStore(path, read_only=True)
    print(f"GDS available: {getattr(store, 'gds_available', '?')}")

    # enumerate chunk files (leaf files under c/, sorted by chunk index)
    files = sorted(glob.glob(str(path / "c" / "**"), recursive=True))
    files = [f for f in files if os.path.isfile(f) and not f.endswith("zarr.json")]
    sizes = [os.path.getsize(f) for f in files]
    assert len(files) == nchunks, f"{len(files)} files != {nchunks} chunks"

    # --- czarr full e2e (the real pipeline) --- guarded: GPULocalStore large-chunk
    # read is suspect (dex ve8x3mkv); don't let it abort the wave isolation.
    arr_g = zarr.open_array(store=store, mode="r")
    try:
        out = arr_g[:]
        e2e_ok = isinstance(out, cp.ndarray) and np.array_equal(_to_np(out), ref)
        print(f"e2e (GPULocalStore) correctness: {e2e_ok}")
        e2e_ms = _med_ms(lambda: arr_g[:]) if e2e_ok else float("nan")
    except Exception as exc:
        print(f"e2e (GPULocalStore) FAILED: {type(exc).__name__}: {exc}")
        e2e_ms = float("nan")

    # --- decode-only on pre-read buffers (isolates decode from read+orchestration) ---
    comps_pre = [cp.asarray(np.frombuffer(open(f, "rb").read(), dtype=np.uint8)) for f in files]
    decode_ms = _med_ms(lambda: decode_blosc_batch(comps_pre))

    # --- WAVE: read all chunks into ONE register-once buffer + batched decode ---
    total = sum(sizes)
    offsets, off = [], 0
    for s in sizes:
        offsets.append(off)
        off += s
    wave_buf = cp.empty(total, dtype=cp.uint8)
    base_ptr = int(wave_buf.data.ptr)
    cufile_runtime.ensure_buf_registered(base_ptr, total)

    def wave():
        reqs = [(files[i], base_ptr + offsets[i], sizes[i], 0) for i in range(nchunks)]
        cufile_runtime.read_into_many(reqs)
        comps = [wave_buf[offsets[i] : offsets[i] + sizes[i]] for i in range(nchunks)]
        return decode_blosc_batch(comps)

    wave_out = wave()
    cp.cuda.runtime.deviceSynchronize()
    # validate wave path bit-exact, per-chunk vs numcodecs (isolates read_into_many
    # correctness for large chunks from the GPULocalStore path)
    from numcodecs import Blosc as _NB

    wave_ok = all(
        np.array_equal(
            cp.asnumpy(wave_out[i]), np.frombuffer(_NB().decode(open(files[i], "rb").read()), dtype=np.uint8)
        )
        for i in range(min(nchunks, 4))
    )
    print(f"WAVE (read_into_many) correctness: {wave_ok}")
    wave_ms = _med_ms(wave) if wave_ok else float("nan")
    cufile_runtime.deregister_buf(base_ptr)

    def gibs(ms):
        return (nbytes / 2**30) / (ms / 1e3)

    print(f"\n{'=' * 64}\n{nbytes / 2**20:.0f}MiB, {nchunks}x {chunk_mib:.0f}MiB chunks, {REPS} reps:")
    print(f"  decode-only (native batched) : {decode_ms:7.1f} ms   ({gibs(decode_ms):.1f} GiB/s)")
    print(f"  czarr e2e   (full pipeline)  : {e2e_ms:7.1f} ms   ({gibs(e2e_ms):.1f} GiB/s)")
    print(f"  WAVE e2e    (reg-once + dec) : {wave_ms:7.1f} ms   ({gibs(wave_ms):.1f} GiB/s)")
    print(f"  CPU (blosc + H2D)            : {cpu_ms:7.1f} ms   ({gibs(cpu_ms):.1f} GiB/s)")
    print(
        f"\n  e2e vs CPU: {cpu_ms / e2e_ms:.2f}x   |  WAVE vs e2e: {e2e_ms / wave_ms:.2f}x   "
        f"|  decode/e2e: {decode_ms / e2e_ms:.0%} (rest = read+orchestration)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
