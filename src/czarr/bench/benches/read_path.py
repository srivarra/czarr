"""Read-path A/B/C: storage->GPU transfer only (no decode). The foundational test.

Does cuFile/GDS actually beat reading into a pinned host buffer + H2D copy (and a
pageable floor) on our storage?  Decode is held out entirely — this isolates the
transfer mechanism, which the literature and our own benches both flag as the
unresolved crux (GDS clearly wins on big reads + real nvidia-fs, ties/loses on
NFS and small reads).

Three arms (param ``impl``), all landing identical device bytes:
* ``gds``      — cuFile ``read_into_many`` straight into a registered device buffer.
* ``bounce``   — threaded read into a pinned host buffer + one H2D copy.
* ``pageable`` — same but pageable host buffer (the floor).

Run on H100/H200 for real GDS; on compat-mode GPUs (A40/A100) the ``gds`` arm may
fall back to a POSIX bounce internally or assert (use --isolate). ``gds_available``
is recorded per row.
"""

from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import cupyx
import numpy as np

from czarr import cufile
from czarr.bench.context import BenchContext
from czarr.bench.fixtures import real_fs_tmpdir
from czarr.bench.registry import benchmark


def _write_raw(chunk_mib: int, n_chunks: int) -> tuple[list[str], list[int], bytes]:
    """Write N equal-size raw files on a real FS; return (paths, sizes, first-4KiB ref)."""
    d = real_fs_tmpdir() / f"rawread_{chunk_mib}mib_{n_chunks}c"
    d.mkdir(parents=True, exist_ok=True)
    size = chunk_mib << 20
    block = np.random.default_rng(0).integers(0, 256, size, dtype=np.uint8)
    raw = block.tobytes()
    paths, sizes = [], []
    for i in range(n_chunks):
        p = d / f"chunk_{i:04d}.bin"
        if not (p.exists() and p.stat().st_size == size):
            p.write_bytes(raw)
        paths.append(str(p))
        sizes.append(size)
    return paths, sizes, raw[:4096]


def _threaded_readinto(paths: list[str], host, offsets: list[int], sizes: list[int]) -> None:
    """Parallel unbuffered file reads into slices of one host buffer (mirrors read_into_many)."""
    mv = memoryview(host)

    def rd(i: int) -> None:
        with open(paths[i], "rb", buffering=0) as fh:
            fh.readinto(mv[offsets[i] : offsets[i] + sizes[i]])

    with ThreadPoolExecutor(max_workers=min(32, len(paths))) as ex:
        list(ex.map(rd, range(len(paths))))


@benchmark(
    "read-path",
    params={"chunk_mib": 64, "n_chunks": 16, "impl": "gds"},
    sweep={"impl": ["gds", "bounce", "pageable"], "chunk_mib": [8, 64, 256]},
)
def read_path(ctx: BenchContext):
    """Storage->GPU transfer only: gds (cuFile) vs bounce (pinned+H2D) vs pageable."""
    chunk_mib, n_chunks, impl = ctx.p("chunk_mib"), ctx.p("n_chunks"), ctx.p("impl")
    paths, sizes, ref0 = _write_raw(chunk_mib, n_chunks)
    total = sum(sizes)
    offsets, off = [], 0
    for s in sizes:
        offsets.append(off)
        off += s
    dev = cp.empty(total, dtype=cp.uint8)
    regime = {"chunk_mib": chunk_mib, "n_chunks": n_chunks, "impl": impl, "gds_available": False}

    if impl == "gds":
        cufile.ensure_driver_open()
        regime["gds_available"] = bool(cufile.is_available())
        base = int(dev.data.ptr)
        cufile.ensure_buf_registered(base, total)

        def body():
            cufile.read_into_many([(paths[i], base + offsets[i], sizes[i], 0) for i in range(n_chunks)])
            ctx.sync()
            return dev
    elif impl == "bounce":
        host = cupyx.empty_pinned(total, dtype=np.uint8)

        def body():
            _threaded_readinto(paths, host, offsets, sizes)
            dev.set(host)
            ctx.sync()
            return dev
    else:  # pageable
        host = np.empty(total, dtype=np.uint8)

        def body():
            _threaded_readinto(paths, host, offsets, sizes)
            dev.set(host)
            ctx.sync()
            return dev

    def verify(out) -> bool:
        return bytes(cp.asnumpy(out[:4096])) == ref0

    return ctx.plan(body=body, nbytes=total, verify=verify, regime=regime)
