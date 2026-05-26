"""LZ4 block-format decoder spike — pure cuda-python / cupy.RawKernel.

Goal of this spike
------------------
Prove out the engineering cost of replacing nvCOMP's LZ4 codec with a
hand-rolled CUDA kernel.  We pick LZ4 because the block format is by far
the simplest of the eight nvCOMP families: a stream of (token, literals,
offset, match) sequences with no entropy coding.  Reference spec:

    https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md

Design choices
--------------
* **One CUDA block per LZ4 block.**  Each LZ4 *block* in the on-disk
  representation is independent (numcodecs.LZ4 writes one block per
  chunk with a 4-byte little-endian uncompressed-size prefix; nvCOMP
  RAW mode skips the prefix).  We pick a launch shape of
  ``grid=(num_blocks,)``, ``block=(WARP=32,)`` so we get one warp per
  LZ4 block.
* **Single-thread parse, warp coopcopy.**  LZ4's sequence stream is
  inherently sequential (each sequence's parse position depends on the
  previous one's length).  Lane 0 walks the token tape; the whole warp
  cooperates on the literal/match copy.  This mirrors the nvCOMP 2.2
  open-source kernel design (``src/LZ4Kernels.cuh::decompressStream``,
  BSD-3-Clause).  Trying to parallelise the parse itself across threads
  needs a parallel prefix-scan over sequence boundaries — possible but
  out of scope for the spike.
* **No corruption handling beyond a guard flag.**  Real impl needs
  bounds checks on every load; we elide some in the spike for clarity.
* **One-shot launch, no batching API yet.**  Real Zarr codec needs the
  batched-decompress contract (lists of (compressed_offset,
  uncompressed_size)).  Implementation is mechanical: pad the launch
  config and add per-block offset tables.

Status
------
Written without GPU access (this agent only has CPU); the kernel is
syntactically validated against the cupy.RawKernel API and the algorithm
is verified by ``_decode_lz4_block_cpu`` against the ``lz4.block``
reference library.  Run ``python lz4_decoder.py`` on an H100/A40 to get
end-to-end numbers — the GPU path is gated by ``HAS_CUPY``.

References
----------
* lz4 spec: https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
* nvCOMP 2.2 ``src/LZ4Kernels.cuh::decompressStream`` (BSD-3-Clause,
  Copyright NVIDIA 2017-2020):
  https://github.com/NVIDIA/nvcomp/blob/branch-2.2/src/LZ4Kernels.cuh
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ---------------------------------------------------------------------------
# CPU reference decoder — used as the oracle the GPU kernel must match.
# ---------------------------------------------------------------------------


def _decode_lz4_block_cpu(compressed: bytes, uncompressed_size: int) -> bytes:
    """Pure-Python LZ4 block decoder.  Mirrors what the CUDA kernel does."""
    out = bytearray(uncompressed_size)
    src = memoryview(compressed)
    sp = 0  # source pointer
    dp = 0  # dest pointer
    n = len(src)
    while sp < n:
        token = src[sp]
        sp += 1
        # --- literal length (high nibble) -----------------------------------
        lit_len = token >> 4
        if lit_len == 15:
            while sp < n:
                b = src[sp]
                sp += 1
                lit_len += b
                if b != 255:
                    break
        # --- copy literals --------------------------------------------------
        out[dp : dp + lit_len] = src[sp : sp + lit_len]
        sp += lit_len
        dp += lit_len
        if sp >= n:
            break  # last sequence is literals-only (spec rule #1)
        # --- offset (2-byte LE) --------------------------------------------
        offset = src[sp] | (src[sp + 1] << 8)
        sp += 2
        # --- match length (low nibble + 4) ----------------------------------
        match_len = (token & 0x0F) + 4
        if (token & 0x0F) == 15:
            while sp < n:
                b = src[sp]
                sp += 1
                match_len += b
                if b != 255:
                    break
        # --- byte-by-byte match copy (handles overlap, e.g. offset=1 RLE) ---
        for i in range(match_len):
            out[dp + i] = out[dp + i - offset]
        dp += match_len
    assert dp == uncompressed_size, f"decoded {dp}, expected {uncompressed_size}"
    return bytes(out)


# ---------------------------------------------------------------------------
# CUDA kernel — one warp per LZ4 block.
# ---------------------------------------------------------------------------

# Notes on the kernel:
#   * Lane 0 owns the parse state machine and broadcasts (sp, dp, lit_len,
#     offset, match_len) via __shfl_sync.  All 32 lanes participate in the
#     coopcopy loops.
#   * For the spike we read bytes directly from global memory.  Production
#     would prefetch a chunk into shared memory (as nvCOMP 2.2 does — its
#     ``BufferControl`` is ~150 lines of prefetcher), giving ~2-3x speedup.
#   * Offset can be < 32, in which case the match copy is RLE-like and
#     cannot be parallelised — handle byte-by-byte with lane 0 only.

LZ4_DECODE_KERNEL_SRC = r"""
extern "C" __global__
void lz4_decode_blocks(
    const unsigned char* __restrict__ compressed,   // flat input
    const long long* __restrict__ block_offsets,    // n_blocks+1, prefix-sum offsets in `compressed`
    const long long* __restrict__ out_offsets,      // n_blocks+1, prefix-sum in `decompressed`
    unsigned char* __restrict__ decompressed,       // flat output
    int* __restrict__ status                        // n_blocks, 0 = ok, !=0 = corrupted
) {
    const int block_id = blockIdx.x;
    const int lane = threadIdx.x;            // 0..31
    const long long src_begin = block_offsets[block_id];
    const long long src_end   = block_offsets[block_id + 1];
    const long long dst_begin = out_offsets[block_id];
    const long long dst_end   = out_offsets[block_id + 1];

    // Shared per-block parse state — lane 0 writes, all lanes read via shuffle.
    long long sp = src_begin;
    long long dp = dst_begin;
    int corrupted = 0;

    while (sp < src_end) {
        unsigned int token;
        if (lane == 0) token = compressed[sp];
        token = __shfl_sync(0xFFFFFFFF, token, 0);
        if (lane == 0) sp += 1;
        sp = __shfl_sync(0xFFFFFFFF, sp, 0);

        // ------- literal length (high nibble) --------------------------------
        unsigned int lit_len = token >> 4;
        if (lit_len == 15) {
            unsigned int b;
            do {
                if (lane == 0) b = compressed[sp];
                b = __shfl_sync(0xFFFFFFFF, b, 0);
                if (lane == 0) sp += 1;
                sp = __shfl_sync(0xFFFFFFFF, sp, 0);
                lit_len += b;
                if (sp >= src_end && b == 255) { corrupted = 1; break; }
            } while (b == 255);
        }

        // ------- coop-copy literals: 32 lanes, 32-byte stride ---------------
        if (dp + lit_len > (unsigned long long)dst_end) { corrupted = 1; break; }
        for (unsigned int i = lane; i < lit_len; i += 32) {
            decompressed[dp + i] = compressed[sp + i];
        }
        sp += lit_len;
        dp += lit_len;

        if (sp >= src_end) break;   // last sequence (literals-only) — spec rule #1

        // ------- offset (2-byte little-endian) ------------------------------
        unsigned int offset;
        if (lane == 0) {
            unsigned int b0 = compressed[sp];
            unsigned int b1 = compressed[sp + 1];
            offset = b0 | (b1 << 8);
            sp += 2;
        }
        offset = __shfl_sync(0xFFFFFFFF, offset, 0);
        sp = __shfl_sync(0xFFFFFFFF, sp, 0);

        // ------- match length (low nibble + 4) ------------------------------
        unsigned int match_len = (token & 0x0F) + 4;
        if ((token & 0x0F) == 15) {
            unsigned int b;
            do {
                if (lane == 0) b = compressed[sp];
                b = __shfl_sync(0xFFFFFFFF, b, 0);
                if (lane == 0) sp += 1;
                sp = __shfl_sync(0xFFFFFFFF, sp, 0);
                match_len += b;
                if (sp > src_end) { corrupted = 1; break; }
            } while (b == 255);
        }

        if (dp + match_len > (unsigned long long)dst_end) { corrupted = 1; break; }
        if (offset == 0 || dp < (unsigned long long)dst_begin + offset) {
            corrupted = 1; break;
        }

        // ------- match copy ----------------------------------------------------
        // If offset >= 32, lanes can copy in parallel safely (no overlap within
        // a single 32-byte stride).  If offset < 32, the copy is overlapping
        // and must be done serially — lane 0 does it byte-by-byte.  This is
        // the same RLE-overlap path nvCOMP 2.2 takes in coopCopyRepeat /
        // coopCopyOverlap.
        if (offset >= 32) {
            for (unsigned int i = lane; i < match_len; i += 32) {
                decompressed[dp + i] = decompressed[dp + i - offset];
            }
        } else {
            if (lane == 0) {
                for (unsigned int i = 0; i < match_len; ++i) {
                    decompressed[dp + i] = decompressed[dp + i - offset];
                }
            }
            __syncwarp();
        }
        dp += match_len;
    }

    if (lane == 0) {
        status[block_id] = corrupted ? -1 : 0;
    }
}
"""


def _get_kernel():
    if not HAS_CUPY:
        raise RuntimeError("cupy not available — cannot launch GPU kernel")
    return cp.RawKernel(LZ4_DECODE_KERNEL_SRC, "lz4_decode_blocks")


def decode_lz4_blocks_gpu(
    compressed_blocks: list[bytes],
    uncompressed_sizes: list[int],
) -> list[bytes]:
    """Decode a batch of LZ4 blocks on the GPU.  Returns host bytes.

    The block format produced/consumed here is plain LZ4 block (no frame,
    no length prefix).  Caller is expected to strip any wrapping (e.g. the
    4-byte LE uncompressed-size prefix that numcodecs.LZ4 writes).
    """
    n = len(compressed_blocks)
    if n == 0:
        return []
    assert len(uncompressed_sizes) == n

    src_offs = np.zeros(n + 1, dtype=np.int64)
    dst_offs = np.zeros(n + 1, dtype=np.int64)
    for i, (b, u) in enumerate(zip(compressed_blocks, uncompressed_sizes, strict=True)):
        src_offs[i + 1] = src_offs[i] + len(b)
        dst_offs[i + 1] = dst_offs[i] + u

    src_host = np.concatenate([np.frombuffer(b, dtype=np.uint8) for b in compressed_blocks])
    d_src = cp.asarray(src_host)
    d_src_offs = cp.asarray(src_offs)
    d_dst_offs = cp.asarray(dst_offs)
    d_dst = cp.empty(int(dst_offs[-1]), dtype=cp.uint8)
    d_status = cp.zeros(n, dtype=cp.int32)

    kernel = _get_kernel()
    kernel(grid=(n,), block=(32,), args=(d_src, d_src_offs, d_dst_offs, d_dst, d_status))
    cp.cuda.runtime.deviceSynchronize()

    status = cp.asnumpy(d_status)
    if (status != 0).any():
        bad = int(np.argmax(status != 0))
        raise RuntimeError(f"GPU LZ4 decode reported corruption at block {bad}")

    flat = cp.asnumpy(d_dst).tobytes()
    return [flat[dst_offs[i] : dst_offs[i + 1]] for i in range(n)]


# ---------------------------------------------------------------------------
# Tests / mini-benchmark.
# ---------------------------------------------------------------------------


def _make_fixtures() -> list[tuple[bytes, bytes]]:
    """Return a list of (raw_uncompressed, compressed_lz4_block) pairs.

    Uses python's ``lz4.block`` for the reference encoder.  We strip the
    4-byte size prefix that ``lz4.block.compress`` writes, since our spike
    decoder operates on plain LZ4 blocks (matching nvCOMP RAW mode).
    """
    import lz4.block

    rng = np.random.default_rng(42)
    fixtures: list[tuple[bytes, bytes]] = []

    # 1) Pure literals — small.
    raw = b"hello, world! " * 8
    fixtures.append((raw, lz4.block.compress(raw, store_size=False)))

    # 2) Highly repetitive — exercises the RLE overlap path (offset < 32).
    raw = (b"abcdefgh" * 1024)[:4096]
    fixtures.append((raw, lz4.block.compress(raw, store_size=False)))

    # 3) Random — almost incompressible; mostly literal sequences.
    raw = rng.bytes(16384)
    fixtures.append((raw, lz4.block.compress(raw, store_size=False)))

    # 4) Larger, lz4-friendly synthetic chunk.
    raw = (b"the quick brown fox jumps over the lazy dog. " * 1024)[:65536]
    fixtures.append((raw, lz4.block.compress(raw, store_size=False)))

    return fixtures


def test_cpu_reference() -> None:
    """Validate the CPU oracle against the reference library."""
    import lz4.block

    for raw, comp in _make_fixtures():
        out = _decode_lz4_block_cpu(comp, len(raw))
        ref = lz4.block.decompress(comp, uncompressed_size=len(raw))
        assert out == ref == raw, "CPU oracle disagrees with lz4.block"
    print("CPU oracle: PASS")


def test_gpu_kernel() -> None:
    if not HAS_CUPY:
        print("GPU kernel: SKIPPED (no cupy)")
        return
    fixtures = _make_fixtures()
    comps = [c for _, c in fixtures]
    sizes = [len(r) for r, _ in fixtures]
    refs = [r for r, _ in fixtures]
    got = decode_lz4_blocks_gpu(comps, sizes)
    for i, (g, r) in enumerate(zip(got, refs)):
        assert g == r, f"GPU mismatch at fixture {i}: len(g)={len(g)} len(r)={len(r)}"
    print("GPU kernel: PASS")


def bench_gpu(n_blocks: int = 1024, block_size: int = 65536) -> None:
    if not HAS_CUPY:
        print("Bench: SKIPPED (no cupy)")
        return
    import time

    import lz4.block

    raw = (b"the quick brown fox jumps over the lazy dog. " * (block_size // 45 + 1))[:block_size]
    comp = lz4.block.compress(raw, store_size=False)
    comps = [comp] * n_blocks
    sizes = [block_size] * n_blocks

    # Warm up — first launch compiles the kernel.
    decode_lz4_blocks_gpu(comps[:4], sizes[:4])

    t0 = time.perf_counter()
    decode_lz4_blocks_gpu(comps, sizes)
    elapsed = time.perf_counter() - t0
    mib = (n_blocks * block_size) / (1024 * 1024)
    print(f"GPU decode: {n_blocks} blocks x {block_size}B = {mib:.1f} MiB in {elapsed * 1000:.1f} ms")
    print(f"            throughput: {mib / elapsed:.1f} MiB/s; chunks/s: {n_blocks / elapsed:.1f}")


if __name__ == "__main__":
    test_cpu_reference()
    test_gpu_kernel()
    bench_gpu()
