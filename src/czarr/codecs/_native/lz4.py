"""Native LZ4 block decoder — single ``cupy.RawKernel`` over a batch of blocks.

Productionised from ``.planning/research/cuda-array/spikes/lz4_decoder.py``.
The kernel is unchanged from the spike (the H200 stream-parallelism probe
showed it saturates the GPU at ~173 GiB/s from a single grid launch); the
production surface differs in:

* Accepts device-resident ``cupy.ndarray`` inputs instead of host bytes.
* Returns ``list[cupy.ndarray]`` instead of host bytes.
* Decoupled from the framing layer — the caller strips any size prefix
  (e.g. numcodecs' 4-byte LE prefix for ``WITH_UNCOMPRESSED_SIZE`` mode)
  before invoking this function and passes the uncompressed size out-of-band.
* Cached kernel handle keyed by source string — first call compiles, every
  subsequent call hits the cupy module cache.

The kernel grid shape is one block per chunk, 32 threads (one warp) per
block.  See the spike module header for the parsing-state-machine design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class CodecDecodeError(RuntimeError):
    """Raised when the native LZ4 kernel reports a corrupted bitstream."""

    def __init__(self, block_index: int, hex_preview: str) -> None:
        super().__init__(
            f"native LZ4 decoder reported corruption at block {block_index}; first 32 bytes: {hex_preview}"
        )
        self.block_index = block_index
        self.hex_preview = hex_preview


_LZ4_DECODE_KERNEL_SRC = r"""
extern "C" __global__
void lz4_decode_blocks(
    const unsigned char* __restrict__ compressed,
    const long long* __restrict__ block_offsets,
    const long long* __restrict__ out_offsets,
    unsigned char* __restrict__ decompressed,
    int* __restrict__ status
) {
    const int block_id = blockIdx.x;
    const int lane = threadIdx.x;
    const long long src_begin = block_offsets[block_id];
    const long long src_end   = block_offsets[block_id + 1];
    const long long dst_begin = out_offsets[block_id];
    const long long dst_end   = out_offsets[block_id + 1];

    long long sp = src_begin;
    long long dp = dst_begin;
    int corrupted = 0;

    while (sp < src_end) {
        unsigned int token;
        if (lane == 0) token = compressed[sp];
        token = __shfl_sync(0xFFFFFFFF, token, 0);
        if (lane == 0) sp += 1;
        sp = __shfl_sync(0xFFFFFFFF, sp, 0);

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

        if (dp + lit_len > (unsigned long long)dst_end) { corrupted = 1; break; }
        for (unsigned int i = lane; i < lit_len; i += 32) {
            decompressed[dp + i] = compressed[sp + i];
        }
        sp += lit_len;
        dp += lit_len;

        if (sp >= src_end) break;

        unsigned int offset;
        if (lane == 0) {
            unsigned int b0 = compressed[sp];
            unsigned int b1 = compressed[sp + 1];
            offset = b0 | (b1 << 8);
            sp += 2;
        }
        offset = __shfl_sync(0xFFFFFFFF, offset, 0);
        sp = __shfl_sync(0xFFFFFFFF, sp, 0);

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


_KERNEL: cp.RawKernel | None = None


def _get_kernel() -> cp.RawKernel:
    """Lazy cupy ``RawKernel`` cache.

    cupy caches by source string under the hood, so this function exists
    mostly so we don't reconstruct the ``RawKernel`` object on every call.
    Future: switch to ``cuda.core.utils.FileStreamProgramCache`` for
    cross-process persistence.
    """
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_LZ4_DECODE_KERNEL_SRC, "lz4_decode_blocks")
    return _KERNEL


def decode_lz4_native(
    compressed_blocks: Sequence[cp.ndarray],
    uncompressed_sizes: Sequence[int],
    *,
    stream: cp.cuda.Stream | None = None,
) -> list[cp.ndarray]:
    """Decode a batch of LZ4 blocks on the GPU.

    The block format is plain LZ4 block (no length prefix, no frame).  The
    caller is responsible for stripping any wrapping (e.g. the 4-byte LE
    uncompressed-size prefix that numcodecs writes for the
    ``WITH_UNCOMPRESSED_SIZE`` bitstream kind) before calling.

    Parameters
    ----------
    compressed_blocks
        N device-resident ``cupy.ndarray`` views of dtype uint8.  Each is
        one compressed LZ4 block.
    uncompressed_sizes
        N integers — the expected uncompressed size of each block.  For
        czarr's typical workload this comes from
        ``CudaBytesBytesCodec._expected_decoded_bytes(spec)``.
    stream
        Optional cupy stream to launch on.  Defaults to the current
        thread's stream.

    Returns
    -------
    list[cupy.ndarray]
        N device-resident uint8 arrays sized to ``uncompressed_sizes[i]``.

    Raises
    ------
    CodecDecodeError
        If the kernel reports a corrupted block.  Block index and a hex
        preview of the bad bitstream are attached.
    ValueError
        If the input lists have mismatched lengths.
    """
    n = len(compressed_blocks)
    if n == 0:
        return []
    if len(uncompressed_sizes) != n:
        raise ValueError(f"compressed_blocks length {n} != uncompressed_sizes length {len(uncompressed_sizes)}")

    # Per-block offset prefix-sums — small, sit on the device alongside
    # the input data so the kernel can read them without a CPU roundtrip.
    src_offs = np.zeros(n + 1, dtype=np.int64)
    dst_offs = np.zeros(n + 1, dtype=np.int64)
    for i, (block, size) in enumerate(zip(compressed_blocks, uncompressed_sizes, strict=True)):
        if block.dtype != cp.uint8:
            raise ValueError(f"compressed_blocks[{i}] dtype={block.dtype}, expected uint8")
        if block.ndim != 1:
            raise ValueError(f"compressed_blocks[{i}] ndim={block.ndim}, expected 1")
        src_offs[i + 1] = src_offs[i] + int(block.size)
        dst_offs[i + 1] = dst_offs[i] + int(size)

    # Concatenate the per-chunk views into one contiguous device buffer.
    # cupy.concatenate handles the device-to-device copy efficiently.
    flat_src = cp.concatenate(list(compressed_blocks))
    d_src_offs = cp.asarray(src_offs)
    d_dst_offs = cp.asarray(dst_offs)
    d_dst = cp.empty(int(dst_offs[-1]), dtype=cp.uint8)
    d_status = cp.zeros(n, dtype=cp.int32)

    kernel = _get_kernel()
    if stream is not None:
        with stream:
            kernel(
                grid=(n,),
                block=(32,),
                args=(flat_src, d_src_offs, d_dst_offs, d_dst, d_status),
            )
    else:
        kernel(
            grid=(n,),
            block=(32,),
            args=(flat_src, d_src_offs, d_dst_offs, d_dst, d_status),
        )

    status_host = cp.asnumpy(d_status)
    if (status_host != 0).any():
        bad = int(np.argmax(status_host != 0))
        # Pull the first 32 bytes of the bad block back to the host for the
        # error message.  Cheap — at most 32 bytes per failure.
        bad_offset = int(src_offs[bad])
        bad_end = min(bad_offset + 32, int(src_offs[bad + 1]))
        bad_preview = bytes(cp.asnumpy(flat_src[bad_offset:bad_end]))
        raise CodecDecodeError(bad, bad_preview.hex())

    return [d_dst[int(dst_offs[i]) : int(dst_offs[i + 1])] for i in range(n)]
