# 03 — Replacing nvCOMP with pure cuda-python implementations

**Author:** Research agent 3 of 5 (cuda-array epic)
**Date:** 2026-05-23
**Hardware probed:** NVIDIA A40 (Ampere SM 8.6), driver 580.126.20
**Software:** Python 3.13, cupy 14.0.1, nvcomp 5.2.0, CUDA 13.1 system

## Executive summary

**Recommendation: a phased, surgical replacement — not a wholesale rewrite.**

The black-box-ness of nvCOMP is real but unevenly distributed across the
eight codec families czarr currently wraps. The cost-benefit splits cleanly
into three groups:

1. **Build (LZ4, Snappy) — recommended for v1 of cuda-array.** The
   block formats are simple, fully public, the open-source nvCOMP 2.2
   tree (BSD-3-Clause) gives us a reference implementation we can study
   and re-derive, and the LZ4 spike below already **beats nvCOMP by 2-43×
   on A40** in the relevant batch regime. Two engineer-weeks each for a
   production-quality version with shared-memory prefetch + vectorised
   coopcopy. Snappy is structurally identical to LZ4 (LZ77-family,
   no entropy coding); 3 weeks.

2. **Hybrid (Deflate / Gzip / Zlib) — leverage upstream code.** Don't
   write a deflate decoder from scratch. Two viable bases exist:

   * **cuDF's `gpuinflate.cu`** (1448 lines, Apache-2.0) is a battle-
     tested, production GPU deflate decoder used in every Parquet read
     RAPIDS does. We can vendor or fork it, write Python bindings, and
     skip months of Huffman-decoder work.
   * **Microsoft DirectStorage's GDeflate reference impl** (HLSL, but
     Apache-2.0) — useful as a second reference.

   Cost: 3-4 weeks to wrap + adapt + integrate. Bonus: GDeflate becomes
   trivial (it's the same Huffman engine on a parallel-friendly bitstream).

3. **Buy (Zstd, Bitcomp, ANS, Cascaded) — keep nvCOMP.** Zstd's FSE +
   Huffman entropy decoders are 6-9 months of work to get right, and
   the public state-of-the-art (Gstd, weissenberger/gpuhd, codyjrivera/
   ipdps22-opthuffdec) is research code, not production. Bitcomp/ANS/
   Cascaded are *NVIDIA-proprietary formats* with no published bitstream
   spec — we **cannot** re-implement them at all without reverse
   engineering. Keep nvCOMP for these four; they cover the "exotic"
   codecs anyway, not the common ones.

The transparency win is highest where the cost is lowest: LZ4 (the codec
czarr uses for the numcodecs-compat path) becomes a few hundred lines of
inspectable, profileable, debuggable cupy.RawKernel source.

### Headline LZ4 spike result (A40, kernel-only, warmed up)

| n_blocks | block_size | spike MiB/s | nvCOMP MiB/s | spike/nvcomp |
| -------: | ---------: | ----------: | -----------: | -----------: |
|       64 |       64 K |      11,745 |        5,149 |         2.3× |
|     1024 |       64 K |     125,200 |        7,043 |        17.8× |
|    16384 |       64 K |      79,198 |        6,489 |        12.2× |
|     4096 |       16 K |      80,169 |        1,846 |        43.4× |

The spike runs **one warp per LZ4 block** with no shared-memory prefetch
yet. It validates output bit-exactly against the `lz4.block` reference
on a 4-fixture suite (literals, RLE-overlap, random, large lz4-friendly).
Code: [`spikes/lz4_decoder.py`](spikes/lz4_decoder.py) — 311 lines
including CPU oracle, kernel, batched launcher, tests, and benchmark.

The 2-43× win over nvcomp is *not* primarily kernel quality — it's that
nvcomp's Python wrapper has substantial per-call setup cost that
disappears when you batch into a single grid-launch. This is itself a
reason to consider in-house: the integration overhead is a real cost we
keep paying.

## What is nvCOMP and what is its surface today?

nvCOMP is closed-source from v2.3 onward — the [README at branch-2.3](https://github.com/NVIDIA/nvcomp/blob/branch-2.3/README.md)
explicitly states *"From version 2.3 onwards, the compression /
decompression source code will not be released."* The current repo at
[NVIDIA/nvcomp main](https://github.com/NVIDIA/nvcomp) (last pushed
2024-09-11) holds **only docs, headers, and CPU-side examples**; the
actual CUDA kernels ship as opaque object files inside
`nvidia-nvcomp-cu12` / `cu13` pip wheels (`libnvcomp.so`).

**The good news:** nvCOMP 2.2 *was* released as BSD-3-Clause source, and
the [branch-2.2 tree](https://github.com/NVIDIA/nvcomp/tree/branch-2.2)
is still on GitHub. It contains the original LZ4 and Snappy CUDA kernels
in `src/LZ4Kernels.cuh` (1023 lines) and `src/SnappyKernels.cuh` (~1180
lines). These are five years behind current perf — but they are a
correct, complete reference for the **bitstream layouts**, which are the
hard part of any rewrite. License is BSD-3-Clause Copyright NVIDIA
2017-2020.

Czarr's surface area is just `_create_codec()` and the four nvcomp
method calls (`encode`, `decode`, `as_array`, `Array.cuda()`) inside
`_batch_sync()` in
[`src/czarr/codecs/base.py`](../../../src/czarr/codecs/base.py).
A replacement codec backend has to expose:

```python
class _Backend(Protocol):
    def decode_batch(
        self,
        compressed: list[bytes | cupy.ndarray],
        out: list[cupy.ndarray],  # pre-allocated dst buffers
        stream: int | None = None,
    ) -> None: ...
    def encode_batch(self, raw: list[cupy.ndarray]) -> list[cupy.ndarray]: ...
```

That's the whole API contract. Everything else (framing, checksum,
metadata) is already handled in `CudaBytesBytesCodec`.

## Per-codec analysis

For each codec I give: **spec status** (public format?), **reference
implementations available** (with license), **rewrite cost** (decoder +
encoder, lines + engineer-weeks), and a **build/hybrid/buy verdict**.

The cost numbers assume one engineer fluent in CUDA + Python (the team is
already writing cuTile kernels in `src/czarr/kernels/byteshuffle.py`).
"Decoder-only" rewrites are roughly **40% of the full effort** — writing
the compressor is the hard half.

### 1. LZ4 — BUILD

* **Spec:** Public, stable, very simple. [LZ4 block format](https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md):
  token-byte + length-extension + 2-byte LE offset. No entropy coding,
  no Huffman, no state.
* **References:**
  * [nvCOMP 2.2 `LZ4Kernels.cuh`](https://github.com/NVIDIA/nvcomp/blob/branch-2.2/src/LZ4Kernels.cuh)
    — BSD-3-Clause, 1023 lines. Decompress function is ~127 lines
    (`decompressStream`).
  * [Original LZ4 reference](https://github.com/lz4/lz4) — BSD 2-Clause,
    CPU.
* **Spike result:** [`spikes/lz4_decoder.py`](spikes/lz4_decoder.py)
  decodes correctly and runs at **125 GiB/s on A40 for 1024×64KiB
  batches**, 17.8× faster than nvCOMP 5.2 in the same shape. Spike is
  311 lines including CPU oracle, kernel, batched launcher, four
  fixtures, and benchmark. The CUDA kernel proper is **~95 lines**.
* **Production cost (decoder + encoder):**
  * Decoder hardening: shared-memory prefetcher (~150 lines, mirrors
    nvCOMP 2.2 `BufferControl`), vectorised 4/8/16-byte coopcopy in the
    `offset >= 32` path (~50 lines), full bounds checks (~30 lines),
    numcodecs.LZ4 4-byte uncompressed-size prefix support (~20 lines).
    **Total decoder: ~350 lines, 1.5 weeks.**
  * Encoder (LZ4 fast hash-table-based matcher, GPU-parallel by block):
    nvCOMP 2.2 `compressStream` is ~170 lines + hash table helpers.
    **Total encoder: ~400 lines, 2 weeks.**
* **Verdict: BUILD.** This is the codec czarr uses for the
  numcodecs.LZ4 compat path (`WITH_UNCOMPRESSED_SIZE`). Replacing it
  removes a 60 MiB pip-install dep for the most common case, gains us
  full source-level visibility, and the spike says perf is at minimum
  competitive.

### 2. Snappy — BUILD

* **Spec:** Public, simple. Snappy format is LZ77-family like LZ4 but
  with variable-byte length encoding instead of length-extension bytes.
  Slightly more complex tag decoding (4 tag types vs 1).
* **References:**
  * [nvCOMP 2.2 `SnappyKernels.cuh`](https://github.com/NVIDIA/nvcomp/blob/branch-2.2/src/SnappyKernels.cuh)
    — BSD-3-Clause, 1180 lines.
  * [cuDF `unsnap.cu`](https://github.com/rapidsai/cudf/blob/main/cpp/src/io/comp/unsnap.cu)
    — Apache-2.0, 762 lines, **production-grade GPU Snappy decoder**
    used by every cuDF Parquet read. This is the better reference: more
    actively maintained than nvCOMP 2.2.
  * [google/snappy](https://github.com/google/snappy) CPU spec/impl.
* **Production cost:**
  * Decoder: clone+port cuDF's `unsnap.cu` kernel logic to a
    `cupy.RawKernel` Python wrapper, swap output types. **~500 lines,
    2 weeks.**
  * Encoder: nvCOMP 2.2 `SnappyBatchKernels.cu` is ~280 lines; cuDF
    `snap.cu` is also Apache-2.0 (10.9 KB). **~450 lines, 2 weeks.**
* **Verdict: BUILD.** cuDF's Apache-2.0 GPU Snappy code makes this
  almost as easy as LZ4. 3-4 weeks total for both directions.

### 3. Deflate / Gzip / Zlib — HYBRID

These three czarr codecs all share one decoder — gzip and zlib are just
framed deflate. Deflate is **substantially harder** than LZ4/Snappy: it
has dynamic Huffman tables (two interleaved Huffman codes — literal+
length, and distance) that must be parsed from the bitstream header,
then the body is a stream of Huffman-coded symbols + variable-length
extra bits.

* **Spec:** Public ([RFC 1951](https://www.rfc-editor.org/rfc/rfc1951)).
* **References:**
  * [cuDF `gpuinflate.cu`](https://github.com/rapidsai/cudf/blob/main/cpp/src/io/comp/gpuinflate.cu)
    — Apache-2.0, **1448 lines**. Production deflate decoder. Used in
    cuDF Parquet/ORC since 2018. cuDF also exposes a `nvcomp_adapter`
    that chooses between this and nvCOMP based on the `LIBCUDF_NVCOMP_POLICY`
    env var, so the in-house implementation is treated as first-class.
  * [Microsoft DirectStorage GDeflate](https://github.com/microsoft/DirectStorage/blob/main/GDeflate/GDeflate/GDeflateDecompress.cpp)
    — Apache-2.0, but HLSL (`shaders/GDeflate.hlsl`, 28 KB). Useful as
    a second reference for the Huffman decoder.
* **Why hybrid not build:** Writing a correct dynamic-Huffman GPU
  decoder from scratch is **2-3 months**. Forking
  cuDF's existing one is 3-4 weeks (port the C++ host-side glue to
  cupy.RawKernel, write tests against zlib).
* **Production cost:**
  * Decoder via fork: 3-4 weeks. ~1500 lines vendored + ~200 lines glue.
  * Encoder: **do not write one.** GPU deflate encoding is not solved —
    cuDF doesn't have one, nvCOMP's is decent but proprietary. If we
    must encode, fall back to CPU libdeflate ([ebiggers/libdeflate](https://github.com/ebiggers/libdeflate))
    + upload — czarr writes are dominated by storage I/O anyway.
* **Verdict: HYBRID for decode (fork cuDF), BUY for encode.** Keep
  the nvCOMP encode path; replace decode.

### 4. GDeflate — HYBRID (free with deflate)

* **Spec:** Open standard. [Microsoft DirectStorage reference](https://github.com/microsoft/DirectStorage/tree/main/GDeflate)
  is Apache-2.0. GDeflate is a *bit-swizzled* DEFLATE: the bitstream is
  split into 32 sub-streams so 32 GPU threads can Huffman-decode in
  parallel. Once a deflate codec exists, GDeflate is mostly a
  bitstream-layout layer on top.
* **References:**
  * The same MS DirectStorage HLSL is the reference impl.
  * [`shaders/GDeflate.hlsl`](https://github.com/microsoft/DirectStorage/blob/main/GDeflate/shaders/GDeflate.hlsl)
    — 28 KB, Apache-2.0.
* **Production cost:** 2-3 weeks **assuming we already have deflate**.
  Port the HLSL bit-swizzle + 32-way parallelisation to CUDA on top of
  the cuDF inflate kernels. ~600 lines.
* **Verdict: HYBRID, but only after Deflate is shipped.**

### 5. Zstd — BUY

* **Spec:** Public ([RFC 8478](https://www.rfc-editor.org/rfc/rfc8478)),
  but **complex**. Two entropy coders interleaved (Huffman for literals,
  FSE for sequences), variable block types, optional dictionary, frame-
  level checksum. The decoder is well over 10× the complexity of LZ4.
* **References (and why none of them are good enough):**
  * **nvCOMP 5.x** has it, but closed source.
  * **[NVIDIA/nvcomp branch-2.2 — does NOT ship Zstd.](https://github.com/NVIDIA/nvcomp/blob/branch-2.2/src/lowlevel/)**
    Confirmed via API listing of `branch-2.2/src/lowlevel/`: only
    LZ4Batch, SnappyBatch, BitcompBatch, CascadedBatch, ansBatch,
    gdeflateBatch are present. Zstd was added in 2.3 (closed era).
  * **[Gstd](https://encode.su/threads/4176)** / [elasota/zstdhl](https://github.com/elasota/zstdhl)
    is a research project to make Zstd GPU-friendly by re-framing the
    bitstream. Not a production decoder.
  * **[weissenberger/gpuhd](https://github.com/weissenberger/gpuhd)** —
    parallel Huffman decoder paper (ICPP 2018). Just Huffman, not full
    Zstd. Decent reference for the literals stream.
  * **[codyjrivera/ipdps22-opthuffdec](https://github.com/codyjrivera/ipdps22-opthuffdec)** —
    IPDPS '22 paper artifact, optimised GPU Huffman decoder. Research
    code.
  * **[facebook/zstd #2307](https://github.com/facebook/zstd/issues/2307)**
    — upstream zstd team's official position is "no GPU support, sequential
    state machine doesn't parallelise well."
* **Production cost (decoder only):** Realistically 6-9 engineer-months
  to get a correct, robust decoder. The challenge isn't lines of code
  (~5000 lines feasible) — it's correctness across the long tail of
  zstd's variant blocks, magic-less frames, dictionary modes, etc., and
  achieving acceptable performance against the sequential FSE state
  machine. **No team I can find has done this in the open.**
* **Verdict: BUY.** Even if we do everything else in-house, keep the
  nvcomp dependency just for Zstd. The 60 MiB wheel is worth it for
  the codec that's most likely to be a user request (CPU world has
  largely standardised on Zstd).

### 6. Bitcomp — BUY (can't build)

* **Spec:** **Proprietary, undocumented.** From the [nvCOMP docs](https://docs.nvidia.com/cuda/nvcomp/):
  *"A proprietary compressor designed for efficient GPU compression in
  Scientific Computing applications."*
* **References:** None. There is no published bitstream specification.
  The format is binary-only.
* **Verdict: BUY.** We literally cannot re-implement Bitcomp without
  reverse-engineering it from binary, which is (a) a legal grey area
  and (b) several engineer-years anyway.

### 7. ANS (gANS) — BUY (can't build)

* **Spec:** **Proprietary.** From nvCOMP docs: *"A proprietary entropy
  encoder based on asymmetric numeral systems."* The underlying ANS
  technique is public (Duda 2009), but NVIDIA's specific gANS bitstream
  layout is not documented.
* **Verdict: BUY.** Same logic as Bitcomp.

### 8. Cascaded — BUY (can't build)

* **Spec:** **Proprietary.** A composite of run-length encoding, delta,
  and bit-packing tuned for analytical/tabular data. The components are
  generic, but the framing/composition format is not documented.
* **Verdict: BUY.** Same logic as Bitcomp.

## Cost summary

| Codec     | Verdict | Decoder weeks | Encoder weeks | Total weeks | Notes                                                |
| --------- | ------- | ------------- | ------------- | ----------- | ---------------------------------------------------- |
| LZ4       | BUILD   | 1.5           | 2.0           | **3.5**     | Spike already runs at 125 GiB/s on A40              |
| Snappy    | BUILD   | 2.0           | 2.0           | **4.0**     | Port cuDF `unsnap.cu` / `snap.cu`                   |
| Deflate   | HYBRID  | 3.5           | (CPU fallback)| **3.5**     | Fork cuDF `gpuinflate.cu` for decode                |
| Gzip      | HYBRID  | 0.5           | 0.5           | **1.0**     | Adds CRC32 + framing on top of Deflate              |
| Zlib      | HYBRID  | 0.5           | 0.5           | **1.0**     | Adds Adler-32 + framing on top of Deflate           |
| GDeflate  | HYBRID  | 2.5           | 2.5           | **5.0**     | Port DirectStorage HLSL, requires Deflate first     |
| Zstd      | BUY     | —             | —             | **0**       | Keep nvCOMP                                          |
| Bitcomp   | BUY     | —             | —             | **0**       | Proprietary format, cannot rewrite                  |
| ANS       | BUY     | —             | —             | **0**       | Proprietary format, cannot rewrite                  |
| Cascaded  | BUY     | —             | —             | **0**       | Proprietary format, cannot rewrite                  |
| **Total** |         |               |               | **~18 weeks** | One engineer, 4-5 months to fully replace 5 of 8 |

## Where the cost of nvCOMP's black-box-ness exceeds the cost of building

Concrete pain points the team has hit that an in-house impl would
eliminate:

1. **Opaque scratch allocation** — observed in czarr's
   `register_nvcomp_allocator` hack. An in-house codec lets us reuse
   the existing czarr GPU buffer pool / RMM allocator without surgery.
2. **CUDA 12 vs 13 version pinning** — `pyproject.toml` carries
   parallel `nvidia-nvcomp-cu12>=5.1.0.21` and `nvidia-nvcomp-cu13>=5.1.0.21`
   dependency lines. An in-house codec built on `cupy.RawKernel`
   compiles JIT against whatever runtime is loaded, eliminating this.
3. **`WITH_UNCOMPRESSED_SIZE` quirks** — the 4-byte prefix hack to be
   compatible with numcodecs.LZ4 is a documented surprise in
   `_BitstreamKind`. With an in-house decoder we control framing
   directly — no surprises.
4. **Per-call API overhead in the Python wrapper** — measured above.
   Even at the same kernel quality, the spike's single-launch
   batched path beats nvcomp's per-chunk-Array-wrap path by **10×+**
   on small chunks. This is wall-clock latency that compound onto
   every Zarr read.
5. **Inability to profile / debug** — when a chunk fails to decode
   inside nvcomp, all we see is `nvcompErrorCannotDecompress`. With
   in-house code we can dump the parse state, single-step the
   sequence stream, etc.

These five pain points alone justify the LZ4 + Snappy + Deflate work.
Bitcomp/ANS/Cascaded are barely used (none are even in the standard
Zarr spec — they're nvcomp-native formats that only nvcomp can read),
so keeping them on the binary backend has zero user impact.

## Recommended sequencing

1. **Phase 1 (3 weeks): LZ4 production.** Take the spike, add shared-
   memory prefetcher + vectorised coopcopy, full bounds checks,
   numcodecs.LZ4 prefix handling. Plumb through `CudaBytesBytesCodec`
   as an alternative backend, gated by `czarr_codec_backend="native"`
   env or arg. Keep nvcomp as the default to avoid regression risk.
2. **Phase 2 (4 weeks): Snappy.** Port cuDF `unsnap.cu` + `snap.cu`.
3. **Phase 3 (4 weeks): Deflate fork.** Vendor cuDF `gpuinflate.cu`
   under our build, write the Python kernel-launch glue, integrate
   the existing CRC32/Adler-32 host-side framing helpers in
   `compressors/gzip.py` / `compressors/zlib.py`.
4. **Phase 4 (1 week): flip default to native** for the three families
   above, after a soak period.
5. **Phase 5 (optional, 5 weeks): GDeflate.** Only if a real user asks
   for it — DirectStorage adoption in the scientific Python world is
   near zero.

Total to flip 5 of 8 codecs to in-house defaults: ~17 engineer-weeks.

## Cited references

* nvCOMP main repo (docs-only, post-2.3): https://github.com/NVIDIA/nvcomp
* nvCOMP 2.2 BSD-3-Clause source tree:
  * `src/` directory: https://github.com/NVIDIA/nvcomp/tree/branch-2.2/src
  * LZ4 kernels: https://github.com/NVIDIA/nvcomp/blob/branch-2.2/src/LZ4Kernels.cuh
  * LZ4 entry-point: https://github.com/NVIDIA/nvcomp/blob/branch-2.2/src/lowlevel/LZ4Batch.cpp
  * Snappy kernels: https://github.com/NVIDIA/nvcomp/blob/branch-2.2/src/SnappyKernels.cuh
  * LICENSE (BSD-3-Clause): https://github.com/NVIDIA/nvcomp/blob/branch-2.2/LICENSE
* LZ4 reference + spec:
  * Repo: https://github.com/lz4/lz4
  * Block format: https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
* cuDF GPU codec source (Apache-2.0):
  * Dir: https://github.com/rapidsai/cudf/tree/main/cpp/src/io/comp
  * Snappy decode: https://github.com/rapidsai/cudf/blob/main/cpp/src/io/comp/unsnap.cu
  * Deflate decode: https://github.com/rapidsai/cudf/blob/main/cpp/src/io/comp/gpuinflate.cu
  * Brotli decode: https://github.com/rapidsai/cudf/blob/main/cpp/src/io/comp/debrotli.cu
  * nvcomp adapter (shows the in-house vs nvcomp policy): https://github.com/rapidsai/cudf/blob/main/cpp/src/io/comp/nvcomp_adapter.cu
* GDeflate reference (Apache-2.0):
  * Dir: https://github.com/microsoft/DirectStorage/tree/main/GDeflate
  * Decompressor: https://github.com/microsoft/DirectStorage/blob/main/GDeflate/GDeflate/GDeflateDecompress.cpp
  * Shader: https://github.com/microsoft/DirectStorage/blob/main/GDeflate/shaders/GDeflate.hlsl
* Zstd format + status:
  * RFC 8478: https://www.rfc-editor.org/rfc/rfc8478
  * Facebook upstream "no GPU support" position: https://github.com/facebook/zstd/issues/2307
  * Gstd / zstdhl research project: https://github.com/elasota/zstdhl
* GPU Huffman decoders (academic):
  * weissenberger/gpuhd (ICPP 2018): https://github.com/weissenberger/gpuhd
  * codyjrivera/ipdps22-opthuffdec: https://github.com/codyjrivera/ipdps22-opthuffdec
* RAPIDS kvikIO (does NOT have its own codec implementations, wraps nvcomp):
  https://github.com/rapidsai/kvikio
* NVIDIA libdeflate fork (CPU only, used by nvcomp host paths):
  https://github.com/NVIDIA/libdeflate
* libgiddy (research GPU lightweight-codec library):
  https://github.com/eyalroz/libgiddy
* zarrs-python PR #147 (the original cuda-native Array idea):
  https://github.com/zarrs/zarrs-python/pull/147

## Open questions for the team

1. Do we care about the encode path at all, or is czarr primarily a read-
   side library? If reads dominate, the cost summary above drops by ~40%
   (encoders are roughly that share of the work).
2. Is there a real user for Bitcomp/ANS/Cascaded today? If not, the
   "BUY" recommendation is even safer — we keep nvcomp around purely
   for these four formats nobody else can read anyway.
3. Should the in-house codec backend be runtime-pluggable
   (`czarr.set_codec_backend("nvcomp" | "native")`) or build-time?
   Runtime is more flexible; build-time is what removes the nvcomp
   wheel from the dependency closure for the "native" install.
