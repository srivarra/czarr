"""Probe: which CPU codec ↔ nvCOMP algorithm pairs roundtrip via RAW bitstream?

For each candidate pair: encode bytes with CPU codec, push compressed bytes to
GPU, decode with nvCOMP in RAW mode (or NVCOMP_NATIVE if RAW fails), compare.

Outcomes inform which nvCOMP codecs we can register under numcodecs/zarr codec
names for transparent CPU→GPU interop.
"""

import cupy as cp
import numcodecs
import numpy as np
from nvidia import nvcomp


def _try_decode(name: str, compressed: bytes, original: bytes, *, nvcomp_alg: str, kind):
    try:
        codec = nvcomp.Codec(algorithm=nvcomp_alg, bitstream_kind=kind)
    except Exception as e:
        return f"codec_init FAIL: {e}"
    nv_in = nvcomp.as_array(np.frombuffer(compressed, dtype=np.uint8)).cuda()
    try:
        decoded = codec.decode([nv_in])[0]
    except Exception as e:
        return f"decode FAIL: {type(e).__name__}: {str(e)[:120]}"
    out = cp.asarray(decoded).view(cp.uint8).get().tobytes()
    if out == original:
        return f"OK ({len(compressed)} → {len(out)} bytes)"
    return f"MISMATCH (got {len(out)} bytes, expected {len(original)}, first16 got={out[:16].hex()} exp={original[:16].hex()})"


def main() -> int:
    rng = np.random.default_rng(0)
    # Compressible payload — random int32 cast to bytes, 256 KiB
    payload = rng.integers(0, 64, 65536, dtype=np.int32).tobytes()
    print(f"payload: {len(payload)} bytes")

    # Build all (cpu_codec, nvcomp_alg, [bitstream_kinds_to_try]) candidates
    cases = [
        ("Zstd", numcodecs.Zstd(level=3), "Zstd", [nvcomp.BitstreamKind.RAW, nvcomp.BitstreamKind.NVCOMP_NATIVE]),
        (
            "LZ4",
            numcodecs.LZ4(),
            "LZ4",
            [nvcomp.BitstreamKind.RAW, nvcomp.BitstreamKind.WITH_UNCOMPRESSED_SIZE, nvcomp.BitstreamKind.NVCOMP_NATIVE],
        ),
        (
            "GZip",
            numcodecs.GZip(level=5),
            "Deflate",
            [nvcomp.BitstreamKind.RAW, nvcomp.BitstreamKind.WITH_UNCOMPRESSED_SIZE, nvcomp.BitstreamKind.NVCOMP_NATIVE],
        ),
        (
            "Zlib",
            numcodecs.Zlib(level=5),
            "Deflate",
            [nvcomp.BitstreamKind.RAW, nvcomp.BitstreamKind.WITH_UNCOMPRESSED_SIZE, nvcomp.BitstreamKind.NVCOMP_NATIVE],
        ),
    ]

    # Strip-then-decode variants for GZip + Zlib (handle header/trailer ourselves)
    print(f"{'CPU codec':<14} {'-> nvCOMP':<14} {'bitstream':<25} result")
    print("-" * 110)
    for label, cpu_codec, nvcomp_alg, kinds in cases:
        comp = cpu_codec.encode(np.frombuffer(payload, dtype=np.uint8))
        comp_bytes = bytes(comp)
        for kind in kinds:
            result = _try_decode(label, comp_bytes, payload, nvcomp_alg=nvcomp_alg, kind=kind)
            print(f"{label:<14} {nvcomp_alg:<14} {kind.name:<25} {result}")

        # Header-strip retries
        if label == "GZip":
            # gzip frame: 10-byte header + DEFLATE + 8-byte trailer (CRC32 + ISIZE)
            stripped = comp_bytes[10:-8]
            for kind in kinds:
                result = _try_decode(label, stripped, payload, nvcomp_alg=nvcomp_alg, kind=kind)
                print(f"{label + ' (strip)':<14} {nvcomp_alg:<14} {kind.name:<25} {result}")
        elif label == "Zlib":
            # zlib frame: 2-byte header + DEFLATE + 4-byte trailer (Adler-32)
            stripped = comp_bytes[2:-4]
            for kind in kinds:
                result = _try_decode(label, stripped, payload, nvcomp_alg=nvcomp_alg, kind=kind)
                print(f"{label + ' (strip)':<14} {nvcomp_alg:<14} {kind.name:<25} {result}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
