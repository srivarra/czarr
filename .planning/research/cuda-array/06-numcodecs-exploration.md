# numcodecs — architectural inspiration for czarr

This report explores the `zarr-developers/numcodecs` repository in depth and
distils the patterns that should (and should not) carry across to czarr's
GPU-native codec layer. The scope is deliberately wide: the codec ABC, the
registry, the per-codec implementation conventions, the numcodecs→Zarr v3
bridge that now lives inside `zarr-python`, the build system, and the
backward-compatibility fixture pattern. Companion document to
`01-cuda-core-api.md` through `05-lifetime-interop.md` and `SUMMARY.md`; this
one is the missing "what does the upstream reference look like" perspective.

References are to `zarr-developers/numcodecs@main` unless stated otherwise.
The package source moved under `src/` recently — every codec file lives at
`src/numcodecs/<name>.{py,pyx}`.

---

## 1. High-level architecture

### 1.1 One codec kind, two encode/decode methods

numcodecs predates Zarr v3 and exposes a single, narrow contract:

```python
class Codec(ABC):
    codec_id: str | None = None
    def encode(self, buf): ...
    def decode(self, buf, out=None): ...
    def get_config(self): ...
    @classmethod
    def from_config(cls, config): ...
```

(`src/numcodecs/abc.py:36-67`). The "kind" — Bytes→Bytes vs Array→Array vs
Array→Bytes — is *implicit*. A numcodecs codec just promises:

* `encode(buf)` accepts any object supporting the new-style buffer protocol
  and returns *something* with a buffer protocol.
* `decode(buf, out=None)` symmetrically; `out` is a writable buffer that, if
  provided, must be exactly the right size.

There is no per-method type contract beyond "buffer-like in, buffer-like
out". Whether the codec is an ndarray-shaped filter (e.g. `Delta`) or a
bytes-shaped compressor (e.g. `Zstd`) is a runtime decision the *caller*
makes. This is the looseness Zarr v2 lived with: filters and compressors
were stored in separate slots, but called through the same interface.

### 1.2 Zarr v3's three-codec split

Zarr v3 tightened this up. The v3 spec mandates three kinds, each with a
clearly typed boundary, and `zarr-python` materialises them as ABCs:

| Kind                 | Input                | Output               | Used as                |
| -------------------- | -------------------- | -------------------- | ---------------------- |
| `ArrayArrayCodec`    | shaped NDBuffer      | shaped NDBuffer      | filters                |
| `ArrayBytesCodec`    | shaped NDBuffer      | flat byte Buffer     | "the" serialiser       |
| `BytesBytesCodec`    | flat byte Buffer     | flat byte Buffer     | compressors, checksums |

A v3 codec chain is `[ArrayArrayCodec*, ArrayBytesCodec, BytesBytesCodec*]`
— filters first, then the (mandatory) serialiser (default `bytes`), then
compressors / checksums. Compare with numcodecs's free-form chain, which
left the encoded-bytes boundary up to the caller's discipline.

### 1.3 Bridging the two

The bridge lives in **zarr-python**, not numcodecs. As of zarr 3.1.3 the
`numcodecs.zarr3` module is just a deprecation shim
(`src/numcodecs/zarr3.py:1-67`):

```python
msg = (
    "The numcodecs.zarr3 module is deprecated... "
    f"Import {name} via zarr.codecs.numcodecs.{name} instead."
)
```

The real bridge is `zarr.codecs.numcodecs._codecs` in zarr-python
(320 lines). Its skeleton:

```python
@dataclass(frozen=True)
class _NumcodecsCodec(Metadata):
    codec_name: str
    codec_config: dict[str, JSON]

    @cached_property
    def _codec(self) -> Numcodec:
        return get_numcodec(self.codec_config)  # delegates to numcodecs.get_codec

class _NumcodecsBytesBytesCodec(_NumcodecsCodec, BytesBytesCodec):
    async def _decode_single(self, chunk_data, chunk_spec):
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._codec.decode, chunk_data, chunk_spec.prototype,
        )
    ...
```

(`zarr-developers/zarr-python:src/zarr/codecs/numcodecs/_codecs.py:120-148`).
Three points worth absorbing:

1. **The bridge is one class per Zarr-v3 kind**, with the kind chosen by
   reading the codec's behaviour, not by introspecting numcodecs. The
   bridge author picked which numcodecs codec maps to which Zarr-kind, and
   wrote it down in a hand-curated subclass list (`Blosc/LZ4/Zstd/...` are
   `_NumcodecsBytesBytesCodec`; `Delta/BitRound/PackBits/...` are
   `_NumcodecsArrayArrayCodec`; `PCodec/ZFPY` are
   `_NumcodecsArrayBytesCodec`).
2. **`asyncio.to_thread`** is the offload hook. Every numcodecs codec call
   inside zarr v3 is wrapped in `asyncio.to_thread(...)` so the CPU work
   does not block the asyncio loop. There is no streaming or batching at
   this layer.
3. **`evolve_from_array_spec`** is how the bridge lazily fills in `dtype`,
   `elementsize`, and similar fields when the user doesn't pass them
   explicitly. See `Shuffle.evolve_from_array_spec`, `FixedScaleOffset`,
   `Quantize`, `AsType` (all in the same file). This is the v3 plumbing
   that lets you write `codecs=[FixedScaleOffset(offset=1000, scale=10)]`
   without having to repeat `dtype="<f8"`.

czarr should follow this exact split when it ships a numcodecs-bridge of
its own (see §5 below). For codecs we hand-write directly against the v3
ABCs (the current `czarr.codecs.compressors.LZ4`,
`czarr.codecs.filters.Shuffle`, etc.), the bridge isn't needed — those are
already v3-native.

---

## 2. The `Codec` base class contract

### 2.1 Required surface

From `src/numcodecs/abc.py:36-119`:

* `codec_id: str` — class attribute, the persisted identifier
  (e.g. `"zstd"`, `"shuffle"`, `"vlen-utf8"`). Two classes sharing a
  `codec_id` must be byte-for-byte compatible.
* `encode(self, buf) -> buffer-like` — abstract.
* `decode(self, buf, out=None) -> buffer-like` — abstract.
* `get_config(self) -> dict` — returns a JSON-serialisable dict including
  `"id"`. Default implementation walks `self.__dict__` and returns
  everything not prefixed with `_`. Subclasses override when fields need
  custom serialisation (`Delta.get_config`, `FixedScaleOffset.get_config`,
  `AsType.get_config` all override to encode dtypes as `.str`).
* `from_config(cls, config) -> Codec` — class method. Default
  implementation is `cls(**config)`. Note: the framework strips the `"id"`
  key *before* calling `from_config` (see `registry.get_codec`).
* `__eq__` and `__repr__` are provided in the base; both walk
  `get_config()` / `self.__dict__`. Subclasses override `__repr__` when the
  config has too many fields to print legibly (`Blosc`, `Categorize`,
  `JSON`).

The default `get_config`/`from_config` is what makes adding a codec a
near-trivial exercise: as long as `__init__` accepts the same kwargs your
`__dict__` will hold, the round-trip works for free.

### 2.2 What's *not* in the contract

* No `compute_encoded_size`. numcodecs callers don't pre-allocate output
  buffers; they let `encode` allocate.
* No "kind" / "is_fixed_size". Filters and compressors are
  indistinguishable.
* No async. Every codec is sync. The bridge to v3 adds async with
  `asyncio.to_thread`.
* No streaming. Each `encode`/`decode` is a one-shot call over the whole
  buffer. (Internally, `numcodecs.zstd.stream_decompress` handles unknown
  content-size; see `src/numcodecs/zstd.pyx:268-356`. But it's still a
  blocking call.)
* No batch API. Each chunk is one call.

The last two matter for czarr: nvCOMP's whole performance story is batched
GPU calls. Wrapping `nvcomp.Codec.encode(chunk)` per-chunk would hide that.
czarr already addresses this at the `CudaBytesBytesCodec` level by relying
on Zarr v3's batched `encode/decode` (with shape `Iterable[(buf, spec)]`),
not the `numcodecs.Codec.encode(buf)` one-shot surface.

---

## 3. Registration mechanism

### 3.1 The simple half — `register_codec`

`src/numcodecs/registry.py:55-72`:

```python
codec_registry: dict[str, Codec] = {}

def register_codec(cls, codec_id=None):
    if codec_id is None:
        codec_id = cls.codec_id
    codec_registry[codec_id] = cls
```

A plain module-level dict, keyed on codec id. `numcodecs/__init__.py`
registers every shipped codec at import time
(`src/numcodecs/__init__.py:31-148`):

```python
from numcodecs.zstd import Zstd
register_codec(Zstd)
from numcodecs.lz4 import LZ4
register_codec(LZ4)
...
```

### 3.2 The interesting half — entry-point discovery

`src/numcodecs/registry.py:13-22`:

```python
entries: dict[str, EntryPoints] = {}

def run_entrypoints():
    entries.clear()
    eps = entry_points()
    entries.update({e.name: e for e in eps.select(group="numcodecs.codecs")})

run_entrypoints()
```

`get_codec` first looks in `codec_registry`; on a miss it consults
`entries`, lazily imports, and caches into `codec_registry`. Third parties
publish codecs by adding an entry point under `numcodecs.codecs` in their
own package metadata. The repo contains a working example for the test
suite at `tests/package_with_entrypoint-0.1.dist-info/entry_points.txt`:

```ini
[numcodecs.codecs]
test = package_with_entrypoint:TestCodec
```

and `tests/package_with_entrypoint/__init__.py`:

```python
from numcodecs.abc import Codec

class TestCodec(Codec):
    codec_id = "test"
    def encode(self, buf): pass
    def decode(self, buf, out=None): pass
```

The test (`tests/test_entrypoints.py`) just calls
`numcodecs.registry.get_codec({"id": "test"})` and asserts the codec is
loaded.

### 3.3 Two registries, not one

Crucially, **`numcodecs.registry` and `zarr.registry` are independent**.
zarr-python has its own `register_codec` (`zarr.registry.register_codec`)
used by Zarr v3 codecs to register themselves under their `codec_name`
(NOT `codec_id`; the v3 spec calls the field `name`). czarr already uses
this: see `src/czarr/codecs/__init__.py:16-53`:

```python
from zarr.registry import register_codec
...
for _cls in (ANS, Bitcomp, ..., Zstd, LZ4, Gzip, Zlib, Shuffle, Delta, ...):
    register_codec(_cls.codec_name, _cls)
```

What zarr.registry *also* understands, since 3.1.x, is "give me the
numcodec under this name" via `get_numcodec` (used by
`_NumcodecsCodec._codec` in the bridge). That call delegates back to
`numcodecs.get_codec`. The bridge is what unifies the two registries.

**Implication for czarr**: if we want a codec available under both
`numcodecs.get_codec({"id": "czarr_lz4"})` *and*
`zarr.registry.get_codec("czarr_lz4")`, we register in both places. Today
we only register on the zarr side. That's correct for Zarr v3-native
codecs (the user reads/writes through zarr, not numcodecs). But if someone
wants to manually `numcodecs.get_codec({"id": "lz4"})` and have *our* LZ4
come back, we'd need an entry-point declaration in `pyproject.toml`:

```toml
[project.entry-points."numcodecs.codecs"]
lz4 = "czarr.codecs.compressors.lz4:LZ4"
```

That probably *should not* happen because it'd shadow `numcodecs.LZ4` for
any user who installs czarr — which would be surprising. Cleaner is the
status quo: czarr codecs are v3-native and only register with
`zarr.registry`.

---

## 4. Per-codec implementation patterns

This section walks the codecs the team has identified as relevant — LZ4
and Zstd because we wrap them via nvCOMP, Blosc because of its batched
multi-codec design, the filters because we mirror them on GPU, and the
exotic ones (`vlen-utf8`, `categorize`) because we don't and shouldn't.

### 4.1 Compressors

#### Zstd — straight libzstd wrapper

`src/numcodecs/zstd.pyx:1-460`. Wraps the official libzstd C API.

* `compress(source, level, checksum)` → `bytes`. Allocates
  `ZSTD_compressBound(source_size)` up front (line 134), runs
  `ZSTD_compress2` under `nogil` (line 162), then `PyBytes_RESIZE` down to
  the actual compressed size (line 174). Returns a single zstd frame; the
  content size is embedded in the frame header.
* `decompress(source, dest=None)` → `bytes` / `out`. First calls
  `findTotalContentSize` (line 396-444) which sums `ZSTD_getFrameContentSize`
  across all concatenated frames in the buffer. If unknown size and no
  `out`, falls back to `stream_decompress` (line 268-356) which grows the
  buffer in 128 KB chunks via `realloc`.
* No size-prefix outside the standard frame header. The output buffer is
  exactly the libzstd frame.

This maps cleanly to czarr's `Zstd` codec which uses
`_BitstreamKind.NVCOMP_NATIVE` by default (nvCOMP's own chunked format,
incompatible with libzstd) but exposes `_BitstreamKind.RAW` for
interoperability with numcodecs's output. The "RAW" bitstream produces a
single libzstd frame per chunk — byte-identical to
`numcodecs.zstd.compress(buf, level)`.

#### LZ4 — *with* a 4-byte little-endian size prefix

`src/numcodecs/lz4.pyx:42-119`. This is the one to look at carefully.

```cython
dest = PyBytes_FromStringAndSize(NULL, dest_size + sizeof(uint32_t))
dest_ptr = PyBytes_AS_STRING(dest)
store_le32(<uint8_t*>dest_ptr, source_size)       # ← prefix
dest_start = dest_ptr + sizeof(uint32_t)

with nogil:
    compressed_size = LZ4_compress_fast(source_ptr, dest_start,
                                        source_size, dest_size, acceleration)
```

So the on-disk layout is:

```
| uint32_le: uncompressed_size | LZ4_block_compressed_payload |
```

`store_le32` and `load_le32` are tiny inline helpers in
`src/numcodecs/_utils.pxd`. This is the format czarr's LZ4 needs to match
when `_BitstreamKind.WITH_UNCOMPRESSED_SIZE` is set — and indeed
`src/czarr/codecs/compressors/lz4.py:20` does exactly that:

```python
_bitstream_kind: ClassVar[_BitstreamKind] = _BitstreamKind.WITH_UNCOMPRESSED_SIZE
```

Two implementation tricks worth stealing:

1. **Allocate over-large then `PyBytes_RESIZE`.** `LZ4_compressBound`
   gives a worst-case bound; the actual compressed length is shorter, so
   the codec resizes the bytes object in place. This avoids a copy. The
   equivalent in cupy is `cp.empty(bound).resize(actual)` — but resizing
   a cupy array deallocates and reallocates. cuda-python managed memory
   buffers have the same problem. For czarr the established pattern is to
   over-allocate and slice; if the resulting buffer is large the slice
   keeps holding the parent allocation. RMM with the suballocating pool
   keeps this cheap.
2. **`with nogil`** around the C call. The Python equivalent for cupy is
   nothing — we already run in a no-GIL stream context. But the *pattern*
   transfers: keep Python objects on the cold path, do C/CUDA work in the
   hot path.

The empty-input edge case is not explicitly handled. `LZ4_compress_fast`
with `source_size=0` returns 1 (a single literal-block header byte), which
`PyBytes_RESIZE` then crops down. We should test that nvCOMP RAW LZ4 also
round-trips through an empty chunk; numcodecs's behaviour is the oracle.

#### Blosc — wraps a meta-codec that already does sharding

`src/numcodecs/blosc.pyx:1-617`. This is the most complex codec in the
package. Blosc itself is a multi-codec meta-format (it embeds blosclz /
lz4 / zstd / zlib / snappy / lz4hc internally) and a sharder (its "blocks"
are essentially what Zarr v3 calls shards).

Key behaviours:

* The compressor and shuffle filter are fused into one bitstream. Blosc
  applies SHUFFLE/BITSHUFFLE on the way in, picks block sizes, threads the
  blocks, and produces a single binary blob. The decoder reverses
  everything from the header.
* Two API entry points: `blosc_compress(...)` uses the library's global
  context and a process-wide thread pool (lines 372-389), and
  `blosc_compress_ctx(...)` is the thread-safe variant with caller-managed
  threads (line 391-393). `_get_use_threads()` (line 480-505) picks
  between them based on whether the process has forked / is multi-threaded.
* `_cbuffer_sizes(source)` (line 200-222) parses the Blosc header to
  recover `nbytes` (uncompressed), `cbytes` (compressed), `blocksize`. This
  enables decoding into a caller-supplied buffer of the right size without
  any out-of-band metadata.

For czarr the takeaway is more about Blosc's existence than its source:
**Blosc is a counterexample to "one codec = one algorithm"**. It already
fuses filter + multi-algorithm compression + threaded sharding behind one
API. nvCOMP's batched API (`encode_batch(chunks, algorithm)`) is broadly
isomorphic. We could imagine a future `czarr.Blosc` that:

* Accepts the same `cname`/`clevel`/`shuffle`/`blocksize` config as
  `numcodecs.Blosc`.
* Maps `cname="lz4"|"zstd"|"snappy"|"zlib"` to the corresponding nvCOMP
  algorithm.
* Applies the shuffle (we already have a CUDA byteshuffle kernel).
* Emits a Blosc-1 header so the output is consumable by stock Blosc on the
  host.

The Blosc-1 header is fully documented and small (16 bytes). Format
fidelity is the main effort; the rest is plumbing through our existing
nvCOMP wrappers. *Speculative — not on the v0.1 roadmap.*

#### Gzip / Zlib — pure-Python stdlib wrappers

`src/numcodecs/gzip.py` and `src/numcodecs/zlib.py`. These are illustrative
because they show the *minimum* amount of code needed to make a codec:

```python
class Zlib(Codec):
    codec_id = 'zlib'
    def __init__(self, level=1):
        self.level = level
    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)
        return _zlib.compress(buf, self.level)
    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)
        dec = _zlib.decompress(buf)
        return ndarray_copy(dec, out)
```

That's the whole codec. No `get_config`/`from_config`/`__eq__`/`__repr__`
— all inherited.

Both codecs emit the standard library's gzip/zlib bitstream, which means
gzip wraps the deflate payload with a 10-byte header + CRC32 + ISIZE
trailer, and zlib wraps it with a 2-byte header + Adler32 trailer. These
are exactly the framing layers that czarr's `_frame_strip_head` /
`_frame_strip_tail` / `_wrap_frame` / `_unwrap_frame` hooks on
`CudaBytesBytesCodec` handle (see `src/czarr/codecs/base.py:1-19` and the
nvCOMP Deflate codec's raw bitstream which omits the zlib/gzip wrapper).
The hooks are correctly named for the cross-reference: numcodecs gives us
the wrapped form, nvCOMP gives us deflate-only, czarr glues them.

### 4.2 Filters

The filters are *much* simpler than the compressors — small,
pure-numpy/Python, no C extensions. They're the right oracles for czarr's
Phase 2 CCCL filter implementations.

#### Shuffle

`src/numcodecs/shuffle.py` plus the byte-level kernel in
`src/numcodecs/_shuffle.pyx:7-29`:

```cython
cpdef void _doShuffle(const unsigned char[::1] src, unsigned char[::1] des,
                      Py_ssize_t element_size) noexcept nogil:
    count = len(src) // element_size
    for i in range(count):
        offset = i * element_size
        for byte_index in range(element_size):
            j = byte_index * count + i
            des[j] = src[offset + byte_index]
```

Read the indexing carefully. For `element_size=4`, the bytes of element
`i` go to positions `(0*count+i, 1*count+i, 2*count+i, 3*count+i)` — all
byte-0 first, then all byte-1, etc. czarr's `byteshuffle_batched`
(`src/czarr/kernels/byteshuffle.py`) must produce the same byte order.
The numcodecs reference is the test oracle: take a random `np.uint32`
array of length 1024, run it through `numcodecs.Shuffle(elementsize=4)`,
and verify czarr's encoder produces byte-identical output. The decode
side is the inverse permutation.

Empty-input edge: `numcodecs.Shuffle.encode(np.array([], dtype=np.uint8))`
returns an empty buffer (the `count = len // element_size` is 0 and the
loop doesn't execute). czarr should match.

#### Delta

`src/numcodecs/delta.py:46-78`:

```python
def encode(self, buf):
    arr = ensure_ndarray(buf).view(self.dtype).reshape(-1, order='A')
    enc = np.empty_like(arr, dtype=self.astype)
    enc[0] = arr[0]
    enc[1:] = np.diff(arr)
    return enc

def decode(self, buf, out=None):
    enc = ensure_ndarray(buf).view(self.astype).reshape(-1, order='A')
    dec = np.empty_like(enc, dtype=self.dtype)
    np.cumsum(enc, out=dec)
    return ndarray_copy(dec, out)
```

Two non-obvious details:

* **`reshape(-1, order='A')`** flattens in whatever memory order the array
  has. For C-contig arrays it's row-major; for F-contig arrays it's
  column-major. The encoding is fundamentally 1-D; the codec doesn't know
  about array shape. czarr's Delta
  (`src/czarr/codecs/filters/delta.py:50`) already uses
  `arr.ravel()` which has the same C-order assumption for typical inputs.
  We should explicitly document the F-order behaviour (or refuse F-order
  inputs) so the oracle test stays meaningful.
* **`astype`** lets the user pick a narrower output dtype (signed int when
  values fit). Decoding casts back to `dtype`. czarr already supports both
  fields.

`numcodecs.Delta.get_config` returns `{'id': 'delta', 'dtype': '<f4',
'astype': '<i2'}` — strings, not dtype objects, for JSON serialisation.
czarr's `to_dict` already converts via `str(self.dtype)`.

#### FixedScaleOffset and Quantize

`src/numcodecs/fixedscaleoffset.py:88-105` and
`src/numcodecs/quantize.py:54-78`. Both are simple `(arr - offset) *
scale → round → cast` filters. They differ only in how they pick the
quantum:

* `FixedScaleOffset(offset, scale, dtype, astype)` — user picks `scale`
  directly.
* `Quantize(digits, dtype, astype)` — derives `scale = 2**ceil(log2(10**digits))`
  internally. The rounding is to the nearest power-of-2 quantum, which
  the docstring example illustrates.

Float32 implementations on the GPU are straightforward: `cp.around((x -
offset) * scale).astype(astype)`. czarr already implements
FixedScaleOffset (`src/czarr/codecs/filters/fixedscaleoffset.py`); we do
not yet have Quantize, which would be a single-file addition with the
scale derivation copied verbatim.

#### BitRound

`src/numcodecs/bitround.py:50-80`. The IEEE-754 mantissa truncation. The
bit-twiddling reference:

```python
maskbits = bits - self.keepbits      # bits to drop
mask = (all_set >> maskbits) << maskbits
half_quantum1 = (1 << (maskbits - 1)) - 1
b += ((b >> maskbits) & 1) + half_quantum1   # round to even
b &= mask
```

The `(b >> maskbits) & 1` is the round-to-even tiebreaker — it adds 1
extra if the bit just *above* the truncation point is set. czarr's
`src/czarr/codecs/filters/bitround.py:70-77` does the same:

```python
half = cp.uint64(1) << cp.uint64(shift - 1)
mask = ~((cp.uint64(1) << cp.uint64(shift)) - cp.uint64(1))
bits = (bits.astype(cp.uint64) + half) & mask
```

— but **omits the round-to-even bit**. Compare with numcodecs's `+
((b >> maskbits) & 1)`. That is a real divergence: for halfway values
(the truncated bit is exactly at half), numcodecs rounds to even, czarr
rounds up. The numcodecs output for a halfway value is one ULP smaller
roughly half the time. **This needs a bit-exact test**, and the fix is to
add the tiebreaker:

```python
tiebreak = (bits >> cp.uint64(shift)) & cp.uint64(1)
bits = (bits.astype(cp.uint64) + half + tiebreak) & mask
```

Logging this in the wishlist under "bit-exact compat" (§7).

#### PackBits

`src/numcodecs/packbits.py:29-79`. Encodes a boolean array into bits in a
uint8 array. The output layout:

```
| u8: n_bits_padded | np.packbits(arr) bytes |
```

The first byte stores how many bits were padded to round up to a whole
byte. Decoder reads that, runs `np.unpackbits`, strips the padding.

`np.packbits` itself runs left-to-right (`np.packbits([1,0,1,0,0,0,0,0])`
= `[0b10100000]` = `[0x80 | 0x20]` = `[128 + 32]` = `160`). To match this
on the GPU we'd want a `cp.packbits` (which exists — `cupy.packbits` has
the same MSB-first convention). czarr doesn't currently have PackBits;
it's a single-call wrapper around `cp.packbits` plus the 1-byte padding
prefix. *Low effort, low demand.*

### 4.3 Exotic codecs we don't support (and shouldn't)

* **VLen / VLenUTF8 / VLenBytes** (`src/numcodecs/vlen.pyx:43-185`) emit a
  parquet-style framing (`u32 n_items | (u32 len_i | bytes_i)*`) but the
  encoder loops over Python objects per-string. Same on the read side.
  GPU implementations are doable in principle (prefix-sum lengths, then
  concatenate), but the user-visible operation is "I have a Python list
  of strings", and the CPython↔CUDA crossing per string dominates. If
  someone needs variable-length strings on GPU, they should use Arrow /
  cuDF, not Zarr. **Skip.**
* **Categorize** (`src/numcodecs/categorize.py:53-83`) has a per-label
  Python loop. Same story as VLen. **Skip.**
* **JSON / MsgPack / Pickle / Base64** are escape hatches for arbitrary
  Python objects encoded as bytes. There is no GPU path for "JSON-encode
  a Python list". These are for metadata sidecars, not data. **Skip.**

### 4.4 Recent additions — PCodec, ZFPY, AsType

#### PCodec (`pcodec`)

`src/numcodecs/pcodec.py:1-115`. This is a Python wrapper around the
`pcodec` rust library (`from pcodec import standalone`). It's an
`ArrayBytesCodec` in v3 terms (input: typed array, output: opaque bytes).
The config has six parameters covering `mode_spec` (auto / classic),
`delta_spec` (auto / none / try_consecutive / try_lookback), and paging.

PCodec is the only non-trivial example in numcodecs of an `ArrayBytesCodec`
(zfpy is the other). Both consume a typed array and emit bytes; the bridge
classes are `_NumcodecsArrayBytesCodec` (see `zarr-python:_codecs.py:185`).
This pattern is what we'd use if we ever wrapped a GPU codec that
inherently transforms array→bytes (e.g. a future GPU CABAC entropy coder).

#### ZFPY (`zfpy`)

`src/numcodecs/zfpy.py:54-104`. Wraps the `zfpy` Python module which in
turn wraps libzfp. Like PCodec, it's an ArrayBytes codec. Three modes:
fixed-accuracy (tolerance), fixed-rate (rate), fixed-precision (precision).
The encoder takes a *typed* numpy array and produces bytes via
`zfpy.compress_numpy(buf, write_header=True, ...)`.

ZFP has a CUDA backend in libzfp itself, so a true `czarr.ZFP` is feasible
and would slot in as an `ArrayBytesCodec`. Demand is modest (climate
science / large floating-point arrays). *Wishlist item.*

#### AsType

`src/numcodecs/astype.py:43-67`. Pure dtype-conversion filter:

```python
def encode(self, buf):
    arr = ensure_ndarray(buf).view(self.decode_dtype)
    return arr.astype(self.encode_dtype)
def decode(self, buf, out=None):
    enc = ensure_ndarray(buf).view(self.encode_dtype)
    dec = enc.astype(self.decode_dtype)
    return ndarray_copy(dec, out)
```

GPU implementation is one line: `cp.asarray(x).astype(target_dtype)`.
**Low effort, low demand, but useful as a building block.** It also
illustrates the bridge's `resolve_metadata` (see
`zarr-python:_codecs.py:255-263`): an `ArrayArrayCodec` whose output
dtype differs from its input must tell the chain about that. czarr's
filters do this through `compute_encoded_size` plus shape preservation
(see `Delta.compute_encoded_size`); for AsType we'd also need the dtype
change.

### 4.5 Checksum codecs — framing pattern

`src/numcodecs/checksum32.py:50-100`. All four (`CRC32`, `Adler32`,
`Fletcher32`, `JenkinsLookup3`, optionally `CRC32C`) inherit
`Checksum32`:

```python
def encode(self, buf):
    arr = ensure_contiguous_ndarray(buf).view('u1')
    checksum = self.checksum(arr) & 0xFFFFFFFF
    enc = np.empty(arr.nbytes + 4, dtype='u1')
    if self.location == 'start':
        checksum_view, payload_view = enc[:4], enc[4:]
    else:
        checksum_view, payload_view = enc[-4:], enc[:-4]
    checksum_view.view('<u4')[0] = checksum
    ndarray_copy(arr, payload_view)
    return enc
```

The framing is consistent: **4 bytes of little-endian checksum, then the
payload (or vice versa).** Whether the checksum is prepended or appended
is configurable per instance, with sensible per-codec defaults:

* CRC32 / Adler32 / JenkinsLookup3 — `location = 'start'`.
* CRC32C — `location = 'end'` (Zarr v3 spec mandates the trailer
  position).
* Fletcher32 — always appended (the codec overrides the framing entirely
  to match the netCDF/HDF5 convention).

This is exactly the pattern czarr's gzip/zlib codecs use for their CRC32
trailer (gzip) and Adler32 trailer (zlib) — see
`CudaBytesBytesCodec._frame_strip_tail` / `_wrap_frame`. The numcodecs
file is the cleanest CPU reference for "framing layer is a 4-byte
checksum somewhere predictable".

---

## 5. Things czarr should adopt

### 5.1 Lazy bridge classes (only if/when we ship CPU-fallback codecs)

If czarr ever ships a numcodecs-compatible CPU fallback so users can
read their CPU-written stores on a no-GPU machine, the pattern at
`zarr-python:src/zarr/codecs/numcodecs/_codecs.py:120-170` is the right
template. Three classes, one per Zarr-v3 kind, all sharing a
`_NumcodecsCodec` base that holds the codec config and lazily
instantiates the underlying `Numcodec` via `cached_property`. **Don't
fork it — re-use it.** czarr can publish v3-native GPU codecs *and* let
users fall back to `zarr.codecs.numcodecs.LZ4` on CPU when the
GPU codec isn't available. The bridge gives them that for free as long
as the on-disk bitstream matches.

This is the v3 codec configuration JSON that the bridge expects:

```json
{
  "name": "numcodecs.lz4",
  "configuration": {"id": "lz4", "acceleration": 1}
}
```

czarr's LZ4 currently emits `{"name": "lz4", "configuration":
{"acceleration": 1}}` — the unprefixed name. That's correct for a
zarr-native codec (it shadows the `numcodecs.lz4` form). The two coexist:
zarr v3 stores can encode either way. The byte payload is identical, so a
reader of either form gets the right data.

### 5.2 Entry-point plug-in pattern

If we want third parties to extend czarr's codec set without forking, we
can publish a discriminator entry-point group:

```toml
[project.entry-points."czarr.codecs"]
my_codec = "my_pkg.codecs:MyCodec"
```

with `czarr.codecs.__init__` doing:

```python
from importlib.metadata import entry_points
for ep in entry_points().select(group="czarr.codecs"):
    cls = ep.load()
    zarr.registry.register_codec(cls.codec_name, cls)
```

This is exactly the numcodecs pattern (`src/numcodecs/registry.py:16-22`).
Optional / low priority — Zarr v3's own
`zarr.registry.register_codec_pipeline` and entry-point group already
exists, and external packages can use it directly. We don't need a czarr
group.

### 5.3 Backward-compatibility fixtures

`tests/common.py:155-244` defines `check_backwards_compatibility`. The
pattern:

```python
fixture_dir = os.path.join('fixture', codec_id, prefix or '')
# 1. Save each input array as fixture/<codec_id>/array.<NN>.npy (one-time)
# 2. For each codec config, save:
#    fixture/<codec_id>/codec.<JJ>/config.json
#    fixture/<codec_id>/codec.<JJ>/encoded.<NN>.dat
# 3. On test run: load the fixtures, decode, assert equality.
```

The `fixture/` directory is committed to git. The first run writes
fixtures; subsequent runs verify they decode the same way. **This is
exactly what czarr needs for bitstream-compat tests against numcodecs.**

A concrete czarr adaptation:

```
.tests/fixture/
    numcodecs_lz4/
        array.00.npy        ← random float32, written once
        codec.00/
            config.json     ← {"id": "lz4", "acceleration": 1}
            encoded.00.dat  ← numcodecs.LZ4().encode(array.00)
```

The czarr test then loads `array.00.npy`, runs
`czarr.codecs.compressors.LZ4().encode_chunk(...)` (with
`_bitstream_kind=WITH_UNCOMPRESSED_SIZE`), and asserts the output is
byte-identical to `encoded.00.dat`. Conversely, the test takes
`encoded.00.dat`, decodes it through czarr's LZ4, and asserts the result
matches `array.00.npy`.

This catches both directions: CPU-write→GPU-read and GPU-write→CPU-read.
The fixtures are generated on a host with `numcodecs` installed (no GPU
required); the tests run on the GPU host. **High-priority adoption** —
without these fixtures we cannot make a credible "byte-compatible" claim.

### 5.4 The `ensure_contiguous_ndarray` shape-flattening pattern

`src/numcodecs/compat.py:64-114` makes every numcodecs codec safe to
assume "I have a contiguous 1-D buffer of bytes": it flattens contiguous
arrays, views datetime/timedelta as int64, rejects non-contiguous input,
and enforces an optional `max_buffer_size`. Today czarr's codecs accept
Zarr v3 `Buffer` / `NDBuffer` directly — fine for v3-native. If we ever
expose a numcodecs-style direct call (`czarr.LZ4().encode(cupy_array)`),
this is the entry-point pattern to mirror.

---

## 6. Things czarr should NOT adopt

### 6.1 Per-element Python loops

`src/numcodecs/vlen.pyx:88-119` and `src/numcodecs/categorize.py:62-66`
both loop in Python or at-best in Cython over per-item operations. On
modern CPUs with vector ISAs these are already slow; on a GPU they
collapse to "one thread launches a kernel per element" which is
catastrophic. czarr should refuse to ship codecs that don't have a
batched per-chunk formulation.

### 6.2 Synchronous mutex-protected global library context

`src/numcodecs/blosc.pyx:480-507` and the `get_mutex()` /
`_get_use_threads()` pair. Blosc has a thread-pool global state; the
codec uses a process-wide `multiprocessing.Lock` to serialise access.
Even on the host this is annoying (no parallel encode/decode unless you
use the `_ctx` variant). On the GPU the analogue would be a CUDA context
lock, which kills the whole point of stream-overlapped pipelines. **Use
per-call contexts, never globals.** czarr's nvCOMP wrapper does this
correctly via `nvcomp.Codec(...)` instances; we should stay there.

### 6.3 Forced full-array allocation on decode

`src/numcodecs/zstd.pyx:268-356` (`stream_decompress`) allocates a
worst-case 128 KB output and grows it with `realloc`. That's fine for
CPU. For GPU we always know the output size from the chunk metadata
(Zarr stores `shape`, the codec stores `dtype`, the math is
deterministic), so `compute_encoded_size` should return an exact number
and the caller should pre-allocate. No grow loops.

### 6.4 Sync-only API surface

numcodecs's `encode`/`decode` are blocking. The bridge wraps them in
`asyncio.to_thread`. czarr's Phase 2 design (per `04-cuda-array-architecture.md`
and `SUMMARY.md`) pipelines decode with read; if the codec's "encode"
is a synchronous blocking call, the pipeline degenerates. czarr's
`CudaBytesBytesCodec.encode` is already an `async def` that runs nvCOMP
on the active CUDA stream and returns control immediately. Keep it that
way; do not paper over with `asyncio.to_thread`.

### 6.5 `np.dtype('O')` object arrays anywhere in the codec contract

VLen, Categorize, JSON, MsgPack, Pickle, Base64 all traffic in
`dtype=object` numpy arrays. Object arrays on the GPU don't make sense —
cupy doesn't support them at all. czarr's public API should reject object
arrays loudly at the surface.

---

## 7. Codec wishlist — numcodecs codecs czarr could add

Scored on (a) GPU-friendliness, (b) user demand, (c) implementation
effort. Each on a 1-5 scale; high is "good".

| Codec              | Type            | GPU-friendly | User demand | Effort (inv) | Notes                              |
| ------------------ | --------------- | -----------: | ----------: | -----------: | ---------------------------------- |
| Quantize           | Array→Array     |            5 |           3 |            5 | One-liner. Wraps FixedScaleOffset. |
| AsType             | Array→Array     |            5 |           2 |            5 | Pure cast. Trivial.                |
| PackBits           | Array→Array     |            4 |           2 |            4 | `cp.packbits` + 1-byte prefix.     |
| CRC32 / CRC32C     | Bytes→Bytes     |            5 |           4 |            3 | cuda-python has no native CRC32C; needs custom kernel or cuCollections. |
| Fletcher32         | Bytes→Bytes     |            4 |           2 |            3 | Custom kernel. netCDF interop.     |
| Adler32            | Bytes→Bytes     |            3 |           2 |            3 | Sequential-feel; can be parallelised. |
| BZ2                | Bytes→Bytes     |            1 |           2 |            1 | No GPU BZ2 implementation exists. Skip. |
| LZMA               | Bytes→Bytes     |            1 |           1 |            1 | Same as BZ2. Skip.                 |
| Blosc (host-compat) | Bytes→Bytes    |            3 |           4 |            2 | Wrap nvCOMP, emit Blosc-1 header.  |
| ZFP                | Array→Bytes     |            4 |           3 |            2 | libzfp has CUDA backend.           |
| PCodec             | Array→Bytes     |            2 |           2 |            1 | Rust crate, no GPU port yet.       |
| VLen / Categorize  | Array→Bytes     |            1 |           1 |            1 | Skip — fundamentally CPU.          |
| JSON / MsgPack     | Array→Bytes     |            1 |           1 |            1 | Skip.                              |

**v0.1 candidates from this list**: Quantize (cheap), AsType (cheap), and
CRC32C (already on the v0.1 roadmap per `SUMMARY.md`). PackBits is a
freebie if a real user shows up. ZFP is a genuine v1+ research project.

---

## 8. Bitstream compatibility test pattern

The numcodecs `check_backwards_compatibility` (`tests/common.py:155-244`)
is the right template adapted for our needs:

```python
# Generated once on a host with `numcodecs` available, committed to git.
# fixture/
#     czarr_lz4/
#         array.00.npy         ← np.random.bytes(8192).view(np.float32)
#         array.01.npy         ← np.linspace(0, 1, 1024).astype("<f4")
#         codec.00/
#             config.json      ← {"id": "lz4", "acceleration": 1}
#             encoded.00.dat   ← numcodecs.LZ4().encode(array.00)
#             encoded.01.dat   ← numcodecs.LZ4().encode(array.01)
```

The test:

```python
import json, numpy as np
from pathlib import Path

import numcodecs
from czarr.codecs.compressors import LZ4 as CzarrLZ4

FIXTURE = Path(__file__).parent / "fixture" / "czarr_lz4"

@pytest.mark.parametrize("arr_path", sorted(FIXTURE.glob("array.*.npy")))
def test_lz4_bitstream_compat(arr_path):
    arr = np.load(arr_path)
    for codec_dir in sorted(FIXTURE.glob("codec.*")):
        cfg = json.loads((codec_dir / "config.json").read_text())
        encoded_path = codec_dir / f"encoded.{arr_path.stem.split('.')[-1]}.dat"
        expected_bytes = encoded_path.read_bytes()

        # 1. CPU-write → GPU-read
        gpu_codec = CzarrLZ4(**{k: v for k, v in cfg.items() if k != "id"})
        decoded = gpu_codec.decode_single_chunk(expected_bytes)  # returns cupy array
        np.testing.assert_array_equal(cp.asnumpy(decoded), arr.view(np.uint8))

        # 2. GPU-write → CPU-read
        produced_bytes = gpu_codec.encode_single_chunk(cp.asarray(arr.view(np.uint8)))
        # 2a. Bit-identical to numcodecs
        assert bytes(produced_bytes) == expected_bytes
        # 2b. Decodes through numcodecs
        cpu_codec = numcodecs.get_codec(cfg)
        roundtrip = cpu_codec.decode(produced_bytes)
        np.testing.assert_array_equal(np.frombuffer(roundtrip, dtype=arr.dtype), arr.ravel())
```

Three kinds of assertion:

1. **CPU→GPU read**: take the numcodecs output, decode through our codec,
   assert the array matches. This is the *practical* compat check —
   users wrote with numcodecs on CPU, they want to read on GPU.
2. **GPU→CPU read**: take our encoded output, decode through numcodecs.
   This is the dual.
3. **Byte equality**: our encoded output is byte-identical to numcodecs's
   for the same input. This is *aspirational*; for LZ4 with the
   uncompressed-size prefix it should hold (the framing is fixed), but
   the LZ4 block payload can legitimately differ between implementations
   that pick different match strategies. **Mark these as `xfail` for
   compressors and `assert_equal` for filters and checksums.**

For filters (Shuffle, Delta, BitRound, FixedScaleOffset) byte equality is
the *whole point*: there's only one correct answer per (input, config)
pair. For compressors only the decoded data needs to match; the
intermediate bytes are allowed to differ as long as the codec self-tests
the round-trip.

A `scripts/regen_fixtures.py` (run once when adding or changing a codec)
iterates `{codec_id: [arrays], codec_id: [configs]}` dicts, calls
`numcodecs.get_codec(cfg).encode(arr)` for each pair, and writes
`array.NN.npy` / `codec.JJ/config.json` / `codec.JJ/encoded.NN.dat`. The
fixtures are small (kilobytes per codec) so committing them is fine.
Regenerate, eyeball the diff, commit whenever the wire format could
change.

---

## 9. Summary

* **Codec ABC**: tiny — `encode(buf)` and `decode(buf, out=None)` plus
  `codec_id`, `get_config`, `from_config`. The default `__eq__` walks
  config, the default `__repr__` walks `__dict__`. czarr's
  `BytesBytesCodec`/`ArrayArrayCodec` subclassing already covers this for
  Zarr v3 — we don't need to re-expose the numcodecs ABC.
* **Registry**: two-tier — explicit `register_codec(cls)` plus
  entry-point discovery under `numcodecs.codecs`. zarr-python has its own
  `zarr.registry` for Zarr v3 codecs. The two are bridged through
  `zarr.codecs.numcodecs` (in zarr-python), which is the deprecated
  successor to `numcodecs.zarr3`. czarr should register only with
  `zarr.registry` (already does); the numcodecs entry point is
  unnecessary as long as we ship v3-native codecs.
* **Per-codec patterns**: LZ4 prepends a 4-byte LE uncompressed-size
  header (`src/numcodecs/lz4.pyx:91-95`); Zstd uses standard libzstd
  frames; Blosc wraps everything in a single multi-codec meta-format;
  checksums prepend or append 4 bytes LE; filters are tiny numpy
  expressions. czarr's existing implementations already match the
  important ones; the BitRound rounding tiebreaker is the one bug-shaped
  divergence to fix.
* **Adopt**: the fixture-based bitstream-compat test pattern
  (§5.3, §8); the lazy bridge classes only if we ever ship a CPU
  fallback; the `ensure_contiguous_ndarray` flattening discipline.
* **Don't adopt**: synchronous Python-loop codecs (VLen, Categorize,
  JSON, MsgPack, Pickle); global library state with process locks
  (Blosc's `get_mutex`); object dtype anywhere on the codec contract;
  worst-case allocate-and-grow loops on decode.
* **Wishlist**: Quantize, AsType, CRC32/CRC32C (already on the roadmap),
  PackBits, Fletcher32, host-compat Blosc, ZFP. Skip the object-codecs.

The next concrete action this report unlocks is the fixture-based
bitstream-compat test. Once we have `numcodecs.LZ4().encode(arr)`
byte-checked against `czarr.LZ4(WITH_UNCOMPRESSED_SIZE).encode(arr)`, the
"byte-for-byte compatible" claim from `SUMMARY.md` stops being
aspirational.
