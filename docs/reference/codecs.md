# czarr.codecs

Every codec accepts `chunk_size` (nvCOMP internal chunk, default 64 KiB), `checksum_policy` (default `Checksum.NO_COMPUTE_NO_VERIFY`), `device_id`, and `cuda_stream` (any `__cuda_stream__`-compliant object). Compat codecs additionally accept `level` / `checksum` for metadata round-trip with their CPU equivalents.

## Compressors — native bitstream

nvCOMP-only formats, maximum throughput. Registered as zarr codec entry points, so stores round-trip by name.

::: czarr.codecs.compressors.native
    options:
      members: [ANS, Bitcomp, Cascaded, Deflate, GDeflate, Snappy]

## Compressors — compat bitstream

Bit-identical with libzstd / liblz4 / libdeflate / libz. After [`configure_gpu`][czarr.configure_gpu] they shadow the CPU codecs in zarr's registry.

::: czarr.codecs.compressors.zstd

::: czarr.codecs.compressors.lz4

::: czarr.codecs.compressors.gzip

::: czarr.codecs.compressors.zlib

::: czarr.codecs.compressors.blosc

## Filters

::: czarr.codecs.filters.shuffle

::: czarr.codecs.filters.delta

::: czarr.codecs.filters.fixedscaleoffset

::: czarr.codecs.filters.bitround

## Sharding

::: czarr.codecs.sharding.CzarrShardingCodec

## Base machinery

::: czarr.codecs.base.CudaBytesBytesCodec

::: czarr.codecs.base.Checksum
