# czarr

GPU-native reading and writing of Zarr v3 arrays — nvCOMP codecs, GPUDirect Storage via cuFile, and an explicit low-level read path.

**~20 GiB/s** end-to-end reads on H100 + GDS. **10-30×** faster than CPU codecs at multi-megabyte chunks.

## Two tiers, one substrate

czarr exposes the same GPU read machinery at two altitudes — pick per call site, mix freely.

**Tier 1 — drop-in zarr.** One call plugs GPU codecs, a batched decode pipeline, and GPU buffer prototypes into zarr's official extension points. Existing code and existing CPU-written stores work unchanged:

```python
import czarr, zarr

czarr.configure_gpu()
arr = zarr.open_array("existing_store.zarr")
out = arr[:]          # cupy.ndarray, decoded on the GPU
```

**Tier 2 — explicit.** `czarr.core.Array` opens an array with a single metadata read and exposes the staged path — plan → coalesced cuFile reads → one batched nvCOMP decode — with per-call knobs and zero global state:

```python
from czarr.core import Array

arr = Array.open("existing_store.zarr")
out = arr[0:4, :, 8:24]               # cupy.ndarray
```

`CudaZarrArray` (tier 1's array type) bridges the two: basic-indexing reads route through a cached tier-2 `Array` automatically, and everything else falls back to zarr's machinery.

## Where to go

- New to czarr → [Getting started](getting-started.md)
- Existing CPU-written stores → [GPU-decode existing stores](how-to/read-cpu-stores.md)
- Reads without global state → [Explicit reads](how-to/explicit-reads.md)
- GPUDirect Storage → [GDS setup](how-to/gds.md)
- Why two tiers → [Architecture](explanation/architecture.md)
- Signatures → [API reference](reference/czarr.md)

## Installation

CUDA-version-specific wheels make the CUDA build an explicit choice:

```bash
pip install "czarr[cu12]"     # CUDA 12 systems
pip install "czarr[cu13]"     # CUDA 13 systems
```

!!! warning "Pick a CUDA extra"

    The base `pip install czarr` installs no CUDA wheels and fails at import. Linux only.
