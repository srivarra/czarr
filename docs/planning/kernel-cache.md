# persistent kernel cache wiring

Wire up cross-process kernel caches so cold-start JIT cost evaporates after the first run on a node.

## Motivation

Each fresh Python process currently pays JIT compile cost for:

| Kernel | Backend | Trigger | Cold cost (A40) |
|---|---|---|---|
| `bitunshuffle_blosc` | `cupy.RawKernel` | First Blosc bitshuffle decode | ~200 ms |
| `byteunshuffle_kernel` (cuTile) | `cuda.tile.compile_tile` | First `ct.launch` per `(typesize, nelem)` shape | 300-500 ms |
| `byteshuffle_kernel` (cuTile) | Same | First encode at a shape | 300-500 ms |
| Any future RawKernels / cuTile kernels | Same | First call | 100-500 ms |

On a SLURM benchmark sweep (`bench.zarr.pipeline_sweep` etc.), every job pays >1 second of JIT before useful work. On a long-running serving process it amortises away — but for short jobs, batch reruns, and CI it's pure waste.

## Locked decisions

1. **One cache root**: `${CZARR_KERNEL_CACHE:-$HOME/.cache/czarr/kernels}` (overridable; defaults to user home so multi-user nodes don't collide).
2. **Per-backend subdirs**: `cupy/`, `cutile/`, `cuda-core/` so each backend's atomic-write semantics stay independent.
3. **No deletion / eviction logic at our layer**: each backend's cache handles its own LRU (cuda.core's `FileStreamProgramCache` does it; cupy's CUPY_CACHE_DIR uses its own scheme).
4. **Activation in `configure_gpu()`**: single entrypoint that already sets the rest of the runtime knobs.
5. **No tests requiring cold cache**: tests run with the cache active; we don't artificially evict to measure cold cost.

## Architecture

```
configure_gpu(kernel_cache: Path | bool | None = None)
   │
   ├─ cupy cache dir   ──►  os.environ["CUPY_CACHE_DIR"]   = <root>/cupy
   │
   ├─ cuTile cache dir ──►  cuda.tile.config.cache_dir     = <root>/cutile
   │                        (or context.config.cache_dir,
   │                        depending on the active version's API)
   │
   └─ cuda.core kernels ──► via FileStreamProgramCache on
                            future cuda.core.Program-backed kernels
                            (no kernels there today — wire the substrate
                            so it's ready for Phase 3 of the buffer epic)
```

`kernel_cache` parameter behaviour:
- `None` (default) — use `$CZARR_KERNEL_CACHE` env var if set, else default location.
- `False` — explicitly disable persistent caching for this process.
- `Path` — use this directory.
- `True` — equivalent to `None` (kept for symmetry).

## Phases

### Phase 0 — cuTile compile cache wiring

- Probe `cuda.tile` for the right cache-dir hook (could be `ct.config.cache_dir`, `ct.set_cache_dir()`, or an env var depending on cuda-tile version).
- Add to `configure_gpu` so the first call materialises the dir and points the compiler at it.
- Verify a second process (same Python interpreter via `subprocess.run` in a test) gets a cache hit — measure compile time both passes.

Deliverable: cuTile kernels survive process restart.

### Phase 1 — cupy RawKernel cache wiring

- Set `CUPY_CACHE_DIR` env var inside `configure_gpu` (or use `cupy.cuda.compiler.set_persistent_cache_dir` if available).
- Same subprocess-restart test for `czarr.kernels.bitshuffle._BITUNSHUFFLE_KERNEL`.

Deliverable: RawKernels survive process restart.

### Phase 2 — `FileStreamProgramCache` for future cuda.core.Program kernels

- Initialise a `FileStreamProgramCache` instance pointed at `<root>/cuda-core`.
- Expose via `czarr.kernels._cache` so any future kernel written with `cuda.core.Program` can be wired through (`program._cache = ...`).
- No kernels use it today, but the substrate is there for the buffer epic + future codecs.

Deliverable: `czarr.kernels.get_program_cache()` returns an initialised cache.

### Phase 3 — bench: cold vs warm JIT cost

- New `bench/kernels/jit_cold_warm.py`:
  - Spawn a fresh subprocess, time `import czarr; czarr.configure_gpu(); arr[:]` end-to-end.
  - Repeat with cache pre-populated.
  - Report cold ms / warm ms / cache hit rate per backend.
- SLURM wrapper for both A40 (local) and H100.

Deliverable: documented cold→warm savings, numbers in `docs/planning/kernel-cache-results.md`.

## Out of scope

- Eviction policy (each backend has its own).
- Distributed cache (NFS-shared kernel cache across users) — defer; `FileStreamProgramCache` is atomic for readers but we'd need to validate concurrent writes from multiple users.
- CUDA module-load cache (`CUDA_CACHE_PATH`) — that's nvcc's own cache for non-JIT compiles; orthogonal.
