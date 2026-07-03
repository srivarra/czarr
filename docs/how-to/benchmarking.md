# Benchmarking with czarr-bench

`czarr-bench` produced every number in these docs. Results append as JSON-lines under `bench/results/`, keyed by `(bench, params, gpu, commit)`.

```bash
pip install "czarr[cu12,bench]"
czarr-bench list
```

## Running

```bash
czarr-bench run zarr-read --param impl=lowlevel --param chunk_mib=128
czarr-bench run read-path --param impl=gds --reps 10 --warmup 3
czarr-bench sweep zarr-read          # the declared impl x chunk_mib grid
```

Each bench validates its output against a CPU reference before timing; a wrong result fails instead of reporting. Sweeps run each parameter point in a child process, so an abort or a global-state mutation in one point cannot affect the rest.

## Benches

| Bench | Measures |
|---|---|
| `read-path` | Storage-to-GPU transfer only: cuFile GDS vs pinned bounce vs pageable |
| `blosc-e2e` | End-to-end blosc store read, GPU decode vs CPU + H2D |
| `zarr-read` | Full reads through lowlevel, the tier-1 fast path, the zarr pipeline, and kvikio `GDSStore` |

The kvikio arm requires kvikio (`uv run --with kvikio-cu12 czarr-bench ...`); it is a comparison target, not a dependency.

## Profiling

`--nsys` wraps the run in Nsight Systems. The harness brackets the timed region with `cudaProfilerStart/Stop`, and the pipeline stages carry NVTX ranges:

```bash
czarr-bench run zarr-read --param impl=lowlevel --nsys
```

## Pitfalls

- cuFile cannot read tmpfs. The fixtures detect and avoid it, but check `$TMPDIR` when numbers look impossible.
- GDS versus compat mode changes every conclusion; each result row records `gds_available`.
