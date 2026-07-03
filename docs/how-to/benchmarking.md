# Benchmarking with czarr-bench

**Goal:** reproduce czarr's published numbers on your hardware, or measure a change before trusting it.

Every performance claim in these docs comes from `czarr-bench`; results land as append-only JSON-lines in `bench/results/` and are tracked in git.

## Install and list

```bash
pip install "czarr[cu12,bench]"      # bench extra adds the typer CLI
czarr-bench list
```

## Run one bench

```bash
czarr-bench run zarr-read --param impl=lowlevel --param chunk_mib=128
czarr-bench run read-path --param impl=gds --reps 10 --warmup 3
```

Each bench validates bit-exactness against a CPU reference **before** timing — a fast-but-wrong result fails instead of reporting.

## Sweep the declared axes

```bash
czarr-bench sweep zarr-read          # impl x chunk_mib grid
czarr-bench sweep read-path
```

Sweeps isolate each parameter point in a child process by default, so a C-level abort or global-state mutation (`configure_gpu`, cuFile driver config) can't poison the rest of the grid.

## The benches

| Bench | Question it answers |
|---|---|
| `read-path` | Does cuFile/GDS beat pinned-bounce and pageable transfer on this storage? (decode excluded) |
| `blosc-e2e` | End-to-end blosc store read: GPU decode vs CPU + H2D |
| `zarr-read` | The four read stacks head-to-head: lowlevel vs tier-1 fast path vs zarr pipeline vs kvikio `GDSStore` |

The `zarr-read` kvikio arm needs kvikio installed (`uv run --with kvikio-cu12 czarr-bench ...`) — it is a comparison baseline, not a czarr dependency.

## Profiling

`--nsys` wraps the run in Nsight Systems (the harness brackets the timed region with `cudaProfilerStart/Stop`, and czarr's stages carry NVTX ranges):

```bash
czarr-bench run zarr-read --param impl=lowlevel --nsys
```

## Pitfalls

- **Never bench against tmpfs** — GDS reads return zeros from it. The bench fixtures refuse tmpfs and pick a real filesystem automatically, but a leaked `$TMPDIR` is the first thing to check when numbers look impossible.
- **Real GDS vs compat mode** changes every conclusion; each result row records `gds_available`.
- Results are keyed by `(bench, params, gpu, commit)` — diff them across commits with plain git tooling or pandas.
