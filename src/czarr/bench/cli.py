"""``czarr-bench`` — the unified benchmark CLI (typer).

    czarr-bench list
    czarr-bench run <name> [--param k=v ...] [--reps N --warmup M]
                           [--nsys] [--ncu] [--isolate] [--json out.jsonl]
    czarr-bench sweep <name> --param chunk_mib=8,128,256 --param store=gpu,local

``--isolate`` runs each param-point in a child process: a C-level ``SIGABRT``
(libcufile assert) or a global-state mutation (``configure_gpu``, nvcomp temp)
then can't take down the rest of a sweep.  Sweeps isolate by default.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from typing import Annotated

import typer

from .output import Row, render_table, write_jsonl
from .registry import BenchSpec, load_all

app = typer.Typer(add_completion=False, help="czarr GPU benchmark harness")

_ROW_SENTINEL = "__CZARR_ROW__"


def _parse_params(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise typer.BadParameter(f"expected k=v, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _spec_or_exit(name: str) -> BenchSpec:
    reg = load_all()
    if name not in reg:
        typer.secho(f"unknown bench {name!r}. known: {sorted(reg)}", fg="red", err=True)
        raise typer.Exit(2)
    return reg[name]


@app.command("list")
def list_benches() -> None:
    """List registered benchmarks with their params and sweep axes."""
    reg = load_all()
    if not reg:
        typer.echo("(no benchmarks registered)")
        return
    for name in sorted(reg):
        s = reg[name]
        typer.secho(name, fg="cyan", bold=True)
        typer.echo(f"    {s.doc}")
        if s.params:
            typer.echo(f"    params: {s.params}")
        if s.sweep:
            typer.echo(f"    sweep:  {dict(s.sweep)}")


def _run_one(spec: BenchSpec, overrides: dict[str, str], *, reps: int, warmup: int, nsys: bool, ncu: bool) -> Row:
    from .harness import run_bench  # deferred: imports cupy

    params = spec.resolve(overrides)
    return run_bench(spec, params, reps=reps, warmup=warmup, nsys=nsys, ncu=ncu)


def _run_isolated(name: str, overrides: dict[str, str], *, reps: int, warmup: int, nsys: bool, ncu: bool) -> Row:
    """Re-exec ``run`` for one param-point in a child process; parse its row."""
    cmd = [sys.executable, "-m", "czarr.bench", "run", name, "--reps", str(reps), "--warmup", str(warmup), "--emit-row"]
    if nsys:
        cmd.append("--nsys")
    if ncu:
        cmd.append("--ncu")
    for k, v in overrides.items():
        cmd += ["--param", f"{k}={v}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        typer.secho(f"[isolate] {name} {overrides} exited {proc.returncode}", fg="red", err=True)
        typer.echo(proc.stderr, err=True)
        raise typer.Exit(proc.returncode)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_ROW_SENTINEL):
            d = json.loads(line[len(_ROW_SENTINEL) :])
            return Row(
                **{k: d[k] for k in ("bench", "params", "median_ms", "gib_s", "reps", "warmup")},
                gpu=d.get("gpu"),
                gpu_mem_gib=d.get("gpu_mem_gib"),
                gds_active=d.get("gds_active"),
                commit=d.get("commit"),
                extra={k: v for k, v in d.items() if k not in Row.__slots__},
            )
    raise typer.Exit(1)


@app.command()
def run(
    name: str,
    param: Annotated[list[str] | None, typer.Option(help="override: k=v (repeatable)")] = None,
    reps: int = 5,
    warmup: int = 2,
    nsys: Annotated[bool, typer.Option(help="cudaProfilerStart/Stop window for nsys")] = False,
    ncu: Annotated[bool, typer.Option(help="collect Nsight Compute kernel metrics")] = False,
    isolate: Annotated[bool, typer.Option(help="run in a child process")] = False,
    json_out: Annotated[str, typer.Option("--json", help="append row to this JSONL file")] = "",
    emit_row: Annotated[bool, typer.Option(hidden=True, help="print row JSON sentinel (isolate child)")] = False,
) -> None:
    """Run one benchmark and print its result row."""
    spec = _spec_or_exit(name)
    overrides = _parse_params(param or [])
    row = (
        _run_isolated(name, overrides, reps=reps, warmup=warmup, nsys=nsys, ncu=ncu)
        if isolate
        else _run_one(spec, overrides, reps=reps, warmup=warmup, nsys=nsys, ncu=ncu)
    )
    typer.echo(render_table([row]))
    path = write_jsonl(row) if not json_out else write_jsonl(row, __import__("pathlib").Path(json_out))
    typer.secho(f"-> {path}", fg="green", err=True)
    if emit_row:
        print(_ROW_SENTINEL + json.dumps(row.as_record()))


@app.command()
def sweep(
    name: str,
    param: Annotated[list[str] | None, typer.Option(help="axis: k=v1,v2,v3 (repeatable)")] = None,
    reps: int = 5,
    warmup: int = 2,
    nsys: bool = False,
    ncu: bool = False,
    isolate: Annotated[bool, typer.Option(help="run each point in a child process")] = True,
) -> None:
    """Sweep a benchmark over the cartesian product of --param axes."""
    spec = _spec_or_exit(name)
    axes = {k: v.split(",") for k, v in _parse_params(param or []).items()} or {
        k: [str(x) for x in vs] for k, vs in spec.sweep.items()
    }
    if not axes:
        typer.secho(f"{name}: no sweep axes (pass --param k=v1,v2 or declare a sweep)", fg="red", err=True)
        raise typer.Exit(2)
    keys = list(axes)
    rows: list[Row] = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        overrides = dict(zip(keys, combo, strict=True))
        typer.secho(f"--- {name} {overrides} ---", fg="yellow", err=True)
        try:
            row = (
                _run_isolated(name, overrides, reps=reps, warmup=warmup, nsys=nsys, ncu=ncu)
                if isolate
                else _run_one(spec, overrides, reps=reps, warmup=warmup, nsys=nsys, ncu=ncu)
            )
            rows.append(row)
        except typer.Exit:
            typer.secho(f"  point failed, continuing sweep: {overrides}", fg="red", err=True)
    typer.echo("\n" + render_table(rows))


if __name__ == "__main__":
    app()
