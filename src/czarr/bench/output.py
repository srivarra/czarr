"""Result row schema + sinks: a table to stdout and JSON-lines to disk.

One row per ``(bench, params, gpu, commit)``.  JSON-lines (append-only) make
results trackable across hardware and commits with no database — diff them in
git, load them with pandas.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parents[3] / "bench" / "results"


@dataclass(slots=True)
class Row:
    """One benchmark result: timing, the regime it ran in, and telemetry."""

    bench: str
    params: dict[str, Any]
    median_ms: float
    gib_s: float
    reps: int
    warmup: int
    gpu: str | None = None
    gpu_mem_gib: float | None = None
    gds_active: bool | None = None
    commit: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # regime + telemetry + ncu metrics

    def as_record(self) -> dict[str, Any]:
        """Flat dict for JSON output — ``extra`` keys promoted to top level."""
        d = asdict(self)
        d.update(d.pop("extra"))  # flatten regime/telemetry to top level
        return d


def git_commit() -> str | None:
    """Short HEAD hash so each row is tied to the code that produced it."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def write_jsonl(row: Row, path: Path | None = None) -> Path:
    """Append ``row`` as one JSON line (default: ``bench/results/<name>.jsonl``)."""
    path = path or (RESULTS_DIR / f"{row.bench}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row.as_record()) + "\n")
    return path


def render_table(rows: list[Row]) -> str:
    """Compact fixed-width table; one line per row."""
    if not rows:
        return "(no rows)"
    lines = []
    for r in rows:
        params = " ".join(f"{k}={v}" for k, v in r.params.items())
        regime = " ".join(
            f"{k}={v}" for k, v in r.extra.items() if k in ("compressed_ratio", "compressed_mib", "ncu_skipped")
        )
        lines.append(
            f"{r.bench:<18} {params:<28} {r.median_ms:8.2f} ms  {r.gib_s:6.1f} GiB/s  "
            f"[{r.gpu or '?'} gds={r.gds_active} {regime}]"
        )
    return "\n".join(lines)
