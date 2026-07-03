"""``@benchmark`` registry — name → bench function + declared params/sweep.

Kills the loose-script sprawl: every perf question is a registered name, not a
new file.  Importing :mod:`czarr.bench.benches` fires the decorators so the
registry is populated by the time the CLI reads it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .context import BenchContext, BenchPlan

BenchFn = Callable[[BenchContext], BenchPlan]


@dataclass(slots=True)
class BenchSpec:
    """A registered benchmark: its function plus declared params and sweep axes."""

    name: str
    fn: BenchFn
    params: dict[str, Any] = field(default_factory=dict)  # name -> default
    sweep: dict[str, list[Any]] = field(default_factory=dict)  # name -> axis values
    doc: str = ""

    def resolve(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge CLI overrides onto declared defaults; reject unknown params."""
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise KeyError(f"{self.name}: unknown param(s) {sorted(unknown)}; known: {sorted(self.params)}")
        merged = dict(self.params)
        for k, v in overrides.items():
            merged[k] = _coerce(self.params[k], v)
        return merged


REGISTRY: dict[str, BenchSpec] = {}


def _coerce(default: Any, value: Any) -> Any:
    """Coerce a string CLI value to the type of the declared default."""
    if not isinstance(value, str) or isinstance(default, str):
        return value
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def benchmark(
    name: str,
    *,
    params: dict[str, Any] | None = None,
    sweep: dict[str, list[Any]] | None = None,
) -> Callable[[BenchFn], BenchFn]:
    """Register ``fn`` under ``name`` with its declared params and sweep axes."""

    def deco(fn: BenchFn) -> BenchFn:
        if name in REGISTRY:
            raise ValueError(f"duplicate benchmark name: {name!r}")
        REGISTRY[name] = BenchSpec(
            name=name,
            fn=fn,
            params=dict(params or {}),
            sweep=dict(sweep or {}),
            doc=(fn.__doc__ or "").strip().split("\n")[0],
        )
        return fn

    return deco


def load_all() -> dict[str, BenchSpec]:
    """Import bench modules (firing decorators) and return the registry."""
    from . import benches  # noqa: F401  -- side-effect import populates REGISTRY

    return REGISTRY
