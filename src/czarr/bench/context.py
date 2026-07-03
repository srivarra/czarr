"""The handle a benchmark function receives — params + helpers, no boilerplate.

A registered bench is a function ``fn(ctx) -> BenchPlan``.  It pulls resolved
params off ``ctx``, builds whatever it needs (usually via ``ctx.fixture``), and
returns a :class:`BenchPlan` describing the timed region, the correctness check,
and the byte count.  The harness drives everything else (warmup, reps, median,
GiB/s, telemetry, output) so the bench itself stays declarative.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cupy as cp
import numpy as np


def to_np(x: Any) -> np.ndarray:
    """Host ndarray from a cupy or numpy array (for correctness checks)."""
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


@dataclass(slots=True)
class BenchPlan:
    """What a bench hands back: the timed body, its check, and the byte count.

    ``body`` is the region we time (must end device work + return its result so
    ``verify`` can inspect it).  ``verify`` gates timing — it runs once on a
    fresh ``body()`` result before the timing loop and a falsy/raising return
    fails the bench (never report fast-but-wrong).  ``nbytes`` is the logical
    (uncompressed) payload for GiB/s.  ``regime`` carries context that flips the
    verdict (compressed ratio, chunk size, …) onto every result row.
    """

    body: Callable[[], Any]
    nbytes: int
    verify: Callable[[Any], bool] | None = None
    regime: dict[str, Any] = field(default_factory=dict)
    setup: Callable[[], None] | None = None  # run once before warmup (not timed)


@dataclass(slots=True)
class BenchContext:
    """Resolved params + helpers handed to every bench function."""

    params: dict[str, Any]
    fixture: Any  # czarr.bench.fixtures.Fixtures — late-bound to avoid import cycle

    def p(self, key: str) -> Any:
        """Resolved value of a declared param."""
        return self.params[key]

    @staticmethod
    def sync() -> None:
        """Block until all device work issued so far completes."""
        cp.cuda.runtime.deviceSynchronize()

    @staticmethod
    def to_np(x: Any) -> np.ndarray:
        """Host ndarray from a cupy/numpy result (for the correctness check)."""
        return to_np(x)

    def plan(
        self,
        body: Callable[[], Any],
        nbytes: int,
        *,
        verify: Callable[[Any], bool] | None = None,
        regime: dict[str, Any] | None = None,
        setup: Callable[[], None] | None = None,
    ) -> BenchPlan:
        """Build the :class:`BenchPlan` the harness will drive (sugar over the dataclass)."""
        return BenchPlan(body=body, nbytes=nbytes, verify=verify, regime=regime or {}, setup=setup)
