"""czarr.lowlevel — explicit, staged GPU read-path plumbing.

Design: ``.planning/lowlevel-api-design.md``.  Stages:

* :mod:`czarr.lowlevel.plan` — derive-once metadata → coalesced byte ranges
* :mod:`czarr.lowlevel.coalesce` — range fusion primitives
* ``io`` / ``decode`` — cuFile reads and nvCOMP batch decode (next steps)

``czarr.core.Array`` composes these; each stage is also callable and
benchable on its own.
"""

from czarr.lowlevel.coalesce import ByteRange, FusedRead, coalesce_ranges, slice_into_outputs
from czarr.lowlevel.plan import DecodePlan, ReadRequest, ShardSpec, normalize_selection, open_plan, plan_from_metadata

__all__ = [
    "ByteRange",
    "DecodePlan",
    "FusedRead",
    "ReadRequest",
    "ShardSpec",
    "coalesce_ranges",
    "normalize_selection",
    "open_plan",
    "plan_from_metadata",
    "slice_into_outputs",
]
