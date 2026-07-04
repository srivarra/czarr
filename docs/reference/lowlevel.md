---
icon: lucide/wrench
description: The staged read plumbing — planning, coalescing, I/O, decode.
tags:
  - API
---

# czarr.lowlevel

The staged read plumbing. Each stage is callable and benchable on its own; `read_array` composes them.

## Planning

::: czarr.lowlevel.open_plan

::: czarr.lowlevel.plan_from_metadata

::: czarr.lowlevel.DecodePlan

::: czarr.lowlevel.ReadRequest

::: czarr.lowlevel.normalize_selection

## Coalescing

::: czarr.lowlevel.coalesce_ranges

::: czarr.lowlevel.ByteRange

::: czarr.lowlevel.FusedRead

## I/O and decode

::: czarr.lowlevel.io.read

::: czarr.lowlevel.decode.decode

::: czarr.lowlevel.decode.read_array
