"""GPU compute kernels — shuffle / bitshuffle / checksum primitives.

Pure compute primitives consumed by codec ABCs.  Keeping them separate
from the codec classes lets a single kernel back multiple codec
entry-points (e.g. a Bitshuffle filter + a Blosc(bitshuffle) container).

Populated incrementally in Phases 3 / 5 of the pipeline refactor
(epic ianyfe7m).
"""
