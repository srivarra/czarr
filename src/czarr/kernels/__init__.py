"""GPU compute kernels — bitshuffle primitives.

Pure compute primitives consumed by codec ABCs.  Keeping them separate
from the codec classes lets a single kernel back multiple codec
entry-points (e.g. a Blosc(bitshuffle) container).
"""
