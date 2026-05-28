"""Native cuda.compute filter implementations.

Each module exposes ``decode_<codec>_native`` / ``encode_<codec>_native``
functions that the filter codec dispatches to when ``backend="cccl"``.
Output must be bit-identical to the cupy backend for the same filter.

(Compressors are pure nvCOMP wrappers — czarr no longer ships a
hand-written native compressor.  These modules serve the filters:
Delta and FixedScaleOffset.)
"""
