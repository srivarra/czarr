"""Native cuda-python codec implementations.

Each module exposes a top-level ``decode_<codec>_native`` / ``encode_<codec>_native``
function that the public codec class dispatches to when ``backend="native"``.
The wire format must be bit-identical to nvCOMP / numcodecs for the same
codec — see ``docs/planning/research/cuda-array/08-two-tier-codecs.md`` §2
for the bitstream-identity rule.
"""
