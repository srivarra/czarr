"""GPU-native zarr v3 codec pipeline.

Populated in Phases 1 + 2 of the pipeline refactor (epic ianyfe7m).
Will export :class:`CzarrPipeline`, plus :class:`StreamPool`,
:class:`PinnedHostPool`, and :class:`DeviceBufferPool` primitives.
"""
