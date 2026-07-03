"""Importing this package fires the ``@benchmark`` decorators (registry side-effect).

Add a module here and import it below to register its benches.
"""

from __future__ import annotations

from . import blosc, read_path

__all__ = ["blosc", "read_path"]
