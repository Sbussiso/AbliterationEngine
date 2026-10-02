"""Migrate-only alias: `eng` -> package `abliteration_engine` (legacy shim).

NO logic lives here. Delete this shim when the migration ends.
"""
from abliteration_engine.data import (
    BUILTIN_MARKERS,
    MARKERS_FP_EXPLICIT_V1,
    resolve_markers,
    resolve_probe_set,
)

__all__ = ["BUILTIN_MARKERS", "MARKERS_FP_EXPLICIT_V1", "resolve_markers",
           "resolve_probe_set"]