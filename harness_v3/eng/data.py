"""Migrate-only alias: `eng` -> package `abliteration_engine` (FTT-20 shim).

NO logic lives here. Delete this shim when FTT-20 ends.
"""
from abliteration_engine.data import (
    BUILTIN_MARKERS,
    MARKERS_FP_EXPLICIT_V1,
    resolve_markers,
    resolve_probe_set,
)

__all__ = ["BUILTIN_MARKERS", "MARKERS_FP_EXPLICIT_V1", "resolve_markers",
           "resolve_probe_set"]