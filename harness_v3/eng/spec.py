"""Migrate-only alias: `eng` -> package `abliteration_engine` (FTT-20 shim).

Keeps the freeze-review test harness and FTT-20 Colab stubs (which import
`eng.spec` / `eng.cli` via PYTHONPATH) working while the package lands.
NO logic lives here. Delete this shim when FTT-20 ends.
"""
from abliteration_engine.spec import SpecError, load_spec, sha256_file, spec_hash

__all__ = ["SpecError", "load_spec", "sha256_file", "spec_hash"]