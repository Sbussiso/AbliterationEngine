"""Migrate-only alias: `eng` -> package `abliteration_engine` (legacy shim).

NO logic lives here. Delete this shim when the migration ends.
"""
from abliteration_engine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())