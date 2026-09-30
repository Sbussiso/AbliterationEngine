"""Migrate-only alias: `eng` -> package `abliteration_engine` (FTT-20 shim).

NO logic lives here. Delete this shim when FTT-20 ends.
"""
from abliteration_engine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())