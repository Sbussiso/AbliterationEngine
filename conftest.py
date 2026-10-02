"""Pytest path wiring: make `src/` (and repo root) importable in-place
without installation.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
for p in (_HERE, _SRC):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)