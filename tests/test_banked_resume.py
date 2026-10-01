"""Banked-resume unit tests (2026-09-30, 3x Colab registry drops).

run_ladder must reuse COMPLETE prior-session probes_<name>.json instead of
re-running edit+save+verify+probe for variants a killed session already
finished — provided the banked file passes: (1) both sides exactly
n_probes rows, (2) every row has grader fields (refused flag NOT None,
output text present), (3) the LADDER selection payload records which
variants were banked (provenance for the paper).

Fail-safe: anything corrupt/incomplete/partial re-runs normally.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import abliteration_engine.edits as edits  # noqa: E402
from abliteration_engine import core  # noqa: E402

import tempfile  # noqa: E402


def _mk_spec(n_probes=64):
    return {
        "probe_sets": {
            "n_probes": n_probes,
            "harmful": "builtin:primary64_harmful",
            "harmless": "builtin:primary64_harmless",
        },
        "patient": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
        "run_card": {"run_number": 2, "patient": "qwen2.5-1.5b"},
    }


def _mk_out_dir(tmp, n_probes=64, missing_field=None, full=True):
    """Fake engine out_dir with baseline + a complete-ish probes_wd_B.json."""
    out = os.path.join(tmp, "eng_run_002_qwen2.5-1.5b")
    os.makedirs(out, exist_ok=True)
    rows = []
    for i in range(n_probes):
        row = {"i": i, "prompt": f"p{i}", "output": "t" * 20,
               "refused": 0, "degenerate": False, "gen_s": 8.0}
        if missing_field and i == 3:
            row.pop(missing_field)
        rows.append(row)
    json.dump({"harmful": rows, "harmless": [dict(r) for r in rows]},
              open(os.path.join(out, "probes_wd_B.json"), "w"))
    json.dump({"harmful": rows, "harmless": [dict(r) for r in rows]},
              open(os.path.join(out, "probes_baseline.json"), "w"))
    return out


def test_banked_resume_complete_file_is_reused(monkeypatch=None):
    with tempfile.TemporaryDirectory() as tmp:
        out = _mk_out_dir(tmp)
        spec = _mk_spec()
        # point core._out_dir at our fake tree
        monkeypatch_out = tmp
        orig = core._out_dir
        core._out_dir = lambda spec, create=False: monkeypatch_out + out[len(tmp):] \
            if False else out  # noqa: E731
        try:
            s = edits._banked_variant_summary(spec, "wd_B")
            assert s is not None, "complete file must be reused"
            assert s["banked_resume"] is True
            assert s["edit_info"]["banked_resume"]
        finally:
            core._out_dir = orig


def test_banked_resume_incomplete_file_reruns():
    for missing in ("output", "refused"):
        with tempfile.TemporaryDirectory() as tmp:
            out = _mk_out_dir(tmp, missing_field=missing)
            spec = _mk_spec()
            orig = core._out_dir
            core._out_dir = lambda spec, create=False: out  # noqa: E731
            try:
                s = edits._banked_variant_summary(spec, "wd_B")
                assert s is None, f"missing {missing} must force rerun"
            finally:
                core._out_dir = orig


def test_banked_resume_partial_rows_rerun():
    with tempfile.TemporaryDirectory() as tmp:
        out = _mk_out_dir(tmp, n_probes=50)  # session died mid-sweep
        spec = _mk_spec(n_probes=64)
        orig = core._out_dir
        core._out_dir = lambda spec, create=False: out  # noqa: E731
        try:
            assert edits._banked_variant_summary(spec, "wd_B") is None
        finally:
            core._out_dir = orig


def test_banked_resume_missing_file_rerun():
    with tempfile.TemporaryDirectory() as tmp:
        out = _mk_out_dir(tmp)
        os.remove(os.path.join(out, "probes_wd_B.json"))
        spec = _mk_spec()
        orig = core._out_dir
        core._out_dir = lambda spec, create=False: out  # noqa: E731
        try:
            assert edits._banked_variant_summary(spec, "wd_B") is None
        finally:
            core._out_dir = orig