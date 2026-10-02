"""Banked-resume unit tests (2026-09-30, 3x Colab registry drops).

run_ladder must reuse COMPLETE prior-session probes_<name>.json instead of
re-running edit+save+verify+probe for variants a killed session already
finished — provided the banked file passes: (1) both sides exactly
n_probes rows, (2) every row has grader fields (refused flag NOT None,
output text present), (3) the LADDER selection payload records which
variants were banked (provenance for the paper).

Fail-safe: anything corrupt/incomplete/partial re-runs normally.

CPU-CI note: tests/test_phase_ports.py locks that edits (torch-bound) is
imported only on GPU paths; these tests exercise the pure-resume branch of
run_ladder, so torch is stubbed with a lightweight fake — the resume path
touches summaries and file naming only, never tensor ops.
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.nn = types.SimpleNamespace(
        Parameter=lambda *a, **k: types.SimpleNamespace())
    _fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    _fake_torch.__version__ = "0-fake"
    _fake_torch.cuda.is_available = lambda: False
    _fake_torch.from_numpy = lambda *a, **k: None
    _fake_torch.eye = lambda *a, **k: None
    _fake_torch.outer = lambda *a, **k: None
    sys.modules["torch"] = _fake_torch

import tempfile  # noqa: E402

import abliteration_engine.edits as edits  # noqa: E402,F811
from abliteration_engine import core  # noqa: E402


def _mk_spec(n_probes=64):
    return {
        "probe_sets": {
            "n_probes": n_probes,
            "harmful": "builtin:primary64_harmful",
            "harmless": "builtin:primary64_harmless",
            "refusal_markers": "builtin:fp_explicit_v1",
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

# ---- provenance (code review: stale banked probes were reused blindly) ----

def _with_out(out):
    orig = core._out_dir
    core._out_dir = lambda spec, create=False: out  # noqa: E731
    return orig


def _bank_with_provenance(out, prov):
    p = os.path.join(out, "probes_wd_B.json")
    d = json.load(open(p))
    d["_provenance"] = prov
    json.dump(d, open(p, "w"))


def test_banked_resume_provenance_mismatch_reruns():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        out = _mk_out_dir(tmp)
        spec = _mk_spec()
        spec["decoding"] = {"max_new_tokens": 200, "seed": 0}
        spec["ladder"] = {"variants": ["wd_B"]}
        old = edits.variant_provenance(spec, "wd_B", [],
                                       {"dir_B": np.ones(4)})
        new = edits.variant_provenance(spec, "wd_B", [],
                                       {"dir_B": np.arange(4.0)})
        assert old["fingerprint"] != new["fingerprint"]
        _bank_with_provenance(out, old)
        orig = _with_out(out)
        try:
            assert edits._banked_variant_summary(spec, "wd_B", new) is None, \
                "stale probes (different stage-A direction) must re-run"
            s = edits._banked_variant_summary(spec, "wd_B", old)
            assert s is not None and s["banked_provenance"] == "verified"
        finally:
            core._out_dir = orig


def test_banked_resume_legacy_file_flagged_unverified():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        out = _mk_out_dir(tmp)  # no _provenance key: pre-fix banked file
        spec = _mk_spec()
        spec["decoding"] = {"max_new_tokens": 200, "seed": 0}
        spec["ladder"] = {"variants": ["wd_B"]}
        prov = edits.variant_provenance(spec, "wd_B", [],
                                        {"dir_B": np.ones(4)})
        orig = _with_out(out)
        try:
            s = edits._banked_variant_summary(spec, "wd_B", prov)
            assert s is not None and s["banked_provenance"] == "unverified"
        finally:
            core._out_dir = orig


def test_marker_mode_change_changes_fingerprint():
    import numpy as np
    spec = _mk_spec()
    spec["decoding"] = {"max_new_tokens": 200, "seed": 0}
    spec["ladder"] = {"variants": ["wd_B"]}
    spec["probe_sets"]["refusal_markers"] = "builtin:fp_explicit_v1"
    a = edits.variant_provenance(spec, "wd_B", [], {"dir_B": np.ones(4)})
    spec["probe_sets"]["marker_mode"] = "v2"
    b = edits.variant_provenance(spec, "wd_B", [], {"dir_B": np.ones(4)})
    assert a["fingerprint"] != b["fingerprint"]


def test_select_variant_honors_degenerate_max():
    def cands():
        return [{"variant": "wd_B", "ladder_index": 0, "refusal_rate": 0.1,
                 "benign_preserved": 1.0, "degenerate_total": 1}]
    sel, _ = edits.select_variant(cands(), 1.0, 0.25, degenerate_max=0)
    assert sel["passes_gate"] is False
    sel, ok = edits.select_variant(cands(), 1.0, 0.25, degenerate_max=1)
    assert sel["passes_gate"] is True and ok is True
