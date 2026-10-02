"""Smoke-run regression (research-workstation, 2026-10-01 fresh-T4
Colab smoke of tutorials/01): a fresh-clone session without lm_eval used to
die silently inside `abliterate mmlu` (empty stdout, rc=3, reason buried in
mmlu_log.txt). The loud precheck contract now: rc=4 + sentinel "4" +
actionable REFUSING line, before any GPU spend. rc=3 stays reserved for
genuine parse failures (v2 sentinel contract).

Skipped where lm-eval IS installed: the blocked-import scenario can only be
simulated in an environment that truly lacks the module (CI / local venv).
Sessions that installed lm-eval (e.g. after Tutorial 1's install cell) get
the real precheck pass-through instead — which is the intended behavior.
"""
import importlib.util
import json

import pytest


@pytest.mark.skipif(importlib.util.find_spec("lm_eval") is not None,
                    reason="lm_eval installed -> blocked-import path "
                           "unreachable; precheck pass-through is intended")
def test_mmlu_precheck_loud_missing_lm_eval(monkeypatch, tmp_path):
    import abliteration_engine.mmlu as mmlu_mod
    from abliteration_engine import core

    monkeypatch.setattr(core, "_out_dir",
                        lambda spec, create=False: str(tmp_path))
    monkeypatch.setattr(core, "eng_base", lambda: str(tmp_path))
    (tmp_path / "selection.json").write_text(json.dumps({
        "selected": "wd_B",
        "selected_variant_dir": "/tmp/definitely-absent",
        "gate": "passed"}))

    rc = mmlu_mod.mmlu_phase("specs/run001_parity.yaml")

    assert rc == 4
    assert (tmp_path / "mmlu_exit_code.txt").read_text() == "4"
    log = (tmp_path / "mmlu_log.txt").read_text()
    assert "REFUSING: lm_eval not importable" in log