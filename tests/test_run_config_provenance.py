"""Run-006 provenance gap (post-Colab review): run_config.json recorded
spec_hash but not the effective hook scope or ladder variant list itself
— a reader auditing artifacts alone could not tell a hook-only run from
a selected-scope run without resolving the spec. Pinned: write_run_config
always embeds hooks.scope (defaulting selected) and ladder.variants.
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from abliteration_engine import core  # noqa: E402
from abliteration_engine.spec import load_spec  # noqa: E402

RUN001 = os.path.join(REPO, "specs", "run001_parity.yaml")
RECREATION = os.path.join(REPO, "specs",
                          "qwen25_0p5b_run000_recreation.yaml")


def _cfg_for(spec_path, monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.chdir(tmp)
    spec = load_spec(spec_path)
    core.write_run_config(spec, tmp)
    return json.load(open(os.path.join(tmp, "run_config.json")))


def test_run001_config_records_scope_selected(monkeypatch):
    cfg = _cfg_for(RUN001, monkeypatch)
    assert cfg["hooks"]["scope"] == "selected"
    assert cfg["ladder_variants"], "run-001 must record its ladder list"


def test_recreation_config_records_scope_all_and_empty_ladder(monkeypatch):
    cfg = _cfg_for(RECREATION, monkeypatch)
    assert cfg["hooks"]["scope"] == "all"
    assert cfg["ladder_variants"] == []