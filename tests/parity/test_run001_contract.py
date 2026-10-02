"""Run-001 parity contract (spec §5, CPU-safe subset).

Baseline metrics from the published Run 001
(tests/fixtures parity anchors, frozen from run qwen2.5-0.5b-002): refusal 0.875,
benign_preserved 0.9375, hook-ablated refusal 0.0. The parity verb must
read those values and MUST NOT fake parity_ok=true when v3 artifacts are
absent (honest-missing is exit 1, not a silent pass).
"""
import json
import os

import pytest

from abliteration_engine import cli

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BASELINE = os.path.join(REPO, "tests", "fixtures")
SPEC = os.path.join(REPO, "specs", "run001_parity.yaml")


def _run_parity():
    rc = 0
    try:
        rc = cli.main(["parity", "--baseline", BASELINE, "--spec", SPEC])
    finally:
        pass
    return rc


def test_baseline_metrics_present(capsys):
    """Run 001 baseline metrics are what we verified on the freeze."""
    rc = _run_parity()
    out = capsys.readouterr().out
    data = json.loads(out[out.index("{"):out.rindex("}") + 1])
    by_artifact = {c["artifact"]: c for c in data["checks"]}
    base = by_artifact["probes_baseline.json"]["baseline"]
    assert base["refusal_rate"] == pytest.approx(0.875)
    assert base["benign_preserved"] == pytest.approx(0.9375)
    hook = by_artifact["probes_hook_ablated.json"]["baseline"]
    assert hook["refusal_rate"] == pytest.approx(0.0)
    # No v3 artifacts exist yet -> honest parity failure
    assert data["parity_ok"] is False
    assert rc == 1


def test_parity_honest_missing_not_pass(capsys):
    rc = _run_parity()
    out = capsys.readouterr().out
    data = json.loads(out[out.index("{"):out.rindex("}") + 1])
    missing = [c for c in data["checks"]
               if c.get("v3") is None or c.get("missing")]
    assert missing, "expected honest missing-v3-artifact entries"
    assert rc == 1  # exit 1, never faked parity_ok=true