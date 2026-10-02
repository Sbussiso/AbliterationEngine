"""Packaged-CLI contract tests (packaging).

--spec must be accepted before OR after the verb; a missing --spec exits 2
(argparse usage error); all shipped specs plan + validate on CPU; `run`
refuses without the HITL quota acknowledgment (hard contract,
FREEZE_HANDOFF.md).
"""
import os

import pytest

from abliteration_engine import cli

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = [os.path.join(REPO, "specs", p)
         for p in ("run001_parity.yaml", "qwen25_7b.yaml")]


def test_plan_both_flag_orders():
    for argv in (["--spec", SPECS[0], "plan"], ["plan", "--spec", SPECS[0]]):
        rc = cli.main(argv)
        assert rc == 0, argv


def test_missing_spec_exits_2():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["plan"])
    assert excinfo.value.code == 2


def test_unknown_verb_exits_2():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["frobnicate", "--spec", SPECS[0]])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("spec", SPECS)
def test_validate_all_shipped_specs(spec):
    assert cli.main(["--spec", spec, "validate"]) == 0


@pytest.mark.parametrize("spec", SPECS)
def test_plan_all_shipped_specs_cpu_no_model_load(spec):
    assert cli.main(["plan", "--spec", spec]) == 0


def test_run_refuses_without_quota_flag(capsys):
    rc = cli.main(["run", "--spec", SPECS[0]])
    assert rc == 2
    out = capsys.readouterr().out
    assert "REFUSING" in out
    assert "--i-know-this-spends-quota" in out