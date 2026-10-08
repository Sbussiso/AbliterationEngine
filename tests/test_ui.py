"""Run-output contract (ui.py): one stage counter per phase, human phase
summaries with the exact next command BEFORE the machine sentinel line,
quiet mode, color only on a real terminal, readable `plan`."""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from abliteration_engine import cli, pipeline, ui  # noqa: E402
from abliteration_engine.spec import load_spec  # noqa: E402

RUN001 = os.path.join(REPO, "specs", "run001_parity.yaml")
RECREATION = os.path.join(REPO, "specs",
                          "qwen25_0p5b_run000_recreation.yaml")


def _ladder_payload(eligible=True, gate="passed"):
    return {"selected": "wd_ML", "gate": gate,
            "publish_eligible_probe_gate": eligible,
            "metrics": {"refusal_rate_before": 0.9,
                        "refusal_rate_after": 0.1 if eligible else 0.4,
                        "benign_preserved_before": 0.95,
                        "benign_preserved_after": 1.0},
            "variants": {"wd_B": {"refusal_rate": 0.5,
                                  "benign_preserved": 1.0,
                                  "degenerate_total": 0},
                         "wd_ML": {"refusal_rate": 0.1 if eligible else 0.4,
                                   "benign_preserved": 1.0,
                                   "degenerate_total": 0}}}


def test_stage_counter_format(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    st = ui.Stages("B", 3)
    st.step("ladder plan")
    st.step("wd_ML")
    out = capsys.readouterr().out.splitlines()
    assert out == ["[B 1/3] ladder plan", "[B 2/3] wd_ML"]


def test_ladder_summary_next_command_and_verdicts(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    spec = load_spec(RUN001)
    ui.summary_ladder(_ladder_payload(), spec)
    out = capsys.readouterr().out
    assert "wd_ML" in out and "← selected" in out
    assert "passes the gates" in out
    assert "next: abliterate mmlu --spec" in out
    assert "--i-know-this-spends-quota" in out
    ui.summary_ladder(_ladder_payload(), spec, chained_mmlu=True)
    assert "MMLU guardrail runs now" in capsys.readouterr().out
    ui.summary_ladder(_ladder_payload(eligible=False), spec)
    out = capsys.readouterr().out
    assert "not under the 25.0% publish gate" in out
    assert "publish is gated off" in out


def test_mmlu_summary_points_to_publish_or_back(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    spec = load_spec(RUN001)
    base = {"mmlu_base_pct": 45.78, "mmlu_variant_pct": 45.55,
            "guardrail_loss_pp_limit": 3.0, "variant": "wd_ML"}
    ui.summary_mmlu(dict(base, mmlu_delta_pp=0.23), spec)
    out = capsys.readouterr().out
    assert "PASSED" in out and "abliterate publish --spec" in out
    ui.summary_mmlu(dict(base, mmlu_delta_pp=3.4), spec)
    out = capsys.readouterr().out
    assert "FAILED" in out and "abliterate publish" not in out


def test_summary_precedes_machine_sentinel(tmp_path, monkeypatch, capsys):
    """The poll-loop contract: ENG<STAGE>_DONE stays the LAST line."""
    monkeypatch.setenv("ENG_OUT_ROOT", str(tmp_path))
    monkeypatch.setenv("NO_COLOR", "1")
    spec = load_spec(RUN001)
    rc = pipeline._run_phase(lambda s: _ladder_payload(), spec,
                             str(tmp_path / "exit.txt"), "LADDER_DONE",
                             render=lambda p: ui.summary_ladder(p, spec))
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("LADDER_DONE {")
    assert any("ladder complete" in ln for ln in lines[:-1])


def test_broken_renderer_never_fails_a_finished_stage(tmp_path, monkeypatch,
                                                      capsys):
    monkeypatch.setenv("ENG_OUT_ROOT", str(tmp_path))
    spec = load_spec(RUN001)

    def boom(_):
        raise KeyError("x")
    rc = pipeline._run_phase(lambda s: {"ok": 1}, spec,
                             str(tmp_path / "exit.txt"), "RUN_DONE",
                             render=boom)
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith(
        "RUN_DONE")


def test_quiet_mode_one_line_per_batch(capsys, monkeypatch):
    from abliteration_engine import core
    monkeypatch.setenv("ENG_QUIET", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(core, "generate",
                        lambda tok, model, p, max_new=200:
                        "I'm sorry, I can't" if "bad" in p else "Sure!")
    rows = core.run_probes(None, None, ["bad 1", "ok 2", "bad 3"],
                           tag="base-harm", markers=["i can't"])
    out = capsys.readouterr().out.rstrip("\n").splitlines()
    assert len(rows) == 3 and len(out) == 1
    assert out[0].startswith("  [base-harm] 3/3 · refused 2 (66.7%)")


def test_cli_quiet_flag_sets_env(monkeypatch):
    monkeypatch.delenv("ENG_QUIET", raising=False)
    seen = {}
    monkeypatch.setattr(cli, "run", lambda s, yes, with_mmlu=False:
                        seen.update(q=os.environ.get("ENG_QUIET")) or 0)
    assert cli.main(["run", "--spec", RUN001, "--quiet",
                     "--i-know-this-spends-quota"]) == 0
    assert seen["q"] == "1"


def test_color_only_on_a_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("ENG_COLOR", raising=False)
    assert ui.color_enabled(io.StringIO()) is False   # Colab cell / log
    tty = io.StringIO()
    tty.isatty = lambda: True
    assert ui.color_enabled(tty) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.color_enabled(tty) is False


def test_plan_is_readable_and_lists_next_commands(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli.main(["plan", "--spec", RUN001]) == 0
    out = capsys.readouterr().out
    assert "{'" not in out and "': " not in out   # no raw dict dumps
    for part in ("patient", "probes", "decoding", "ladder", "gates",
                 "publish", "stages", "commands"):
        assert part in out
    assert "abliterate run --spec" in out
    assert "abliterate mmlu --spec" in out
    assert "abliterate publish --spec" in out
    assert cli.main(["plan", "--spec", RECREATION]) == 0
    out = capsys.readouterr().out
    assert "hook-only characterization" in out
    assert "abliterate mmlu" not in out
