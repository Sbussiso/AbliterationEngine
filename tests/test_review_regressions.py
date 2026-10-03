"""Regression pins for the engine code review (CPU-only unless noted).

Each test names the defect it pins:
- publish KeyError'd on every engine-produced run_config.json ("layer"
  block was never written) and rendered a 3.0pp limit as "30pp";
- publish could pair the card with a non-selected variant dir / MMLU run,
  and the identity gate was hardcoded to one account;
- variant weight dirs were shared across runs (/content/<variant>);
- the run sentinel read "0" after stage A while the ladder still ran;
- ladder/mmlu ran without --i-know-this-spends-quota; flags given before
  the verb were silently clobbered by subparser defaults;
- MMLU exited 0 on a failed guardrail and reused banked lm-eval results
  without checking which model produced them;
- spec validation let max_new_tokens / n_probes go missing until GPU time;
- directions.readout_norm (opt-in fix for the double final-norm on
  direction B) is validated and never injected.
Torch-only checks (hook detach, readout norm math) importorskip torch.
"""
import json
import os
import sys
import types
from unittest import mock

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

import test_publish_contract as pc  # noqa: E402

from abliteration_engine import cli, core  # noqa: E402
from abliteration_engine.spec import SpecError, load_spec, spec_hash  # noqa: E402

RUN001 = os.path.join(REPO, "specs", "run001_parity.yaml")


def _spec_variant(tmp_path, mutate):
    payload = yaml.safe_load(open(RUN001))
    mutate(payload)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(payload))
    return str(p)


# ---- spec validation ------------------------------------------------------

@pytest.mark.parametrize("drop", [("decoding", "max_new_tokens"),
                                  ("probe_sets", "n_probes"),
                                  ("probe_sets", "n_pairs")])
def test_spec_requires_keys_gpu_stages_index(tmp_path, drop):
    p = _spec_variant(tmp_path, lambda d: d[drop[0]].pop(drop[1]))
    with pytest.raises(SpecError, match=drop[1]):
        load_spec(p)


@pytest.mark.parametrize("bad", [-1, 0.5, "0", True])
def test_spec_rejects_bad_degenerate_max(tmp_path, bad):
    p = _spec_variant(tmp_path,
                      lambda d: d["gates"].update({"degenerate_max": bad}))
    with pytest.raises(SpecError, match="degenerate_max"):
        load_spec(p)


def test_readout_norm_absent_never_injected():
    s = load_spec(RUN001)
    assert "directions" not in s
    assert spec_hash(s).startswith("54590e5d")  # Run-001 freeze unchanged
    assert core.readout_norm_mode(s) == "double"


def test_readout_norm_single_roundtrip_and_invalid(tmp_path):
    p = _spec_variant(tmp_path, lambda d: d.update(
        {"directions": {"readout_norm": "single"}}))
    s = load_spec(p)
    assert s["directions"] == {"readout_norm": "single"}
    assert core.readout_norm_mode(s) == "single"
    for bad in ({"readout_norm": "triple"}, {"oops": 1}, ["single"]):
        p = _spec_variant(tmp_path, lambda d: d.update({"directions": bad}))
        with pytest.raises(SpecError):
            load_spec(p)


# ---- variant dirs ------------------------------------------------------------

def test_variant_dirs_are_run_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("ENG_OUT_ROOT", str(tmp_path))
    monkeypatch.delenv("ENG_VARBASE", raising=False)
    a = {"run_card": {"run_number": 1, "patient": "p-a"}}
    b = {"run_card": {"run_number": 2, "patient": "p-b"}}
    da, db = core.variant_dir(a, "wd_ML"), core.variant_dir(b, "wd_ML")
    assert da != db
    assert da == os.path.join(str(tmp_path), "eng_run_001_p-a_variants",
                              "wd_ML")
    # sibling of (not inside) the artifacts dir
    assert not da.startswith(core._out_dir(a) + os.sep)
    monkeypatch.setenv("ENG_VARBASE", str(tmp_path / "vb"))
    assert core.variant_dir(a, "wd_ML").startswith(str(tmp_path / "vb"))


# ---- CLI ---------------------------------------------------------------------

@pytest.mark.parametrize("verb", ["ladder", "mmlu"])
def test_gpu_verbs_require_quota_flag(verb, capsys):
    assert cli.main([verb, "--spec", RUN001]) == 2
    assert "--i-know-this-spends-quota" in capsys.readouterr().out


def test_pre_verb_flags_not_clobbered(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "bundle",
                        lambda s, out_dir: seen.update(out=out_dir) or 0)
    cli.main(["--out-dir", "XYZ", "--spec", RUN001, "bundle"])
    assert seen["out"] == "XYZ"
    monkeypatch.setattr(
        cli, "run",
        lambda s, yes, with_mmlu=False: seen.update(ran=True) or 0)
    # quota flag BEFORE the verb must still pass the quota gate
    assert cli.main(["--i-know-this-spends-quota", "--spec", RUN001,
                     "run"]) == 0
    assert seen.get("ran") is True


# ---- pipeline sentinel ----------------------------------------------------------

def test_sentinel_stays_running_between_stage_a_and_ladder(tmp_path,
                                                            monkeypatch):
    import abliteration_engine
    from abliteration_engine import pipeline

    monkeypatch.setenv("ENG_OUT_ROOT", str(tmp_path))
    spec = load_spec(RUN001)
    exit_f = core.sentinel_exit()
    seen = {}
    monkeypatch.setattr(core, "from_spec", lambda s: {"ok": 1})

    def fake_ladder(s, ctx):
        seen["during_ladder"] = open(exit_f).read()
        return {"selected": "wd_B"}
    fake_edits = types.ModuleType("abliteration_engine.edits")
    fake_edits.run_ladder = fake_ladder
    monkeypatch.setitem(sys.modules, "abliteration_engine.edits", fake_edits)
    monkeypatch.setattr(abliteration_engine, "edits", fake_edits,
                        raising=False)
    assert pipeline.run_pipeline(spec) == 0
    assert seen["during_ladder"] == "running", \
        "poller would read stage-A '0' as done while the ladder runs"
    assert open(exit_f).read() == "0"


# ---- MMLU ------------------------------------------------------------------------

def _mmlu_env(tmp_path, monkeypatch, sel_extra=None):
    from abliteration_engine import mmlu as mmlu_mod

    monkeypatch.setattr(core, "_out_dir",
                        lambda spec, create=False: str(tmp_path))
    monkeypatch.setattr(core, "eng_base", lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "lm_eval", types.ModuleType("lm_eval"))
    sel = {"selected": "wd_ML", "gate": "passed",
           "selected_variant_dir": str(tmp_path / "wd_ML"),
           "selected_provenance": {"fingerprint": "f1"}}
    sel.update(sel_extra or {})
    (tmp_path / "selection.json").write_text(json.dumps(sel))
    return mmlu_mod


def _fake_lm_eval(accs, calls):
    """subprocess.run stand-in writing an lm-eval results file."""
    def run(cmd, capture_output=True, text=True):
        odir = cmd[cmd.index("--output_path") + 1]
        model_args = cmd[cmd.index("--model_args") + 1]
        tag = "base" if odir.endswith("base") else "variant"
        calls.append(tag)
        os.makedirs(os.path.join(odir, "m"), exist_ok=True)
        with open(os.path.join(odir, "m", "results_1.json"), "w") as f:
            json.dump({"results": {"mmlu": {"acc,none": accs[tag]}},
                       "config": {"model_args": model_args}}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def test_mmlu_guardrail_failure_exits_nonzero(tmp_path, monkeypatch):
    mmlu_mod = _mmlu_env(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(mmlu_mod.subprocess, "run",
                        _fake_lm_eval({"base": 0.60, "variant": 0.50}, calls))
    rc = mmlu_mod.mmlu_phase(RUN001)
    assert rc == 6
    assert (tmp_path / "mmlu_exit_code.txt").read_text() == "6"
    summ = json.loads((tmp_path / "mmlu_summary.json").read_text())
    assert summ["guardrail_30pp"] is False and summ["variant"] == "wd_ML"


def test_mmlu_banked_reuse_requires_matching_provenance(tmp_path,
                                                        monkeypatch):
    mmlu_mod = _mmlu_env(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(mmlu_mod.subprocess, "run",
                        _fake_lm_eval({"base": 0.60, "variant": 0.59}, calls))
    assert mmlu_mod.mmlu_phase(RUN001) == 0
    assert calls == ["base", "variant"]

    # same selection -> both sides reused, no lm-eval spend
    calls.clear()
    assert mmlu_mod.mmlu_phase(RUN001) == 0
    assert calls == []

    # a different stage-B result now occupies the same variant dir path:
    # the base side is reused, the variant side must be re-evaluated
    # (a spec_sha256-only change must NOT force a re-eval)
    _mmlu_env(tmp_path, monkeypatch,
              {"selected_provenance": {"fingerprint": "f1",
                                       "spec_sha256": "edited"}})
    assert mmlu_mod.mmlu_phase(RUN001) == 0
    assert calls == []
    _mmlu_env(tmp_path, monkeypatch,
              {"selected_provenance": {"fingerprint": "f2"}})
    assert mmlu_mod.mmlu_phase(RUN001) == 0
    assert calls == ["variant"]
    stale = [d for d in os.listdir(tmp_path / "mmlu_results")
             if d.startswith("variant.stale-")]
    assert stale, "stale variant results must be moved aside, not mixed"


# ---- publish ------------------------------------------------------------------------

def _publish(tmp_path, monkeypatch, cfg_mutate=None, mmlu_mutate=None,
             who=None, vdir=None, artifacts_mutate=None,
             publish_overrides=None):
    from abliteration_engine import publish as P

    out_dir = pc._eng_root_env(tmp_path, monkeypatch)
    mmlu_p, vdir_default = pc._fake_run_artifacts(out_dir, tmp_path)
    cfg_p = out_dir / "run_config.json"
    if cfg_mutate:
        cfg = json.loads(cfg_p.read_text())
        cfg_mutate(cfg)
        cfg_p.write_text(json.dumps(cfg))
    (out_dir / "layer_coherence.json").write_text(json.dumps(
        {"final_layer": 23,
         "best": {"decoder_layer": 17, "coherence": 0.664},
         "readout_space": {"coherence": 0.71}}))
    if mmlu_mutate:
        m = json.loads(mmlu_p.read_text())
        mmlu_mutate(m)
        mmlu_p.write_text(json.dumps(m))
    if artifacts_mutate:
        artifacts_mutate(out_dir)
    spec = pc._spec_file(tmp_path, monkeypatch, **(publish_overrides or {}))
    api = pc._fake_hub(tmp_path, files=("README.md", "config.json",
                                        "model.safetensors.index.json",
                                        "refusal_direction.npy"))
    if who is not None:
        api.whoami = lambda: who
    with mock.patch("huggingface_hub.HfApi", return_value=api):
        rc = P.publish_phase(spec, str(vdir or vdir_default), str(mmlu_p),
                             assume_publish=True)
    return rc, vdir_default


def test_publish_without_layer_block_uses_layer_coherence(tmp_path,
                                                          monkeypatch):
    rc, vdir = _publish(tmp_path, monkeypatch,
                        cfg_mutate=lambda c: c.pop("layer"))
    assert rc == 0
    card = (vdir / "README.md").read_text()
    assert "| chosen decoder layer | 17 / 24 " in card
    assert "| coherence (final-layer readout space, direction B) | 0.71 |" \
        in card
    assert "under 3.0pp" in card and "30pp" not in card


def test_from_spec_writes_layer_block():
    import inspect
    src = inspect.getsource(core.from_spec)
    assert '"layer": {"decoder_layer": L_star' in src


def test_publish_refuses_non_selected_variant_dir(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(AssertionError, match="not the selected variant"):
        _publish(tmp_path, monkeypatch, vdir=other)


def test_publish_refuses_mmlu_of_other_variant(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="MMLU summary evaluated"):
        _publish(tmp_path, monkeypatch,
                 mmlu_mutate=lambda m: m.update(variant="wd_ML"))


def test_publish_identity_is_namespace_owner(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="identity check failed"):
        _publish(tmp_path, monkeypatch, who={"name": "someone-else"})
    rc, _ = _publish(tmp_path, monkeypatch,
                     who={"name": "alice", "orgs": [{"name": "sbussiso"}]})
    assert rc == 0


def test_card_describes_only_the_published_variant(tmp_path, monkeypatch):
    """The card rendered the whole edit_desc dict (every variant's
    description as a Python repr) instead of the published one."""
    rc, vdir = _publish(tmp_path, monkeypatch)
    card = (vdir / "README.md").read_text()
    assert "{'wd_B'" not in card
    line = next(ln for ln in card.splitlines()
                if "persistent weight edit (published artifact" in ln)
    assert "`wd_B`" in line and "lm_head readout-space" in line
    assert "wd_ML" not in line
    assert "Hermes" not in card  # attribution only via publish.card_byline
    assert "Arditi et al. (2024) refusal-direction method" in card


def test_card_byline_and_hook_scope_all(tmp_path, monkeypatch):
    rc, vdir = _publish(
        tmp_path, monkeypatch,
        cfg_mutate=lambda c: c.update(hooks={"scope": "all"}),
        publish_overrides={"card_byline": "lab agent profile"})
    card = (vdir / "README.md").read_text()
    assert "(lab agent profile) on" in card
    assert "| hook (inference-time, all layers) |" in card
    assert "at every decoder layer" in card


def test_card_names_ara_method_for_ara_variant(tmp_path, monkeypatch):
    def to_ara(out_dir):
        sel = json.loads((out_dir / "selection.json").read_text())
        sel.update(selected="ara_8",
                   ladder_ara={"layers": [3, 4]})
        (out_dir / "selection.json").write_text(json.dumps(sel))
        (out_dir / "selection_candidates.json").write_text(json.dumps(
            [{"variant": "ara_8", "refusal_rate": 0.05,
              "benign_preserved": 1.0, "degenerate_total": 0}]))
    rc, vdir = _publish(
        tmp_path, monkeypatch, artifacts_mutate=to_ara,
        cfg_mutate=lambda c: c.update(ladder_ara={"rank": 8}),
        mmlu_mutate=lambda m: m.update(variant="ara_8"))
    assert rc == 0
    card = (vdir / "README.md").read_text()
    assert "produced with Arbitrary-Rank Ablation (ARA, Weidmann 2026)" in card
    assert "does not remove that direction" in card
    assert "`ara_8`" in card and "rank-8 LoRA" in card


# ---- torch-only (skipped in CPU CI) ------------------------------------------------

def test_scope_all_hooks_fully_detached():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "nn") or not hasattr(torch.nn, "Linear"):
        pytest.skip("torch stub in sys.modules")
    layers = [torch.nn.Linear(4, 4) for _ in range(3)]
    hook = core.AblationHook(torch.ones(4), "cpu", torch.float32)
    for m in layers:
        hook.attach(m)
    x = torch.randn(2, 4)
    for m in layers:
        m(x)
    assert hook.calls == 3
    hook.detach()
    for m in layers:
        m(x)
    assert hook.calls == 3, "scope=all hooks survived detach()"


# ---- second review ------------------------------------------------------------

def test_chained_mmlu_sentinel_reads_running_until_mmlu_ends(tmp_path,
                                                             monkeypatch):
    """run --with-mmlu: exit_code.txt read '0' after the ladder while MMLU
    still ran; it must read 'running' until MMLU writes the final code."""
    import abliteration_engine
    from abliteration_engine import mmlu as mmlu_mod
    from abliteration_engine import pipeline

    monkeypatch.setenv("ENG_OUT_ROOT", str(tmp_path))
    exit_f = core.sentinel_exit()
    seen = {}
    monkeypatch.setattr(core, "from_spec", lambda s: {"ok": 1})
    fake_edits = types.ModuleType("abliteration_engine.edits")
    fake_edits.run_ladder = lambda s, ctx: {"selected": "wd_B"}
    monkeypatch.setitem(sys.modules, "abliteration_engine.edits", fake_edits)
    monkeypatch.setattr(abliteration_engine, "edits", fake_edits,
                        raising=False)

    def fake_mmlu(path):
        seen["during_mmlu"] = open(exit_f).read()
        return 6
    monkeypatch.setattr(mmlu_mod, "mmlu_phase", fake_mmlu)
    assert pipeline  # imported for its run_pipeline used by cli.run
    assert cli.main(["run", "--spec", RUN001, "--with-mmlu",
                     "--i-know-this-spends-quota"]) == 6
    assert seen["during_mmlu"] == "running"
    assert open(exit_f).read() == "6"


def test_mmlu_rerun_removes_stale_summary(tmp_path, monkeypatch):
    mmlu_mod = _mmlu_env(tmp_path, monkeypatch)
    (tmp_path / "mmlu_summary.json").write_text('{"guardrail_30pp": true}')

    def broken_lm_eval(cmd, capture_output=True, text=True):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr(mmlu_mod.subprocess, "run", broken_lm_eval)
    assert mmlu_mod.mmlu_phase(RUN001) == 3
    assert not (tmp_path / "mmlu_summary.json").exists(), \
        "old passing summary survived a failed re-run"


def test_mmlu_summary_records_variant_fingerprint(tmp_path, monkeypatch):
    mmlu_mod = _mmlu_env(tmp_path, monkeypatch)
    monkeypatch.setattr(mmlu_mod.subprocess, "run",
                        _fake_lm_eval({"base": .6, "variant": .59}, []))
    assert mmlu_mod.mmlu_phase(RUN001) == 0
    summ = json.loads((tmp_path / "mmlu_summary.json").read_text())
    assert summ["variant_fingerprint"] == "f1"


def _with_selection_fp(fp):
    def mutate(out_dir):
        sel = json.loads((out_dir / "selection.json").read_text())
        sel["selected_provenance"] = {"fingerprint": fp, "verified": True}
        (out_dir / "selection.json").write_text(json.dumps(sel))
    return mutate


def test_publish_refuses_mmlu_for_other_weights(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="not for the current selected"):
        _publish(tmp_path, monkeypatch,
                 artifacts_mutate=_with_selection_fp("new" * 10),
                 mmlu_mutate=lambda m: m.update(
                     variant_fingerprint="old" * 10))


def test_publish_accepts_matching_fingerprint(tmp_path, monkeypatch):
    rc, _ = _publish(tmp_path, monkeypatch,
                     artifacts_mutate=_with_selection_fp("same" * 8),
                     mmlu_mutate=lambda m: m.update(
                         variant_fingerprint="same" * 8))
    assert rc == 0


def test_publish_refuses_fingerprintless_summary_for_new_selection(
        tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="re-run `abliterate mmlu`"):
        _publish(tmp_path, monkeypatch,
                 artifacts_mutate=_with_selection_fp("x" * 32))


@pytest.mark.parametrize("key,val", [("card_charts", "false"),
                                     ("hf_user", 123),
                                     ("card_byline", ["a"])])
def test_spec_validates_publish_options(tmp_path, key, val):
    p = _spec_variant(tmp_path, lambda d: d["publish"].update({key: val}))
    with pytest.raises(SpecError, match=f"publish.{key}"):
        load_spec(p)


def test_spec_rejects_out_of_range_ara_layers(tmp_path):
    def mutate(d):
        d["ladder"] = {"variants": ["wd_ML", "ara_8"],
                       "ara": {"layers": [3, 24]}}  # 24 decoder layers: 0..23
    p = _spec_variant(tmp_path, mutate)
    with pytest.raises(SpecError, match=r"\[24\] out of range"):
        load_spec(p)
    ok = _spec_variant(tmp_path, lambda d: d.update(
        {"ladder": {"variants": ["wd_ML", "ara_8"],
                    "ara": {"layers": [3, 23]}}}))
    assert load_spec(ok)["ladder"]["ara"]["layers"] == [3, 23]


def test_init_download_failure_is_clean_refusal(tmp_path, capsys):
    from abliteration_engine import init_spec

    class Info:
        sha = "c" * 40
        siblings = []
        card_data = {}

    class Api:
        def model_info(self, *a, **k):
            return Info()

    def download(*a, **k):
        raise PermissionError("gated repo")
    rc = init_spec.init_spec("org/gated", out=str(tmp_path / "s.yaml"),
                             api=Api(), download=download)
    assert rc == 2
    assert "could not download config.json" in capsys.readouterr().err
    assert not (tmp_path / "s.yaml").exists()
