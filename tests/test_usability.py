"""Usability contract (CPU-only): CLI help/errors, `abliterate init`,
publish path defaults, `run --with-mmlu`, load-time compat check, probe ETA.

Real Qwen2.5 config.json files are pinned in tests/fixtures/hf_configs/
(fetched from the hub) so init's structure_expect is checked against the
values the shipped, GPU-proven specs carry — no network in CI.
"""
import json
import os
import sys
import types

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

from abliteration_engine import cli, core, init_spec  # noqa: E402

RUN001 = os.path.join(REPO, "specs", "run001_parity.yaml")
CFG_DIR = os.path.join(HERE, "fixtures", "hf_configs")


# ---- help + errors ----------------------------------------------------------

def test_top_help_lists_every_verb_with_description(capsys):
    with pytest.raises(SystemExit):
        cli.main(["-h"])
    out = capsys.readouterr().out
    for verb, (desc, *_rest) in cli.VERBS.items():
        assert verb in out
    assert "--variant-dir" not in out  # verb flags hidden at top level


def test_verb_help_shows_only_its_flags(capsys):
    with pytest.raises(SystemExit):
        cli.main(["plan", "-h"])
    out = capsys.readouterr().out
    assert "--spec" in out
    for other in ("--mmlu", "--i-know-this-publishes", "--out-dir",
                  "--baseline"):
        assert other not in out


def test_quota_flag_harmless_on_cpu_verbs(capsys):
    assert cli.main(["plan", "--spec", RUN001,
                     "--i-know-this-spends-quota"]) == 0
    with pytest.raises(SystemExit):
        cli.main(["plan", "-h"])
    assert "--i-know-this-spends-quota" not in capsys.readouterr().out


def test_bad_spec_is_one_line_refusal_not_traceback(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("spec_version: 1\n")
    assert cli.main(["validate", "--spec", str(p)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("REFUSING: spec invalid")
    assert "Traceback" not in err and "run_card" in err


@pytest.mark.parametrize("payload", [None, "a: [unclosed"])
def test_missing_or_malformed_spec_refused(tmp_path, capsys, payload):
    p = tmp_path / "s.yaml"
    if payload is not None:
        p.write_text(payload)
    assert cli.main(["plan", "--spec", str(p)]) == 2
    assert "REFUSING" in capsys.readouterr().err


# ---- init ---------------------------------------------------------------------

class _Info:
    def __init__(self, sha, files, license_="apache-2.0"):
        self.sha = sha
        self.siblings = [types.SimpleNamespace(rfilename=f) for f in files]
        self.card_data = {"license": license_}


class _Api:
    def __init__(self, info, who="alice"):
        self._info, self._who = info, who
        self.calls = []

    def model_info(self, model_id, revision=None):
        self.calls.append((model_id, revision))
        return self._info

    def whoami(self):
        if self._who is None:
            raise RuntimeError("not logged in")
        return {"name": self._who}


def _download_from(cfg_file, tok_cfg=None):
    def download(model_id, filename, revision=None):
        if filename == "config.json":
            return cfg_file
        raise FileNotFoundError(filename)
    return download


@pytest.mark.parametrize("cfg_name,spec_name", [
    ("qwen2.5-0.5b-instruct.json", "run001_parity.yaml"),
    ("qwen2.5-7b-instruct.json", "qwen25_7b.yaml"),
])
def test_structure_from_real_config_matches_shipped_specs(cfg_name,
                                                          spec_name):
    cfg = json.load(open(os.path.join(CFG_DIR, cfg_name)))
    got = init_spec.structure_from_config(cfg)
    shipped = yaml.safe_load(open(os.path.join(
        REPO, "specs", spec_name)))["patient"]["structure_expect"]
    for k, v in shipped.items():
        assert got.get(k) == v, (k, got.get(k), v)


def test_init_writes_valid_pinned_spec(tmp_path, capsys):
    sha = "7ae557604adf67be50417f59c2c2f167def9a775"
    api = _Api(_Info(sha, ["config.json", "chat_template.jinja"]))
    out = tmp_path / "spec.yaml"
    rc = init_spec.init_spec(
        "Qwen/Qwen2.5-0.5B-Instruct", out=str(out), api=api,
        download=_download_from(os.path.join(
            CFG_DIR, "qwen2.5-0.5b-instruct.json")))
    assert rc == 0
    assert api.calls == [("Qwen/Qwen2.5-0.5B-Instruct", "main")]
    from abliteration_engine.spec import load_spec
    spec = load_spec(str(out))
    assert spec["patient"]["revision"] == sha
    assert spec["patient"]["structure_expect"]["down_proj_shape"] == \
        [896, 4864]
    assert spec["publish"]["repo_id"] == \
        "alice/Qwen2.5-0.5B-Instruct-abliterated"
    assert spec["run_card"]["patient"] == "qwen2.5-0.5b-instruct"
    out_txt = capsys.readouterr().out
    assert "OK: architecture 'qwen2'" in out_txt
    assert "next: abliterate plan" in out_txt


def test_init_logged_out_leaves_publish_gated_and_refuses_overwrite(
        tmp_path, capsys):
    api = _Api(_Info("a" * 40, ["config.json", "chat_template.jinja"]),
               who=None)
    out = tmp_path / "spec.yaml"
    dl = _download_from(os.path.join(CFG_DIR, "qwen2.5-0.5b-instruct.json"))
    assert init_spec.init_spec("Qwen/x", out=str(out), api=api,
                               download=dl) == 0
    from abliteration_engine.spec import load_spec
    assert load_spec(str(out))["publish"]["repo_id"] is None
    assert init_spec.init_spec("Qwen/x", out=str(out), api=api,
                               download=dl) == 2
    assert "exists" in capsys.readouterr().err


def test_init_warns_on_unknown_arch_and_missing_template(tmp_path, capsys):
    cfg = json.load(open(os.path.join(CFG_DIR, "qwen2.5-0.5b-instruct.json")))
    cfg["model_type"] = "gpt2"
    f = tmp_path / "config.json"
    f.write_text(json.dumps(cfg))
    api = _Api(_Info("b" * 40, ["config.json"]))
    rc = init_spec.init_spec("x/y", out=str(tmp_path / "s.yaml"), api=api,
                             download=_download_from(str(f)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING: architecture 'gpt2' is not one" in out
    assert "no chat template" in out


def test_init_unreachable_hub_is_clean_refusal(capsys):
    class Down(_Api):
        def model_info(self, *a, **k):
            raise ConnectionError("offline")
    rc = init_spec.init_spec("x/y", api=Down(None), download=lambda *a, **k: 0)
    assert rc == 2
    assert capsys.readouterr().err.startswith("REFUSING: could not read")


def test_cli_init_requires_model(capsys):
    assert cli.main(["init"]) == 2
    assert "--model" in capsys.readouterr().err


# ---- run --with-mmlu ------------------------------------------------------------

def test_run_with_mmlu_chains_after_ladder(monkeypatch):
    from abliteration_engine import mmlu, pipeline
    order = []
    finals = []
    monkeypatch.setattr(
        pipeline, "run_pipeline",
        lambda spec, final=True: finals.append(final) or
        order.append("run") or 0)
    monkeypatch.setattr(mmlu, "mmlu_phase",
                        lambda p: order.append("mmlu") or 0)
    assert cli.main(["run", "--spec", RUN001, "--with-mmlu",
                     "--i-know-this-spends-quota"]) == 0
    assert order == ["run", "mmlu"]
    assert finals == [False], "chained run must leave the sentinel running"
    order.clear()
    finals.clear()
    assert cli.main(["run", "--spec", RUN001,
                     "--i-know-this-spends-quota"]) == 0
    assert order == ["run"] and finals == [True]


def test_run_with_mmlu_stops_on_run_failure(monkeypatch):
    from abliteration_engine import mmlu, pipeline
    monkeypatch.setattr(pipeline, "run_pipeline",
                        lambda spec, final=True: 1)
    monkeypatch.setattr(mmlu, "mmlu_phase",
                        lambda p: pytest.fail("mmlu ran after a failed run"))
    assert cli.main(["run", "--spec", RUN001, "--with-mmlu",
                     "--i-know-this-spends-quota"]) == 1


def test_runner_supports_phase_all():
    from abliteration_engine.bundle import RUNNER_TEMPLATE
    assert '"$PHASE" = "all"' in RUNNER_TEMPLATE
    assert "run --with-mmlu" in RUNNER_TEMPLATE


# ---- publish defaults ---------------------------------------------------------------

def test_publish_defaults_paths_from_run_records(tmp_path, monkeypatch):
    from unittest import mock

    import test_publish_contract as pc

    from abliteration_engine import publish as P

    out_dir = pc._eng_root_env(tmp_path, monkeypatch)
    mmlu_p, vdir = pc._fake_run_artifacts(out_dir, tmp_path)
    (out_dir / "mmlu_summary.json").write_text(mmlu_p.read_text())
    (out_dir / "layer_coherence.json").write_text(json.dumps(
        {"final_layer": 23, "best": {"decoder_layer": 17, "coherence": 0.6},
         "readout_space": {"coherence": 0.7}}))
    spec = pc._spec_file(tmp_path, monkeypatch)
    api = pc._fake_hub(tmp_path, files=("README.md", "config.json",
                                        "model.safetensors",
                                        "refusal_direction.npy"))
    with mock.patch("huggingface_hub.HfApi", return_value=api):
        rc = P.publish_phase(spec, assume_publish=True)  # no paths given
    assert rc == 0
    assert (vdir / "README.md").exists()


def test_publish_without_mmlu_summary_is_blocked_with_hint(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    import test_publish_contract as pc
    out_dir = pc._eng_root_env(tmp_path, monkeypatch)
    pc._fake_run_artifacts(out_dir, tmp_path)  # summary NOT in run dir
    spec = pc._spec_file(tmp_path, monkeypatch)
    rc = cli.main(["publish", "--spec", spec, "--i-know-this-publishes"])
    assert rc == 3
    err = capsys.readouterr().err
    assert err.startswith("PUBLISH BLOCKED") and "abliterate mmlu" in err


# ---- load-time compat check + ETA ---------------------------------------------------

def _fake_model(has_down=True):
    lin = types.SimpleNamespace(weight=object())
    mlp = types.SimpleNamespace(down_proj=lin) if has_down else \
        types.SimpleNamespace()
    layer = types.SimpleNamespace(self_attn=types.SimpleNamespace(o_proj=lin),
                                  mlp=mlp)
    inner = types.SimpleNamespace(layers=[layer], norm=object())
    return types.SimpleNamespace(
        model=inner, config=types.SimpleNamespace(model_type="fake"),
        get_output_embeddings=lambda: lin)


def test_compat_check_passes_qwen_layout():
    core.check_patient_compat(types.SimpleNamespace(chat_template="{{x}}"),
                              _fake_model())


def test_compat_check_names_every_problem():
    with pytest.raises(RuntimeError) as e:
        core.check_patient_compat(types.SimpleNamespace(chat_template=None),
                                  _fake_model(has_down=False))
    msg = str(e.value)
    assert "mlp.down_proj" in msg and "chat template" in msg
    assert "'fake' is not supported" in msg


def test_eta_format():
    assert core._fmt_eta(5.4) == "5s"
    assert core._fmt_eta(65) == "1m05s"
