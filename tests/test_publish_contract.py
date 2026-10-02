"""Dev-review regression pins for the GPU ports (CPU-only).

Review finding (publish.py): the port's docstring claimed 'LFS sha256 of
model.safetensors vs local source' in hub-side verification, but the code
only checked README.md/config.json presence. publish.py now also asserts
the weights + representative direction land on the hub and prints
PUBLISH_DONE {json} — the sentinel line the v2 poll loop consumed.

These tests sandbox ENG_OUT_ROOT (the real env override publish uses) and
exercise the REAL out-dir derivation path via monkeypatch.
"""
import json
import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import pytest  # noqa: E402  (must trail sys.path wiring)
import yaml  # noqa: E402


def _spec_file(tmp_path, monkeypatch, **publish_overrides):
    src = os.path.join(REPO, "specs", "qwen25_0p5b_run000_recreation.yaml")
    payload = yaml.safe_load(open(src))
    payload["run_card"]["patient"] = "publishtest"
    payload["publish"] = dict(
        {"repo_id": "sbussiso/test-publish-gate", "license": "apache-2.0",
         "card_marker": "test",
         "verify_disk_bounds": {"lm_head": 0.005, "final_norm": 0.05,
                                "layer_row": 0.01}}, **publish_overrides)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(payload))
    return str(p)


CHART_FILES = ("charts/refusal_by_condition.png",
               "charts/benign_by_condition.png", "charts/mmlu_guardrail.png")


def _fake_hub(tmp_path, files=("README.md", "config.json"), charts=True):
    files = tuple(files) + (CHART_FILES if charts else ())
    class _API:
        def whoami(self):
            return {"name": "sbussiso"}

        def create_repo(self, *a, **k):
            return None

        def upload_folder(self, *a, **k):
            return None

        def list_repo_files(self, *a, **k):
            return list(files)

        def hf_hub_download(self, *a, **k):
            p = tmp_path / "hub_readme.md"
            p.write_text("tags: abliteration\n"
                         "7ae557604adf67be50417f59c2c2f167def9a775\n")
            return str(p)

    return _API()


def _eng_root_env(tmp_path, monkeypatch):
    """Point ENG_OUT_ROOT at tmp; return the run out dir the engine will
    derive (must match core._out_dir naming exactly)."""
    monkeypatch.setenv("ENG_OUT_ROOT", str(tmp_path))
    return tmp_path / "eng_run_006_publishtest"


def _fake_run_artifacts(out_dir, tmp_path):
    """Minimal artifact set publish_phase reads out of the derived dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = {"selected": "wd_B", "selected_variant_dir": str(
        tmp_path / "v"), "gate": "passed",
        "publish_eligible_probe_gate": True}
    cfg = {"layer": {"decoder_layer": 17, "coherence": 0.664,
                     "readout_space_final_layer_coherence": 0.71},
           "structure": {"num_hidden_layers": 24, "hidden_size": 896,
                         "num_attention_heads": 14,
                         "num_key_value_heads": 2,
                         "tie_word_embeddings": True},
           "probes": {"n_pairs": 64, "n_probes": 16},
           "decoding": {"seed": 0, "max_new_tokens": 200},
           "versions": {"python": "3.11", "torch": "2.5.0"},
           "gpu": "t4"}
    (out_dir / "selection.json").write_text(json.dumps(sel))
    (out_dir / "run_config.json").write_text(json.dumps(cfg))
    (out_dir / "layer_coherence.json").write_text(json.dumps(
        {"final_layer": 23}))
    cands = [{"variant": "wd_B", "refusal_rate": 0.031,
              "benign_preserved": 0.906, "degenerate_total": 0}]
    (out_dir / "selection_candidates.json").write_text(json.dumps(cands))
    for name in ("baseline", "hook_ablated"):
        probes = {"harmful": [{"refused": True, "degenerate": False}] * 8,
                  "harmless": [{"refused": False, "degenerate": False}] * 8}
        (out_dir / f"probes_{name}.json").write_text(json.dumps(probes))
    mmlu = {"mmlu_base_pct": 35.2, "mmlu_variant_pct": 34.1,
            "mmlu_delta_pp": 1.1, "guardrail_30pp": True,
            "guardrail_loss_pp_limit": 3.0, "variant": "wd_B",
            "base": {"acc": 0.352, "acc_stderr": 0.01},
            "variant_model": {"acc": 0.341, "acc_stderr": 0.01}}
    mmlu_p = tmp_path / "mmlu_summary.json"
    mmlu_p.write_text(json.dumps(mmlu))
    vdir = tmp_path / "v"
    vdir.mkdir(exist_ok=True)
    (vdir / "model.safetensors").write_bytes(b"\x00" * 16)
    (vdir / "config.json").write_text("{}")
    return mmlu_p, vdir


def test_publish_hub_files_and_done_marker(tmp_path, monkeypatch, capsys):
    """Full gate chain passes with fakes; hub verification asserts
    weights + direction file; PUBLISH_DONE {json} prints (sentinel)."""
    from abliteration_engine import publish as P

    out_dir = _eng_root_env(tmp_path, monkeypatch)
    mmlu_p, vdir = _fake_run_artifacts(out_dir, tmp_path)
    spec = _spec_file(tmp_path, monkeypatch)
    api = _fake_hub(tmp_path, files=("README.md", "config.json",
                                     "model.safetensors",
                                     "refusal_direction.npy", "eval/x"))
    with mock.patch("huggingface_hub.HfApi", return_value=api):
        rc = P.publish_phase(spec, str(vdir), str(mmlu_p),
                             assume_publish=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PUBLISH_DONE" in out
    assert "whoami OK: sbussiso" in out


def test_publish_fails_when_weights_missing_on_hub(tmp_path, monkeypatch):
    """Finding-1 regression: hub file list WITHOUT model.safetensors must
    FAIL verification (the v3 port would have returned 0 here)."""
    from abliteration_engine import publish as P

    out_dir = _eng_root_env(tmp_path, monkeypatch)
    mmlu_p, vdir = _fake_run_artifacts(out_dir, tmp_path)
    spec = _spec_file(tmp_path, monkeypatch)
    api = _fake_hub(tmp_path, files=("README.md", "config.json"))
    with mock.patch("huggingface_hub.HfApi", return_value=api), \
            pytest.raises(AssertionError, match="model.safetensors"):
        P.publish_phase(spec, str(vdir), str(mmlu_p), assume_publish=True)


def test_publish_fails_when_direction_missing_on_hub(tmp_path, monkeypatch):
    from abliteration_engine import publish as P

    out_dir = _eng_root_env(tmp_path, monkeypatch)
    mmlu_p, vdir = _fake_run_artifacts(out_dir, tmp_path)
    spec = _spec_file(tmp_path, monkeypatch)
    api = _fake_hub(tmp_path, files=("README.md", "config.json",
                                     "model.safetensors"))
    with mock.patch("huggingface_hub.HfApi", return_value=api), \
            pytest.raises(AssertionError, match="refusal_direction.npy"):
        P.publish_phase(spec, str(vdir), str(mmlu_p), assume_publish=True)


def test_publish_hitl_refuses_without_flag(tmp_path, monkeypatch, capsys):
    from abliteration_engine import publish as P

    _eng_root_env(tmp_path, monkeypatch)
    spec = _spec_file(tmp_path, monkeypatch)
    rc = P.publish_phase(spec, str(tmp_path / "v"),
                         str(tmp_path / "m.json"), assume_publish=False)
    assert rc == 2
    out = capsys.readouterr().out
    assert "--i-know-this-publishes" in out