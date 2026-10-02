"""Model-card charts (card_charts.py via publish): rendered from the same
artifacts as the card tables, embedded with numeric alt text, uploaded and
hub-verified, byte-deterministic, and optional (text-only card without
matplotlib or with publish.card_charts: false)."""
import builtins
import json
import os
import sys
from unittest import mock

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import test_publish_contract as pc  # noqa: E402

pytest.importorskip("matplotlib")

from abliteration_engine import card_charts  # noqa: E402
from abliteration_engine import publish as P  # noqa: E402

CHARTS = pc.CHART_FILES
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _publish(tmp_path, monkeypatch, hub_charts=True, **pub_overrides):
    out_dir = pc._eng_root_env(tmp_path, monkeypatch)
    mmlu_p, vdir = pc._fake_run_artifacts(out_dir, tmp_path)
    (out_dir / "layer_coherence.json").write_text(json.dumps(
        {"final_layer": 23, "best": {"decoder_layer": 17, "coherence": 0.6},
         "readout_space": {"coherence": 0.7}}))
    spec = pc._spec_file(tmp_path, monkeypatch, **pub_overrides)
    api = pc._fake_hub(tmp_path, files=("README.md", "config.json",
                                        "model.safetensors",
                                        "refusal_direction.npy"),
                       charts=hub_charts)
    with mock.patch("huggingface_hub.HfApi", return_value=api):
        rc = P.publish_phase(spec, str(vdir), str(mmlu_p),
                             assume_publish=True)
    return rc, vdir


def test_charts_rendered_embedded_and_listed(tmp_path, monkeypatch):
    rc, vdir = _publish(tmp_path, monkeypatch)
    assert rc == 0
    for rel in CHARTS:
        data = (vdir / rel).read_bytes()
        assert data.startswith(PNG_MAGIC), rel
    card = (vdir / "README.md").read_text()
    for rel in CHARTS:
        assert f"]({rel})" in card, rel
    # alt text carries the numbers (the charts are not color/image-only)
    assert "![Harmful-prompt refusal by condition: baseline 100.0%" in card
    assert "wd_B 3.1%" in card
    assert "loss 1.10pp against a 3.0pp limit" in card
    assert "- `charts/`" in card


def test_charts_byte_deterministic(tmp_path):
    conds = [{"label": "baseline", "kind": "baseline", "refusal_rate": .9,
              "benign_preserved": .95},
             {"label": "hook", "kind": "hook", "refusal_rate": 0.0,
              "benign_preserved": .95},
             {"label": "wd_ML\nPUBLISHED", "kind": "variant",
              "published": True, "refusal_rate": .2,
              "benign_preserved": 1.0}]
    mmlu = {"base": {"acc": .6, "acc_stderr": .004},
            "variant_model": {"acc": .599, "acc_stderr": .004},
            "guardrail_loss_pp_limit": 3.0}
    digests = []
    for i in range(2):
        out = tmp_path / str(i)
        card_charts.render_card_charts(str(out), conds, .95, .1, .25, mmlu,
                                       "wd_ML", "Qwen2.5-0.5B-Instruct")
        digests.append([(out / rel).read_bytes() for rel in CHARTS])
    assert digests[0] == digests[1], "re-render of same artifacts differs"


def test_chart_missing_on_hub_blocks(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="chart charts/"):
        _publish(tmp_path, monkeypatch, hub_charts=False)


def test_charts_can_be_disabled(tmp_path, monkeypatch):
    rc, vdir = _publish(tmp_path, monkeypatch, hub_charts=False,
                        card_charts=False)
    assert rc == 0
    assert not (vdir / "charts").exists()
    assert "](charts/" not in (vdir / "README.md").read_text()


def test_no_matplotlib_ships_text_only_card(tmp_path, monkeypatch, capsys):
    real_import = builtins.__import__

    def no_mpl(name, *a, **k):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("no matplotlib")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_mpl)
    rc, vdir = _publish(tmp_path, monkeypatch, hub_charts=False)
    assert rc == 0
    assert "](charts/" not in (vdir / "README.md").read_text()
    assert "card charts skipped: matplotlib not installed" in \
        capsys.readouterr().out
