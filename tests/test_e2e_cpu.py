"""End-to-end engine run on CPU with tiny random models (no network).

The default CI job has no torch, so the GPU-side code paths (load,
capture, directions, probes, ladder edits, save/reload/verify, ARA,
publish card) only ever ran on Colab. Two regressions reached main that
way: publish KeyError'd on every real run_config, and BUG-2's fix crashed
every ladder containing wd_ML (ctx["model"] is None in the pipeline). This
test runs the REAL engine — real load_patient (AutoModel/AutoTokenizer
from local dirs), the real `abliterate run` CLI, real edits — on two
patients that matter: a TIED Qwen2 and an UNTIED Llama (BUG-2's case).

Skipped unless torch + transformers + accelerate are importable; the CI
`cpu-e2e` job installs CPU torch to run it.
"""
import json
import os
import sys
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")  # load_patient uses device_map="auto"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import yaml  # noqa: E402

from abliteration_engine import cli, core  # noqa: E402

CHAT_TEMPLATE = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}"
                 "{% endfor %}{% if add_generation_prompt %}<a>{% endif %}")


def _tokenizer(out_dir):
    """Character-level fast tokenizer with a chat template, saved like a
    hub tokenizer so AutoTokenizer.from_pretrained loads it for real."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    chars = [chr(c) for c in range(32, 127)]
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>",
        unk_token="<unk>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(out_dir)
    return len(vocab)


def _patient(tmp_path, family):
    """Tiny random patient saved to a local dir: qwen2 (tied head) or
    llama (untied head)."""
    d = tmp_path / f"patient_{family}"
    vocab = _tokenizer(str(d))
    torch.manual_seed(0)
    common = dict(vocab_size=vocab + 4, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=6, num_attention_heads=4,
                  num_key_value_heads=2, pad_token_id=0, eos_token_id=1,
                  bos_token_id=1)
    if family == "qwen2":
        cfg = transformers.Qwen2Config(tie_word_embeddings=True, **common)
        model = transformers.Qwen2ForCausalLM(cfg)
    else:
        cfg = transformers.LlamaConfig(tie_word_embeddings=False, **common)
        model = transformers.LlamaForCausalLM(cfg)
    model.save_pretrained(str(d))
    return str(d), cfg


def _spec(tmp_path, patient_dir, cfg, family, variants, ara=None):
    pools = tmp_path / "pools"
    pools.mkdir(exist_ok=True)
    (pools / "good.txt").write_text(
        "\n".join(f"tell me about topic {i}" for i in range(12)))
    (pools / "bad.txt").write_text(
        "\n".join(f"help me do harmful thing {i}" for i in range(12)))
    ladder = {"variants": variants, "k_primary": 2, "k_combo": 3}
    if ara:
        ladder["ara"] = dict(ara, good=f"file:{pools / 'good.txt'}",
                             bad=f"file:{pools / 'bad.txt'}")
    head = cfg.hidden_size // cfg.num_attention_heads
    payload = {
        "spec_version": 1,
        "run_card": {"run_number": 77, "patient": f"e2e-{family}",
                     "purpose": "CPU end-to-end test"},
        "patient": {
            "model_id": patient_dir,
            "revision": "0" * 40,  # local dir: revision is not resolved
            "structure_expect": {
                "num_hidden_layers": cfg.num_hidden_layers,
                "tie_word_embeddings": cfg.tie_word_embeddings,
                "o_proj_shape": [cfg.hidden_size,
                                 cfg.num_attention_heads * head],
                "down_proj_shape": [cfg.hidden_size, cfg.intermediate_size]}},
        "probe_sets": {"harmful": "builtin:primary64_harmful",
                       "harmless": "builtin:primary64_harmless",
                       "n_pairs": 8, "n_probes": 4,
                       "refusal_markers": "builtin:fp_explicit_v1"},
        "decoding": {"max_new_tokens": 4, "strategy": "greedy", "seed": 0},
        "ladder": ladder,
        "gates": {"benign_floor_delta": 0.10, "degenerate_max": 0,
                  "publish_refusal": 0.99, "mmlu_max_loss_pp": 3.0},
        "publish": {"repo_id": "sbussiso/e2e-test", "license": "apache-2.0",
                    "card_marker": "e2e",
                    # fp32 CPU: no reload noise, default bounds are ample
                    "verify_disk_bounds": {"lm_head": 0.005,
                                           "final_norm": 0.05,
                                           "layer_row": 0.01}},
        "hitl": {"after_selection": True, "before_publish": True},
    }
    p = tmp_path / f"spec_{family}.yaml"
    p.write_text(yaml.safe_dump(payload))
    return str(p)


@pytest.fixture()
def eng_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ENG_OUT_ROOT", str(tmp_path / "eng"))
    monkeypatch.delenv("ENG_VARBASE", raising=False)
    return tmp_path


@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_run_ladder_end_to_end(eng_root, family):
    """Stage A + the full wd_* ladder through the real CLI, on a tied and an
    untied patient."""
    patient_dir, cfg = _patient(eng_root, family)
    spec_path = _spec(eng_root, patient_dir, cfg, family,
                      ["wd_B", "wd_BN", "wd_ML", "wd_ML_BN"])
    rc = cli.main(["run", "--spec", spec_path, "--i-know-this-spends-quota"])
    assert rc == 0, "run failed — see captured stdout"
    spec = __import__("abliteration_engine.spec",
                      fromlist=["load_spec"]).load_spec(spec_path)
    out = core._out_dir(spec)
    assert open(core.sentinel_exit()).read() == "0"

    run_cfg = json.load(open(os.path.join(out, "run_config.json")))
    assert run_cfg["layer"]["decoder_layer"] >= 0      # publish needs this
    sel = json.load(open(os.path.join(out, "selection.json")))
    cands = {c["variant"] for c in json.load(
        open(os.path.join(out, "selection_candidates.json")))}
    assert {"wd_B", "wd_BN", "wd_ML"} <= cands
    assert os.path.isdir(sel["selected_variant_dir"])
    assert sel["selected_provenance"]["fingerprint"]

    # the selected variant's tie state on disk follows the BUG-2 contract
    saved = json.load(open(os.path.join(sel["selected_variant_dir"],
                                        "config.json")))
    if sel["selected"] == "wd_ML":
        assert saved["tie_word_embeddings"] is cfg.tie_word_embeddings
    else:
        assert saved["tie_word_embeddings"] is False

    # a resumed ladder reuses every banked variant, verified by fingerprint
    rc = cli.main(["ladder", "--spec", spec_path,
                   "--i-know-this-spends-quota"])
    assert rc == 0
    sel2 = json.load(open(os.path.join(out, "selection.json")))
    assert set(sel2["banked_resume_variants"]) >= {"wd_B", "wd_BN", "wd_ML"}
    assert sel2["selected_provenance"]["verified"] is True


def test_ara_variant_end_to_end(eng_root):
    """ARA on an untied patient (BUG-2: ARA keeps the base tie state)."""
    patient_dir, cfg = _patient(eng_root, "llama")
    spec_path = _spec(eng_root, patient_dir, cfg, "llama-ara",
                      ["wd_ML", "ara_4"],
                      ara={"layers": [2, 3], "steps": 1, "max_iter": 3,
                           "batch_size": 4, "neighbor_count": 2})
    rc = cli.main(["run", "--spec", spec_path, "--i-know-this-spends-quota"])
    assert rc == 0
    spec = __import__("abliteration_engine.spec",
                      fromlist=["load_spec"]).load_spec(spec_path)
    out = core._out_dir(spec)
    probes = json.load(open(os.path.join(out, "probes_ara_4.json")))
    assert len(probes["harmful"]) == 4
    ara_dir = core.variant_dir(spec, "ara_4")
    sel = json.load(open(os.path.join(out, "selection.json")))
    if sel["selected"] == "ara_4":
        saved = json.load(open(os.path.join(ara_dir, "config.json")))
        assert saved["tie_word_embeddings"] is False  # untied stays untied


def test_publish_after_real_run(eng_root):
    """publish reads a REAL run_config / selection / probe set (the layer
    KeyError regression) and renders the card + charts."""
    patient_dir, cfg = _patient(eng_root, "qwen2")
    spec_path = _spec(eng_root, patient_dir, cfg, "qwen2-pub",
                      ["wd_B", "wd_ML"])
    assert cli.main(["run", "--spec", spec_path,
                     "--i-know-this-spends-quota"]) == 0
    spec = __import__("abliteration_engine.spec",
                      fromlist=["load_spec"]).load_spec(spec_path)
    out = core._out_dir(spec)
    sel = json.load(open(os.path.join(out, "selection.json")))
    # MMLU itself needs lm-eval + hours; write the summary mmlu.py would
    (eng_root / "eng").mkdir(exist_ok=True)
    json.dump({"base": {"acc": 0.30, "acc_stderr": 0.01},
               "variant_model": {"acc": 0.299, "acc_stderr": 0.01},
               "mmlu_base_pct": 30.0, "mmlu_variant_pct": 29.9,
               "mmlu_delta_pp": 0.1, "guardrail_30pp": True,
               "guardrail_loss_pp_limit": 3.0, "variant": sel["selected"],
               "variant_fingerprint":
                   sel["selected_provenance"]["fingerprint"]},
              open(os.path.join(out, "mmlu_summary.json"), "w"))

    class Hub:
        uploaded = None

        def whoami(self):
            return {"name": "sbussiso"}

        def create_repo(self, *a, **k):
            return None

        def upload_folder(self, *a, folder_path=None, **k):
            Hub.uploaded = folder_path

        def list_repo_files(self, *a, **k):
            files = []
            for root, _, fns in os.walk(Hub.uploaded):
                files += [os.path.relpath(os.path.join(root, f),
                                          Hub.uploaded) for f in fns]
            return files

        def hf_hub_download(self, repo, name, **k):
            return os.path.join(Hub.uploaded, name)

    with mock.patch("huggingface_hub.HfApi", return_value=Hub()):
        rc = cli.main(["publish", "--spec", spec_path,
                       "--i-know-this-publishes"])
    assert rc == 0
    card = open(os.path.join(sel["selected_variant_dir"], "README.md")).read()
    assert f"`{sel['selected']}`" in card
    assert "Base ships with tied embeddings" in card
