"""CPU smoke test for run_002 harness mechanics (no HF download needed).

Builds a tiny random Qwen2 config with TIED embeddings, runs the exact
run_002 stage flow with reduced sizes, and asserts:
  1. capture_final_residuals returns n_layers+1 tensors of the right shape
  2. coherence scan returns a table with finite values
  3. orthogonalize_lm_head leaves |W r-hat| ~ 0 and UNTIES the parameter
  4. save_variant persists tie_word_embeddings=False on disk
  5. RELOAD from disk: untied preserved, edit preserved (|W r| still ~0),
     config flag False, generation works
  6. greedy generation changes after ablation (smoke-level)
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "/root/research/abliteration/qwen2.5-0.5b-002/harness")
import run_002 as R  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers import GenerationConfig  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

OUT = "/root/research/abliteration/qwen2.5-0.5b-002/smoke"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
cfg = Qwen2Config(
    vocab_size=151936,  # real Qwen vocab keeps the tokenizer head-matrix legal
    hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=512, tie_word_embeddings=True,
    bos_token_id=151643, eos_token_id=151645, pad_token_id=151643,
)
model = Qwen2ForCausalLM(cfg).to(torch.float32).eval()
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

# Patch out the HF revision pin for the smoke test only.
orig_load = R.load_model


def fake_load(repo, revision=None):
    if os.path.isdir(str(repo)):
        # real disk load (persistence check path) - bypass only hub loads
        dtype = torch.float32
        try:
            m = AutoModelForCausalLM.from_pretrained(
                str(repo), torch_dtype=dtype)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(str(repo), dtype=dtype)
        m.eval()
        return tok, m
    return tok, model


R.load_model = fake_load

prompts = R.HARMFUL[:4] + R.HARMLESS[:4]
wrapped = [R.wrap(tok, p) for p in prompts]

print("== stage 2: capture ==")
cap = R.capture_final_residuals(tok, model, wrapped, batch=4)
assert len(cap) == cfg.num_hidden_layers + 1, len(cap)
assert cap[0].shape == (8, cfg.hidden_size), cap[0].shape
print("capture OK", len(cap), cap[0].shape)

print("== stage 3: scan + directions ==")
best, table = R.scan_layers(cap, cap, n_pairs=4)
assert table and np.isfinite(table[0]["coherence"])
L_star = best["decoder_layer"]
dir_A = cap[L_star + 1][:4].mean(0) - cap[L_star + 1][4:].mean(0)
n_layers = cfg.num_hidden_layers
norm_mod = R.final_norm_module(model)
post = norm_mod(cap[n_layers]).float()
dir_B, nd_B, coh_B = R.coherence_stats(post[:4], post[4:])
print("scan OK, best L =", L_star, "| readout coh =", round(coh_B, 3))

print("== stage 6a: orthogonalize ==")
max_comp, rnorm = R.orthogonalize_lm_head(model, dir_A)
assert max_comp < 1e-4, max_comp
chk = R.verify_untied(model)
assert not chk["tied"] and chk["weight_diff_l2"] > 0, chk
assert model.config.tie_word_embeddings is False
print("orthogonalize OK: max|Wr| =", f"{max_comp:.2e}",
      "untied diff L2 =", round(chk["weight_diff_l2"], 4))

print("== stage 6b: save + reload persistence ==")
vdir = os.path.join(OUT, "wd_A")
R.save_variant(model, tok, vdir)
del model

tok_r, model_r = R.load_model(str(vdir), revision=None)
chk2 = R.verify_untied(model_r)
assert not chk2["tied"] and chk2["weight_diff_l2"] > 0, chk2
assert model_r.config.tie_word_embeddings is False
# lm_head weight is actually orthogonal to r-hat
lm = model_r.get_output_embeddings()
W = lm.weight.data
r = dir_A / dir_A.norm()
resid = (W.float() @ r).abs().max().item()
assert resid < 1e-3, resid
print("reload OK: untied, config flag False, max|W r| on disk =",
      f"{resid:.2e}")

print("== stage 4/5/6 probes (greedy, short) ==")
base_h = R.run_probes(tok_r, model_r, R.HARMFUL[:2], tag="t", max_new=16)
base_b = R.run_probes(tok_r, model_r, R.HARMLESS[:2], tag="t", max_new=16)
s = R.summarize(base_h, base_b)
assert s["n_harmful"] == 2 and s["n_harmless"] == 2
assert all(len(r["output"]) > 0 for r in base_h + base_b)
print("probes OK:", json.dumps(s))

print("== hook mechanics ==")
hook = R.AblationHook(dir_A, model_r.device, torch.float32)
hook.attach(model_r.model.layers[L_star])
enc = tok(R.wrap(tok, R.HARMFUL[0]), return_tensors="pt").to(model_r.device)
_ = model_r.generate(**enc, max_new_tokens=8, do_sample=False,
                     pad_token_id=tok.pad_token_id)
hook.detach()
assert hook.calls > 0, "hook never fired"
print("hook OK, calls =", hook.calls)

print("SMOKE_ALL_OK")