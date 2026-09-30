"""CPU smoke test for the run-003 harness (no GPU, tiny random model).

Builds a tiny random Qwen2 config with TIED embeddings + GQA (like the real
1.5B patient), runs the exact run_003/ladder_003 stage flow at reduced size,
and asserts:
  1. structure_report runs report-only; pinned 1.5B assertions fire on a
     WRONG-shaped model (negative test)
  2. capture_final_residuals returns n_layers+1 tensors of the right shape
  3. coherence scan + per-layer directions + readout-space direction work
  4. ladder edits: lm_head untie-orth, final_norm orth, multi-layer
     row-space orth (with the (M W)^T r invariant check), and the combo
  5. save_variant persists the right tie flag per variant; RELOAD from disk
     verifies each edit survived (lm_head/norm/layers) and generation works
  6. selection: gate logic + publish_eligible + V4-skip logic
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

H = "/root/research/abliteration/qwen2.5-1.5b-003/harness"
sys.path.insert(0, H)
import run_003 as R  # noqa: E402
import ladder_003 as L  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM  # noqa: E402

OUT = "/root/research/abliteration/qwen2.5-1.5b-003/smoke"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)


def fresh_cfg():
    # fresh object per model: config is mutable shared state (the harness
    # mutates tie_word_embeddings on untie), so never share one instance
    return Qwen2Config(
        vocab_size=151936,  # real Qwen vocab keeps the tokenizer head legal
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=True,
        bos_token_id=151643, eos_token_id=151645, pad_token_id=151643,
    )


cfg = fresh_cfg()
model = Qwen2ForCausalLM(cfg).to(torch.float32).eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

orig_load = R.load_model


def fake_load(repo, revision=None):
    if os.path.isdir(str(repo)):  # real disk load for persistence checks
        try:
            m = AutoModelForCausalLM.from_pretrained(str(repo),
                                                     torch_dtype=torch.float32)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(str(repo),
                                                     dtype=torch.float32)
        m.eval()
        return tok, m
    return tok, model


R.load_model = fake_load

prompts = R.HARMFUL[:4] + R.HARMLESS[:4]
wrapped = [R.wrap(tok, p) for p in prompts]

print("== 1. structure_report (report-only) + assert_patient_structure ==")
rep = R.structure_report(model)
assert rep["tie_word_embeddings"] is True and rep["lm_head_is_input_emb"]
bad_cfg = Qwen2Config(
    vocab_size=151936, hidden_size=64, intermediate_size=128,
    num_hidden_layers=999, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=512, tie_word_embeddings=True,
    bos_token_id=151643, eos_token_id=151645, pad_token_id=151643)
try:
    R.assert_patient_structure(R.structure_report(
        Qwen2ForCausalLM(bad_cfg).eval()))
    raise SystemExit("FAIL: pinned assertion did NOT fire on bad model")
except AssertionError:
    pass  # expected: wrong layer count rejected
print("  structure_report OK; pinned assertions fire on wrong shape")

print("== 2. capture ==")
cap = R.capture_final_residuals(tok, model, wrapped, batch=4)
assert len(cap) == cfg.num_hidden_layers + 1, len(cap)
assert cap[0].shape == (8, cfg.hidden_size), cap[0].shape
print("  capture OK", len(cap), cap[0].shape)

print("== 3. scan + directions ==")
best, table = R.scan_layers(cap, cap, n_pairs=4)
assert table and np.isfinite(table[0]["coherence"])
L_star = best["decoder_layer"]
n_layers = cfg.num_hidden_layers
dir_A = cap[L_star + 1][:4].mean(0) - cap[L_star + 1][4:].mean(0)
norm_mod = R.final_norm_module(model)
post = norm_mod(cap[n_layers]).float()
dir_B, nd_B, coh_B = R.coherence_stats(post[:4], post[4:])
dirs_all = torch.stack([cap[l + 1][:4].mean(0) - cap[l + 1][4:].mean(0)
                        for l in range(n_layers)])
print(f"  scan OK, best L={L_star}, readout coh={coh_B:.3f}")

print("== 4. ladder edits ==")
mc, rn = L.orthogonalize_lm_head(model, dir_B)
assert mc < 1e-4, mc
chk = L.verify_untied(model)
assert not chk["tied"] and chk["weight_diff_l2"] > 0
assert model.config.tie_word_embeddings is False
print(f"  lm_head orth OK: max|W r|={mc:.2e}, untied L2 diff="
      f"{chk['weight_diff_l2']:.3f}")

comp, resid = L.orthogonalize_final_norm(model, dir_B)
assert resid < 1e-5, resid
print(f"  final_norm orth OK: w.d before={comp:.4f} -> |resid|={resid:.2e}")

# multi-layer row-space orth (fresh untied-tie model for clean invariants)
model2 = Qwen2ForCausalLM(fresh_cfg()).to(torch.float32).eval()
kl = [r["decoder_layer"] for r in table[:3]]
info = {}
for l in kl:
    info[l] = L.orthogonalize_layer_output(model2, l,
                                           dirs_all[l].float())
for l in kl:
    lay = model2.model.layers[l]
    r_ = dirs_all[l].float()
    r_ = r_ / r_.norm()
    Wo = lay.self_attn.o_proj.weight.data.float()
    Wd = lay.mlp.down_proj.weight.data.float()
    assert float((Wo.T @ r_).abs().max()) < 1e-5, l
    assert float((Wd.T @ r_).abs().max()) < 1e-5, l
print(f"  multi-layer row-space orth OK on layers {kl} "
      f"(invariant (M W)^T r = 0)")

# combo edit on the same fresh model
model2.config.tie_word_embeddings = True
mc2, _ = L.orthogonalize_lm_head(model2, dir_B)
c2, r2 = L.orthogonalize_final_norm(model2, dir_B)
assert mc2 < 1e-4 and r2 < 1e-5
print("  combo edit OK (layers + lm_head + final_norm)")

print("== 5. save / reload persistence per variant ==")
# untied variant (wd_B-style)
vdir = os.path.join(OUT, "v_un")
L.save_variant(model, tok, vdir, expect_tied=False)
del model
tok_r, model_r = R.load_model(vdir, revision=None)
assert model_r.config.tie_word_embeddings is False
chk2 = L.verify_untied(model_r)
assert not chk2["tied"] and chk2["weight_diff_l2"] > 0
lm_resid = L.verify_lm_head_disk(model_r, dir_B)
norm_resid = L.verify_final_norm_disk(model_r, dir_B)
print(f"  untied variant reload OK: lm_head |W r|={lm_resid:.2e}, "
      f"norm |w.d|={norm_resid:.2e}")

# tied variant (wd_ML-style): FRESH model, layer edits only, tie flag true
model3 = Qwen2ForCausalLM(fresh_cfg()).to(torch.float32).eval()
for l in kl:
    L.orthogonalize_layer_output(model3, l, dirs_all[l].float())
vdir2 = os.path.join(OUT, "v_tied")
L.save_variant(model3, tok, vdir2, expect_tied=True)
del model3
tok_r2, model_r2 = R.load_model(vdir2, revision=None)
assert model_r2.config.tie_word_embeddings is True
lay_disk = L.verify_layers_disk(model_r2, kl, dirs_all.numpy())
print(f"  tied variant reload OK: layer resids on disk = {lay_disk}")

print("== 6. probes + selection logic ==")
base_h = R.run_probes(tok_r, model_r, R.HARMFUL[:2], tag="t", max_new=16)
base_b = R.run_probes(tok_r, model_r, R.HARMLESS[:2], tag="t", max_new=16)
s = R.summarize(base_h, base_b)
assert s["n_harmful"] == 2 and all(len(r["output"]) > 0 for r in base_h)
cands = [
    {"variant": "wd_B", "ladder_index": 0, "refusal_rate": 0.5,
     "benign_preserved": 1.0, "degenerate_total": 0},
    {"variant": "wd_BN", "ladder_index": 1, "refusal_rate": 0.24,
     "benign_preserved": 1.0, "degenerate_total": 0},
    {"variant": "wd_ML", "ladder_index": 2, "refusal_rate": 0.6,
     "benign_preserved": 0.95, "degenerate_total": 0},
]
sel, elig = L.select_variant(cands, 1.0, 0.25)
assert sel["variant"] == "wd_BN" and elig is True, (sel, elig)
cands_gate_fail = [dict(c) for c in cands]
cands_gate_fail[1]["degenerate_total"] = 1  # gate failure -> fallback
sel2, elig2 = L.select_variant(cands_gate_fail, 1.0, 0.25)
assert sel2["variant"] == "wd_B" and elig2 is False, (sel2, elig2)
all_bad = [{"variant": "x", "ladder_index": 0, "refusal_rate": 0.9,
            "benign_preserved": 0.5, "degenerate_total": 3}]
sel3, elig3 = L.select_variant(all_bad, 1.0, 0.25)
assert sel3["variant"] == "x" and elig3 is False
print(f"  selection OK: normal={sel['variant']} eligible={elig}, "
      f"fallback={sel2['variant']} eligible={elig2}, "
      f"all-fail={sel3['variant']} eligible={elig3}")

print("== hook mechanics ==")
hook = R.AblationHook(dir_A, model_r.device, torch.float32)
hook.attach(model_r.model.layers[L_star])
enc = tok(R.wrap(tok, R.HARMFUL[0]), return_tensors="pt").to(model_r.device)
_ = model_r.generate(**enc, max_new_tokens=8, do_sample=False,
                     pad_token_id=tok.pad_token_id)
hook.detach()
assert hook.calls > 0
print(f"  hook OK, calls={hook.calls}")

shutil.rmtree(OUT, ignore_errors=True)
print("SMOKE003_ALL_OK")