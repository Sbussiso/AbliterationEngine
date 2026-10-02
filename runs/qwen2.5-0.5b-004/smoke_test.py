"""CPU smoke test for the ROUND-2 harness (0.5B patient, no GPU, tiny model).

Same checks as the run-002 smoke test but importing the -004 (0.5B-pinned)
harness: extraction/scan/directions, all ladder edits, save/reload
persistence, probe/selection logic. The pinned patient assertions are
negative-tested (must fire on a wrong-shaped model).
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

H = "/root/research/abliteration/qwen2.5-0.5b-004/harness"
sys.path.insert(0, H)
import run_003 as R  # noqa: E402
import ladder_003 as L  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM  # noqa: E402

OUT = "/root/research/abliteration/qwen2.5-0.5b-004/smoke"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)


def fresh_cfg():
    return Qwen2Config(
        vocab_size=151936,
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=True,
        bos_token_id=151643, eos_token_id=151645, pad_token_id=151643,
    )


cfg = fresh_cfg()
model = Qwen2ForCausalLM(cfg).to(torch.float32).eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

orig_load = R.load_model


def fake_load(repo, revision=None):
    if os.path.isdir(str(repo)):
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
print("  structure_report OK")
try:
    R.assert_patient_structure(rep)
    raise SystemExit("FAIL: pinned 0.5B assertions should NOT fire on a tiny model")
except AssertionError:
    print("  pinned 0.5B assertions correctly fire on wrong shape (negative test)")

print("== 2. capture ==")
cap_h = R.capture_final_residuals(tok, model, wrapped[:4], batch=2)
cap_b = R.capture_final_residuals(tok, model, wrapped[4:], batch=4)
assert len(cap_h) == 5 and cap_h[2].shape == (4, 64), (len(cap_h), cap_h[2].shape)
print("  capture OK", len(cap_h), cap_h[2].shape)

print("== 3. scan + directions ==")
best, table = R.scan_layers(cap_h, cap_b, 4)
assert best["decoder_layer"] in (0, 1, 2), best
norm_mod = R.final_norm_module(model)
with torch.inference_mode():
    post_h = norm_mod(cap_h[4].to(model.device)).float().cpu()
    post_b = norm_mod(cap_b[4].to(model.device)).float().cpu()
dir_B, nd_B, coh_B = R.coherence_stats(post_h, post_b)
dirs_all = torch.stack([cap_h[l + 1][:4].mean(0) - cap_b[l + 1][:4].mean(0)
                        for l in range(4)])
print(f"  scan OK, best L={best['decoder_layer']}, readout coh={coh_B:.3f}")

print("== 4. ladder edits ==")
mc, rn = L.orthogonalize_lm_head(model, dir_B)
assert mc < 1e-6, mc
print("  lm_head orth OK: max|W r| =", mc)
comp, resid = L.orthogonalize_final_norm(model, dir_B)
assert resid < 1e-6, resid
print("  final_norm orth OK: |resid| =", resid)
r17 = np.asarray(dirs_all[2])
resid_info = L.orthogonalize_layer_output(model, 2, dirs_all[2].float())
print("  multi-layer row-space orth OK:", resid_info)

print("== 5. save / reload persistence ==")
L.save_variant(model, tok, os.path.join(OUT, "v_untied"), expect_tied=False)
tok_r, model_r = R.load_model(os.path.join(OUT, "v_untied"), revision=None)
r1 = L.verify_lm_head_disk(model_r, dir_B)
r2 = L.verify_final_norm_disk(model_r, dir_B)
print(f"  untied variant reload OK: lm_head |W r|={r1:.2e}, norm |w.d|={r2:.2e}")

cfg2 = fresh_cfg()
model2 = Qwen2ForCausalLM(cfg2).to(torch.float32).eval()
L.orthogonalize_layer_output(model2, 2, dirs_all[2].float())
L.save_variant(model2, tok, os.path.join(OUT, "v_tied"), expect_tied=True)
tok_r2, model_r2 = R.load_model(os.path.join(OUT, "v_tied"), revision=None)
res = L.verify_layers_disk(model_r2, [2], dirs_all.numpy(), bound=1e-2)
print("  tied variant reload OK: layer resids on disk =", res)

print("== 6. probes + selection logic ==")
rows_t = R.run_probes(tok_r, model_r, R.HARMFUL[:2], tag="t", max_new=8)
rows_b = R.run_probes(tok_r, model_r, R.HARMLESS[:2], tag="b", max_new=8)
s = R.summarize(rows_t, rows_b)
print("  summarize OK:", s)
cands = [
    {"variant": "wd_B", "ladder_index": 0, "refusal_rate": 0.5,
     "benign_preserved": 1.0, "degenerate_total": 0},
    {"variant": "wd_BN", "ladder_index": 1, "refusal_rate": 0.2499,
     "benign_preserved": 1.0, "degenerate_total": 0},
]
sel, elig = L.select_variant([dict(c) for c in cands], 1.0, 0.25)
assert sel["variant"] == "wd_BN" and elig is True, (sel, elig)
sel2, elig2 = L.select_variant([dict(c) for c in cands], 1.0, 0.20)
assert sel2["variant"] == "wd_BN" and elig2 is False, (sel2, elig2)
# boundary: exactly 0.25 is NOT eligible (strict <) — document the policy
sel3, elig3 = L.select_variant(
    [{"variant": "x", "ladder_index": 0, "refusal_rate": 0.25,
      "benign_preserved": 1.0, "degenerate_total": 0}], 1.0, 0.25)
assert elig3 is False, (sel3, elig3)
print("  selection OK: below-threshold eligible=%s; at-threshold NOT eligible=%s"
      % (elig, elig3))

print("== hook mechanics ==")
hook = R.AblationHook(dir_B, model2.device, torch.float32)
hook.attach(model2.model.layers[2])
_ = R.run_probes(tok, model2, R.HARMFUL[:1], tag="hook", max_new=4)
hook.detach()
assert hook.calls > 0
print("  hook OK, calls =", hook.calls)

print("SMOKE004_ALL_OK")