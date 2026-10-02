#!/usr/bin/env python
"""Byte-level repro of the published wd_ML_BN rebuild (instance #3 recipe).

Recipe (per qwen2.5-0.5b-002/reapersplit/rebuild_variant.py + FTT-13 publish
record): load base Qwen2.5-1.5B-Instruct @ pinned rev, fp16, CPU; apply
orthogonalize_layer_output at k_combo [19,18,16,17,20] with directions from
the s5 bank layer_directions.npz (directions + decoder_layers keys);
orthogonalize_lm_head (untie) with dir_B from s5 bank; orthogonalize_final_norm
with dir_B; save with the engine's save_variant. Compare model.safetensors
sha256 against the published ae691de6a308bd57a8d6d672548abb726df2bc26466904716da867928bdd12f4.
"""
import hashlib, json, os, sys, time

sys.path.insert(0, "/root/research/abliteration/src")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from abliteration_engine import core  # noqa: E402
from abliteration_engine.edits import (  # noqa: E402
    orthogonalize_layer_output, orthogonalize_lm_head,
    orthogonalize_final_norm, save_variant)

BASE_DIR = "/tmp/fc15/base"
BANK = "/root/research/abliteration-runs/qwen2.5-0.5b-002/eng_run002_pull_s5"
OUT = "/root/research/abliteration-runs/qwen2.5-1.5b-003/publish_local/repro"
EXPECTED = "ae691de6a308bd57a8d6d672548abb726df2bc26466904716da867928bdd12f4"
K_COMBO = [19, 18, 16, 17, 20]

t0 = time.time()
import numpy as np, torch

npz = np.load(os.path.join(BANK, "layer_directions.npz"))
D = npz["directions"]; dl = npz["decoder_layers"]
dirs_all = {int(l): D[i] for i, l in enumerate(dl.tolist())}
dir_B = torch.from_numpy(np.load(os.path.join(BANK, "refusal_direction_B.npy"))).float()
print(f"[t+{time.time()-t0:.0f}s] banks loaded: npz {D.shape}, layers {sorted(dirs_all)}, "
      f"dir_B sha16 {hashlib.sha256(open(os.path.join(BANK,'refusal_direction_B.npy'),'rb').read()).hexdigest()[:16]}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE_DIR)
model = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.float16,
                                              device_map="cpu", low_cpu_mem_usage=True)
print(f"[t+{time.time()-t0:.0f}s] base loaded fp16 cpu; dtype={next(model.parameters()).dtype}", flush=True)

info = {"layers": K_COMBO, "resids": {}}
for l in K_COMBO:
    info["resids"][f"L{l}"] = orthogonalize_layer_output(model, l, torch.from_numpy(dirs_all[l]).float())
mc, comp = orthogonalize_lm_head(model, dir_B)
_, resid = orthogonalize_final_norm(model, dir_B)
info["lm_head_max|W r_B|"] = mc
info["norm_wdB_before"] = comp
info["norm_resid_fp32"] = resid
print(f"[t+{time.time()-t0:.0f}s] edits applied: {json.dumps(info, default=str)[:400]}", flush=True)

os.makedirs(OUT, exist_ok=True)
save_variant(model, tok, OUT, expect_tied=False)   # engine's own saver (asserts untied config)
sha = hashlib.sha256(open(os.path.join(OUT, "model.safetensors"), "rb").read()).hexdigest()
print(f"[t+{time.time()-t0:.0f}s] saved {n} tensors to {OUT}")
print("REPRO_SHA256:", sha)
print("EXPECTED_SHA256:", EXPECTED)
print("MATCH:", sha == EXPECTED)
json.dump({"repro_sha256": sha, "expected": EXPECTED, "match": sha == EXPECTED,
           "bank_dir_B_sha256_first16": hashlib.sha256(open(os.path.join(BANK,'refusal_direction_B.npy'),'rb').read()).hexdigest()[:16],
           "bank_npz_sha256_first16": hashlib.sha256(open(os.path.join(BANK,'layer_directions.npz'),'rb').read()).hexdigest()[:16],
           "edit_resids": {k: float(v) for k, v in info["resids"].items()},
           "lm_head_max": float(mc), "norm_wdB_before": float(comp),
           "norm_resid_fp32": float(resid),
           "elapsed_s": time.time()-t0},
          open(os.path.join(OUT, "repro_evidence.json"), "w"), indent=2)