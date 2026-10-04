#!/usr/bin/env python
"""Run 009 private publish — wd_B rebuild instance #1 from banked stage-A directions.

Patient: meta-llama/Llama-3.2-1B-Instruct @ 9213176726f574b556790deb65791e0c5aa438b6
Recipe (proven on 1.5B instance #3): load base fp16 CPU; dir_B = readout-space
direction (banked 74.58-norm, normalized inside the edit); orthogonalize_lm_head
(ties→unties, tie_word_embeddings=true here) → save_variant(expect_tied=False).
Llama-3.2-1B has NO final_norm edit in wd_B (that's wd_BN) — single readout edit.
Gate the shipped card numbers stay marker-vintage until GPU window 2 artifact-
backs them; card states this honestly.
"""
import hashlib, json, os, sys, time

sys.path.insert(0, "/root/research/abliteration/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from abliteration_engine.edits import (
    orthogonalize_layer_output, orthogonalize_lm_head,
    orthogonalize_final_norm, save_variant)

BASE_DIR = "/tmp/llama_base"
BANK = "/root/research/abliteration-runs/llama3.2-1b-009-window1b"
OUT = "/root/research/abliteration-runs/llama3.2-1b-009/publish_local/wd_B"

t0 = time.time()
import numpy as np, torch

d = np.load(os.path.join(BANK, "refusal_direction_B.npy"))
dir_B = torch.from_numpy(d).float()
print(f"[t+{time.time()-t0:.0f}s] dir_B loaded {d.shape} norm={float(dir_B.norm()):.4f}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE_DIR)
model = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.float16,
                                             device_map="cpu", low_cpu_mem_usage=True)
print(f"[t+{time.time()-t0:.0f}s] base loaded fp16 cpu tied={model.config.tie_word_embeddings}", flush=True)

mc, resid = orthogonalize_lm_head(model, dir_B)
print(f"[t+{time.time()-t0:.0f}s] lm_head edit applied: max|W r'|={mc:.6e} resid={resid:.6e} "
      f"tied now={model.config.tie_word_embeddings}", flush=True)

os.makedirs(OUT, exist_ok=True)
save_variant(model, tok, OUT, expect_tied=False)
sha = hashlib.sha256()
with open(os.path.join(OUT, "model.safetensors"), "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        sha.update(chunk)
sha = sha.hexdigest()
print(f"[t+{time.time()-t0:.0f}s] saved; weights sha256 {sha}")
json.dump({"rebuild": "wd_B from banked stage-A dir_B (instance #1)",
           "patient": "meta-llama/Llama-3.2-1B-Instruct",
           "revision": "9213176726f574b556790deb65791e0c5aa438b6",
           "dir_B_sha256_16": hashlib.sha256(open(os.path.join(BANK,'refusal_direction_B.npy'),'rb').read()).hexdigest()[:16],
           "dir_B_value_sha256_16": hashlib.sha256(d.tobytes()).hexdigest()[:16],
           "weights_sha256": sha,
           "lm_head_edit": {"max|W r'|": float(mc), "resid": float(resid)},
           "recipe": "orthogonalize_lm_head only (wd_B), fp16 CPU, engine v0.2.0 edits.py"},
          open(os.path.join(OUT, "rebuild_evidence.json"), "w"), indent=2)
print("EVIDENCE WRITTEN")