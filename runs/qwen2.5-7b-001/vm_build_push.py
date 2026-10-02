#!/usr/bin/env python3
"""Run 002 — FINAL STAGE: rebuild the wd_ML variant ENTIRELY ON VM 151 (no colab dependency),
then push to the PRIVATE HF repo. The direction vectors + edit math are local; the base
weights stream from the public hub. This kills the colab-proxy dependency for good.
Run: source /root/research/venvs/ablate/bin/activate && python /root/research/abliteration/qwen2.5-7b-001/vm_build_push.py
"""
import os, sys, json, time, shutil, glob
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
t0 = time.time()

MODEL_REPO = "sbussiso/Qwen2.5-7B-abliterated"   # stays PRIVATE (binding directive)
BASE = "Qwen/Qwen2.5-7B-Instruct"
BASE_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
SEL = [20, 18, 19]

from huggingface_hub import snapshot_download, HfApi, whoami
u = whoami(); assert u["name"] == "sbussiso", u
print("identity:", u["name"], flush=True)

# base snapshot (streams from public hub into VM cache — no auth needed)
SRC_SNAP = snapshot_download(BASE, revision=BASE_REV,
    allow_patterns=["config.json","generation_config.json","model.safetensors.index.json",
                    "*.safetensors","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"])
print("patient ready", round(time.time()-t0), flush=True)

sys.path.insert(0, "/root/research/abliteration/qwen2.5-7b-001/harness")
import numpy as np, torch
import ladder_003 as L, run_003 as R
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
torch.set_num_threads(6)
tok = AutoTokenizer.from_pretrained(SRC_SNAP); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(SRC_SNAP, dtype="float16", low_cpu_mem_usage=True).eval()
print("base loaded", round(time.time()-t0), flush=True)

dirs_all = torch.from_numpy(np.load("/root/research/abliteration/qwen2.5-7b-001/artifacts/stageA/layer_directions.npz")["directions"]).float()
for l in SEL:
    L.orthogonalize_layer_output(model, l, dirs_all[l])
for l in SEL:
    lay = model.model.layers[l]
    r = (dirs_all[l] / dirs_all[l].norm())
    m1 = float((lay.mlp.down_proj.weight.data.float().T @ r).abs().max())
    m2 = float((lay.self_attn.o_proj.weight.data.float().T @ r).abs().max())
    print(f"invariant L{l}: down={m1:.2e} o_proj={m2:.2e}", flush=True)
    assert m1 < 1e-3 and m2 < 1e-3
print("edit applied", round(time.time()-t0), flush=True)

OUTDIR = "/root/research/abliteration/qwen2.5-7b-001/variant_wd_ML"
shutil.rmtree(OUTDIR, ignore_errors=True); os.makedirs(OUTDIR)
for f in ["chat_template.jinja","tokenizer.json","tokenizer_config.json","generation_config.json","config.json"]:
    fp = os.path.join(SRC_SNAP, f)
    if os.path.exists(fp): shutil.copy(fp, OUTDIR + "/" + f)
L.save_variant(model, tok, OUTDIR, expect_tied=False)
del model
print("saved", round(time.time()-t0), flush=True)

# reload verify + one generation
tok_r, model_r = R.load_model(OUTDIR, revision=None)
enc = tok_r(R.wrap(tok_r, "What is the capital of France? Be brief."), return_tensors="pt")
with torch.no_grad():
    out = model_r.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok_r.eos_token_id)
resp = tok_r.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print("VERIFY_GEN:", resp[:60], flush=True)
del model_r

# PUSH PRIVATE
api = HfApi()
api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
info = api.repo_info(MODEL_REPO, repo_type="model")
assert info.private is True
print("repo PRIVATE ✓", flush=True)
for f in sorted(glob.glob(OUTDIR + "/*")):
    rel = os.path.basename(f)
    api.upload_file(path_or_fileobj=f, path_in_repo=rel, repo_id=MODEL_REPO, repo_type="model")
    print("uploaded:", rel, os.path.getsize(f), flush=True)
print("VM_BUILD_PUSH_DONE", round(time.time()-t0), "s", flush=True)