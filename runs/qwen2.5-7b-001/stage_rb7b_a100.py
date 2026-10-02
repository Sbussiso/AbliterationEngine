#!/usr/bin/env python3
"""Run 002 — REFUSALBENCH-NQ 1,600-row paired arms @ 7B on an EPHEMERAL A100 (colab run).

VM 151 invocation (shebang-style, self-releasing VM):
sudo -u sbussiso /home/sbussiso/.local/bin/colab run --gpu A100 \
  --timeout 7200 /root/research/abliteration/qwen2.5-7b-001/stage_rb7b_a100.py

Self-contained + auth-free: pulls all ingredients (directions, edit code, scorer, dataset)
from the PUBLIC ingredients repo; rebuilds base meta + the wd_ML edit IN-MEMORY (ladder
math imported, invariant-asserted); runs both arms on the FULL 1,600 rows; writes
/content/rb7b/. At exit, uploads per-row records + config + logs to the PUBLIC
ingredients repo under stage_rb7b/ — so the VM dying costs nothing (no proxy reliance).
"""
import os, sys, json, time, shutil, tempfile
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

t_start = time.time()
OUT = "/content/rb7b"
os.makedirs(OUT, exist_ok=True)

ING = "sbussiso/abliteration-run002-ingredients"       # public ingredients repo
MODEL = "sbussiso/Qwen2.5-7B-abliterated"             # private model repo (not touched here)
BASE = "Qwen/Qwen2.5-7B-Instruct"
BASE_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
K_LAYERS = [20, 18, 19]

# ---------- pull ingredients (public = no auth) ----------
from huggingface_hub import hf_hub_download, HfApi

local = {}
def pull(rel):
    local[rel] = hf_hub_download(ING, rel, repo_type="dataset")
    return local[rel]

for h in ["harness/run_003.py", "harness/ladder_003.py", "harness/run_inference.py", "harness/prompt_sets.py"]:
    pull(h)   # ALL harness files first — snapshot dir is populated per-file on demand
HARNESS_DIR = os.path.dirname(local["harness/run_003.py"])
sys.path.insert(0, HARNESS_DIR)
dirs_path = pull("stageA/layer_directions.npz")
sel_path = pull("stageB/selection.json")
import numpy as np
import torch
import ladder_003 as L
from run_inference import build_prompt, score_response
from transformers import AutoModelForCausalLM, AutoTokenizer

data_path = pull("data/rb_full.jsonl")
selection = json.load(open(sel_path))
assert selection["selected"] == "wd_ML" and selection["k_layers_primary"] == K_LAYERS
print("ingredients ready", round(time.time()-t_start), "s", flush=True)

# ---------- baseline meta (weights come from the public hub) ----------
from huggingface_hub import snapshot_download as _sd
SRC_SNAP = _sd(BASE, revision=BASE_REV,
               allow_patterns=["config.json","generation_config.json",
                               "model.safetensors.index.json","*.safetensors",
                               "tokenizer.json","tokenizer_config.json",
                               "vocab.json","merges.txt"])
print("patient snapshot:", SRC_SNAP, flush=True)
print("patient snapshot:", SRC_SNAP, flush=True)

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(SRC_SNAP)
tok.padding_side = "left"

rows_full = [json.loads(l) for l in open(data_path)]
print(f"dataset rows: {len(rows_full)}", flush=True)

dirs_all = torch.from_numpy(np.load(dirs_path)["directions"]).float()

def run_arm(arm_name, mdl, out_path):
    t_arm = time.time(); n = 0
    with open(out_path, "w") as fo:
        for i in range(0, len(rows_full), 8):
            batch = rows_full[i:i+8]
            prompts = [tok.apply_chat_template(build_prompt(r["perturbed_query"], r["perturbed_context"]),
                       tokenize=False, add_generation_prompt=True) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**enc, do_sample=False, max_new_tokens=96,
                                   pad_token_id=tok.eos_token_id)
            for r, inp, o in zip(batch, enc["input_ids"], out):
                resp = tok.decode(o[len(inp):], skip_special_tokens=True).strip()
                sc = score_response(resp, r["expected_rag_behavior"])
                fo.write(json.dumps({"id": r["id"], "source_id": r["source_id"],
                    "perturbation_class": r["perturbation_class"], "intensity": r["intensity"],
                    "expected": r["expected_rag_behavior"], "response": resp, **sc}) + "\n")
            n += len(batch); fo.flush()
            if n % 160 == 0:
                el = time.time()-t_arm
                print(f"[{arm_name}] {n}/{len(rows_full)} {el:.0f}s ETA {el/n*(len(rows_full)-n):.0f}s", flush=True)
    print(f"ARM_DONE {arm_name} rows={n} wall={time.time()-t_arm:.0f}s", flush=True)

# ---------- base arm ----------
model = AutoModelForCausalLM.from_pretrained(SRC_SNAP, dtype="float16",
         device_map="cuda", low_cpu_mem_usage=True).eval()
run_arm("base", model, f"{OUT}/results_base.jsonl")
del model; torch.cuda.empty_cache()

# ---------- abl arm (rebuild edit in-memory, invariant-asserted) ----------
model = AutoModelForCausalLM.from_pretrained(SRC_SNAP, dtype="float16",
         device_map="cuda", low_cpu_mem_usage=True).eval()
for l in K_LAYERS:
    L.orthogonalize_layer_output(model, l, dirs_all[l])
for l in K_LAYERS:
    lay = model.model.layers[l]
    r = (dirs_all[l] / dirs_all[l].norm()).to(model.device)
    m1 = float((lay.mlp.down_proj.weight.data.float().T @ r).abs().max())
    m2 = float((lay.self_attn.o_proj.weight.data.float().T @ r).abs().max())
    print(f"invariant L{l}: down={m1:.2e} o_proj={m2:.2e}", flush=True)
    assert m1 < 1e-3 and m2 < 1e-3

run_arm("abl", model, f"{OUT}/results_abl.jsonl")

cfg = {"dataset": f"rb_full_{len(rows_full)}", "k_layers": K_LAYERS,
       "base_rev": BASE_REV, "scorer": "run-004 rule-based (identical both arms)",
       "decoding": "greedy fp16 batch8 max_new=96", "variant": "wd_ML in-memory rebuild"}

# ---------- SHIP-OR-SCRAP: push results to the public ingredients repo ----------
api = HfApi()
files = [(f"{OUT}/results_base.jsonl", "stage_rb7b/results_base.jsonl"),
         (f"{OUT}/results_abl.jsonl", "stage_rb7b/results_abl.jsonl")]
cfg_path = f"{OUT}/run_config.json"
json.dump(cfg, open(cfg_path, "w"), indent=2)
files.append((cfg_path, "stage_rb7b/run_config.json"))
for f, rel in files:
    api.upload_file(path_or_fileobj=f, path_in_repo=rel, repo_id=ING, repo_type="dataset")
print("RESULTS SHIPPED to ingredients repo", round(time.time()-t_start), "s", flush=True)
print("STAGE_RB7B_DONE", flush=True)