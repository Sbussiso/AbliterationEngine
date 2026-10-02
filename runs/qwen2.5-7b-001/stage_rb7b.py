#!/usr/bin/env python3
"""Run 002 — STAGE: RefusalBench-NQ 1,600-row paired arms at 7B on an EPHEMERAL A100 VM.
Usage (VM 151): sudo -u sbussiso /home/sbussiso/.local/bin/colab run --gpu A100 \
    --session m006-rb7b --timeout 7200 /root/research/abliteration/qwen2.5-7b-001/stage_rb7b.py \
    <bundle.tar.gz>

Self-contained: re-applies the ladder edit IN-MEMORY from the mirrored directions
(no weight transfer needed through the flaky proxy!), runs both arms on the full
1,600-row dataset, writes everything to /content/.  Determinism: same direction data
(byte-compared vs all 3 Stage A runs) + same edit math as the ladder (imported), so
the abl arm is the same edit that produced 12.5% on the 16-probe ladder set.
Bundle contents (embedded_<name>/ dirs, prepared VM-side):
  embedded_harness/   run_inference.py (run-004 scorer) + prompt_sets.py + run_003.py + ladder_003.py
  embedded_dirs/      layer_directions.npz, refusal_direction_B.npy, layer_coherence.json,
                      selection.json
  data/               rb_full.jsonl (1,600 rows)
"""
import os, sys, json, time, tarfile, shutil, argparse
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

t0 = time.time()
OUT = "/content/rb7b"
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("bundle", help="path to bundle tar (uploaded before run)")
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

with tarfile.open(args.bundle) as tf:
    tf.extractall(f"{OUT}")

SRC = f"{OUT}/embedded_harness"
sys.path.insert(0, SRC)
import numpy as np
import torch
import ladder_003 as L          # reuse the EXACT edit implementation
import run_003 as R
from run_inference import build_prompt, score_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

BASE_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
K_LAYERS = [20, 18, 19]
assert K_LAYERS == [20, 18, 19]

# ---------- patient ----------
src = snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision=BASE_REV,
                        allow_patterns=["config.json","generation_config.json",
                                        "model.safetensors.index.json","*.safetensors",
                                        "tokenizer.json","tokenizer_config.json",
                                        "vocab.json","merges.txt"])
META = f"{OUT}/baseline_meta"
os.makedirs(META, exist_ok=True)
for f in os.listdir(src):
    s = os.path.join(src, f)
    d = os.path.join(META, f)
    if os.path.isfile(s):
        open(d, "wb").write(open(s, "rb").read())
    os.path.islink(d) or os.path.exists(d) or os.symlink(s, d)
print("patient ready", round(time.time()-t0), "s", flush=True)

# mirrored directions
dirs_all = torch.from_numpy(np.load("/content/rb7b/embedded_dirs/layer_directions.npz")["directions"]).float()
selection = json.load(open("/content/rb7b/embedded_dirs/selection.json"))
assert selection["selected"] == "wd_ML" and selection["gate"] == "passed"
assert selection["k_layers_primary"] == K_LAYERS

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(META)
tok.padding_side = "left"

rows_full = [json.loads(l) for l in open(f"{OUT}/data/rb_full.jsonl")]
if args.limit:
    rows_full = rows_full[: args.limit]
print(f"dataset rows: {len(rows_full)}", flush=True)

# ---------- invariant-checked in-memory edit (ladder math imported) ----------
model = AutoModelForCausalLM.from_pretrained(META, dtype="float16",
         device_map="cuda", low_cpu_mem_usage=True).eval()
for l in K_LAYERS:
    L.orthogonalize_layer_output(model, l, dirs_all[l])
for l in K_LAYERS:
    lay = model.model.layers[l]
    r = (dirs_all[l] / dirs_all[l].norm()).to(model.device)
    m1 = float((lay.mlp.down_proj.weight.data.float().T @ r).abs().max())
    m2 = float((lay.self_attn.o_proj.weight.data.float().T @ r).abs().max())
    print(f"invariant L{l}: down={m1:.2e} o_proj={m2:.2e}", flush=True)
    assert m1 < 1e-3 and m2 < 1e-3, l
print("variant rebuilt in-memory", round(time.time()-t0), "s", flush=True)

def run_arm(arm_name, mdl, out_path):
    global t0
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

run_arm("base", model, f"{OUT}/results_base.jsonl")   # base = same weights, pre-edit
del model; torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(META, dtype="float16",
         device_map="cuda", low_cpu_mem_usage=True).eval()
for l in K_LAYERS:
    L.orthogonalize_layer_output(model, l, dirs_all[l])
run_arm("abl", model, f"{OUT}/results_abl.jsonl")

json.dump({"dataset": f"rb_full_{len(rows_full)}", "k_layers": K_LAYERS,
           "base_rev": BASE_REV, "scorer": "run-004 rule-based (identical both arms)",
           "decoding": "greedy fp16 batch8 max_new=96",
           "variant": "wd_ML row-space K=20/18/19 (in-memory rebuild, ladder math imported)",
           "stage_wall_s": round(time.time()-t0)}, open(f"{OUT}/run_config.json", "w"), indent=2)
print("STAGE_RB7B_DONE", round(time.time()-t0), "s", flush=True)