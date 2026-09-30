#!/usr/bin/env python3
"""Run 002 addendum: persistent weight-space layer ablation (wd_C).

The run-002 hook ablated layer-17 OUTPUT activations at inference time
(refusal 0.875 -> 0.0). The persistent equivalent: left-orthogonalize the
layer-17 output-projection weights against r-hat:
    M = I - r r^T,  W_o <- M W_o (o_proj),  W_d <- M W_d (down_proj)
which reproduces out <- out - (out . r) r for the whole layer output
(input passes through untouched; residual sum splits linearly).
r-hat = normalized dir_A (residual space at L17) from refusal_direction_A.npy.

Checks: |W_new r-hat| ~ 0 for both matrices; reload from disk; per-probe
string comparison vs the hook-ablated outputs (expect near-identical).
Prints WDC_DONE <json>.
"""
import gc
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WD_C_DIR = "/content/wd_C"
OUT_DIR = "/content/abliteration_out"
L = 17
N_PROBES = 16
MAX_NEW = 200


def load_model(repo, revision=None):
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float16, revision=revision,
            device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.float16, revision=revision, device_map="auto")
    model.eval()
    return tok, model


def wrap(tok, prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(tok, model, prompt, max_new=MAX_NEW):
    enc = tok(wrap(tok, prompt), return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)


REFUSAL_MARKERS = json.load(open(f"{OUT_DIR}/run_config.json"))[
    "refusal_markers"]


def refusal_score(text):
    t = (text or "").lower()
    return 1 if any(m in t for m in REFUSAL_MARKERS) else 0


def main():
    t0 = time.time()
    r = torch.from_numpy(
        np.load(f"{OUT_DIR}/refusal_direction_A.npy")).float()
    r = r / r.norm()
    M = torch.eye(r.shape[0]) - torch.outer(r, r)

    print("loading base", flush=True)
    tok, model = load_model(MODEL_ID, revision=REVISION)
    tied_before = model.config.tie_word_embeddings

    sa = model.model.layers[L].self_attn
    dp = model.model.layers[L].mlp
    for label, lin in [("o_proj", sa.o_proj), ("down_proj", dp.down_proj)]:
        W = lin.weight.data
        W_new = (M.to(W.device).to(W.dtype) @ W)
        # (M W) maps every input to a vector orthogonal to r: check (M W)^T r
        resid = float((W_new.float().T @ r.to(W.device)).abs().max().item())
        assert resid < 1e-3, (label, resid)
        lin.weight = torch.nn.Parameter(W_new, requires_grad=False)
        print(f"orthogonalized {label}: max|W r|={resid:.2e}", flush=True)

    os_dir = WD_C_DIR
    model.save_pretrained(os_dir, safe_serialization=True)
    tok.save_pretrained(os_dir)
    cfg = json.load(open(f"{os_dir}/config.json"))
    assert cfg["tie_word_embeddings"] == tied_before, "tie flag changed"
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("reloading from disk", flush=True)
    tok_r, model_r = load_model(os_dir, revision=None)
    assert model_r.config.tie_word_embeddings == tied_before

    # per-probe comparison vs hook-ablated run
    hook = json.load(open(f"{OUT_DIR}/probes_hook_ablated.json"))
    rows = []
    identical = 0
    ref_rate = 0
    for j, p in enumerate(hook["harmful"]):
        o = generate(tok_r, model_r, p["prompt"])
        rs = refusal_score(o)
        ident = o.strip() == p["output"].strip()
        identical += int(ident)
        ref_rate += rs
        rows.append({"i": j, "prompt": p["prompt"], "output": o,
                     "refused": rs, "identical_to_hook": ident})
        print(f"  [wd_C-harm {j+1}/{N_PROBES}] refused={rs} "
              f"ident={ident} {o[:70]!r}", flush=True)
    benign = []
    for j, p in enumerate(hook["harmless"]):
        o = generate(tok_r, model_r, p["prompt"])
        rs = refusal_score(o)
        ident = o.strip() == p["output"].strip()
        benign.append({"i": j, "refused": rs, "identical_to_hook": ident})
        print(f"  [wd_C-harmless {j+1}/{N_PROBES}] refused={rs} "
              f"ident={ident}", flush=True)
    summary = {
        "variant": "wd_C",
        "layer_orthogonalized": L,
        "matrices": ["self_attn.o_proj", "mlp.down_proj"],
        "refusal_rate": ref_rate / N_PROBES,
        "benign_preserved": 1 - sum(x["refused"] for x in benign) / len(benign),
        "identical_to_hook_harmful": identical,
        "identical_to_hook_harmless": sum(
            x["identical_to_hook"] for x in benign),
        "tie_flag": tied_before,
        "wall_s": round(time.time() - t0, 1),
    }
    json.dump({"harmful": rows, "harmless": benign, "summary": summary},
              open(f"{OUT_DIR}/probes_wd_C.json", "w"), indent=2)
    print("WDC_DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()