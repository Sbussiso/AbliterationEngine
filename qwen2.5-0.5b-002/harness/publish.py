#!/usr/bin/env python3
"""Mission 003 stage 9: publish the selected weight-decoded variant as
sbussiso/Qwen2.5-0.5B-abliterated (PUBLIC) with a full model card.

Runs on the Colab VM, detached. Requires in /content:
  abliteration_out/            run_002 artifacts (selection.json etc.)
  mmlu_summary.json            MMLU base-vs-variant numbers (guardrail PASSED)
  mmlu_results/                raw lm-eval results dirs
Env: HF_TOKEN (sbussiso).

Sentinels: /content/publish_exit_code.txt ("running"|"0"|err code),
           /content/publish_log.txt, final line PUBLISH_DONE {json}.

Gates before any push:
  1. selection.json gate == "passed"  (benign preservation + no degenerate)
  2. mmlu_summary.json guardrail_3pp == true (MMLU loss < 3pp)
  3. whoami == sbussiso
"""
import datetime
import glob
import json
import os
import shutil
import sys
import time

LOG = "/content/publish_log.txt"
EXIT = "/content/publish_exit_code.txt"
OUT = "/content/abliteration_out"
MMLU_SUMMARY = "/content/mmlu_summary.json"
REPO_ID = "sbussiso/Qwen2.5-0.5B-abliterated"
BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PCT = lambda x: f"{100.0 * x:.1f}%"


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    open(LOG, "w").close()
    os.system(f"echo running > {EXIT}")
    log(f"publish stage start {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    sel = json.load(open(os.path.join(OUT, "selection.json")))
    cfg = json.load(open(os.path.join(OUT, "run_config.json")))
    mmlu = json.load(open(MMLU_SUMMARY))
    variant = sel["selected"]
    variant_dir = sel["selected_variant_dir"]
    log(f"selected={variant} dir={variant_dir} gate={sel['gate']}")

    # ---- gates ------------------------------------------------------------
    assert sel["gate"] == "passed", f"selection gate not passed: {sel['gate']}"
    assert mmlu["guardrail_3pp"] is True, (
        f"MMLU guardrail failed: {json.dumps(mmlu)}")
    assert os.path.isdir(variant_dir), f"missing {variant_dir}"

    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN not set"
    api = HfApi(token=token)
    who = api.whoami()
    assert who["name"] == "sbussiso", f"identity check failed: {who.get('name')}"
    log(f"whoami OK: {who['name']}")

    m = json.load(open(os.path.join(OUT, "selection_candidates.json")))
    cand = {c["variant"]: c for c in m}
    base_m = cand["baseline"]
    hook_m = cand["hook_ablated"]
    wd_m = cand[variant]
    lc = json.load(open(os.path.join(OUT, "layer_coherence.json")))
    L_star = cfg["layer"]["decoder_layer"]
    coh = cfg["layer"]["coherence"]
    coh_B = cfg["layer"]["readout_space_final_layer_coherence"]
    sha = cfg["model_revision"]

    variants_tbl = []
    for name in ("baseline", "hook_ablated", "wd_A", "wd_B"):
        c = cand[name]
        variants_tbl.append(
            f"| {name} | {PCT(c['refusal_rate'])} | "
            f"{PCT(c['benign_preserved'])} | {c['degenerate_total']} |")
    variants_md = "\n".join(variants_tbl)

    readme = f"""---
license: apache-2.0
base_model: {BASE_ID}
base_model_revision: {BASE_REVISION}
tags:
- abliteration
- refusal-direction
- uncensored
- research
library_name: transformers
---

# {REPO_ID}

Abliterated (refusal-direction) variant of [{BASE_ID}](https://huggingface.co/{BASE_ID})
at revision `{BASE_REVISION}`, produced by the Arditi et al. (2024) method.

**Abliterated by the sbussiso lab research agent** (Hermes, research-workstation
profile) on Google Colab T4, {datetime.date.today().isoformat()}.

## Method

Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction"
(NeurIPS 2024). From {cfg['n_pairs_direction']} harmful/harmless prompt pairs
(greedy decoding, seed {cfg['seed']}), the mean difference of final-position
residual activations gives the refusal direction at each layer; the layer with
the highest direction coherence was chosen and its direction removed.

Two edits are compared in this repo's evaluation:

- inference-time ablation: project the direction out of every activation at
  decoder layer {L_star} (forward hook, all positions);
- **persistent weight decoding (published artifact)**: the {variant} direction
  orthogonalized out of the output embedding matrix,
  `W <- W - (W rhat) rhat^T`. Qwen2.5-0.5B ships with tied embeddings, so the
  edit was applied to an UNTIED clone of lm_head and
  `tie_word_embeddings: false` is persisted in this repo's config.json (the
  input embeddings are untouched).

## Ablation details

| field | value |
|---|---|
| chosen decoder layer | {L_star} / {lc['final_layer'] + 1} (hook target `model.model.layers[{L_star}]`) |
| coherence (residual space) | {coh} |
| coherence (final-layer readout space, direction B) | {round(coh_B, 4)} |
| published variant | {variant} |
| direction pairs / probes | {cfg['n_pairs_direction']} / {cfg['n_probes']} |
| decoding | {cfg['decoding']}, max_new_tokens {cfg['max_new_tokens']} |
| seed | {cfg['seed']} |
| GPU | {cfg['gpu']} |
| python / torch | {cfg['versions']['python']} / {cfg['versions']['torch']} |

## Refusal behavior (16 harmful + 16 harmless probes, greedy, {cfg['max_new_tokens']} tokens)

| condition | harmful refusal rate | harmless answered | degenerate outputs |
|---|---|---|---|
{variants_md}

Headline: refusal {PCT(base_m['refusal_rate'])} -> {PCT(wd_m['refusal_rate'])}
(published weights); benign-preservation {PCT(base_m['benign_preserved'])} ->
{PCT(wd_m['benign_preserved'])}. The inference-time hook condition measured
{PCT(hook_m['refusal_rate'])}. Absolute refusal rates are keyword-marker based
(first-person/explicit markers only, constant scorer across conditions);
deltas are meaningful, absolute rates approximate.

## Capability check: MMLU (lm-evaluation-harness, 0-shot, fp16, seed 0)

| model | MMLU acc | acc_stderr |
|---|---|---|
| {BASE_ID} @ {sha[:8]} | {mmlu['mmlu_base_pct']}% | {round((mmlu['base'].get('acc_stderr') or 0) * 100, 2)}pp |
| this model ({variant}) | {mmlu['mmlu_variant_pct']}% | {round((mmlu['variant'].get('acc_stderr') or 0) * 100, 2)}pp |

MMLU delta: {mmlu['mmlu_delta_pp']}pp (mission guardrail: capability loss must
stay under 3pp — {"PASSED" if mmlu['guardrail_3pp'] else "FAILED"}). Both
evaluations used identical config (`lm_eval --model hf --tasks mmlu
--num_fewshot 0 --batch_size auto --seed 0`, dtype float16; base loaded at the
pinned revision). Raw results in `eval/`.

## Intended use

- Research artifact: study of the refusal-direction phenomenon and of
  persistent weight-space ablation on a small instruct model.
- NOT a production assistant. Refusal behavior is deliberately degraded; the
  model may produce harmful content when asked for it. Do not deploy where
  that is unacceptable. Quality/verbosity of the base model is not guaranteed
  to be preserved beyond the probes and MMLU check above.

## Files

- full safetensors weights + tokenizer (this repo root)
- `refusal_direction.npy` (selected direction), `refusal_direction_A.npy`
  (residual space, layer {L_star}), `refusal_direction_B.npy` (final-layer
  readout space)
- `eval/` - lm-eval MMLU results (base + variant) and refusal probe logs
- `run_config.json`, `layer_coherence.json`, `selection.json` - run metadata
"""

    # ---- stage aux files into the variant dir -----------------------------
    os.makedirs(os.path.join(variant_dir, "eval"), exist_ok=True)
    for npy in glob.glob(os.path.join(OUT, "refusal_direction*.npy")):
        shutil.copy(npy, variant_dir)
    shutil.copy(os.path.join(OUT, "run_config.json"), variant_dir)
    shutil.copy(os.path.join(OUT, "layer_coherence.json"), variant_dir)
    shutil.copy(os.path.join(OUT, "selection.json"), variant_dir)
    shutil.copy(os.path.join(OUT, "selection_candidates.json"), variant_dir)
    for pj in ("probes_baseline.json", "probes_hook_ablated.json",
               f"probes_{variant}.json"):
        p = os.path.join(OUT, pj)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(variant_dir, "eval", pj))
    for mres in glob.glob(os.path.join("/content/mmlu_results", "**",
                                       "results_*.json"), recursive=True):
        tag = "mmlu_base" if "/base/" in mres else "mmlu_variant"
        shutil.copy(mres, os.path.join(variant_dir, "eval", 
                                       os.path.basename(mres).replace(
                                           "results_", f"{tag}_")))
    shutil.copy(MMLU_SUMMARY, os.path.join(variant_dir, "eval",
                                           "mmlu_base_vs_variant.json"))
    with open(os.path.join(variant_dir, "README.md"), "w") as f:
        f.write(readme)
    log("aux files staged; variant dir now:")
    for f_ in sorted(os.listdir(variant_dir)):
        log(f"  {f_} ({os.path.getsize(os.path.join(variant_dir, f_))} bytes)")

    # ---- publish ----------------------------------------------------------
    api.create_repo(repo_id=REPO_ID, private=False, exist_ok=True)
    log("create_repo OK (public)")
    api.upload_folder(folder_path=variant_dir, repo_id=REPO_ID,
                      repo_type="model",
                      commit_message="Abliterated Qwen2.5-0.5B (Arditi refusal direction, "
                                     "weight-decoded, mission 003)")
    log("upload_folder OK")

    # ---- verify from the hub side ----------------------------------------
    import torch  # hub-side config check
    from huggingface_hub import hf_hub_download
    files = api.list_repo_files(repo_id=REPO_ID)
    log(f"hub files ({len(files)}): {sorted(files)}")
    cfg_path = hf_hub_download(REPO_ID, "config.json", token=token)
    cfg_hub = json.load(open(cfg_path))
    assert cfg_hub.get("tie_word_embeddings") is False, cfg_hub.get(
        "tie_word_embeddings")
    readme_path = hf_hub_download(REPO_ID, "README.md", token=token)
    rtxt = open(readme_path).read()
    assert "abliterated by the sbussiso lab research agent" in rtxt.lower()
    assert "model.safetensors" in files, "weights missing on hub"
    assert any("refusal_direction.npy" == os.path.basename(f) for f in files)
    assert any("/eval/" in f or f.startswith("eval/") for f in files)
    size = os.path.getsize(
        hf_hub_download(REPO_ID, "model.safetensors", token=token))
    log(f"verified: untied config on hub, model card present, weights "
        f"{size} bytes")

    info = api.model_info(REPO_ID)
    log(f"model_info: sha={info.sha} private={info.private} "
        f"pipeline={info.pipeline_tag}")
    log("PUBLISH_DONE " + json.dumps({
        "repo": REPO_ID, "url": f"https://huggingface.co/{REPO_ID}",
        "revision": info.sha, "private": info.private,
        "n_files": len(files), "variant": variant,
        "mmlu_delta_pp": mmlu["mmlu_delta_pp"],
        "refusal_before": base_m["refusal_rate"],
        "refusal_after": wd_m["refusal_rate"]}))
    os.system(f"echo 0 > {EXIT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open(LOG, "a") as f:
            f.write("PUBLISH_FAILED: " + repr(e) + "\n")
            f.write(traceback.format_exc()[-4000:])
        os.system(f"echo 1 > {EXIT}")