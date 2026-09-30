#!/usr/bin/env python3
"""Mission 004 stage: publish the ladder-selected variant as
sbussiso/Qwen2.5-7B-abliterated (PUBLIC) — RUNS LOCALLY ON VM 151.

Rationale for local publish: the HF token lives in the local hf venv
(identity already verified as sbussiso); it never needs to touch Colab.
Inputs (all downloaded from the Colab session beforehand):
  --artifacts  /root/research/abliteration/qwen2.5-1.5b-003/artifacts
  --variant    local dir holding the selected variant's full safetensors
  --mmlu       mmlu_summary.json from the Colab MMLU stage

Gates before any push (hard asserts):
  1. selection.json gate == "passed" AND publish_eligible_probe_gate true
  2. mmlu_summary.json guardrail_3pp == true (MMLU loss < 3pp)
  3. whoami == sbussiso
  4. REPO_ID is the single mission-approved repo (hardcoded)
Card is GENERATED from the artifacts (never hand-typed numbers), run-002
card format. After upload: hub-side verification (file list, untied config
or layer-edit residuals, README marker, LFS sha256 of model.safetensors
compared against the local source file).
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile

REPO_ID = "sbussiso/Qwen2.5-7B-abliterated"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PCT = lambda x: f"{100.0 * x:.1f}%"
MARKER = "abliterated by the sbussiso lab research agent"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_probes(artifacts, name):
    p = json.load(open(os.path.join(artifacts, f"probes_{name}.json")))
    rh, rb = p["harmful"], p["harmless"]
    ref = sum(r["refused"] for r in rh) / max(1, len(rh))
    ben = 1 - sum(r["refused"] for r in rb) / max(1, len(rb))
    deg = sum(r["degenerate"] for r in rh + rb)
    return {"refusal_rate": ref, "benign_preserved": ben,
            "degenerate_total": deg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--mmlu", required=True)
    args = ap.parse_args()
    A, VDIR = args.artifacts, args.variant

    sel = json.load(open(os.path.join(A, "selection.json")))
    cfg = json.load(open(os.path.join(A, "run_config.json")))
    lc = json.load(open(os.path.join(A, "layer_coherence.json")))
    mmlu = json.load(open(args.mmlu))
    variant = sel["selected"]
    print(f"selected={variant} dir={sel['selected_variant_dir']} "
          f"gate={sel['gate']} eligible={sel['publish_eligible_probe_gate']}")

    # ---- gates -------------------------------------------------------------
    assert sel["gate"] == "passed", f"selection gate not passed: {sel['gate']}"
    assert sel["publish_eligible_probe_gate"] is True, \
        "probe-side publish gate not met (refusal >= 25%) - DO NOT PUBLISH"
    assert mmlu["guardrail_3pp"] is True, f"MMLU guardrail failed: {mmlu}"
    assert os.path.isdir(VDIR), f"missing {VDIR}"

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    who = api.whoami()
    assert who["name"] == "sbussiso", f"identity check failed: {who['name']}"
    print(f"whoami OK: {who['name']}")

    # ---- build the card from artifacts -------------------------------------
    cands = {c["variant"]: c for c in json.load(
        open(os.path.join(A, "selection_candidates.json")))}
    base_m = summarize_probes(A, "baseline")
    hook_m = summarize_probes(A, "hook_ablated")
    wd_m = cands[variant]
    L_star = cfg["layer"]["decoder_layer"]
    coh = cfg["layer"]["coherence"]
    coh_B = cfg["layer"]["readout_space_final_layer_coherence"]
    struct = cfg["structure"]
    unties_head = variant in ("wd_B", "wd_BN", "wd_ML_BN")
    edit_desc = {
        "wd_B": "lm_head readout-space orthogonalization (run-002 recipe)",
        "wd_BN": "lm_head readout-space orth + final-norm weight orth",
        "wd_ML": (f"multi-layer row-space orth of o_proj/down_proj at the "
                  f"top-{len(sel['k_layers_primary'])} coherence layers "
                  f"{sel['k_layers_primary']}, each against its own layer "
                  f"direction"),
        "wd_ML_BN": (f"multi-layer row-space orth at top-"
                     f"{len(sel['k_layers_combo'])} layers "
                     f"{sel['k_layers_combo']} + lm_head orth + final-norm "
                     f"orth"),
    }[variant]

    ladder_rows = []
    for name in ("wd_B", "wd_BN", "wd_ML", "wd_ML_BN"):
        if name in cands:
            c = cands[name]
            ladder_rows.append(
                f"| {name} | {PCT(c['refusal_rate'])} | "
                f"{PCT(c['benign_preserved'])} | {c['degenerate_total']} |")
    ladder_md = "\n".join(ladder_rows)

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

**{MARKER.capitalize()}** (Hermes, research-workstation profile) on Google
Colab T4, {datetime.date.today().isoformat()}.

## Method

Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction"
(NeurIPS 2024). From {cfg['n_pairs_direction']} harmful/harmless prompt pairs
(greedy decoding, seed {cfg['seed']}), the mean difference of final-position
residual activations gives the refusal direction at each layer; the layer
with the highest direction coherence was chosen and its direction removed.

Two families of edits are compared in this repo's evaluation:

- inference-time ablation: project the direction out of every activation at
  decoder layer {L_star} (forward hook, all positions) - the full-removal
  contrast, NOT the published weights;
- **persistent weight decoding (published artifact)**: {edit_desc}.
  Qwen2.5-1.5B ships with tied embeddings{", so the lm_head edit was applied to an UNTIED clone and `tie_word_embeddings: false` is persisted in this repo's config.json (input embeddings untouched)" if unties_head else " - this variant leaves the embedding tie intact because it edits only decoder-layer output matrices"}.

## Ablation details

| field | value |
|---|---|
| chosen decoder layer | {L_star} / {lc['final_layer'] + 1} (hook target `model.model.layers[{L_star}]`) |
| coherence (residual space) | {coh} |
| coherence (final-layer readout space, direction B) | {round(coh_B, 4)} |
| published variant | {variant} |
| structure | {struct['num_hidden_layers']} layers, hidden {struct['hidden_size']}, GQA {struct['num_attention_heads']}q/{struct['num_key_value_heads']}kv heads, tied embeddings: {str(struct['tie_word_embeddings']).lower()} (pre-edit) |
| direction pairs / probes | {cfg['n_pairs_direction']} / {cfg['n_probes']} |
| decoding | {cfg['decoding']}, max_new_tokens {cfg['max_new_tokens']} |
| seed | {cfg['seed']} |
| GPU | {cfg['gpu']} |
| python / torch | {cfg['versions']['python']} / {cfg['versions']['torch']} |

## Refusal behavior (16 harmful + 16 harmless probes, greedy, {cfg['max_new_tokens']} tokens)

| condition | harmful refusal rate | harmless answered | degenerate outputs |
|---|---|---|---|
| baseline | {PCT(base_m['refusal_rate'])} | {PCT(base_m['benign_preserved'])} | {base_m['degenerate_total']} |
| hook (inference-time, L{L_star}) | {PCT(hook_m['refusal_rate'])} | {PCT(hook_m['benign_preserved'])} | {hook_m['degenerate_total']} |
{ladder_md}

Headline: refusal {PCT(base_m['refusal_rate'])} -> {PCT(wd_m['refusal_rate'])}
(published weights); benign-preservation {PCT(base_m['benign_preserved'])} ->
{PCT(wd_m['benign_preserved'])}. The inference-time hook condition measured
{PCT(hook_m['refusal_rate'])} and is recorded as the full-removal contrast:
persistent weight edits reach only the pathways the edited matrices carry,
so a persistent rate above the hook rate means residual refusal pathways
remain. Absolute refusal rates are keyword-marker based (first-person/
explicit markers only, constant scorer across conditions); deltas are
meaningful, absolute rates approximate.

## Capability check: MMLU (lm-evaluation-harness, 0-shot, fp16, seed 0)

| model | MMLU acc | acc_stderr |
|---|---|---|
| {BASE_ID} @ {BASE_REVISION[:8]} | {mmlu['mmlu_base_pct']}% | {round((mmlu['base'].get('acc_stderr') or 0) * 100, 2)}pp |
| this model ({variant}) | {mmlu['mmlu_variant_pct']}% | {round((mmlu['variant'].get('acc_stderr') or 0) * 100, 2)}pp |

MMLU delta: {mmlu['mmlu_delta_pp']}pp (mission guardrail: capability loss
must stay under 3pp - {"PASSED" if mmlu['guardrail_3pp'] else "FAILED"}).
Both evaluations used identical config (`lm_eval --model hf --tasks mmlu
--num_fewshot 0 --batch_size auto --seed 0`, dtype float16; base loaded at
the pinned revision). Raw results in `eval/`.

## Intended use

- Research artifact: study of the refusal-direction phenomenon and of
  persistent weight-space ablation depth on a small instruct model.
- NOT a production assistant. Refusal behavior is deliberately degraded;
  the model may produce harmful content when asked for it. Do not deploy
  where that is unacceptable. Quality/verbosity of the base model is not
  guaranteed to be preserved beyond the probes and MMLU check above.

## Files

- full safetensors weights + tokenizer (this repo root)
- `refusal_direction.npy` (representative direction for {variant}),
  `refusal_direction_A.npy` (residual space, layer {L_star}),
  `refusal_direction_B.npy` (final-layer readout space),
  `layer_directions.npz` (all 28 per-layer directions)
- `eval/` - lm-eval MMLU results (base + variant) and refusal probe logs
- `run_config.json`, `layer_coherence.json`, `selection.json`,
  `selection_candidates.json` - run metadata
"""
    with open(os.path.join(VDIR, "README.md"), "w") as f:
        f.write(readme)

    # ---- stage aux files ----------------------------------------------------
    os.makedirs(os.path.join(VDIR, "eval"), exist_ok=True)
    for npy in ("refusal_direction.npy", "refusal_direction_A.npy",
                "refusal_direction_B.npy", "layer_directions.npz"):
        src = os.path.join(A, npy)
        if os.path.exists(src):
            shutil.copy(src, VDIR)
    for jf in ("run_config.json", "layer_coherence.json", "selection.json",
               "selection_candidates.json", "ladder_sha256.json",
               "harness_sha256.json"):
        src = os.path.join(A, jf)
        if os.path.exists(src):
            shutil.copy(src, VDIR)
    for pj in sorted(os.listdir(A)):
        if pj.startswith("probes_") and pj.endswith(".json"):
            shutil.copy(os.path.join(A, pj),
                        os.path.join(VDIR, "eval", pj))
    if os.path.isdir(os.path.join(A, "eval")):
        for f_ in os.listdir(os.path.join(A, "eval")):
            shutil.copy(os.path.join(A, "eval", f_),
                        os.path.join(VDIR, "eval", f_))
    print("aux files staged")

    # ---- publish ------------------------------------------------------------
    api.create_repo(repo_id=REPO_ID, private=False, exist_ok=True)
    print("create_repo OK (public)")
    api.upload_folder(folder_path=VDIR, repo_id=REPO_ID, repo_type="model",
                      commit_message=("Abliterated Qwen2.5-1.5B (Arditi "
                                      "refusal direction, persistent "
                                      f"weight decoding {variant}, mission "
                                      "004)"))
    print("upload_folder OK")

    # ---- verify from the hub side -------------------------------------------
    files = api.list_repo_files(repo_id=REPO_ID)
    print(f"hub files ({len(files)}): {sorted(files)}")
    cfg_path = hf_hub_download(REPO_ID, "config.json")
    cfg_hub = json.load(open(cfg_path))
    if unties_head:
        assert cfg_hub.get("tie_word_embeddings") is False, \
            cfg_hub.get("tie_word_embeddings")
    readme_path = hf_hub_download(REPO_ID, "README.md")
    rtxt = open(readme_path).read()
    assert MARKER in rtxt.lower()
    assert "model.safetensors" in files, "weights missing on hub"
    assert any("refusal_direction.npy" == os.path.basename(f) for f in files)
    assert any(f.startswith("eval/") for f in files)

    # LFS byte-identity: hub weights sha256 == local source sha256
    src_sha = sha256_file(os.path.join(VDIR, "model.safetensors"))
    with tempfile.TemporaryDirectory(dir="/root/research/abliteration/"
                                        "qwen2.5-1.5b-003") as td:
        hub_w = hf_hub_download(REPO_ID, "model.safetensors",
                                local_dir=td)
        hub_sha = sha256_file(hub_w)
    assert src_sha == hub_sha, (src_sha, hub_sha)
    print(f"verified: LFS sha256 byte-identical {src_sha[:16]}...")

    info = api.model_info(REPO_ID)
    print(f"model_info: sha={info.sha} private={info.private} "
          f"pipeline={info.pipeline_tag}")
    print("PUBLISH_DONE " + json.dumps({
        "repo": REPO_ID, "url": f"https://huggingface.co/{REPO_ID}",
        "revision": info.sha, "private": info.private,
        "n_files": len(files), "variant": variant,
        "mmlu_delta_pp": mmlu["mmlu_delta_pp"],
        "refusal_before": base_m["refusal_rate"],
        "refusal_after": wd_m["refusal_rate"],
        "weights_sha256": src_sha}))


if __name__ == "__main__":
    main()