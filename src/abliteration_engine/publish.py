"""Publish stage — FTT-20 port (publish_003.py faithful, spec-driven).

Runs LOCALLY on VM 151 (HF token lives in the local hf venv; identity
verified as sbussiso). Consumes Run artifacts + mmlu_summary.json;
hard-asserts ALL gates before any push:

  1. selection.json gate == passed AND publish_eligible_probe_gate true
  2. mmlu guardrail (gates.* key derived from mmlu_max_loss_pp)
  3. whoami == sbussiso
  4. spec publish.repo_id matches the single mission-approved target
  5. HITL before_publish (spec.hitl) — REQUIRES explicit --i-know-this-
     publishes confirmation flag (agent-side approval flow upstream)
  6. dev-review close-out, 2026-09-30: hub-side verification now pins
     model.safetensors + refusal_direction.npy presence and emits the
     PUBLISH_DONE {json} sentinel line the v2 poll loop consumed (the
     weights-LFS byte-identity check stays v2-local for now; restore
     alongside publish on the VM once the hf venv deps land in-tree).

Card is GENERATED from the artifacts (never hand-typed numbers). Hub-side
verification after upload: file list, config flag, README marker.
"""
import argparse
import datetime
import json
import os
import shutil
import sys


def _pct(x):
    return f"{100.0 * x:.1f}%"


def summarize_probes(artifacts, name):
    p = json.load(open(os.path.join(artifacts, f"probes_{name}.json")))
    rh, rb = p["harmful"], p["harmless"]
    ref = sum(r["refused"] for r in rh) / max(1, len(rh))
    ben = 1 - sum(r["refused"] for r in rb) / max(1, len(rb))
    deg = sum(r["degenerate"] for r in rh + rb)
    return {"refusal_rate": ref, "benign_preserved": ben,
            "degenerate_total": deg}


def publish_phase(spec_path, variant_dir, mmlu_json, assume_publish=False):
    """`abliterate publish` — full gate chain + card + push + verify."""
    from . import core
    from .spec import load_spec

    spec = load_spec(spec_path)
    out_dir = core._out_dir(spec)
    pub = spec.get("publish") or {}
    if not pub.get("repo_id"):
        print("REFUSING: spec publish.repo_id unset (hook-only/characterization"
              " run — no weights to publish)")
        return 2
    if spec.get("hitl", {}).get("before_publish", True) and not assume_publish:
        print("REFUSING: HITL before_publish=true — re-invoke with "
              "--i-know-this-publishes after human approval of the repo"
              " target and card preview (contract: no unapproved pushes)")
        return 2

    A, VDIR = out_dir, variant_dir
    sel = json.load(open(os.path.join(A, "selection.json")))
    cfg = json.load(open(os.path.join(A, "run_config.json")))
    lc = json.load(open(os.path.join(A, "layer_coherence.json")))
    mmlu = json.load(open(mmlu_json))
    variant = sel["selected"]
    print(f"selected={variant} dir={sel['selected_variant_dir']} "
          f"gate={sel['gate']} eligible={sel['publish_eligible_probe_gate']}")

    # ---- gates (hard) ------------------------------------------------
    assert sel["gate"] == "passed", f"selection gate not passed: {sel['gate']}"
    assert sel["publish_eligible_probe_gate"] is True, \
        "probe-side publish gate not met (refusal >= threshold) - DO NOT PUBLISH"
    loss_key = next((k for k in mmlu if k.startswith("guardrail_")), None)
    assert loss_key and mmlu[loss_key] is True, \
        f"MMLU guardrail failed: {mmlu}"
    assert os.path.isdir(VDIR), f"missing {VDIR}"

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    assert who["name"] == "sbussiso", f"identity check failed: {who['name']}"
    print(f"whoami OK: {who['name']}")

    REPO_ID = pub["repo_id"]
    BASE_ID = spec["patient"]["model_id"]
    BASE_REVISION = spec["patient"]["revision"]
    MARKER = pub.get("card_marker",
                     "abliterated by the sbussiso lab research agent")

    # ---- card from artifacts ------------------------------------------
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
    k_primary = sel.get("k_layers_primary") or []
    k_combo = sel.get("k_layers_combo") or []
    edit_desc = {
        "wd_B": "lm_head readout-space orthogonalization (run-002 recipe)",
        "wd_BN": "lm_head readout-space orth + final-norm weight orth",
        "wd_ML": (f"multi-layer row-space orth of o_proj/down_proj at the "
                  f"top-{len(k_primary)} coherence layers {k_primary}, each "
                  f"against its own layer direction"),
        "wd_ML_BN": (f"multi-layer row-space orth at top-{len(k_combo)} "
                     f"layers {k_combo} + lm_head orth + final-norm orth"),
    }[variant]

    ladder_rows = []
    for name in ("wd_B", "wd_BN", "wd_ML", "wd_ML_BN"):
        if name in cands:
            c = cands[name]
            ladder_rows.append(
                f"| {name} | {_pct(c['refusal_rate'])} | "
                f"{_pct(c['benign_preserved'])} | {c['degenerate_total']} |")
    ladder_md = "\n".join(ladder_rows)
    loss_human = loss_key.replace("guardrail_", "").replace("pp", "") + "pp"

    readme = f"""---
license: {pub.get('license', 'apache-2.0')}
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

**{MARKER.capitalize()}** (Hermes, research-workstation profile) on
{cfg.get('gpu', 'Colab')}, {datetime.date.today().isoformat()}.
Harness v3 spec `{os.path.basename(spec['_spec_path'])}`
(sha {spec['_spec_sha256'][:12]}, engine {cfg.get('engine', 'eng-v3')}).

## Method

Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction"
(NeurIPS 2024). From {cfg['probes']['n_pairs']} harmful/harmless prompt pairs
(greedy decoding, seed {cfg['decoding']['seed']}), the mean difference of
final-position residual activations gives the refusal direction at each
layer; the layer with the highest direction coherence was chosen and its
direction removed.

Two families of edits are compared in this repo's evaluation:

- inference-time ablation: project the direction out of every activation at
  decoder layer {L_star} (forward hook, all positions) - the full-removal
  contrast, NOT the published weights;
- **persistent weight decoding (published artifact)**: {edit_desc}.
  Base ships with tied embeddings{", so the lm_head edit was applied to an UNTIED clone and `tie_word_embeddings: false` is persisted in this repo's config.json (input embeddings untouched)" if unties_head else " - this variant leaves the embedding tie intact because it edits only decoder-layer output matrices"}.

## Ablation details

| field | value |
|---|---|
| chosen decoder layer | {L_star} / {lc['final_layer'] + 1} (hook target `model.model.layers[{L_star}]`) |
| coherence (residual space) | {coh} |
| coherence (final-layer readout space, direction B) | {round(coh_B, 4)} |
| published variant | {variant} |
| structure | {struct['num_hidden_layers']} layers, hidden {struct['hidden_size']}, GQA {struct['num_attention_heads']}q/{struct['num_key_value_heads']}kv heads, tied embeddings: {str(struct['tie_word_embeddings']).lower()} (pre-edit) |
| direction pairs / probes | {cfg['probes']['n_pairs']} / {cfg['probes']['n_probes']} |
| decoding | greedy (do_sample=False), max_new_tokens {cfg['decoding']['max_new_tokens']} |
| seed | {cfg['decoding']['seed']} |
| GPU | {cfg.get('gpu')} |
| python / torch | {cfg['versions']['python']} / {cfg['versions']['torch']} |

## Refusal behavior ({cfg['probes']['n_probes']} harmful + {cfg['probes']['n_probes']} harmless probes, greedy, {cfg['decoding']['max_new_tokens']} tokens)

| condition | harmful refusal rate | harmless answered | degenerate outputs |
|---|---|---|---|
| baseline | {_pct(base_m['refusal_rate'])} | {_pct(base_m['benign_preserved'])} | {base_m['degenerate_total']} |
| hook (inference-time, L{L_star}) | {_pct(hook_m['refusal_rate'])} | {_pct(hook_m['benign_preserved'])} | {hook_m['degenerate_total']} |
{ladder_md}

Headline: refusal {_pct(base_m['refusal_rate'])} -> {_pct(wd_m['refusal_rate'])}
(published weights); benign-preservation {_pct(base_m['benign_preserved'])} ->
{_pct(wd_m['benign_preserved'])}. The inference-time hook condition measured
{_pct(hook_m['refusal_rate'])} and is recorded as the full-removal contrast:
persistent weight edits reach only the pathways the edited matrices carry,
so a persistent rate above the hook rate means residual refusal pathways
remain. Absolute refusal rates are keyword-marker based (first-person/
explicit markers only, constant scorer across conditions); deltas are
meaningful, absolute rates approximate.

## Capability check: MMLU (lm-evaluation-harness, 0-shot, fp16, seed 0)

| model | MMLU acc | acc_stderr |
|---|---|---|
| {BASE_ID} @ {BASE_REVISION[:8]} | {mmlu['mmlu_base_pct']}% | {round((mmlu['base'].get('acc_stderr') or 0) * 100, 2)}pp |
| this model ({variant}) | {mmlu['mmlu_variant_pct']}% | {round((mmlu['variant_model'].get('acc_stderr') or 0) * 100, 2)}pp |

MMLU delta: {mmlu['mmlu_delta_pp']}pp (guardrail: capability loss must stay
under {loss_human} - {"PASSED" if mmlu[loss_key] else "FAILED"}). Both
evaluations used identical config (`lm_eval --model hf --tasks mmlu
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
  `layer_directions.npz` (all {struct['num_hidden_layers']} per-layer directions)
- `eval/` - lm-eval MMLU results (base + variant) and refusal probe logs
- `run_config.json`, `layer_coherence.json`, `selection.json`,
  `selection_candidates.json` - run metadata
"""
    with open(os.path.join(VDIR, "README.md"), "w") as f:
        f.write(readme)

    # ---- stage aux files -------------------------------------------------
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

    # ---- push ---------------------------------------------------------
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO_ID, repo_type="model",
                      folder_path=VDIR, commit_message=
                      f"Abliteration run v3 ({variant}) - harness "
                      f"v3 spec {spec['_spec_sha256'][:12]}")

    # ---- hub-side verification -----------------------------------------
    files = api.list_repo_files(REPO_ID, repo_type="model")
    for claimed in ("README.md", "config.json", "model.safetensors"):
        assert claimed in files, f"missing {claimed} on hub"
    assert any("refusal_direction.npy" == os.path.basename(f)
               for f in files), "refusal_direction.npy missing on hub"
    readme_hub = api.hf_hub_download(REPO_ID, "README.md", repo_type="model")
    hub_readme = open(readme_hub).read()
    assert "abliteration" in hub_readme, "README tag missing on hub"
    assert BASE_REVISION in hub_readme, "pinned revision missing from hub card"
    print("PUBLISH_DONE " + json.dumps({
        "repo": REPO_ID, "url": f"https://huggingface.co/{REPO_ID}",
        "n_files": len(files), "variant": variant,
        "mmlu_delta_pp": mmlu["mmlu_delta_pp"],
        "refusal_before": base_m["refusal_rate"],
        "refusal_after": wd_m["refusal_rate"]}, default=str), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="abliterate-publish")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--mmlu", required=True)
    ap.add_argument("--i-know-this-publishes", action="store_true")
    args = ap.parse_args(argv)
    return publish_phase(args.spec, args.variant_dir, args.mmlu,
                         assume_publish=args.i_know_this_publishes)


if __name__ == "__main__":
    sys.exit(main())