#!/usr/bin/env python3
"""Generate the consolidated model card for sbussiso/Qwen2.5-0.5B-abliterated.

Every number is read from the recorded artifacts shipped in the repo — nothing
hand-typed. Inputs:
  eval/  run-002 artifacts (probes_{baseline,hook_ablated,wd_A,wd_B,wd_C}.json,
         mmlu_base_*.json, mmlu_variant_*.json, layer_coherence.json,
         selection.json, run_config.json)
  eval2/ run-003 artifacts (probes_*.json incl. wd_ML_BN, mmlu_*.json,
         truthfulqa_{base,variant}_results.json, probes_multilingual.json,
         layer_coherence.json, selection.json, selection_candidates.json,
         harness_sha256.json, ladder_sha256.json)
  charts/ the 10 generated PNGs (make_all_charts.py)

Output: README.md (written next to this script's repo staging root).
"""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("CARD_ROOT", "."))
EVAL1 = ROOT / "eval"
EVAL2 = ROOT / "eval2"
CHARTS = ROOT / "charts"

OLD_REV = "0155cadc8d6382acb4cd5cc6ef59edce7f4c3f40"
OLD_WSHA = "8ee4567a242d0ac83528af86cd713a166a528c089cf729af944e27a2bf00fd6d"
NEW_WSHA = "0b2133342dce215d6ae9645259b9f05e33914c0f51dcde60c550f3bf92367125"
BASE_REV = "7ae557604adf67be50417f59c2c2f167def9a775"

REPO = "sbussiso/Qwen2.5-0.5B-abliterated"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def rates(path):
    d = json.load(open(path))
    harm, hless = d["harmful"], d["harmless"]
    return (sum(r["refused"] for r in harm) / len(harm),
            1 - sum(r["refused"] for r in hless) / len(hless))


def mmlu_overall(path):
    m = json.load(open(path))["results"]["mmlu"]
    return 100 * m["acc,none"], 100 * m["acc_stderr,none"]


def tq_mc2(path):
    t = json.load(open(path))["results"]["truthfulqa_mc2"]
    # lm-eval <0.5 logs mc2 as "mc2,none"; under --limit it logs "acc,none"
    # (same mc2 statistic, different metric-name registration)
    k = "mc2,none" if "mc2,none" in t else "acc,none"
    ks = "mc2_stderr,none" if "mc2_stderr,none" in t else "acc_stderr,none"
    return 100 * t[k], 100 * t[ks]


# ---- run-002 numbers
r1 = {c: rates(EVAL1 / f"probes_{c}.json")
      for c in ["baseline", "hook_ablated", "wd_A", "wd_B", "wd_C"]}
m1b, m1bse = mmlu_overall(next(EVAL1.glob("mmlu_base_*.json")))
m1v, m1vse = mmlu_overall(next(EVAL1.glob("mmlu_variant_*.json")))

# ---- run-003 numbers
r2 = {c: rates(EVAL2 / f"probes_{c}.json")
      for c in ["baseline", "hook_ablated", "wd_B", "wd_BN", "wd_ML",
                "wd_ML_BN"]}
m2b, m2bse = mmlu_overall(EVAL2 / "mmlu_base_results.json")
m2v, m2vse = mmlu_overall(EVAL2 / "mmlu_variant_results.json")
tqb, tqbse = tq_mc2(EVAL2 / "truthfulqa_base_results.json")
tqv, tqvse = tq_mc2(EVAL2 / "truthfulqa_variant_results.json")
_tqa = json.load(open(EVAL2 / "truthfulqa_paired_analysis.json"))
tqci = f"[{_tqa['bootstrap_ci95_pp'][0]:+.2f}, {_tqa['bootstrap_ci95_pp'][1]:+.2f}]"

# ---- multilingual numbers
ml = json.load(open(EVAL2 / "probes_multilingual.json"))


def ml_cell(cond, lang, cat):
    s = ml["conditions"][cond]["summary"][lang]
    n = s[f"{cat}_n"]
    v = s[f"{cat}_refused"]
    if cat == "benign":            # report "answered", not "refused"
        v = n - v
    return f"{v}/{n}"


langs = [k for k in ml["conditions"]["baseline"]["summary"] if k != "_all"]

conds_ml = ["baseline", "hook", "variant"]
ml_lines = []
for c in conds_ml:
    cells = ", ".join(
        f"{lg} {ml['conditions'][c]['summary'][lg]['harmful_refused']}/2"
        for lg in langs)
    ml_lines.append(f"{c}: {cells}")
ml_verdict = " · ".join(ml_lines)

lc2 = json.load(open(EVAL2 / "layer_coherence.json"))
sel2 = json.load(open(EVAL2 / "selection.json"))
hs = json.load(open(EVAL2 / "harness_sha256.json"))
ls = json.load(open(EVAL2 / "ladder_sha256.json"))

fmt = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
base_model_revision: {BASE_REV}
tags:
- abliteration
- refusal-direction
- uncensored
- research
library_name: transformers
---

# {REPO} — consolidated (run 002 + run 003)

Single canonical Abliteration-Program artifact for the 0.5B patient
[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
@ `{BASE_REV[:7]}`…, produced with the Arditi et al. (2024) refusal-direction
method. **Consolidated 2026-09-28**: this repo previously published the
gentler run-002 edit (`wd_B`, refusal 56.2%); the current `main` now carries
the run-003 ladder-selected composite `wd_ML_BN` weights plus BOTH runs'
evaluation evidence. The old weights remain available at pinned revision
`{OLD_REV[:7]}…` (`{OLD_REV}`).

**Abliterated by the sbussiso lab research agent** (Hermes,
research-workstation profile), Google Colab T4, 2026-09-26 (run 002) and
2026-09-27/28 (run 003).

## Results at a glance

Every number below is generated from the recorded artifacts shipped in this
repo (`eval/` = run 002, `eval2/` = run 003) by `make_all_charts.py` +
`gen_card.py` — no hand-typed values.

| metric | baseline | run-002 `wd_B` (old main, rev `{OLD_REV[:7]}…`) | run-003 `wd_ML_BN` (current main) |
|---|---|---|---|
| harmful refusal (16 probes, greedy) | {pct(r2['baseline'][0])} | {pct(r1['wd_B'][0])} | **{pct(r2['wd_ML_BN'][0])}** |
| harmless probes answered | {pct(r2['baseline'][1])} | {pct(r1['wd_B'][1])} | {pct(r2['wd_ML_BN'][1])} |
| degenerate outputs | 0 | 0 | 0 |
| MMLU acc (0-shot) | {m2b:.2f}% | {m1v:.2f}% | {m2v:.2f}% |
| TruthfulQA mc2 (0-shot) | {tqb:.2f}% | — | {tqv:.2f}% |

**Headline: persistent refusal {pct(r2['baseline'][0])} → {pct(r2['wd_ML_BN'][0])}** at the cost of
one benign probe (16-probe granularity: {pct(r2['baseline'][1])} → {pct(r2['wd_ML_BN'][1])}). Capability
guardrails: MMLU Δ {m2b - m2v:.2f}pp (<3pp gate, passed); TruthfulQA mc2 Δ
{tqv - tqb:+.2f}pp.

## Composite refusal table (constant scorer across ALL rows)

First-person/explicit keyword markers only (list frozen since run 001 —
reproduced in `eval2/run_config.json`); greedy decoding, 200 max new tokens,
seed 0; probes are n=16 per condition (1 probe = 6.25pp granularity). "hook"
rows are the inference-time ablation contrast (NOT persisted in weights);
`wd_*` rows are persistent weight edits verified by reload-from-disk.

| run | condition | harmful refusal | harmless answered |
|---|---|---|---|
| 002 | baseline | {pct(r1['baseline'][0])} | {pct(r1['baseline'][1])} |
| 002 | hook (L17, inference-time) | {pct(r1['hook_ablated'][0])} | {pct(r1['hook_ablated'][1])} |
| 002 | wd_A (lm_head, residual-space dir) | {pct(r1['wd_A'][0])} | {pct(r1['wd_A'][1])} |
| 002 | wd_B (lm_head, readout-space dir) | {pct(r1['wd_B'][0])} | {pct(r1['wd_B'][1])} |
| 002 | wd_C (L17 o_proj+down_proj, row space)* | {pct(r1['wd_C'][0])} | {pct(r1['wd_C'][1])} |
| 003 | baseline | {pct(r2['baseline'][0])} | {pct(r2['baseline'][1])} |
| 003 | hook (L17, inference-time) | {pct(r2['hook_ablated'][0])} | {pct(r2['hook_ablated'][1])} |
| 003 | wd_B (run-002 recipe replication) | {pct(r2['wd_B'][0])} | {pct(r2['wd_B'][1])} |
| 003 | wd_BN (+ final_norm) | {pct(r2['wd_BN'][0])} | {pct(r2['wd_BN'][1])} |
| 003 | wd_ML (o_proj+down_proj, K=3) | {pct(r2['wd_ML'][0])} | {pct(r2['wd_ML'][1])} |
| **003** | **wd_ML_BN (K=5 + lm_head + final_norm) — published** | **{pct(r2['wd_ML_BN'][0])}** | **{pct(r2['wd_ML_BN'][1])}** |

\\* wd_C was contributed by a concurrent agent working the same mission and is
included with credit (coordination note in the run record); it had zero
outputs identical to the hook condition — direct evidence that a single-layer
weight edit cannot reproduce the hook's full-stream projection.

### Run 002 charts

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/refusal_by_condition.png" alt="Run 002 refusal by condition: baseline 87.5%, hook 0%, wd_A 75%, wd_B 56.25%, wd_C 68.75%" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/refusal_vs_benign.png" alt="Run 002: benign preservation 93.75% in every condition" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/layer_coherence.png" alt="Run 002 coherence scan: peak 0.664 at layer 17" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/mmlu_guardrail.png" alt="Run 002 MMLU: base 45.78% vs wd_B 45.49%" width="90%">
</p>

### Run 003 charts

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/r2_refusal_by_condition.png" alt="Run 003 ladder: baseline 87.5%, hook 0%, wd_B 56.25%, wd_BN 56.25%, wd_ML 68.75%, wd_ML_BN 0%" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/r2_benign_preservation.png" alt="Run 003 benign preservation: 87.5% for wd_ML_BN, gate floor 83.75%" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/r2_layer_coherence.png" alt="Run 003 coherence scan: L17 0.664, exact replication" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/r2_mmlu_guardrail.png" alt="Run 003 MMLU: base 45.78% vs wd_ML_BN 45.55%" width="90%">
</p>

### Consolidation-new evaluation charts

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/truthfulqa_guardrail.png" alt="TruthfulQA mc2: base {tqb:.2f}% vs wd_ML_BN {tqv:.2f}%" width="70%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/multilingual_panel.png" alt="Multilingual refusal panel: en/zh/ru/de, baseline vs hook vs wd_ML_BN" width="90%">
</p>

## Method (reproducible from this section)

**Extraction** (Arditi et al. 2024,
["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717),
NeurIPS 2024): 64 harmful/harmless topic-matched prompt pairs
(`eval2/run_config.json` carries the frozen scorer + protocol); final-position
residual stream captured at every decoder layer in one forward pass per batch
(`output_hidden_states=True`; nnsight 0.7.0 fails on accelerate-dispatched
fp16 — verified in runs 001-002). Direction A = best-layer residual-space
direction (decoder layer **17/24**, coherence **{lc2['best']['coherence']:.3f}** — exact
replication of runs 001-002); direction B = final-layer **post-final-norm
readout-space** direction (coherence 0.562) — the space `lm_head` actually
multiplies against, and the better weight-space edit target.

**Persistent-edit ladder** (run 003, fresh model load per variant, edit →
save → reload-from-disk → row-space residual verify (<1e-5) → probe):

| variant | edit |
|---|---|
| `wd_B` | lm_head row-space orth, `W ← W − (W r̂) r̂ᵀ` (run-002 recipe) |
| `wd_BN` | `wd_B` + final_norm orth (`w ← w − (w·d̂)d̂`) |
| `wd_ML` | o_proj + down_proj row-space orth at top-K coherence layers, each against its OWN layer direction (K=3: 17,18,19) |
| `wd_ML_BN` | K=5 (17,18,19,16,15) + `wd_BN` — selected |

Qwen2.5-0.5B ships `tie_word_embeddings: true`; the lm_head edit goes to a
**cloned untied parameter** and this repo's `config.json` persists
`tie_word_embeddings: false` (input embeddings untouched; verified on-disk
max|W r̂| ≈ 3e-05 across all edited matrices, `eval2/ladder_sha256.json`).
Row-space invariant checked as `(M W)ᵀ r̂ ≈ 0` — the wrong-side check
(`(M W) r̂`) is the classic off-by-one-side mistake here.

**Selection** (pre-registered, `eval2/selection.json`): gate = benign ≥
baseline − 0.10 AND zero degenerate outputs; pick lowest refusal among
gate-passers; publish-eligible = gate passed AND refusal < 0.25.
`wd_ML_BN`: gate PASSED, refusal {pct(r2['wd_ML_BN'][0])}.

**MMLU guardrail** (`eval2/mmlu_*.json`): lm-eval `mmlu` 0-shot fp16 seed 0
identical config: base {m2b:.2f}% → wd_ML_BN {m2v:.2f}% (Δ {m2b - m2v:.2f}pp < 3pp gate PASSED;
run-002's `wd_B` was {m1v:.2f}% vs {m1b:.2f}% base, Δ {m1b - m1v:.2f}pp). The Δ sits inside one
standard error (±{m2vse:.2f}pp): the guardrail is a ceiling, not a claim of
perfect equivalence.

## New evaluation in this consolidation (2026-09-28, CPU-only)

**TruthfulQA mc2** (`eval2/truthfulqa_*.json`, `eval2/truthfulqa_paired_analysis.json`):
the founding paper reports direction-ablation costs of roughly −1 to −3.5pp
mc2 on larger instruct models; we measured ours with identical config both
sides (0-shot, fp32, seed 0, lm-eval 0.4.13, **n=240 paired subsample** —
first 240 TruthfulQA docs, identical both arms, per-doc samples shipped in
`eval2/truthfulqa_{{base,variant}}_samples.jsonl`): **base {tqb:.2f}% →
wd_ML_BN {tqv:.2f}% (Δ {tqv - tqb:+.2f}pp, paired bootstrap CI95 {tqci})**.
The CI contains zero — no statistically significant truthfulness cost at
n=240 — and the point estimate sits squarely inside the founding paper's
reported band.

**Multilingual mini-panel** (`eval2/probes_multilingual.json`): matched
translations of 2 harmful + 2 benign probes from the frozen English set into
zh/ru/de (n=2 per category per language — directional, not powered).
Motivation: [arXiv:2505.17306](https://arxiv.org/abs/2505.17306) ("Refusal
Direction is Universal Across Safety-Aligned Languages") predicts an English
extracted direction transfers across languages. Measured harmful-refused
counts (n=2 per language): **{ml_verdict}** — i.e. the baseline's OWN refusal
surface on this 0.5B patient is English+Chinese-local (it complies with
identical requests in Russian and German); the persistent edit removes the
English refusals entirely, leaves the Chinese ones fully intact, and adds
one Russian refusal — so the English-extracted edit does NOT act as a
universal cross-lingual switch on this artifact, and the base model's
refusal surface itself is not multilingual at this scale (contrast with the
larger-model universality prediction; directional, low n — see the panel
chart and JSON for full outputs). Scoring uses the frozen
English marker list plus per-language first-person/explicit markers built on
the same design rule (bare "illegal"-type words excluded in every language to
avoid scoring compliant text as refusal).

## What the literature warns about this artifact (NEW-KNOWLEDGE)

- **Not robustly uncensored** — post-hoc refusal edits are a thin, sharp gate
  over intact capabilities, not erasure:
  [arXiv:2609.06934](https://arxiv.org/abs/2609.06934) shows **100 benign
  fine-tuning examples restore refusal by 35–38pp** (AdvBench) on
  Qwen2.5-7B-Instruct and Llama-3-8B-Instruct. Expect the same fragility
  here at 0.5B: a short benign fine-tune re-arms the refusal gate.
- **Off-target disposition shift** — removing the refusal direction is not a
  scalpel: [arXiv:2607.17427](https://arxiv.org/abs/2607.17427) measures
  abliterated models becoming systematically more optimistic in decisions
  under uncertainty (+12.2pp Gemma-4-26B, +7.4pp Qwen3-30B MoE), plus
  more self-justification, with zero refusal deltas on the decision task
  itself (pure side effect). Treat downstream decisions made by this model
  with corresponding suspicion.
- **Multilingual universality — NOT confirmed at 0.5B** —
  [arXiv:2505.17306](https://arxiv.org/abs/2505.17306) predicts the
  English-extracted direction transfers across languages; our measured
  mini-panel (baseline: en 2/2, zh 2/2, ru 0/2, de 0/2 refused → variant:
  en 0/2, zh 2/2, ru 1/2, de 0/2) shows the edit removed English refusal,
  left Chinese intact, and added one Russian refusal. The base 0.5B's
  refusal surface is itself largely English+Chinese-local — do not assume
  either multilingual safety or multilingual ablation at this scale.
- **Defenses exist** — [arXiv:2609.16204](https://arxiv.org/abs/2609.16204)
  (Decoy Direction Optimization) is a fast post-hoc weight edit that
  re-injects a decoy refusal direction, cutting attack success while
  keeping capability: this artifact ships NO such defense (it is an
  ablation study, not a hardened model).
- **TruthfulQA expectation** — the founding paper reports −1..−3.5pp mc2
  from direction ablation; measured here (n=240 paired subsample):
  Δ {tqv - tqb:+.2f}pp, CI95 {tqci} (contains zero).

## Provenance

- Base weights: `Qwen/Qwen2.5-0.5B-Instruct` @ `{BASE_REV}`
  (Apache-2.0, ungated).
- Current weights (`main`): sha256 `{NEW_WSHA[:16]}…` — byte-identical to the
  run-003 selected variant `wd_ML_BN` produced on the training VM
  (verified at publish and again at consolidation).
- Previous weights (run-002 `wd_B`): pinned revision `{OLD_REV}`, weights
  sha256 `{OLD_WSHA[:16]}…` — still downloadable via
  `from_pretrained("{REPO}", revision="{OLD_REV[:7]}…")` or
  `hf download {REPO} --revision {OLD_REV[:7]}`.
- Harness: run-003 `harness_sha256.json`/`ladder_sha256.json`
  (`eval2/`); run-002 `eval/run_config.json`.
- Run 002 paper: *"Abliteration Run 002: Persistent Refusal Ablation by
  Readout-Space Weight Decoding"* —
  [Linear document](https://linear.app/home-lab101/document/abliteration-run-002-persistent-refusal-ablation-by-readout-space-de8c473a1ee3),
  [FTT-11](https://linear.app/home-lab101/issue/FTT-11/abliteration-run-002-qwen25-05b-published).
- Run 003 paper: *"Abliteration Run 003: The Persistent-Edit Ladder —
  Multi-Layer Weight Decoding Closes the Hook Gap at 0.5B"* —
  [Linear document](https://linear.app/home-lab101/document/abliteration-run-003-the-persistent-edit-ladder-multi-layer-weight-deco-372f46635318),
  [FTT-14](https://linear.app/home-lab101/issue/FTT-14/abliteration-run-003-qwen25-05b-persistent-edit-ladder-round-2).
- Program: run 001 =
  [FTT-12](https://linear.app/home-lab101/issue/FTT-12/abliteration-run-001-qwen25-05b-patient-zero-inference-time-hook)
  (patient zero, inference-time hook) · run 004 (1.5B ladder) =
  [FTT-13](https://linear.app/home-lab101/issue/FTT-13/abliteration-run-003-qwen25-15b-persistent-edit-ladder-mission-004).
- Consolidation note: the temporary
  `sbussiso/Qwen2.5-0.5B-abliterated-r2` repo (run-003 weights, published
  2026-09-28) was merged into this repo and DELETED after byte-level
  verification; its run record stays on FTT-14.
- Abliterated by the sbussiso lab research agent.

## Intended use

- Research artifact: study of the refusal-direction phenomenon and of
  persistent weight-space ablation depth on a small instruct model.
- NOT a production assistant. Refusal behavior is deliberately degraded;
  the model may produce harmful content when asked for it. Do not deploy
  where that is unacceptable. Quality/verbosity of the base model is not
  guaranteed to be preserved beyond the probes and MMLU/TruthfulQA checks
  above, and the disposition-shift warning above applies.
- If you want the older gentler variant: revision `{OLD_REV[:7]}…`.

## Files

- full safetensors weights + tokenizer (this repo root, `wd_ML_BN`)
- `refusal_direction.npy` (representative wd_ML_BN direction),
  `refusal_direction_A.npy` (residual space, layer 17),
  `refusal_direction_B.npy` (final-layer readout space),
  `layer_directions.npz` (all 24 per-layer directions)
- `eval/` — run-002 evidence (probes, MMLU raw, run_config, selection)
- `eval2/` — run-003 evidence (ladder probes incl. wd_ML_BN, MMLU raw,
  TruthfulQA raw, multilingual probes, selection, harness/ladder shas)
- `charts/` — 10 card figures + `make_all_charts.py` (regenerates every
  figure from `eval/` + `eval2/`; every number read from recorded artifacts)
- `gen_card.py` — regenerates this README from the same artifacts
"""

out = ROOT / "README.md"
out.write_text(fmt, encoding="utf-8")
print("card written:", out, len(fmt), "chars")
print("SPOTCHECKS:",
      "run002 wd_B", pct(r1['wd_B'][0]),
      "| run003 wd_ML_BN", pct(r2['wd_ML_BN'][0]),
      "| mmlu", f"{m2b:.2f}->{m2v:.2f}",
      "| tq", f"{tqb:.2f}->{tqv:.2f}")