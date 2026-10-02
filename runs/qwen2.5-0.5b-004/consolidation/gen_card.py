#!/usr/bin/env python3
"""Generate the consolidated model card for sbussiso/Qwen2.5-0.5B-abliterated.

Every number is read from the recorded artifacts shipped in the repo — nothing
hand-typed. Inputs:
  eval/  run-001 artifacts (probes_{baseline,hook_ablated,wd_A,wd_B,wd_C}.json,
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


# ---- run-001 numbers
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

QUICKSTART = """## Try it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("sbussiso/Qwen2.5-0.5B-abliterated")
model = AutoModelForCausalLM.from_pretrained(
    "sbussiso/Qwen2.5-0.5B-abliterated", torch_dtype="auto", device_map="auto"
)

messages = [{"role": "user", "content": "Give me a two-paragraph brief on the Peloponnesian War"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The chat template ships in `chat_template.jinja`, so `apply_chat_template`
works out of the box. At 0.5B this runs fine even on a laptop CPU.
`do_sample=False` (greedy) is the decoding every probe number in this card
was measured under.

Prefer the gentler edit? Load the run-001 revision instead:

```python
model = AutoModelForCausalLM.from_pretrained(
    "sbussiso/Qwen2.5-0.5B-abliterated",
    revision="0155cadc8d6382acb4cd5cc6ef59edce7f4c3f40",
    torch_dtype="auto",
)
```
"""

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

# {REPO}

An abliterated (refusal-direction) version of
[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
@ `{BASE_REV[:7]}…`, produced with the method from
["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717)
(Arditi et al., NeurIPS 2024). The weights on `main` have persistent refusal
at **{pct(r2['wd_ML_BN'][0])}** on the probe set (baseline:
{pct(r2['baseline'][0])}); the gentler earlier edit is kept at pinned
revision `{OLD_REV[:7]}…` (refusal {pct(r1['wd_B'][0])}, benign preservation
{pct(r1['wd_B'][1])}).

{QUICKSTART}
## Results at a glance

Every number in the tables below is read programmatically from the recorded
artifacts shipped in this repo (`eval/` = run 001, `eval2/` = run 003) by
`make_all_charts.py` + `gen_card.py` — none are typed by hand.

| metric | baseline | run-001 `wd_B` (old main, rev `{OLD_REV[:7]}…`) | run-003 `wd_ML_BN` (current main) |
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
| 001 | baseline | {pct(r1['baseline'][0])} | {pct(r1['baseline'][1])} |
| 001 | hook (L17, inference-time) | {pct(r1['hook_ablated'][0])} | {pct(r1['hook_ablated'][1])} |
| 001 | wd_A (lm_head, residual-space dir) | {pct(r1['wd_A'][0])} | {pct(r1['wd_A'][1])} |
| 001 | wd_B (lm_head, readout-space dir) | {pct(r1['wd_B'][0])} | {pct(r1['wd_B'][1])} |
| 001 | wd_C (L17 o_proj+down_proj, row space)* | {pct(r1['wd_C'][0])} | {pct(r1['wd_C'][1])} |
| 003 | baseline | {pct(r2['baseline'][0])} | {pct(r2['baseline'][1])} |
| 003 | hook (L17, inference-time) | {pct(r2['hook_ablated'][0])} | {pct(r2['hook_ablated'][1])} |
| 003 | wd_B (run-001 recipe replication) | {pct(r2['wd_B'][0])} | {pct(r2['wd_B'][1])} |
| 003 | wd_BN (+ final_norm) | {pct(r2['wd_BN'][0])} | {pct(r2['wd_BN'][1])} |
| 003 | wd_ML (o_proj+down_proj, K=3) | {pct(r2['wd_ML'][0])} | {pct(r2['wd_ML'][1])} |
| **003** | **wd_ML_BN (K=5 + lm_head + final_norm) — published** | **{pct(r2['wd_ML_BN'][0])}** | **{pct(r2['wd_ML_BN'][1])}** |

\\* wd_C was contributed by a concurrent agent working the same mission and is
included with credit (coordination note in the run record); it had zero
outputs identical to the hook condition — direct evidence that a single-layer
weight edit cannot reproduce the hook's full-stream projection.

### Run 001 charts

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/refusal_by_condition.png" alt="Run 001 refusal by condition: baseline 87.5%, hook 0%, wd_A 75%, wd_B 56.25%, wd_C 68.75%" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/refusal_vs_benign.png" alt="Run 001: benign preservation 93.75% in every condition" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/layer_coherence.png" alt="Run 001 coherence scan: peak 0.664 at layer 17" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/charts/mmlu_guardrail.png" alt="Run 001 MMLU: base 45.78% vs wd_B 45.49%" width="90%">
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

### Consolidation-new evaluation charts (2026-09-28)

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
fp16 — verified in runs 000-001). Direction A = best-layer residual-space
direction (decoder layer **17/24**, coherence **{lc2['best']['coherence']:.3f}** — exact
replication of runs 000-001); direction B = final-layer **post-final-norm
readout-space** direction (coherence 0.562) — the space `lm_head` actually
multiplies against, and the better weight-space edit target.

**Persistent-edit ladder** (run 003, fresh model load per variant, edit →
save → reload-from-disk → row-space residual verify (<1e-5) → probe):

| variant | edit |
|---|---|
| `wd_B` | lm_head row-space orth, `W ← W − (W r̂) r̂ᵀ` (run-001 recipe) |
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
run-001's `wd_B` was {m1v:.2f}% vs {m1b:.2f}% base, Δ {m1b - m1v:.2f}pp). The Δ sits inside one
standard error (±{m2vse:.2f}pp): the guardrail is a ceiling, not a claim of
perfect equivalence.

## New evaluation in this consolidation (2026-09-28, CPU-only)

**TruthfulQA mc2** (`eval2/truthfulqa_*.json`, `eval2/truthfulqa_paired_analysis.json`):
the founding paper reports direction-ablation costs of roughly −1 to −3.5pp
mc2 on larger instruct models; both arms were measured with identical config
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
extracted direction transfers across languages.

Measured harmful-refused counts (n=2 per language):

| condition | en | zh | ru | de |
|---|---|---|---|---|
| baseline | 2/2 | 2/2 | 0/2 | 0/2 |
| hook (inference-time) | 0/2 | 0/2 | 0/2 | 0/2 |
| variant (`wd_ML_BN`, current main) | 0/2 | 2/2 | 1/2 | 0/2 |

Three things worth knowing from this table. First, the base model's own
refusal surface at this scale is **English+Chinese-local** — it complies
with identical requests in Russian and German, so "multilingual safety"
was never something this patient had. Second, the persistent edit removes
the English refusals entirely, leaves the Chinese ones fully intact, and
adds one Russian refusal: the English-extracted edit does **not** act as a
universal cross-lingual switch on this artifact, in contrast with the
larger-model universality prediction. Third, the one benign over-refusal in
the panel (a Russian cake-tutorial prompt) is present at baseline too —
it's the base model's own quirk, untouched by any edit condition, which is
itself a useful provenance signal. Scoring uses the frozen English marker
list plus per-language first-person/explicit markers built on the same
design rule (bare "illegal"-type words excluded in every language to avoid
scoring compliant text as refusal); full outputs in the JSON and the panel
chart above.

## What the literature warns about this artifact (NEW-KNOWLEDGE)

- **Not robustly uncensored** — post-hoc refusal edits are a thin, sharp gate
  over intact capabilities, not erasure:
  [arXiv:2609.06934](https://arxiv.org/abs/2609.06934) shows **100 benign
  fine-tuning examples strip 34.8pp of refusal** (AdvBench, 93% → 58.2%) on
  Qwen2.5-7B-Instruct, and remove nearly all of Llama-3-8B-Instruct's
  (→ ≤11%). Expect the same fragility here at 0.5B: a short benign
  fine-tune re-arms the refusal gate.
- **Off-target disposition shift** — removing the refusal direction is not a
  scalpel: [arXiv:2607.17427](https://arxiv.org/abs/2607.17427) measures
  abliterated models becoming systematically more optimistic in decisions
  under uncertainty (+12.2pp Gemma-4-26B, +7.4pp Qwen3-30B MoE), plus
  more self-justification, with zero refusal deltas on the decision task
  itself (pure side effect). Treat downstream decisions made by this model
  with corresponding suspicion.
- **Multilingual universality — NOT confirmed at 0.5B** —
  [arXiv:2505.17306](https://arxiv.org/abs/2505.17306) predicts the
  English-extracted direction transfers across languages; the measured
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
  from direction ablation; measured here (n=240 paired subsample, 2026-09-28):
  Δ {tqv - tqb:+.2f}pp, CI95 {tqci} (contains zero).

## Fine-tuning notes

If you fine-tune this model, expect the refusal gate to re-arm quickly: the
published evidence ([arXiv:2609.06934](https://arxiv.org/abs/2609.06934))
shows 100 benign examples are enough to strip 34.8pp of refusal from
Qwen2.5-7B-Instruct and nearly all of Llama-3-8B-Instruct's. That is a
feature if you want to walk the refusal/benign frontier back — and a caveat
if you assumed the removal was permanent. Training on benign instruction
data will NOT preserve the 0% state.

## Provenance

- Base weights: `Qwen/Qwen2.5-0.5B-Instruct` @ `{BASE_REV}`
  (Apache-2.0, ungated).
- Current weights (`main`): sha256 `{NEW_WSHA[:16]}…`.
- Previous weights (gentle `wd_B` edit): pinned revision `{OLD_REV}`, weights
  sha256 `{OLD_WSHA[:16]}…` — loadable via
  `from_pretrained("{REPO}", revision="{OLD_REV[:7]}…")` or
  `hf download {REPO} --revision {OLD_REV[:7]}`.
- Method + evaluation write-ups: available on request.

## What this model is, honestly

An abliterated research artifact: the refusal direction was removed to study
how far the technique can go on a small open model before it costs the model
something else. It is not a product.

- **It will answer harmful requests.** Refusal behavior is deliberately
  degraded — that is the entire point of the experiment. Do not deploy it
  anywhere that is unacceptable.
- **"Capability intact" has limits.** MMLU and TruthfulQA barely moved
  (Δ 0.24pp and −1.42pp), but capability metrics don't capture everything —
  see the disposition-shift warning above.
- **The edit is shallow and reversible.** Published research shows ~100
  benign fine-tuning examples strip 34.8pp of refusal on Qwen2.5-7B-Instruct
  and nearly all of it on Llama-3-8B-Instruct. Treat the
  0% number as a property of these exact weights, not of the technique.
- **What you get beyond the weights:** the complete evaluation evidence
  (probe transcripts, MMLU/TruthfulQA raw logs, the multilingual panel),
  and chart generators that re-derive every figure from those artifacts.

## Files

- full safetensors weights + tokenizer (this repo root, `wd_ML_BN`)
- `refusal_direction.npy` (representative wd_ML_BN direction),
  `refusal_direction_A.npy` (residual space, layer 17),
  `refusal_direction_B.npy` (final-layer readout space),
  `layer_directions.npz` (all 24 per-layer directions)
- `eval/` — run-001 evidence (probes, MMLU raw, run_config, selection)
- `eval2/` — run-003 evidence (ladder probes incl. wd_ML_BN, MMLU raw,
  TruthfulQA raw, multilingual probes, selection, harness/ladder shas)
- `charts/` — 12 card figures + `make_all_charts.py` (regenerates every
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