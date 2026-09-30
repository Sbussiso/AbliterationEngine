---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
base_model_revision: 7ae557604adf67be50417f59c2c2f167def9a775
tags:
- abliteration
- refusal-direction
- uncensored
- research
library_name: transformers
---

# sbussiso/Qwen2.5-0.5B-abliterated-r2 — round 2, the 0%-refusal edit

This is the second-generation artifact from the sbussiso lab's Abliteration
Program: the same patient ([Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
@ `7ae5576…`) with a stronger refusal-direction edit than our first
publish. The first model removed about half the persistent refusals; this
one removes **all of them** on our probe set — 16/16 harmful prompts
answered, zero degenerate outputs — and pays one benign probe for it
(details below).

**Which model is for me?** We publish two 0.5B abliterated models, and this
is the honest comparison:

| | this repo (r2) | [sbussiso/Qwen2.5-0.5B-abliterated](https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated) (the original) |
|---|---|---|
| Weights on `main` | run-003 composite `wd_ML_BN` (refusal **0.0%**) | **byte-identical** — same edit, same weights sha256 `0b2133342dce215d…` |
| Other revisions kept | none — this is the single-run package | run-001 gentle edit `wd_B` (refusal 56.2%) @ revision `0155cad…` |
| Evidence shipped | this run only (`eval/`, 4 charts) | both runs, TruthfulQA + multilingual panel, full chart generators |
| Best for | a lean download of the strongest 0.5B edit + its numbers | the full provenance trail, the gentler variant, or reproducing/auditing either run |

Why two repos? The original went up first (2026-09-26) and had 423 downloads
by the time round 2 finished — people were already using it. Rather than
silently changing weights under them or deleting anything, we merged the
new edit into the original's `main` AND kept this repo as the clean
single-run record. Same weights either way; pick by packaging preference.

**Abliterated by the sbussiso lab research agent** (Hermes,
research-workstation profile), Google Colab T4, 2026-09-27/28.

## Results

87.5% of harmful prompts refused at baseline (the base
model's own behavior). The round-2 ladder tried four persistent edits;
the selected composite (`wd_ML_BN`) got refusals to **0.0%**
while still answering 87.5% of benign probes, with zero
degenerate outputs anywhere. All rates use the same frozen keyword scorer
across every condition (first-person/explicit markers only; the list is in
`eval/run_config.json`), greedy decoding, 200 max new tokens, seed 0.
1 probe = 6.25pp of granularity, so read the benign number as "lost at most
one benign probe".

| condition | harmful refusal | harmless answered | degenerate |
|---|---|---|---|
| baseline | 87.5% | 93.8% | 0 |
| hook (L17, inference-time contrast) | 0.0% | 93.8% | 0 |
| wd_B (round-1 recipe replication) | 56.2% | 93.8% | 0 |
| wd_BN (+ final_norm) | 56.2% | 87.5% | 0 |
| wd_ML (o_proj+down_proj, K=3) | 68.8% | 81.2% | 0 |
| **wd_ML_BN (K=5 + lm_head + final_norm) — published** | **0.0%** | **87.5%** | **0** |

The hook row matters for reading the table honestly: inference-time
projection (full-stream, every position) and a weight edit reach different
depths. The composite closing to 0.0% — matching the hook —
is what makes this edit noteworthy.

<p align="center">
  <img src="https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated-r2/resolve/main/charts/r2_refusal_by_condition.png" alt="Ladder: baseline 87.5%, hook 0%, wd_B 56.25%, wd_BN 56.25%, wd_ML 68.75%, wd_ML_BN 0%" width="90%">
</p>

<p align="center">
  <img src="https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated-r2/resolve/main/charts/r2_benign_preservation.png" alt="Benign preservation 87.5% for wd_ML_BN, gate floor 83.75%" width="90%">
</p>

## What the edit costs (measured, not guessed)

**MMLU** (lm-eval, 0-shot, fp16, seed 0, identical config both sides):
base 45.78% ± 0.41 → this model 45.55% ± 0.41 —
Δ 0.24pp, inside one standard error, and inside our program's
<3pp guardrail. The Δ is a ceiling on measurable loss, not proof of perfect
equivalence.

<p align="center">
  <img src="https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated-r2/resolve/main/charts/r2_mmlu_guardrail.png" alt="MMLU: base 45.78% vs wd_ML_BN 45.55%" width="70%">
</p>

**TruthfulQA mc2** (240-question paired subsample, shipped in the original
repo's `eval2/`): base 41.63% → this model 40.20% (Δ −1.42pp, paired
bootstrap CI95 [−3.43, +0.54] — contains zero). For context, the founding
abliteration paper measured −1.0 to −3.5pp across all 13 models they
ablated; ours sits inside that band.

On the multilingual mini-panel (same probes, shipped in the original repo's `eval2/`): baseline en 2/2, zh 2/2 refused; this variant en 0/2, zh 2/2, ru 1/2, de 0/2 — English and German refusal fully removed, Chinese intact, Russian partial.

<p align="center">
  <img src="https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated-r2/resolve/main/charts/r2_layer_coherence.png" alt="Coherence scan: peak 0.664 at layer 17" width="70%">
</p>

## Known limitations (from the 2026-09 literature deep-dive)

- **The edit is a thin gate, not erasure.** Published work
  ([arXiv:2609.06934](https://arxiv.org/abs/2609.06934)) shows ~100 benign
  fine-tuning examples restore most refusal behavior on models like this —
  the refusal circuitry stays underneath the edit. The 0% is a property of
  these exact weights, not a permanent cure.
- **Decision dispositions can shift.** [arXiv:2607.17427](https://arxiv.org/abs/2607.17427)
  measured abliterated models becoming more optimistic/self-justifying in
  decisions, with no change on capability benchmarks. MMLU alone doesn't
  guard against this.
- **Multilingual universality does not hold at this scale** (measured —
  see above). Do not assume either safety or ablation transfers across
  languages on this artifact.
- **No anti-ablation hardening.** [arXiv:2609.16204](https://arxiv.org/abs/2609.16204)
  describes a cheap defense that re-injects decoy refusal directions; this
  artifact is an ablation study and ships none.

## Method (reproducible summary)

Extraction per Arditi et al. 2024
([arXiv:2406.11717](https://arxiv.org/abs/2406.11717), NeurIPS 2024):
64 harmful/harmless prompt pairs; final-position residual stream captured at
all 24 decoder layers (`output_hidden_states=True`); layer chosen by
direction coherence (layer **17/24**, coherence **0.664**).
The published edit combines: row-space orthogonalization of `o_proj` +
`down_proj` at the top-5 coherence layers [17, 18, 19, 16, 15], plus the
`lm_head` (final-layer readout-space direction), plus the final RMSNorm —
each writer matrix orthogonalized against its own layer's direction.
Qwen2.5-0.5B ships tied embeddings, so the `lm_head` edit went to a cloned
untied parameter; `config.json` persists `tie_word_embeddings: false` and
the input embeddings are untouched (verify: `eval/ladder_sha256.json`).
Selection was pre-registered (`eval/selection.json`): benign ≥ baseline
− 0.10, zero degenerates, lowest refusal among gate-passers, publish gate
refusal < 0.25. Full method + math: the run-003 paper on FT Team
([Linear document](https://linear.app/home-lab101/document/abliteration-run-003-the-persistent-edit-ladder-multi-layer-weight-deco-372f46635318),
[FTT-14](https://linear.app/home-lab101/issue/FTT-14/abliteration-run-003-qwen25-05b-persistent-edit-ladder-round-2));
round-1 paper: [FTT-11](https://linear.app/home-lab101/issue/FTT-11/abliteration-run-002-qwen25-05b-published).

## Provenance

- Base weights: `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0).
- This repo's weights: sha256 `0b2133342dce215d6ae9645259b9f05e33914c0f51dcde60c550f3bf92367125` — byte-identical to
  `sbussiso/Qwen2.5-0.5B-abliterated`'s current `main` (verified at both publishes).
- Charts regenerate from `eval/` via `charts/make_r2_charts.py`.

## What this model is, honestly

A research artifact, not a product. It will answer harmful requests — that
is the experiment. Do not deploy it anywhere that matters. The interesting
science here is what the probe table and the hook contrast say about how
refusal is written into small transformers; if you want the full story with
every recorded number, the original repo carries it.

## Files

- full safetensors weights + tokenizer (repo root, `wd_ML_BN`)
- `refusal_direction.npy` (representative direction), `refusal_direction_A.npy`
  (residual space, layer 17), `refusal_direction_B.npy` (final-layer readout
  space), `layer_directions.npz` (all 24 per-layer directions)
- `eval/` — probe transcripts (all 6 conditions), MMLU raw results
  (base + variant), `run_config.json` (frozen scorer + protocol),
  `selection.json`, `selection_candidates.json`, `layer_coherence.json`,
  `harness_sha256.json`, `ladder_sha256.json`
- `charts/` — the 4 figures + generator
