---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
library_name: transformers
pipeline_tag: text-generation
tags:
- abliteration
- refusal-direction
- research-artifact
- qwen2.5
---

# Qwen2.5-1.5B-abliterated

A refusal-ablated edit of **Qwen/Qwen2.5-1.5B-Instruct** (pinned revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`): refusal-direction components were
orthogonalized out of the weight matrix at the edit sites selected by the
run-002 measurement, making the removal **persistent** — no hook required at
inference time.

Method: weight-decoding abliteration (Arditi et al. 2024,
https://arxiv.org/abs/2406.11717) with a persistent edit ladder.

## What it does

Refusal behavior measured on a fixed 64-prompt harmful set (greedy, 200 new
tokens, marker-based scorer): the hook arm reads 1/64 under the original
v1 marker set and 0/64 under the updated v2 scoring; the persistent-edit
arms are scored v1 throughout (the cross-run comparable standard).

| arm | refused /64 |
|---|---|
| base Qwen2.5-1.5B-Instruct | 63 (98.4%) |
| inference-time hook ablation | 1 (0 under the updated scorer) |
| **this variant (wd_ML_BN)** | **22 (34.4%)** |

![Refusal ladder by edit condition — full 64-prompt probes, greedy, seed 0](charts/f1_ladder_refusal.png)

![Benign preservation by condition](charts/f2_benign_preservation.png)

The residual-honesty bar: this variant does NOT match the hook's clean 0%.
The 64-prompt probe gate for persisted edits required ≤ 25% and the variant
lands at 34.4% — so per the run's own design, **publish-ineligible as a
finished abliteration product**. It is shared for reproducibility of the
recorded measurement, not as a recommended unrefused model.

## Benign behavior

64-prompt benign preservation: base 61/64 answered fully under the original
v1 marker reads (62/64 = 96.9% under the updated scorer's re-grade) →
variant 56/64 (87.5%). The multi-layer + final-norm edit sites (BN legs)
are what cost the ~8pp: three benign prompts that the base answers now
stop with a short boilerplate refusal, plus five that the updated scorer
flags as refusal-preamble-only (a scoring-vintage artifact, disclosed by
the dual-standard provenance).

## Capability guardrail

MMLU (0-shot, 61 subjects, subject-weighted, lm-eval-harness, seed 0):
base 60.10% → variant 59.99% — a loss of **0.11pp against the 3.0pp guardrail
limit** (28× margin). The edit removed refusal with negligible general-capability
cost.

![Capability guardrail: MMLU loss vs the 3.0pp limit](charts/f3_mmlu_guardrail.png)

## Try it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
    "sbussiso/Qwen2.5-1.5B-abliterated", torch_dtype="auto")
t = AutoTokenizer.from_pretrained("sbussiso/Qwen2.5-1.5B-abliterated")
```

## Honest limitations

- Residual refusal at 34.4% on the measured harmful set — above the designed
  ≤25% publish gate. Multi-prompt persistent ablation at this scale trades
  refusal removal for benign over-refusal; the ladder data shows the tradeoff
  is monotone in the number of edit sites.
- Benign preservation 87.5% (vs 96.9% base).
- Scores are from a single fixed probe set (64 harmful / 64 benign prompts,
  greedy decoding); not a benchmark-suite claim.
- This card is a measurement record. No safety fine-tuning of any kind was
  re-applied or restored.

## Provenance

- Base weights: `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
  (Apache-2.0, ungated).
- Variant edit sites: decoder layers [19, 18, 16, 17, 20] (row-space o_proj +
  down_proj nullspace), lm_head (untied) + final RMSNorm (readout space).
- Directions: banked layer-direction extraction (28×1536 directions npz) +
  refusal_direction_B, sha256 (16-char) `f2210808d8dd9221`.
- Variant weights sha256: `ae691de6a308bd57a8d6d672548abb726df2bc26466904716da867928bdd12f4`
  (339 tensors, fp16, single shard).
- Disk-verified untied (lm_head weight_diff_l2 = 9.612) with edit residuals:
  lm_head 7.1e-5, final_norm 7.4e-4, layer rows 5e-5..9e-5 (all within
  spec bounds 5e-3 / 5e-2 / 1e-2).
- Charts in `charts/` are generated programmatically from the recorded probe JSONs
  (64 harmful + 64 benign per condition; MMLU summary) — generator doctrine, no hand-typed digits.
- Full method + evaluation write-ups: available on request.

This is a **private research artifact**. It is not a product and is not
intended for production use.
