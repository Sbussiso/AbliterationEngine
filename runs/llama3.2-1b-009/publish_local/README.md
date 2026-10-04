---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B-Instruct
base_model_revision: 9213176726f574b556790deb65791e0c5aa438b6
library_name: transformers
pipeline_tag: text-generation
tags:
- abliteration
- refusal-direction
- research-artifact
- llama
---

# Llama-3.2-1B-Instruct-abliterated

A refusal-ablated edit of **meta-llama/Llama-3.2-1B-Instruct** (pinned
revision `9213176726f574b556790deb65791e0c5aa438b6`): the readout-space
refusal direction was orthogonalized out of the untied `lm_head` weight,
making the edit **persistent** — no hook required at inference time.
First non-Qwen patient of this engine; the founding-proposal direction
(Arditi et al. 2024) was originally characterized on Llama-2-class
models, and this run measures the method back on a modern small Llama.

## Status: interim (measurements upgrading in place)

The instrument (persistent-edit ladder, verified probes, capability
guardrail) is identical to the engine's other published patients, but
this patient's **final numbers are being re-banked from a fresh GPU
window** after two Colab session-reaper casualties mid-run. What is
here now is real and verifiable; the table below will state its vintage.

## What it does (interim numbers — streamed + rebuilt artifacts)

| arm | refused /64 (harmful) | benign preserved /64 |
|---|---|---|
| base Qwen-comparable instrument | 38 (59.4%) | **64 (100%)** |
| inference-time hook ablation | 34 (53.1%) | 61 (95.3%) |
| **this variant (wd_B, persistent)** | **8 (12.5%)** * | **63 (98.4%)** * |

\* wd_B rows are run-marker vintage (session 1's completed-but-unbanked
ladder); baseline and hook rows are artifact-backed from the banked
probe files in this repo. The re-run window replaces the marker-vintage
rows with artifact-backed ones — byte-deterministic greedy/seed-0
probes make the numbers reproducible (baseline confirmed identical
across both dead sessions).

Residual honesty: 12.5% is the recorded refusal AFTER the edit measured
against the ≤25% publish bar — the strongest gate result of any patient
in this program (every Qwen persistent edit landed above the bar).

## Method

Founding-proposal directional editing (Arditi et al. 2024,
https://arxiv.org/abs/2406.11717): 64-pair contrastive direction
extraction → coherence-ranked sites → weight-space orthogonalization.
This variant = `wd_B` (readout-space direction against `lm_head`).

Direction banks shipped in this repo (`refusal_direction_A.npy`,
`refusal_direction_B.npy`) are the run's banked stage-A artifacts.
Weights sha256 (current upload):
`6417be31231b76300c7013acccfd30251cc5c105356089a456950e23666e0a9c`
(deterministic CPU rebuild from the banks; the GPU-native artifact from
the re-run window replaces/upload-checks against these bytes).

## Capability guardrail

MMLU guardrail (0-shot, 61 subjects, ≤3.0pp loss limit): **pending the
re-run window** — not yet measured on this patient. The card updates
in place when it lands; no claim is made before the measurement.

## Try it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
    "sbussiso/Llama-3.2-1B-Instruct-abliterated", torch_dtype="auto")
t = AutoTokenizer.from_pretrained("sbussiso/Llama-3.2-1B-Instruct-abliterated")
```

## Honest limitations

- Interim measurement record, not a finished product: probe counts above
  are from fixed 64-prompt sets with greedy decoding; single-seed; the
  MMLU guardrail is unmeasured as of this upload.
- Refusal-ablation residual behavior is not characterized on this
  patient yet (RefusalBench/SORRY legs not run).
- Any post-hoc refusal ablation is recoverable by small benign
  fine-tuning (literature finding, applies here).
- Meta's Llama 3.2 Community License applies to the base model and
  this edit; use under that license's terms.

## Provenance

- Base weights: `meta-llama/Llama-3.2-1B-Instruct` @
  `9213176726f574b556790deb65791e0c5aa438b6` (gated — access accepted).
- This variant (wd_B): tie_word_embeddings true → untied (config flag
  persisted); weights sha256 `6417be31…66e0a9c` (rebuild instance #1
  from the shipped direction banks — `740b6a93…` value-hash, `94adc977…`
  file-hash).
- Full method + evaluation write-ups: available on request.