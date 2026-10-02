---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
base_model_revision: a09a35458c702b33eeacc393d103063234e8bc28
tags:
- abliteration
- refusal-direction
- safety
- qwen2.5
library_name: transformers
---

# Qwen2.5-7B-abliterated

An abliterated variant of **Qwen/Qwen2.5-7B-Instruct** (revision `a09a35458c702b33eeacc393d103063234e8bc28`, untied embeddings, bf16 base). The refusal direction identified causally per layer is removed from the **output row spaces** of three MLP `down_proj` matrices (layers 20, 18, 19 — order = primary, second, third) via deterministic orthogonal projection: after the edit, `(W r) ≈ 0` for the refusal direction `r` at each edited site, so projecting a hidden state through these layers cannot produce the refusal-conditional component while all other directions pass through unchanged.

The model answers harmful-policy probes it previously refused, while **every capability and behavior guardrail measured holds at baseline**: identical MMLU accuracy, 100% benign-prompt compliance, zero degenerate outputs, and grounded-refusal calibration moved by less than one percentage point on a 1,600-instance paired benchmark. No fine-tuning was applied; the weights differ from base only by the three output-row projections, so the edit is exactly invertible from the shipped direction vectors.

## Measured results

All numbers come from the paired evaluation harness in [`sbussiso/abliteration-run002-ingredients`](https://huggingface.co/datasets/sbussiso/abliteration-run002-ingredients) (per-instance records, scoring code, and chart generators included there) — every claim below is backed by a recorded JSON artifact in that repo's `evidence/` tree.

| Instrument | Base 7B | This model | Delta |
|---|---|---|---|
| Harmful-probe refusal (16-prompt battery) | 93.75% | **12.5%** | −81.25pp |
| Benign-prompt compliance (16 prompts) | 100% | **100%** | 0 |
| Degenerate outputs | 0 | **0** | 0 |
| MMLU 0-shot (base == variant, identical acc to 16 decimals) | 71.77% ±0.36 | 71.77% ±0.36 | **Δ 0.00pp** |
| RefusalBench-NQ, category-correct refusals, n=1,600 paired | 4.88% | 4.00% | Δ −0.88pp, 95% CI [−1.75, −0.06], McNemar p = 0.059 |

RefusalBench (arXiv:2510.10390) measures selective refusal on deliberately flawed knowledge questions — refusing *more* than base means worse discernment, refusing *less* means worse calibration. Here the edited model's discrimination moves within one point of base with the confidence interval touching zero, i.e. the removal does not produce a broader indiscriminate-compliance regime on this instrument. Binary flawed-question refusal moved 36.75% → 35.25% (same direction, same magnitude).

## Method summary

1. **Refusal-direction identification** — per-layer mean difference of residual-stream activations between harmful-refusal and benign-completion records (15/16 harmful prompts refused at baseline), coherence-weighted selection across all 28 layers; layer 20 selected as the primary causal site (baseline refusal 93.75% → 18.75% with the single-layer hook ablation, benign 100%).
2. **Persistent edit construction** — for each selected layer `l`, compute the orthogonal projector `I − r rᵀ` in the row space and apply it to the corresponding `down_proj` weight: the layer can never re-introduce the refusal component. Applied at three layers (20, 18, 19) chosen from the post-edit probe sweep (wd_B readout-space edit reached only 50% removal — the row-space multi-layer combination wins for this architecture/revision).
3. **Verification** — each edited site asserts `‖Wᵀ r‖∞ < 1×10⁻³` post-edit; the released directions file is fp32 `(28, 3584)` and the edit recomputes bit-identically (three independent rebuilds produced identical probe tables).

## Files

- `model-0000{1..4}-of-00004.safetensors` + `model.safetensors.index.json` — the edited weights (bf16, 339 tensors, total 15,231,233,024 bytes as in the index)
- `config.json`, `generation_config.json`, tokenizer files, `chat_template.jinja` — unchanged from the base instruct revision
- Full evaluation evidence (Stage A direction identification, Stage B variant selection, per-instance probe records, MMLU paired results, RefusalBench per-row records): [`sbussiso/abliteration-run002-ingredients`](https://huggingface.co/datasets/sbussiso/abliteration-run002-ingredients)

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("sbussiso/Qwen2.5-7B-abliterated")
model = AutoModelForCausalLM.from_pretrained(
    "sbussiso/Qwen2.5-7B-abliterated", device_map="auto", dtype="bfloat16")
```

## Limitations and intended use

- Intended for research on refusal-direction geometry and safety-behavior measurement. The persistent edit weakens the refusal channel specifically; other safety-trained behaviors (e.g. training-time alignment beyond the residual-stream refusal pathway) are not evaluated exhaustively here.
- The RefusalBench result is near the significance boundary (p = 0.059); treat the −0.88pp category-movement as "no measurable degradation" rather than a proven improvement.
- The evaluation battery covers English harmful-policy probes plus a four-language benign/harmful mini-panel; broader non-English or adversarial safety behavior is unmeasured.
- MMLU and the benign batteries are unchanged, but long-context behavior and multilingual instruction-following at length are not re-benchmarked.
- This release ships the **wd_ML** selection (the strongest guardrail profile in the Stage B sweep). A gentler single-layer variant is recoverable from the shipped directions file by editing only layer 20.

## Citation

```bibtex
@misc{qwen25_7b_abliterated_2026,
  title        = {Qwen2.5-7B-abliterated: a refusal-direction-removed Qwen2.5-7B-Instruct},
  author       = {S'Bussiso Dube},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated}},
}
```