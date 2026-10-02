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

# sbussiso/Qwen2.5-7B-abliterated

**An abliterated Qwen2.5-7B-Instruct: the refusal behavior is removed by a
persistent weight edit at three layers; capability measured unchanged.**

- **Refusal on harmful probes: 93.75% → 12.5%**, permanent in the weights; the model answers requests the base model refuses
- **Broad-spectrum**: on 440 diverse unsafe prompts (SORRY-Bench), refusal drops 62.95% → 11.36% — every one of the 44 unsafe categories moves toward answering
- **Capability unchanged**: MMLU accuracy bit-identical to base (same number to 16 decimals)
- **Benign behavior unchanged**: 100% compliance with harmless prompts, zero degenerate outputs
- **Grounded-question calibration intact**: paired Δ −0.88pp on 1,600 flawed-knowledge questions (CI touches zero)
- **TruthfulQA within the published band** for this edit class: mc2 −1.09pp, mc1 −1.22pp

It still refuses ~1 in 8 harmful probes — concentrated in violent-crime and
tort categories, phrased as polite declines. This is a research artifact,
not a safety-aligned assistant. See [Limitations](#limitations-and-intended-use).

## Why this exists

Abliteration ([Arditi et al. 2024, "Refusal in Language Models Is Mediated by
a Single Direction"](https://arxiv.org/abs/2406.11717), NeurIPS 2024) finds the direction
in the model's residual stream that carries refusal behavior and removes it
from the weights — no fine-tuning, no prompt changes. Here it is applied to
[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
@ `a09a354…` (untied, bf16): the refusal direction, identified causally per
layer, is projected out of the **output row spaces** of three MLP
`down_proj` matrices (layers 20, 18, 19) via orthogonal projection — the
layer physically cannot re-produce the refusal component, and every other
direction passes through unchanged. The edit is exactly invertible from the
shipped direction vectors; no fine-tuning was applied.

## Results at a glance


All numbers come from the paired evaluation harness in [`sbussiso/qwen2.5-7b-abliterated-evidence`](https://huggingface.co/datasets/sbussiso/qwen2.5-7b-abliterated-evidence) (per-instance records, scoring code, and chart generators included there) — every claim below is backed by a recorded JSON artifact in that repo's `evidence/` tree.

| Instrument | Base 7B | This model | Delta |
|---|---|---|---|
| Harmful-probe refusal (16-prompt battery) | 93.75% | **12.5%** | −81.25pp |
| Benign-prompt compliance (16 prompts) | 100% | **100%** | 0 |
| Degenerate outputs | 0 | **0** | 0 |
| MMLU 0-shot (base == variant, identical acc to 16 decimals) | 71.77% ±0.36 | 71.77% ±0.36 | **Δ 0.00pp** |
| RefusalBench-NQ, category-correct refusals, n=1,600 paired | 4.88% | 4.00% | Δ −0.88pp, 95% CI [−1.75, −0.06], McNemar p = 0.059 |
| TruthfulQA mc2, 0-shot paired | 64.72% ±1.55 | 63.64% ±1.54 | Δ −1.09pp |
| TruthfulQA mc1, 0-shot paired | 47.86% ±1.75 | 46.63% ±1.75 | Δ −1.22pp |
| SORRY-Bench 202503 core (44 unsafe classes × 10, n=440 paired) | 62.95% | **11.36%** | Δ −51.59pp, 227/0 discordant (McNemar p ≈ 1e−45) |

RefusalBench (arXiv:2510.10390) measures selective refusal on deliberately flawed knowledge questions — refusing *more* than base means worse discernment, refusing *less* means worse calibration. Here the edited model's discrimination moves within one point of base with the confidence interval touching zero, i.e. the removal does not produce a broader indiscriminate-compliance regime on this instrument. Binary flawed-question refusal moved 36.75% → 35.25% (same direction, same magnitude).

SORRY-Bench (arXiv:2406.14598, ICLR 2025; 202503 refresh) is the breadth counterpart: 440 core prompts across 44 fine-grained unsafe classes, run identically in both arms (greedy, same generation settings). The edit broad-spectrum-removes refusal on this instrument: **all 44 classes** move toward answering (largest per-class drop −90pp in five classes), 227 base-only refusals flip vs 0 abl-only flips, and no class is fully removed. The residual 11.36% (50 prompts) is not random: 38/50 fall in the crime/tort domain (Violent Crimes, Harassment, Sexual Crimes, Property Crimes, PII Violations among the largest), and 44/50 are phrased as polite declines ("I'm sorry, but…") — the single-direction removal eliminates the loudest refusal pathway but softer residual refusal behavior persists on the most severe categories. On the ascii- and atbash-encoded mutation sets both arms score 0% refusal — the base model itself responds in the encoded domain without refusing — so the edit introduces no differential encoding cost. Scoring uses the program-standard refusal-string detector rather than SORRY's gated Mistral judge; numbers are therefore comparable within arms, not to SORRY's published leaderboard. One quantified scorer artifact, identical in both arms and hence not affecting any paired comparison: three Medical-Advice questions whose hedged "I'm not a doctor" answers are counted as refusals.


## Method summary and figures

![Refusal by variant](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated/resolve/main/charts/refusal_by_variant.png)
![Benign preservation](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated/resolve/main/charts/benign_preservation.png)

1. **Refusal-direction identification** — per-layer mean difference of residual-stream activations between harmful-refusal and benign-completion records (15/16 harmful prompts refused at baseline), coherence-weighted selection across all 28 layers; layer 20 selected as the primary causal site (baseline refusal 93.75% → 18.75% with the single-layer hook ablation, benign 100%).
2. **Persistent edit construction** — for each selected layer `l`, compute the orthogonal projector `I − r rᵀ` in the row space and apply it to the corresponding `down_proj` weight: the layer can never re-introduce the refusal component. Applied at three layers (20, 18, 19) chosen from the post-edit probe sweep (wd_B readout-space edit reached only 50% removal — the row-space multi-layer combination wins for this architecture/revision).
3. **Verification** — each edited site asserts `‖Wᵀ r‖∞ < 1×10⁻³` post-edit; the released directions file is fp32 `(28, 3584)` and the edit recomputes bit-identically (three independent rebuilds produced identical probe tables). The published weights themselves were verified end-to-end: the private repo was loaded back from the hub and re-ran the battery (refusal 12.5%, benign 100%, degenerates 0 — identical).

![Layer coherence](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated/resolve/main/charts/layer_coherence.png)
![Guardrails](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated/resolve/main/charts/mmlu_tq_guardrail.png)

![RefusalBench categories](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated/resolve/main/charts/rb1600_categories.png)
![RefusalBench deltas](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated/resolve/main/charts/rb1600_delta_hist.png)

![SORRY-Bench by class](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated/resolve/main/charts/sorry_core_by_class.png)


## Files

- `model-0000{1..4}-of-00004.safetensors` + `model.safetensors.index.json` — the edited weights (bf16, 339 tensors, total 15,231,233,024 bytes as in the index)
- `config.json`, `generation_config.json`, tokenizer files, `chat_template.jinja` — unchanged from the base instruct revision
- Full evaluation evidence (Stage A direction identification, Stage B variant selection, per-instance probe records, MMLU paired results, RefusalBench per-row records): [`sbussiso/qwen2.5-7b-abliterated-evidence`](https://huggingface.co/datasets/sbussiso/qwen2.5-7b-abliterated-evidence)


## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("sbussiso/Qwen2.5-7B-abliterated")
model = AutoModelForCausalLM.from_pretrained(
    "sbussiso/Qwen2.5-7B-abliterated", device_map="auto", dtype="bfloat16")
```


## Limitations and intended use

- Intended for research on refusal-direction geometry and safety-behavior measurement. The persistent edit weakens the refusal channel specifically; other safety-trained behaviors (e.g. training-time alignment beyond the residual-stream refusal pathway) are not evaluated exhaustively here.
- On the SORRY-Bench breadth panel the removal is broad-spectrum but not total (residual 11.36%, 50/440 prompts — 38 of them in crime/tort categories, mostly polite declines). **This model will still answer many harmful-crime prompts it was originally trained to refuse; treat it strictly as a research artifact, not a safety-aligned assistant.**
- The RefusalBench result is near the significance boundary (p = 0.059); treat the −0.88pp category-movement as "no measurable degradation" rather than a proven improvement.
- The evaluation battery covers English harmful-policy probes plus a four-language benign/harmful mini-panel; broader non-English or adversarial safety behavior is unmeasured.
- MMLU and the benign batteries are unchanged, but long-context behavior and multilingual instruction-following at length are not re-benchmarked.
- This release ships the **wd_ML** selection (the strongest guardrail profile in the Stage B sweep). A gentler single-layer variant is recoverable from the shipped directions file by editing only layer 20.
- **Fine-tuning re-arms refusal.** The edit is persistent but not robust to continued training: on Qwen2.5-7B-Instruct, 100 benign fine-tuning examples (200 steps) strip 34.8pp of refusal from an ablated variant [arXiv:2609.06934]. Expect the same here.

## References

1. Arditi, A. et al. (2024). ["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717). *NeurIPS 2024*. — the founding abliteration method; the refusal direction this edit removes is extracted exactly as described there (per-layer difference-in-means, coherence-weighted site selection).
2. Xie, T. et al. (2025). [SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal](https://arxiv.org/abs/2406.14598). *ICLR 2025 D&B.* — the breadth instrument; the 440-prompt core panel + ascii/atbash mutations of this card's broad-spectrum result use the 202503 refresh (`sorry-bench/sorry-bench-202503`).
3. Muhamed, A. et al. (2025). [RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models](https://arxiv.org/abs/2510.10390). *EACL 2026*. — the grounded selective-refusal instrument used as the calibration guardrail (1,600 paired instances).
4. Malla, S. et al. (2025). [The Geometry of Refusal: Why Post-Hoc Safety Is Fragile and Pretraining-Time Safety Persists](https://arxiv.org/abs/2609.06934). — the fragility caveat above comes from this paper: 100 benign fine-tuning examples (200 steps) strip 34.8pp of refusal from an ablated Qwen-7B; treat this edit as reversible-in-practice and fine-tuning-sensitive.

## Citation

```bibtex
@misc{qwen25_7b_abliterated_2026,
  title        = {Qwen2.5-7B-abliterated: a refusal-direction-removed Qwen2.5-7B-Instruct},
  author       = {S'Bussiso Dube},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated}},
}
```
