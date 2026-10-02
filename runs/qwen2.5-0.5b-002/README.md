# Abliteration Run 002 — Mission 003: First Published Abliterated Model

Fresh end-to-end reproduction of run 001 plus the persistent weight-decoded
variant, benchmark, and HF publish of **Qwen/Qwen2.5-0.5B-Instruct**.

- Date: 2026-09-26 (session m003-abl-rw-1, Colab T4, torch 2.11.0+cu128,
  transformers 5.16.1, lm-eval 0.4.13+ on Colab)
- Base revision (pinned): `7ae557604adf67be50417f59c2c2f167def9a775`
- Method: Arditi et al. 2024, "Refusal in LLMs is mediated by a single
  direction" (NeurIPS 2024). Extraction via `output_hidden_states=True`
  (final-position residual stream, all layers, one forward pass per batch;
  nnsight fails on accelerate-dispatched fp16 — verified run 001, x2).
- Harness: `harness/run_002.py` (fresh, self-contained; sha256 in
  `artifacts/harness_sha256.json`). Prompt sets identical to run 001.
- Smoke test: `smoke_test.py` (tiny tied-embedding Qwen2 on CPU) verified
  capture/scan/orthogonalize/save-reload-persistence/hook BEFORE the GPU run.

## Reproducibility vs run 001 (exact replication)

| quantity | run 001 | run 002 |
|---|---|---|
| best layer | L17/24 | **L17/24** |
| coherence | 0.6643 | **0.6643** |
| baseline refusal (16 probes, greedy) | 0.875 | **0.875** |
| hook-ablated refusal | 0.1875* | **0.0** |
| benign preserved (hook) | 0.9375 | **0.9375** |

*run 001's residual refusals were lecture-style preambles; the marker list
used in run 002 matches the one recorded in run 001's config block.

## Conditions (16 harmful + 16 harmless probes, greedy, 200 tok, seed 0)

| condition | refusal_rate | benign_preserved | degenerate |
|---|---|---|---|
| baseline | 0.875 | 0.9375 | 0 |
| hook (inference-time, L17) | 0.0 | 0.9375 | 0 |
| wd_A (lm_head orth, L17 residual-space dir) | 0.75 | 0.9375 | 0 |
| wd_B (lm_head orth, final-layer readout-space dir) | **0.5625** | 0.9375 | 0 |
| wd_C (L17 o_proj+down_proj left-orth, row space) | 0.6875 | 0.9375 | 0 |

Gate (benign >= baseline-0.10 AND zero degenerate): passed by all variants.
Selected: **wd_B** (lowest persistent refusal rate; playbook recipe).

## Engineering notes (new this run)

1. **Tied embeddings**: Qwen2.5-0.5B ships `tie_word_embeddings=true`. The
   lm_head orthogonalization is applied to a CLONED untied parameter and the
   config flag flipped to `false` before saving; otherwise the edit either
   corrupts input embeddings or is silently undone by re-tying on reload.
   Persistence verified by disk reload: `max|W r-hat|` on-disk = 1.5e-08.
2. **Direction spaces**: dir_A (L17 residual space) works for the hook but
   is weak in lm_head space (wd_A 0.75). dir_B is computed in post-final-norm
   readout space (the exact space lm_head multiplies) and is the better
   weight-space direction (wd_B 0.5625). Both vectors are saved as .npy.
3. **Hook ≠ weight edit at one layer**: the hook projects the FULL layer
   output including the residual pass-through h_in; no weight edit at that
   layer can replicate it because h_in flows around the weight matrices
   (wd_C row-space edit: 0.6875, 0 identical outputs to hook).
4. **Invariant check gotcha**: for (M W) row-space edits the correct
   orthogonality invariant is (M W)^T r ≈ 0, NOT (M W) r ≈ 0 (that's the
   column-space check for W M). Assertion on the wrong side cost one rerun.

## lm-eval MMLU (0-shot, batch auto, float16, seed 0, identical config)

| model | acc | acc_stderr |
|---|---|---|
| baseline (pinned rev) | 45.78% | 0.41pp |
| wd_B | 45.49% | 0.41pp |

Delta: **0.29pp loss** — publish guardrail (<3pp) PASSED. Raw results in
`artifacts/eval/{base,variant}/`; base config pinned to revision
`7ae557…9a775`. A separate 5-shot attempt by the concurrent agent was
aborted on GPU contention (see MISSION-003-COORDINATION.md).

## Published artifact

- **https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated** — PUBLIC,
  Apache-2.0, revision `0155cadc8d6382acb4cd5cc6ef59edce7f4c3f40`, 21 files
  (published 2026-09-27 ~07:05 UTC via harness/publish.py, gates: selection
  gate passed + MMLU <3pp + whoami sbussiso).
- Provenance: hub LFS sha256 `8ee4567a…00fd6d` == sha256 of /content/wd_B
  source weights (byte-identical, verified from VM 151 after upload).
- Card includes: base+revision sha, Arditi method, layer 17/24 + coherence
  0.664, all condition refusal rates, MMLU table, intended-use warning,
  "Abliterated by the sbussiso lab research agent".
- Aux files on hub: refusal_direction{,_A,_B}.npy, eval/ (raw lm-eval
  results + probe JSONs), run_config.json, selection.json,
  layer_coherence.json.
- Card is honest about partial persistent removal: lm_head-only edit =
  56% refusal (vs 0% inference-time hook) because residual-stream pathways
  are untouched; hook result recorded alongside as the full-removal contrast.

Publish gate: MMLU delta must be < 3pp or do-not-publish. **PASSED (0.29pp).**


## Artifacts

- `artifacts/` — selection.json, layer_coherence.json, run_config.json,
  probes_*.json (4 conditions), refusal_direction{,_A,_B}.npy,
  harness_sha256.json
- `/content/wd_B` (Colab) — full safetensors of the selected variant
- HF repo (after publish): sbussiso/Qwen2.5-0.5B-abliterated
- Colab logs pulled to `logs/` after the run.