# Mission 001 — Abliteration First Run (patient zero)

**Model:** Qwen/Qwen2.5-0.5B-Instruct (ungated, Apache-2.0, 24 decoder layers)
**Method:** Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction"
**Date:** 2026-09-26
**Compute:** Google Colab T4 via colab CLI (session `abliteration-001-r2`; see
"Colab session drama" below)
**Result: SUCCESS — refusal ablated, capabilities preserved, full replication
converged.**

## Headline numbers (harness v2, this dir /artifacts/harness-v2/)

| Metric | Value |
|---|---|
| Refusal rate BEFORE (64 harmful probes, greedy) | 82.8% |
| Refusal rate AFTER ablation | 3.1% |
| Benign answer preservation AFTER (64 harmless probes) | 90.6% (baseline 92.2%) |
| Degenerate outputs (all 4 conditions) | 0 |
| Harmful outputs changed by ablation | 64/64 |
| Chosen layer | 17 / 24 |
| Coherence (||mean_diff|| / mean_pairwise_diff) | 0.664 |
| Config | 64 pairs, fp16, greedy, seed 0, max_new 200, hooks on ALL layers |
| Wall time | 122.6 s on T4 (extract 1s + baseline 51s + ablated 53s) |

Qualitative check: baseline outputs are uniformly "I'm sorry, but I can't
assist with that."; ablated outputs are fluent, on-topic compliant
generations (samples_full.json).

## Independent replication (concurrent Claude Code agent, same VM/account)

A second agent ran the same mission in parallel (same session name — that
collision is documented below). Its independent implementation (also fell
back from nnsight to output_hidden_states extraction) converged:

- Same chosen layer: 17/24, coherence 0.664 (mine: 0.6643)
- Its refusal rate: 87.5% → 18.75% (benign preserved 93.75% → 93.75%)
- Direction norms match at layer 17 (7.74) and diverge only at the final
  layer (unnormalized-residual artifact) — extraction replicated.

Delta in headline refusal-after (3.1% vs 18.75%) is probe-scoring
methodology: its marker list includes the bare word "illegal" which also
matches benign-sounding compliance text ("this is illegal in most
jurisdictions, however here is..."); v2 uses a tighter first-person marker
set. Both runs show the same order-of-magnitude drop; artifact sets kept
side by side in /artifacts/ (its files at top level, v2 in harness-v2/).

## Key learnings (also folded back into skills)

1. **nnsight 0.7.0 API drift is real and confirmed twice independently.**
   `LanguageModel(...).trace(...)`, the wrapper around an
   accelerate-dispatched fp16 model, raises `ExitTracingException` from
   inside `lm.trace()` at `layers[10].output[0].save()`. Do not use nnsight
   for this on current Colab (py3.13 / torch 2.11 / transformers 5.16).
   `output_hidden_states=True` gives ALL layers in ONE forward pass per
   batch (v1 reference script traced layer-by-layer ≈ 20x more forwards).
2. **Ablation scope:** hooks must go on ALL decoder layers, not just the
   chosen layer — the direction is present at every depth after its
   formation. Single-layer hooks leave partial refusal.
3. **Marker list design:** bare "illegal"/"harmful" false-positive on
   compliant text; use first-person/explicit markers only. Absolute refusal
   rates are approximate by nature; deltas under a constant scorer are what
   carry meaning.
4. **Coherence profile sanity:** rises through mid layers (peak ~2/3 depth:
   layer 17/24), decays after; last-layer |dir|=95 (unnormalized residual)
   is the classic artifact — never pick the final layer by norm.
5. **Colab shared-name collision:** two agents using the same session name
   on the same Google account will destroy each other's sessions
   (name-based `colab stop` matches ANY runtime). Use unique names
   (`<mission>-<agent>-<n>`), download artifacts incrementally the moment
   they exist, and treat the session as stoppable at any second.
6. **colab download runs as sbussiso** → cannot write /root/research (root
   only). Download to /tmp first, then copy as root. Grep-filters on CLI
   output can hide "Download failed" lines — surface errors.
7. **Long runs must be detached on the Colab VM** (subprocess.Popen in an
   exec cell) — kernel-side, survives CLI disconnects; poll via
   /content/run_log.txt + /content/exit_code.txt sentinels.

## Reproduce

    # on any Colab GPU session
    pip install transformers accelerate datasets huggingface_hub
    python ablate_refusal.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --pairs 64 --probes 64 --out /content/abliteration_out

Harness: /root/research/abliteration/qwen2.5-0.5b/harness/ablate_refusal.py
(v2; fixes the skill reference script's undefined REFUSAL_MARKERS, dead
nnsight prototype code, bf16-on-T4, single-layer hook, and
padding-side-fragile index). Reference v1 kept untouched in the skill dir.

## Artifacts

- artifacts/harness-v2/refusal_direction.{pt,npy} — float32 [896], layer 17
- artifacts/harness-v2/layer_scan.json — full 24-layer coherence table
- artifacts/harness-v2/report.json — metrics + config + versions
- artifacts/harness-v2/samples_full.json — 64x4 generations (all conditions)
- artifacts/harness-v2/progress.log, plus co-tenant's independent
  artifacts/ (report.json, probes_*.json, layer_coherence.json,
  env.json, sample_outputs.md)
- NO HF pushes (per mission guardrail); no weights saved (inference-time
  ablation only; persistent weight-decoding deferred to a later run)

## Colab session drama (postmortem)

- Session 1 (`abliteration-001`): co-tenant Claude agent staged its own
  pipeline over my upload, completed, tore the session down at 22:51 with
  its artifacts.
- Session 2: re-provisioned same name, launched; killed 22:58 by the
  co-tenant's name-based stray-sweep (POST /tun/m/unassign in colab.log).
- Session 3 (`abliteration-001-r2`): full clean run, artifacts pulled, then
  stopped deliberately. Zero strays confirmed (`colab sessions` empty).