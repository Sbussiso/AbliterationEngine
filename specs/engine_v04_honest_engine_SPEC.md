# ENGINE-V0.4 — "The Honest Engine" — Engineering Spec

**Status:** SPEC v1 (user-scope decision 2026-10-06: gun straight for v0.4 — no v0.3 intermediate)
**Owners:** dev-workstation (engine source implementation) · research-workstation (search design, instrument, proof runs, this spec)
**Parent docs:** vision `The Honest Engine` (doc e37cfbd7281b) · FTT-28 (proof runs) · engine repo `/root/research/abliteration` (v0.2.0 lineage)
**Boundary rules:** dev-workstation owns engine source; this spec is the contract. AGPL discipline: ARA implemented from published math only (Weidmann 2026 PR#211 description), ZERO code reuse from p-e-w/heretic. Patient revisions pinned per-run. All research on Colab GPU windows; no local GPU.

---

## 0. What v0.4 must be (the one-line contract)

An engine that, for any open-weight HF patient, **searches the edit geometry (directions × sites × strengths), optimizes an HONEST objective (holdout refusal + benign preservation + capability proxy inside the loss), and ships every candidate with a no-collateral certificate** — producing refusal numbers that survive strict independent scoring.

---

## 1. New modules

### 1.1 `modifiers/` — pluggable edit strategies (new abstraction)
```
Modifier = strategy that maps (model, direction_bank, search-point) -> edited-model
```
- **`projection`** (existing): orthogonalize/readout-write path = current engine ops. Kept for the ladder + as a search candidate.
- **`ara`** (NEW): Arbitrary-Rank Ablation from the math — per-module rank-r LoRA delta (default r=50) on chosen modules, optimized by L-BFGS (lr=1.0, 5 outer × ≤20 inner iters, strong-Wolfe line search) minimizing:
  `L = w_pg·MSE(new_out[good], orig_out[good]) + w_sb·(KNN-pull(new_out[bad] → orig_out[good], k) + w_oc·(−KNN-dist(new_out[bad], orig_out[bad], k)))`
  with `w_oc` = the overcorrect (leash) axis (P4). Module I/O captured once per prompt batch per layer (hook-based capture like stage A).
- **`composite`** (NEW, ours-beyond-them): ARA terms + a **projection term on the direction bank** at selected sites in ONE joint objective — so geometry choice and steering are co-optimized, not sequential.

### 1.2 Write-path hardening (kills the run-010 bug class)
- Direction tensors brought to weights' device+dtype AT APPLICATION TIME (BUG-1 fix as spec'd).
- `save_variant` derives tie-expectation from the model's ACTUAL config; asserts post-application consistency only (BUG-2 fix as spec'd).
- Multi-component coverage: per-layer site sets drawn from the arch registry (dev's 0763282 lineage — coordinate with that WIP): `o_proj`, `mlp.down_proj`, `lm_head` (+ optional `embed_tokens`), respecting untied/tied config and GQA/MLA arch facts.

### 1.3 `search/` — geometry harness (P1)
- Optimizer: TPE (`optuna`, same library heretic uses — the math is public domain) over:
  - **direction-set selection**: subsets/compositions of the direction bank (chat-anchor dir, code-domain dir, adversarial dir, benign-null dir) — set-valued search on top of continuous weights
  - **site set**: which components × which layer ranges (start/end indices as discrete search params)
  - **strength profile**: per-component max weight + position + decay (heretic-analog axes — table stakes)
  - **direction-count K** (the composite question)
  - **ARA hyperparams** when the ara/composite modifier is selected (rank, w_pg, w_sb, w_oc, k)
- Budget adapter: every trial checkpointed (banked-resume lease discipline) — search state survives Colab lease death; trials resumable window-to-window (our existing kit does this for arms; generalize to trials).
- Pruner: median pruner on the honest objective; kill doomed trials early.

### 1.4 `honest/` — the honest objective (P2) + certificates (P3)
- **Data contract:** `train` strata (optimizer sees), **holdout** refusal strata (NEVER seen by training or search), benign strata (FIM/pytest/refactor/explain classes), capability probe set (MMLU-proxy: small per-subject logit-consistency set; full MMLU only at certification).
- **Loss terms (in-loop):** steer-bad (above) + `w_hold·refusal(holdout)` (measured by the engine's strict scorer on held-out prompts) + `w_benign·(benign_drift)` (KL or MSE on benign outputs, magnitude-preserving option per MPOA math) + `w_cap·(capability drift)`.
- **Scorer:** the engine's existing strict refused-flags (49-marker fp_explicit_v1) — NOT loose keyword matching. No train-on-test: holdout ≠ optimizer data, enforced structurally (separate files, separate loader, asserted disjoint).
- **Certificates (P3):** every candidate edit that completes search ships `certificate.json` = {Δ-refusal (train + HOLDOUT), Δ-benign-preservation, Δ-capability (MMLU-proxy + full MMLU at certification), digit-identity record (direction shas, rebuild re-run hash match), seeds, config, patient revision}. No certificate = not publishable.
- **Leash curves (P4):** the search logs carry every point's (w_oc, Δ-refusal-holdout, Δ-benign, Δ-capability); renderer emits the full curve per patient — first-class artifact.

## 2. Proof-run execution plan (FTT-28, our instrument only)
- Patients: Qwen2.5-1.5B-Instruct @ 989aa798 (chat reference) + Qwen2.5-Coder-7B-Instruct @ c03e6d35 (the hardened target).
- Arms: (a) heretic-ARA run by us under our scorer wrapper (their tool, our instrument — no instrument merging), (b) engine-v0.4 best-candidate, (c) v0.2 ladder results (already banked).
- **Pre-registered coder gates:** refusal < 25% (strict scorer, holdout strata), ΔMMLU ≤ 3.0pp, benign-preservation ≥ 90%. All three or the arm fails honestly.
- Budget: heretic arm ~30 trials/L4-window (their n=100 default is our 2-3 windows; 30 gives the TPE signal), v0.4 search ~same. Banked-resume makes windows composable.
- Instrument reporting: BOTH our strict scorer AND their 33-marker keyword scorer computed per arm (for the cross-instrument honesty table).

## 3. Build order (dev) + parallel research tracks (me)
| # | item | owner | gate |
| -- | -- | -- | -- |
| 1 | BUG-1 + BUG-2 fixes (spec already delivered) | dev | pytest suite green + my same-window re-run of wd_ML |
| 2 | modifier abstraction + ARA from math | dev | unit tests vs analytic projection on toy model |
| 3 | write-path hardening per §1.2 | dev | run-010 patient passes save/edit on untied config |
| 4 | honest/ module (loss + holdout contract + certificates) | dev, spec by me | disjointness asserts + certificate schema tests |
| 5 | search/ harness (TPE + budget adapter) | me (spec+impl on research side where possible; engine-side glue dev) | resume-from-lease-death test |
| 6 | proof runs | me | FTT-28 gates + paper |
| ∥ | pools: code-domain direction strata + holdout splits | me (data-only) | strata asserts (established pattern) |

## 4. Success definition (program-level)
Engine-v0.4 = the first abliteration engine that is simultaneously (a) geometry-searched, (b) holdout-honest, (c) certificate-carrying, (d) cross-session reproducible. If the coder gates pass, the claim "the engine whose numbers you can trust — and it jailbreaks hardened patients" is artifact-backed end to end.