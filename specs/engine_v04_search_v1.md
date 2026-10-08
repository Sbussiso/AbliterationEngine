# ENGINE-V0.4 · search v1 — design, positioning, roadmap

**Status:** implemented (`abliterate search`, `src/abliteration_engine/search.py`).
CPU-verified end to end on tiny random patients; **not yet run on a real
patient** — run 011 (`specs/qwen25_1p5b_search.yaml`) is the first GPU proof.
**Parent:** `engine_v04_honest_engine_SPEC.md` §1.3 (search harness) + §1.4
(honest objective, certificates).
**AGPL boundary:** designed from Heretic's public README description only; no
Heretic code read or reused. Optuna is an MIT library both projects happen to
use.

## 1. What search v1 does

1. Stage-A artifacts (per-layer difference-of-means directions, coherence
   table, readout-space direction) — stage A runs first if they are missing.
2. The base model is loaded once and measured: harmful + benign refusal on the
   **search split**, its own greedy answers (the KL reference), and its refusal
   on the **sealed holdout**.
3. Each trial applies a search point **in memory** (rank-1 partial projections;
   exact, tested restore), is scored, and is banked in an SQLite Optuna study —
   a dead Colab session resumes at the next trial.
4. The **Pareto front** over three objectives is kept; a stated policy picks
   one point.
5. The pick is materialized, saved, reloaded from disk, verified
   algebraically (`r·W_new == (1−α) r·W_base`), probed on the standard probe
   sets, **certified on the sealed holdout**, and written as `selection.json` —
   `mmlu` and `publish` then run unchanged.

## 2. How it differs from Heretic (by design, not by accident)

| | Heretic (per its README) | search v1 |
|---|---|---|
| search space | per-layer weight-kernel shape (max/min weight, position, distance) + a fractional direction index, per component | contiguous layer **window** · one partial-projection **strength α** (can over-project) · direction **source**: own layer / best layer / **coherence-weighted blend of top-k layers** · attention / MLP / both · optional **output-head (readout) edit** |
| objective | refusal count + KL, co-minimized | **three objectives never blended**: harmful refusal, benign refusal, **answer-trajectory KL** |
| capability signal | KL divergence on harmless prompts | KL(base‖candidate) over the base model's **own multi-token greedy answers**, not one position |
| refusal grading | refusal counting on harmful prompts | the engine's grader, v1 or **v2** (v2 catches "I can't… here's how" compliance a keyword counter scores as a refusal) |
| data hygiene | — | directions ← probe pairs; trials ← disjoint search split; **certificate ← third, sealed holdout** the optimizer never sees (asserted disjoint) |
| output | best trial(s) | the **whole trade-off** (Pareto front + chart) + a stated selection policy + a **certificate.json** whose numbers trace to artifacts |
| reproducibility | — | spec-pinned, fingerprinted, study banked to disk, on-disk algebraic verification of the shipped weights |

## 3. Where Heretic is still ahead (honest gap list → roadmap)

1. **Proof.** Heretic has thousands of community models and published
   comparisons; search v1 has none yet. → run 011 (search vs run-002 ladder vs
   run-008 ARA on the identical instrument), then the coder-7B patient from the
   FTT-28 plan, reporting both our strict/v2 scorer and a plain keyword scorer.
2. **Throughput.** Heretic auto-benchmarks batch size and supports 4-bit
   loading. → auto batch size for generation; optional bnb 4-bit base for the
   search phase (materialize on the full-precision base).
3. **Zero-config.** Heretic runs from a model name. → `abliterate init` already
   writes the spec; next: `abliterate search --model <id>` that inits + searches
   in one step.
4. **After-run UX.** Heretic offers chat / save / upload / benchmark after a
   run. → an `abliterate chat` verb on any variant dir; `--evaluate` for an
   existing model.
5. **Architecture breadth.** → MoE attention-only edits landed; multimodal
   and hybrid families next.

## 4. Where to pull ahead (things only this engine is set up to do)

- **Certified claims:** every published number carries a holdout certificate
  measured by a scorer that cannot be gamed by apology preambles.
- **Composite modifier (spec §1.1):** ARA terms + direction-bank projection in
  one joint objective — search can select it once implemented.
- **Domain-aware directions:** code-domain pools (run-010) as extra
  direction sources for hardened coder patients.
- **Leash curves (spec P4):** the Pareto front *is* the leash curve — publish
  it on the card so users pick their own refusal/drift trade-off.
- **Cross-session reproducibility:** fingerprints + banked trials make a
  search resumable and auditable across Colab windows.

## 5. Spec block

```yaml
search:
  trials: 40              # resumable: raising it continues the study
  seed: 0
  eval_prompts: 24        # search split, per side
  holdout_prompts: 32     # sealed, certificate only (>= 8)
  max_new_tokens: 48
  kl_tokens: 8
  batch_size: 8
  harmful_pool: builtin:ara_bad    # disjoint from the probe sets
  benign_pool: builtin:ara_good
  max_kl: 0.5             # optional selection cap
  space:
    alpha: [0.25, 1.2]
    components: [attn, mlp, both]
    direction_modes: [own, best, blend]
    readout: [false, true]
    blend_k: [2, 5]
```

Selection policy: among Pareto points with no degenerate outputs, benign
refusal within `gates.benign_floor_delta` of the base, and KL under `max_kl`,
pick the lowest harmful refusal (ties → lowest KL). If none is feasible, the
lowest-refusal front point is reported as such and the publish gates judge it.
