# abliteration_engine

Config-driven abliteration harness (engine v3): one parameterized engine +
YAML run specs replace per-run copy-adapted scripts — the drift vector this
package exists to kill (`run_003.py` once existed in 3 run dirs, differing
only in ~10 lines of constants).

**What "abliteration" means here:** refusal behavior in aligned open-weight
LLMs is mediated by a readable direction in the residual stream (Arditi et
al. 2024, *"Refusal in LLMs is mediated by a single direction"*). The engine
captures harmful/harmless contrast activations, scores every layer × readout
position for direction coherence, and applies the selected refusal direction
as either a runtime hook or a persistent weight edit (attention `o_proj` +
MLP `down_proj` orthogonalization) — producing uncensored model variants with
auditable, reproducible provenance instead of one-off scripts.

- **CI:** [![CI](https://github.com/Sbussiso/abliteration/actions/workflows/ci.yml/badge.svg)](https://github.com/Sbussiso/abliteration/actions/workflows/ci.yml)
- **License:** MIT · **Published models:** [sbussiso/Qwen2.5-0.5B-abliterated](https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated), [sbussiso/Qwen2.5-7B-abliterated](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated)

## How it works

A run spec (YAML) is the **only** run-specific input the engine reads. The
pipeline, driven by that spec:

1. **`plan`** — print the full stage plan from a spec, no model load (CPU-safe).
2. **`run`** (GPU) — load the pinned patient, capture all-layer final-position
   residuals over the harmful/harmless probe sets, coherence-scan every
   layer × position, select the strongest refusal direction, probe
   baseline vs hook-ablated behavior.
3. **`ladder`** (GPU) — build the persistent-edit variants (`wd_B`, `wd_BN`,
   `wd_ML`, `wd_ML_BN`; k-layer primaries and combos), each with
   edit → save → reload → on-disk verify → probe. **Banked resume:** variants
   with complete probe files from a prior session are reused instead of
   recomputed (registry-drop survival).
4. **`mmlu`** (GPU) — guardrail: identical lm-eval config both sides
   (full 57-subject MMLU, 0-shot, fp16, seed 0), hard gate on the
   base→variant knowledge loss (default ≤ 3.0pp).
5. **`publish`** (CPU, runs locally) — hard-asserts ALL gates before any
   Hugging Face push: selection gate, benign-preservation floor,
   degenerate-output ceiling, MMLU loss bound, on-disk weight-delta bounds,
   card marker, license. Model card numbers are built from artifacts —
   never hand-typed.
6. **`parity`** — diff a v3 run's artifacts against a known-good baseline
   (Run 001): exact for deterministic JSON metrics, cosine+L1 tolerances for
   fp16-nondeterministic tensors. A parity pass is exit 1 with
   `parity_ok: false` — never a fake pass.
7. **`bundle`** — package the frozen engine + spec + runner into a tarball
   with a per-file sha256 manifest for Colab GPU stages (`uv sync --frozen`
   inside a `--system-site-packages` venv, so CI-validated versions meet
   the host's CUDA torch with no env drift).

### Provenance and safety rails

- Every artifact dir carries `run_config.json` with the normalized spec +
  `spec_sha256` — artifacts are bound to the exact effective config.
- Specs pin the patient's HF revision (40-char sha or `v`-tag only;
  anything else is a load-time error).
- Every GPU-spending verb requires an explicit `--i-know-this-spends-quota`;
  publishing additionally requires `--i-know-this-publishes`. HITL
  checkpoints are declarable in the spec (`hitl:` block) and are enforced
  by the runner contract (tests lock this).
- Scoring v2: on top of the frozen v1 refusal stat (cross-run comparable
  substring match), a post-hoc re-grade layer separates true refusals from
  apology-preamble compliance — the flat matcher's false positives are
  measured, and both classes are reported in probe files
  (`refused`, `degenerate`, per-row generation times).

## Verified results (primary64 probe sets, 64 harmful + 64 harmless; greedy, seeded)

| Patient (run) | Baseline harmful refusal | Edited harmful refusal | Benign over-refusal | Selection gate | MMLU gate |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (Run 001, published) | 14/16 probes (87.5%) | 0/16 (0%) | 1/16 held benign floor (93.75% preserved) | `wd_B` passed | passed |
| Qwen2.5-7B-Instruct (Run 003, published) | 60/64 (93.75%) | 8/64 (12.5%) | 64/64 → 64/64 | `wd_ML` passed | passed |
| Qwen2.5-1.5B-Instruct (Run 002) | 63/64 (98.4%) | 0/64 true refusals post-hook | 3/64 → 0/64 (over-refusal also removed) | in progress | in progress |

Numbers come straight from committed artifacts (`probes_*.json`,
`selection.json`); scoring-v2 re-grade classified every flagged row
(see `qwen2.5-*/` dirs and `harness_v3/FREEZE_HANDOFF.md` for details).
Run 002 final numbers will be updated when its selection + MMLU close.

**Known limitation (documented on purpose):** single-direction ablation
assumes the convenient "refusal lives in one direction" structure. Larger
models can implement refusal redundantly across parallel directions, in
which case a single-direction edit under-removes (visible in the 7B row:
persistent edit needed where the 1.5B hook sufficed). The ladder +
selection gate exists to handle exactly this — the engine never assumes,
it measures.

## Install

```bash
git clone https://github.com/Sbussiso/abliteration.git
cd abliteration
uv sync --extra dev            # CPU-only runtime + dev tools
uv sync --extra gpu            # + CUDA torch/transformers (Colab/GPU hosts)
```

Requires Python ≥ 3.11. CPU-only environments can plan, validate, parity,
and bundle; GPU stages need a CUDA host.

## Usage

```bash
abliterate plan    --spec specs/run001_parity.yaml      # stage plan, no model load
abliterate validate --spec specs/qwen25_7b.yaml        # fail-fast spec check
abliterate run     --spec <spec> --i-know-this-spends-quota   # stage A (GPU)
abliterate ladder  --spec <spec> --i-know-this-spends-quota   # persistent edits (GPU)
abliterate mmlu    --spec <spec> --i-know-this-spends-quota   # guardrail (GPU)
abliterate publish --spec <spec> --variant-dir <dir> --mmlu mmlu_summary.json \
                   --i-know-this-publishes                     # gates + HF push
abliterate parity  --spec <spec> --baseline <v2 dir> --run-dir <v3 dir>
abliterate bundle  --spec <spec> --out-dir bundles             # Colab tarball
```

`--spec` works on the parent parser and every subparser (shared-flag
clobbering is handled); flag order is flexible.

### Writing your own spec

Start from `specs/run001_parity.yaml` — the normalized schema is
`spec_version, run_card, patient, probe_sets, decoding, ladder, gates,
publish, hitl, colab` (validation is fail-fast; the loader hard-asserts
pinned revisions, probe-count sanity, gate ordering, and ladder subsets).
Built-in probe sets (`builtin:primary64_harmful`/`_harmless`) and refusal
markers (`builtin:fp_explicit_v1`) are carried over verbatim from the
harness v2 runs so configs stay behaviorally identical across versions.

Tests demonstrate every spec-driven contract:
`uv run --no-sync pytest tests/ harness_v3/smoke_test.py -q` (60 tests;
packaging, spec validation, runner/publish contract, parity tolerances,
banked-resume semantics, provenance).

## Repository layout

- `src/abliteration_engine/` — the package: `spec` (yaml load/validate),
  `data` (builtin probe sets/markers), `core` (stage A:
  capture→directions→probes), `edits` (stage B ladder + banked resume),
  `pipeline`/`mmlu`/`publish` (FTT-20 GPU-stage ports), `parity`,
  `scoring_v2` (refusal re-grade), `bundle` (Colab packaging), `cli`.
- `harness_v3/` — freeze artifact: `ftt19_spec.md` (engine surface +
  parity tolerances), `FREEZE_HANDOFF.md`, CPU smoke test, shipped specs,
  and the `eng/` migrate-only shim (aliases only; deleted at end of FTT-20).
- `specs/` — shipped run specs (Run 001 parity baseline, 7B, 1.5B resume).
- `qwen2.5-*/` — per-run research dirs. Run 001 (`qwen2.5-0.5b-002`) is the
  parity baseline: its `artifacts/` (probes, selection — committed on
  purpose) is referenced by the test suite and CI.

### Run-dir index

The `qwen2.5-*` dirs follow `<patient>-<run>`; a bare patient dir is the
pre-numbering "patient zero" mission. Most per-run artifacts are
deliberately *untracked* (models/weights/checkpoints never belong in git —
see `.gitignore`); what's committed is the readable record: READMEs, eval
scripts, JSON summaries, and the few artifact files the test suite
hard-references.

| Dir | What it was | Patient | Outcome |
|---|---|---|---|
| `qwen2.5-0.5b` | Mission 001 "patient zero" (harness v2) | 0.5B | historical seed |
| `qwen2.5-0.5b-002` | **Run 001** + parity baseline + eng-v3 live work (banked pulls s3–s5) | 0.5B | published → [0.5B-abliterated](https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated); `artifacts/` feeds CI |
| `qwen2.5-0.5b-004` | **Run 003** round 2: persistent-edit ladder + TruthfulQA/MMLU consolidation | 0.5B | completed |
| `qwen2.5-0.5b-005` | **Run 004**: RefusalBench-NQ selective-refusal paper (`refusalbench/PAPER.md`) | 0.5B | completed |
| `qwen2.5-0.5b-006` | **Run 006**: v3 path validation — exact recreation of Run 000 | 0.5B | completed (parity green) |
| `qwen2.5-7b-001` | **7B patient** (spec `run_number 5`) | 7B | published → [7B-abliterated](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated) |
| `qwen2.5-1.5b-003` | **Run 002**: 1.5B, harness v3 source work + session pulls | 1.5B | in progress (FTT-13) |

Ops scripts at the repo root (`ftt20_*.sh`, `reapersplit.sh`) are
session-scoped Colab watchers/recovery tooling — session artifacts, not
package code; `src/abliteration_engine/` is the only stable surface.

- `tests/` — packaged-CLI contract tests + Run-001 parity contract.

## Operating notes

- **Colab flow:** `abliterate bundle` → upload tarball → `bash runner.sh`
  (`PHASE=run|ladder|mmlu|publish`) → poll `exit_code.txt` /
  `ENG<STAGE>_DONE {json}` / `<stage>_error.txt` sentinels (written by the
  engine, never the runner — a tested contract) → pull artifacts → verify
  per-file hashes against `bundle_meta.json` before trusting results.
- **Watch-loop ops:** phase-banking + banked-resume are what make session
  drops cheap — three registry drops in one day cost ~8 min re-warmup each,
  with zero banked-work loss (probes reproduced row-identical across
  independent instances; see `qwen2.5-0.5b-002/eng_run002_pull*/`).
- **Never fake a gate:** parity/probe/publish failures are loud exits with
  structured reasons; nothing is auto-converted into a pass.

## Background reading

- Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single
  Direction* (residual-stream refusal direction; the ablation method used here)
- Wei et al. 2023 / the broader safety-fine-tuning literature (why the
  refusal direction exists to begin with)

## Status

Engine v3 is running real patients (three Qwen2.5 sizes), with CI-gated
packaging, a frozen cross-run comparable refusal stat, and two published
models. Open work: Run 002 (1.5B) final sweeps + publish; `harness_v3/eng/`
shim deletion at FTT-20 close; MMLU-phase extension for banked resume of
partially-completed sweeps.