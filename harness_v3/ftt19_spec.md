# abliteration engine v3 — spec (FTT-19, freeze candidate)

Status: **DRAFT v0.1 — freeze blocked on @dev-workstation review of split**
Derived from: harness v2 at `/root/research/abliteration/runs/qwen2.5-1.5b-003/harness/`
(414–416-line scripts identical across patients except ~10 lines of constants —
drift verified by diff 2026-09-29: run_003.py copies in runs/qwen2.5-0.5b-004 and
runs/qwen2.5-7b-001 differ ONLY in MODEL_ID, REVISION, and structure asserts).

## 0. Design principle

The engine is a parameterized pipeline; the YAML run spec carries everything
that varied between runs 000–005. Nothing outside `configs/` is run-specific.

## 1. Engine surface (eng/core.py — the 6 stages)

Stage order fixed; every stage is callable standalone for re-runs:

| # | stage | from v2 | inputs | outputs |
|---|-------|---------|--------|---------|
| 1 | `load_patient` | run_003.py: load_model+structure_report | `patient` block | model, tok, structure report; asserts spec structure_expect vs report |
| 2 | `capture` | capture_final_residuals | probe_sets, n_pairs | captures_harm/captures_harmless npz (all layers) |
| 3 | `directions` | coherence_stats+scan_layers | captures | layer_coherence.json, dir_A (residual), dir_B (readout), layer_directions.npz |
| 4 | `probe` | run_probes (baseline + hook) | decoding, markers | probes_baseline.json, probes_hook_ablated.json |
| 5 | `ladder` | ladder_003.py edit/verify fns | ladder block | run_variant loop: edit→save→reload→verify→probe per variant; selection.json |
| 6 | `publish` | publish_003.py | publish block + mmlu block | gates→card gen→hub upload→post-hub verify |

MMLU (`mmlu_eval_003.py`) = stage 5.5, standalone-callable, runs base AND
variant with identical lm-eval config from spec; emits mmlu_summary.json with
`guardrail_3pp`.

## 2. YAML run spec schema (eng/spec.py — dataclass or pydantic at CLI's choice)

```yaml
spec_version: 1
run_card:                      # naming + provenance
  run_number: 6                # canonical lab numbering (FTT-12=000 ...)
  patient: qwen2.5-1b          # slug for dirs/issues
  purpose: free-text one-liner into run_config.json

patient:
  model_id: Qwen/Qwen2.5-1.5B-Instruct
  revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306   # PINNED, mandatory
  structure_expect:            # asserted at load, FAIL FAST on drift
    num_hidden_layers: 28
    num_key_value_heads: 2
    tie_word_embeddings: true
    o_proj_shape: [1536, 1536]
    down_proj_shape: [1536, 8960]

probe_sets:                    # built-ins from eng/data, or file: paths
  harmful: builtin:primary64_harmful
  harmless: builtin:primary64_harmless
  n_pairs: 64                  # direction extraction
  n_probes: 16                 # per condition (harm, benign)
  refusal_markers: builtin:fp_explicit_v1   # constant scorer across runs; may override inline

decoding:
  max_new_tokens: 200
  strategy: greedy             # do_sample=False; sampling = future
  seed: 0

ladder:
  variants: [wd_B, wd_BN, wd_ML, wd_ML_BN]   # subset allowed (skip via spec)
  k_primary: 3                 # wd_ML top-K layers
  k_combo: 5                   # wd_ML_BN top-K layers

gates:                         # selection + publish, all overridable
  benign_floor_delta: 0.10     # benign_preserved >= baseline - delta
  degenerate_max: 0
  publish_refusal: 0.25
  mmlu_max_loss_pp: 3.0

publish:                       # consumed only by stage 6, local-side
  repo_id: sbussiso/<patient>-abliterated   # single approved target
  license: apache-2.0
  card_marker: "abliterated by the sbussiso lab research agent"
  verify_disk_bounds:          # reload verif bounds from ladder_003.py, now spec'd
    lm_head: 0.005
    final_norm: 0.05
    layer_row: 0.01

colab:                         # stage-1-5 runtime target
  gpu: t4                      # t4|l4|a100
  max_runtime_minutes: 240
  upload_harness: true

hitl:                          # human approval checkpoints — agent pauses
  after_selection: true        # before stage 5.5 MMLU + publish prep
  before_publish: true         # user approves repo_id + card before push
```

## 3. Cross-cutting rules

- **Sentinels constant across all runs**: `/content/exit_code.txt` +
  `<STAGE>_DONE {json}` stdout marker + stage-error files. Machine-readable,
  pollable, exactly as v2.
- **Artifact contract** (unchanged names so Run 001 parity is diffable):
  `run_config.json, layer_coherence.json, captures_*.npz,
  layer_directions.npz, refusal_direction_{,A,B.}npy,
  probes_{baseline,hook_ablated,wd_*}.json, selection.json,
  selection_candidates.json, mmlu_summary.json, harness_sha256.json`.
  `run_config.json` gains a `spec_hash` (sha256 of the normalized YAML) —
  provenance binding for the paper.
- **Determinism**: seeds from spec; greedy decoding; pinned revisions
  mandatory (spec validation rejects missing revision).
- **`--dry-run`**: CLI prints the full stage plan (per stage: what would run,
  inputs/outputs, estimated wall time from past runs) with NO model load.
  Dry-run is also the parity-test vehicle (engine-side, dev-workstation).

## 4. NOT expressible in spec (stays agent-side, hard decisions)

- Colab session provisioning & keepalive (skill colab-unsloth-studio).
- HITL protocol itself (checkpoint booleans live in spec; who answers doesn't).
- Post-run model-card audit protocol (FTT-006-style 53/53 checks).
- Run numbering/paper writing (Linear doctrine).

## 5. Parity gate (FTT-20 merge criterion, hard)

`Run 001` (sbussiso/Qwen2.5-0.5B-abliterated, run dir runs/qwen2.5-0.5b-002) is the
baseline. Parity = for a `run001.yaml` spec execution, artifacts equal within:
  - selection.json: identical selected variant + metrics (exact)
  - probes_*.json: refusal_rate/benign_preserved/degenerate identical (exact;
    greedy + seed 0 makes probe outputs deterministic given same GPU class —
    if a GPU-class nondeterminism appears, declare eps per artifact in parity.yaml)
  - refusal directions: cosine(dir_v2, dir_v3) >= 0.999 and
    L1<0.01 (fp16 nondeterminism headroom — T4 fp16 matmul may vary run to run)

## 6. Open questions for dev-workstation (blocking freeze)

RESOLVED 2026-09-29 (packaging PR, commit 7f0927e; details in
FREEZE_HANDOFF.md):
1. src-layout `src/abliteration_engine/` (name `abliteration_engine`);
   `eng/` = migrate-only shim, deletes at end of FTT-20.
2. dataclass + hand-rolled checks (no pydantic); CI YES —
   .github/workflows/ci.yml runs pytest + ruff + plan/validate for every
   shipped spec on every PR (GitHub wiring pending gh auth).
3. Verb set as proposed; subparser CLI, `--spec` accepted pre- or
   post-verb; `--i-know-this-spends-quota` semantics unchanged.
4. BOTH: parity lives as `abliterate parity` (eng subcommand) with the
   Run-001 contract pinned in `tests/parity/test_run001_contract.py`.

1. Package layout — `eng/` with `eng/{core.py, spec.py, data.py, cli.py}` or src-layout `src/eng/...`?
2. Where spec validation lives (pydantic vs dataclass+manual) and whether CI
   runs `--dry-run` on all shipped configs on every PR (recommend yes).
3. CLI verb set: propose `abliterate plan|run|probe|mmlu|publish --spec ...`.
4. Does the parity harness live in-repo (`tests/parity/`) or as an eng subcommand
   (`abliterate parity --spec run001.yaml --baseline <artifacts>`)? Propose the latter.