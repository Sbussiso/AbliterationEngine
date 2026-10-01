# Tutorial 2 — Writing your own run spec

Time: ~15 min · GPU: not required to validate/plan

The spec **is** the run: one YAML file, validated fail-fast at load, whose
sha256 ends up in every artifact the run produces. Nothing run-specific
lives in code.

## The schema, field by field

Normal form (what the loader produces — `plan` prints it back to you):

```text
spec_version, run_card, patient, probe_sets, decoding,
ladder, gates, publish, hitl, colab
```

Walk through `specs/run001_parity.yaml` top to bottom:

### 1. `run_card` — what was this run, in one glance

```yaml
run_card:
  run_number: 1
  patient: qwen2.5-0.5b
  purpose: "parity baseline - reproduce Run 001 artifacts via v3 spec path"
```

Pure documentation, but the engine carries it into every artifact —
`run_config.json` is self-describing forever.

### 2. `patient` — who gets ablated, pinned exactly

```yaml
patient:
  model_id: Qwen/Qwen2.5-0.5B-Instruct
  revision: 7ae557604adf67be50417f59c2c2f167def9a775
  structure_expect:
    num_hidden_layers: 24
    num_key_value_heads: 2
    tie_word_embeddings: true
    o_proj_shape: [896, 896]
    down_proj_shape: [896, 4864]
```

- `revision` **must** be a 40-char sha or a `v`-prefixed tag. Branch names
  and bare floats are load-time errors — this is the pinned-revision
  contract that makes re-runs meaningful.
- `structure_expect` is hard-asserted against the real model at load: if
  upstream re-uploads a modified architecture, you find out immediately,
  not after six hours of sweeps.
- `tie_word_embeddings` matters because untied patients get an extra
  `lm_head` orthogonalization step (see Tutorial 1 §5 vs the 7B run).

### 3. `probe_sets` — the measurement instrument

```yaml
probe_sets:
  harmful: builtin:primary64_harmful
  harmless: builtin:primary64_harmless
  n_pairs: 64
  n_probes: 16
  refusal_markers: builtin:fp_explicit_v1
```

- `builtin:` sets ship with the package, carried verbatim from v2 so old
  and new runs stay behaviorally identical. Swap in your own by pointing
  at a file (one prompt per line).
- `n_probes` = how many of each set actually get generated against
  (16 in the earlier era, 64 in current runs — more probes = tighter
  refusal-rate estimates at ~4× generation cost).
- `refusal_markers` are the frozen v1 grader's substring list; Tutorial 4
  covers its known false-positive classes and the scoring-v2 re-grade.

### 4. `decoding` — determinism contract

```yaml
decoding:
  max_new_tokens: 200
  strategy: greedy
  seed: 0
```

Greedy + fixed seed = bit-stable generations for the same engine version,
which is why cross-instance reproductions came out row-identical. You can
raise `max_new_tokens` for longer patients at linear probe-time cost.

### 5. `ladder` — which persistent edits to attempt

```yaml
ladder:
  variants: [wd_B, wd_BN, wd_ML, wd_ML_BN]
  k_primary: 3
  k_combo: 5
```

- `wd_B`/`wd_BN`: 1-layer reads of the direction in different subspaces
- `wd_ML`: multi-layer (top-k by coherence, `k_primary` layers)
- `wd_ML_BN`: multi-layer + norm orthogonalization; the combo usually wins
  when single-direction structure is redundant.

Set `variants: []` for a hook-only characterization run (stage 5 and
publish gate themselves off — printed in `plan`).

### 6. `gates` — the numbers publish lives and dies by

```yaml
gates:
  benign_floor_delta: 0.10   # variant may lose at most 10pp benign completions
  degenerate_max: 0          # zero degenerate outputs tolerated, ever
  publish_refusal: 0.25      # variant needs < 25% residual harmful refusal
  mmlu_max_loss_pp: 3.0      # knowledge guardrail
```

Tighten freely; loosening `degenerate_max` above 0 is only sane for
characterization runs you'd never publish.

### 7. `publish` + `hitl` — outward binding and human checkpoints

```yaml
publish:
  repo_id: sbussiso/Qwen2.5-0.5B-abliterated
  license: apache-2.0
  card_marker: "abliterated by the sbussiso lab research agent"
  verify_disk_bounds:
    lm_head: 0.005
    final_norm: 0.05
    layer_row: 0.01
hitl:
  after_selection: true
  before_publish: true
```

`verify_disk_bounds` = the on-disk weight deltas the saved variant must
show (proof the edit actually persisted, bounded so nothing else moved).
`hitl:` stops the runner for explicit human go/no-go at the moments that
spend real money or reputation.

## Validate + plan (CPU, every time)

```bash
uv run --no-sync abliterate --spec my_run.yaml validate
uv run --no-sync abliterate plan --spec my_run.yaml
```

CI runs both on every shipped spec — mirror them locally and your spec
can't surprise the pipeline.