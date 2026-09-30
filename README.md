# abliteration_engine

Config-driven abliteration harness (FTT-18/FTT-19 spike, eng v3):
one parameterized engine + YAML run specs replace the per-run
copy-adapted scripts (`run_003.py` / `ladder_003.py` / `mmlu_eval_003.py` /
`publish_003.py` — the drift vector: run_003.py existed in 3 run dirs,
differing only in ~10 lines of constants).

## Layout

- `src/abliteration_engine/` — the package (`spec`, `data`, `core`, `edits`,
  `parity`, `pipeline`*, `mmlu`*, `publish`*; * = FTT-20 GPU-stage stubs)
- `harness_v3/` — freeze-review artifact: spec (`ftt19_spec.md`),
  `FREEZE_HANDOFF.md`, CPU smoke test, shipped run specs, and the
  `eng/` **migrate-only shim** (aliases only; deletes at end of FTT-20)
- `specs/` mirror: `harness_v3/specs/{run001_parity,qwen25_7b}.yaml`
  — the three real program patients encodable as v3 specs
- `qwen2.5-*/` — per-run research dirs (Run 001 = qwen2.5-0.5b-002 is the
  parity baseline, its `artifacts/` is referenced by tests)
- `tests/` — packaged-CLI contract tests + Run-001 parity contract

## Usage

```bash
uv sync --extra dev                # CPU-only runtime + dev tools
pytest -q                          # CPU smoke + packaging + parity contract
abliterate plan --spec harness_v3/specs/run001_parity.yaml     # no model load
abliterate --spec harness_v3/specs/qwen25_7b.yaml validate    # flag order flexible
abliterate run --spec ... --i-know-this-spends-quota          # HITL-gated GPU
```

`--i-know-this-spends-quota` is a hard contract: every GPU-spending verb
refuses without it. HITL checkpoints also live in the spec (`hitl:`
block).

## Provenance

Every run's `run_config.json` carries a `spec_hash` (sha256 of the
normalized spec) binding artifacts to the exact effective config.
Freeze decisions: `harness_v3/FREEZE_HANDOFF.md`; engine surface +
parity tolerances: `harness_v3/ftt19_spec.md`.