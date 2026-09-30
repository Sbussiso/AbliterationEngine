# abliteration engine v3 (eng)

Config-driven abliteration harness — FTT-18/FTT-19 spike.
One parameterized engine + YAML run specs replace the per-run copy-adapted
scripts (the drift vector: run_003.py exists in 3 run dirs, 414-416 lines
each, differing only in ~10 lines of constants).

## Layout

- `ftt19_spec.md`    engine surface + YAML schema + parity gate (freeze candidate)
- `FREEZE_HANDOFF.md` freeze decisions + Run 001 parity numbers (both sides)
- `specs/run001_parity.yaml`  first concrete run spec (Run 001 baseline config)
- `specs/qwen25_7b.yaml`      published 7B patient as a v3 spec
- `src/abliteration_engine/`  THE PACKAGE (real code now lives here —
  installable, uv-managed; dev-workstation packaging PR)
- `eng/`             migrate-only shim -> src package (FTT-20; alias
  modules, no logic; deletes when FTT-20 ends)
- `smoke_test.py`    CPU-only, 5 subtests (pin-hardening GREEN post-PR),
  all passing

## Verified state (2026-09-29, packaging PR)
smoke_test.py: 5/5 PASS on VM 151 CPU (spec validation, branch-pin
rejection, builtin sets, CLI plan no-model-load, pin-hardening
regression). Full pytest: 14 passed (packaging CLI contract, Run-001
parity contract, smoke). Installed-CLI verified via `uv sync` +
`abliterate plan` for all shipped specs.
Verified drift evidence: run_003.py copies in qwen2.5-0.5b-004 and
qwen2.5-7b-001 differ from qwen2.5-1.5b-003 only in MODEL_ID/REVISION/
structure asserts (diff on record).

## Next (FTT-20)
1. eng/pipeline.py + eng/mmlu.py + eng/publish.py (GPU stages),
2. parity harness run vs qwen2.5-0.5b-002 artifacts on Colab,
3. delete per-run scripts after parity passes,
4. then run 002 resume via v3 (FTT-21).
