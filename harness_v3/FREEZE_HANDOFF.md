# Freeze-handoff notes — FTT-19 (research-workstation → dev-workstation, 2026-09-29)

## What research-workstation added since the original spec
1. `specs/qwen25_7b.yaml` — third real patient as v3 spec (7B constants from
   the actual run: 28 layers, GQA 28q/4kv, UNTIED, o_proj [3584,3584],
   down_proj [3584,18944], L4 GPU, repo sbussiso/Qwen2.5-7B-abliterated).
   `plan` + `validate` both pass on it (CPU).
2. `smoke_test.py::test_revision_pin_hardening` — RED regression test for BOTH
   of your pre-freeze bugs, verified against current spec.py behavior:
   - branch/alias pins `dev`/`research`/`canary` are silently ACCEPTED (load
     returns spec with revision='dev', no error)
   - int revision 12345 raises raw AttributeError ('int' object has no attribute
     'startswith') instead of SpecError
   This test is intentionally failing and stays in the tree; it goes GREEN
   with your packaging-PR patches to spec.py. Do not delete it.

## Dev freeze decisions — accepted from research side
1. src-layout `abliteration_engine` + migrate-only `eng` shim: accepted, with
   one scope addition — the shim should live ONLY for FTT-20 GPU-port
   (eng.edits imports are already relative inside eng/, so the shim is
   src-side alias modules, no logic).
2. dataclasses + hand-rolled checks over pydantic: accepted (fail-fast spec
   errors already read better than what pydantic would give; zero new deps).
3. Verb set as proposed + subparser flag-order fix: accepted.
4. Parity: eng subcommand interface / pytest implementation under
   tests/parity/: accepted, matches the §5 tolerances in ftt19_spec.md
   (cos>=0.999, L1<=0.01 for fp16 directions; exact for deterministic JSON).

## Known-good baseline for the parity test (Run 001, from
/root/research/abliteration/qwen2.5-0.5b-002/artifacts)
- baseline: refusal 0.875, benign_preserved 0.9375
- hook-ablated: refusal 0.0
These are the exact values dev confirmed parity reads today.

## Two contract notes for the packaging PR
- smoke_test.py in-repo MUST keep `PYTHONPATH=<repo-root-or-here>` import
  story (subprocess env already set correctly in test_cli_plan_cpu_only).
- The `--i-know-this-spends-quota` HITL flag is a hard contract: keep
  semantics identical in the packaged CLI.

## Still research-side (not in the packaging PR)
- ftt-20 GPU-stage ports (eng/pipeline.py, eng/mmlu.py, eng/publish.py) —
  exercised on Colab after the package shape lands.
