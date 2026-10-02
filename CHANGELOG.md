# Changelog

Run-by-run record of the abliteration program. Directory names encode the
**mission** that produced them (`qwen2.5-<patient>-<NNN>` where NNN is the
mission id); the authoritative run record inside each directory carries its
own `run_number` — the index below is the reconciliation, so nobody has to
reverse-engineer the mapping again.

| Date | Mission | Dir | Run | Patient | Outcome |
|---|---|---|---|---|---|
| 2026-09-26 | 001 | `qwen2.5-0.5b` | 000 harness v2 | 0.5B | patient zero; seeds the program |
| 2026-09-26/27 | 003 | `qwen2.5-0.5b-002` | **Run 001** (FTT-11) | 0.5B | **published** [`Qwen2.5-0.5B-abliterated`](https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated) (gentle edit at rev `0155cad`; r2 card replaced it 09-28) |
| 2026-09-27/28 | 005 | `qwen2.5-0.5b-004` | **Run 003** r2 (FTT-14) | 0.5B | persistent ladder reaches **0.0%** refusal (`wd_ML_BN`), published card rev on HF `main` |
| 2026-09-28/29 | — | `qwen2.5-0.5b-005` | **Run 004** (FTT-16) | 0.5B | refusalbench selective-refusal study ([paper](qwen2.5-0.5b-005/refusalbench/PAPER.md)) |
| 2026-09-27…10-01 | 004 | `qwen2.5-1.5b-003` | **Run 002** (FTT-13) | 1.5B | ladder reaches 34.4% — publish gated off by design ([paper](papers/ftt13_run002_LINEARBODY.md)) |
| 2026-09-29/30 | — | `qwen2.5-7b-001` | 7B (spec `run_number 5`) | 7B | **published** [`Qwen2.5-7B-abliterated`](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated) (persistent 3-layer edit, 12.5% refusal) |
| 2026-09-30/10-01 | — | `qwen2.5-0.5b-006` | **Run 006** (v3 validation, recreates Run 000) | 0.5B | engine-v3 end-to-end path check |

Notes:
- Two HF artifacts exist: the two published models above (0.5B card ships
  charts + evidence; 7B card links an evidence dataset).
- The 1.5B verdict: inference-time hook reaches ~0% refusal, but no
  persistent weight edit meets the ≤25% publish bar at this scale —
  the deliverable is the [run paper](papers/ftt13_run002_LINEARBODY.md),
  not a model.
