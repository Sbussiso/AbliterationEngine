# Tutorials

Hands-on guides for `abliteration_engine`. Every tutorial is written
against the real CLI surface and the specs shipped in this repo — if a
tutorial says a command works, it ran in this repo's history.

| # | Tutorial | Time | Needs GPU |
|---|---|---|---|
| 1 | [Your first ablated model (full mission loop)](01_first_ablated_model.md) | ~40 min | yes (Colab T4) |
| 2 | [Writing your own run spec](02_writing_a_spec.md) | ~15 min | no |
| 3 | [Surviving session kills: banked resume + phase split](03_banked_resume_ops.md) | ~20 min | yes |
| 4 | [Reading probe files + scoring v2](04_probe_files_and_scoring.md) | ~15 min | no |
| 5 | [Publishing: gates, model card, HF push](05_publishing.md) | ~10 min | no |

**Start with #1 if you want to see the whole machine move; #2 if you're
porting a new patient.**

## The 60-second mental model

1. A **run spec** (YAML) names the patient model, the probe sets, the edit
   ladder, and the publish gates.
2. `abliterate run` captures activations, finds the refusal direction,
   and measures refusal behavior before/after a live hook.
3. `abliterate ladder` builds **persistent** weight-edit variants; the
   selection gate picks the best one that preserves benign behavior.
4. `abliterate mmlu` verifies the model didn't get dumber (≤3pp loss).
5. `abliterate publish` re-checks every gate locally and only then
   touches Hugging Face.

Between steps 3 and 5, nothing happens by accident: GPU verbs demand
`--i-know-this-spends-quota`, publishing demands
`--i-know-this-publishes`, and a failed gate exits with a readable
error instead of a shrug.