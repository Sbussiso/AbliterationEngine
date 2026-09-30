#!/usr/bin/env python3
"""eng — configurable abliteration engine v3 (FTT-18/FTT-19 spike).

One parameterized pipeline replacing the copy-adapted per-run scripts
(run_003.py / ladder_003.py / mmlu_eval_003.py / publish_003.py).
Everything that varied between runs 000-005 is data in a YAML run spec
(eng/spec.py); everything that stayed constant lives here as code.

Method (unchanged): Arditi et al. 2024, "Refusal in LLMs is mediated by a
single direction" (NeurIPS 2024). Extraction via output_hidden_states=True
(all layers in one forward pass per batch); inference-time ablation via a
plain PyTorch forward hook; persistent weight edits live in eng/edits.py
(stage B ladder, FTT-20).

Sentinels (constant across all runs, pollable):
  $ENG_OUT/exit_code.txt        "running" | "0" | "<err code>"
  stdout final line             ENG<STAGE>_DONE {json}
  $ENG_OUT/<stage>_error.txt    traceback tail on stage failure
"""