"""abliteration_engine — config-driven abliteration harness (eng v3).

One parameterized pipeline replacing the copy-adapted per-run scripts
(run_003.py / ladder_003.py / mmlu_eval_003.py / publish_003.py).
Everything that varied between runs 000-005 is data in a YAML run spec
(abliteration_engine/spec.py); everything that stayed constant lives here
as code.

Method (unchanged): Arditi et al. 2024, "Refusal in LLMs is mediated by a
single direction" (NeurIPS 2024). Extraction via output_hidden_states=True
(all layers in one forward pass per batch); inference-time ablation via a
plain PyTorch forward hook; persistent weight edits live in edits.py
(stage B ladder).

Sentinels (constant across all runs, pollable):
  $ENG_OUT/exit_code.txt        "running" | "0" | "<err code>"
  stdout final line             ENG<STAGE>_DONE {json}
  $ENG_OUT/<stage>_error.txt    traceback tail on stage failure
"""
__version__ = "0.1.0"

__all__ = ["__version__"]