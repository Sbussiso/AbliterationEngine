#!/bin/bash
# r2-variant arm: load weights from the VM's verified variant dir (sha 0b2133342dce215d),
# with config/tokenizer from the local variant dir itself.
set -x
cd /root/research/abliteration/qwen2.5-0.5b-004/consolidation
V=/root/research/venvs/ablate/bin/python

$V -m lm_eval --model hf \
  --model_args pretrained=/root/research/abliteration/qwen2.5-0.5b-004/variant_wd_ML_BN,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size auto --seed 0 \
  --output_path tq_r2 2>&1 | tail -8

echo TQ2_DONE
