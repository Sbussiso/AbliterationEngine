#!/bin/bash
# TruthfulQA mc2 v4 (final): base @ pinned rev vs r2-variant wd_ML_BN (local dir).
# fp32 (native fast CPU path) + batch 8 + 8 threads, bounded RSS ~3.4 GB
# (earlyoom SIGTERMed the v1 fp32/auto attempt at 8.3 GB RSS, journal 13:18).
# Identical config both sides. Runs both legs; TQ_DONE marker at end.
set -x
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /root/research/abliteration/qwen2.5-0.5b-004/consolidation
V=/root/research/venvs/ablate/bin/python
VAR=/root/research/abliteration/qwen2.5-0.5b-004/variant_wd_ML_BN

$V -m lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,revision=7ae557604adf67be50417f59c2c2f167def9a775,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 8 --seed 0 \
  --output_path tq_base

$V -m lm_eval --model hf \
  --model_args pretrained=$VAR,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 8 --seed 0 \
  --output_path tq_r2

echo TQ_DONE