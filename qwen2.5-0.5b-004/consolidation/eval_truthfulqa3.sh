#!/bin/bash
# TruthfulQA mc2 v3: fp32 + bounded batch 8. v2 (bf16) projected ~12h: this CPU
# has no AVX512-BF16, so bf16 runs emulated; fp32 matmul is the native fast path.
# The original fp32 failure was batch_size=auto ramping RSS to 8.3 GB -> earlyoom
# SIGTERM; batch 8 keeps RSS ~2-3 GB. Identical config both sides.
set -x
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /root/research/abliteration/qwen2.5-0.5b-004/consolidation
V=/root/research/venvs/ablate/bin/python
VAR=/root/research/abliteration/qwen2.5-0.5b-004/variant_wd_ML_BN

$V -m lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,revision=7ae557604adf67be50417f59c2c2f167def9a775,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 16 --seed 0 \
  --output_path tq_base

$V -m lm_eval --model hf \
  --model_args pretrained=$VAR,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 16 --seed 0 \
  --output_path tq_r2

echo TQ_DONE
