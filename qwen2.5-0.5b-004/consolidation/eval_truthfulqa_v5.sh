#!/bin/bash
# TruthfulQA mc2 v5 (final): base @ pinned rev vs r2-variant wd_ML_BN (local dir).
# fp32 (native fast CPU path) + batch 8 + 8 threads, bounded RSS ~3.5-4 GB.
# History: v1 fp32/auto died (earlyoom, RSS 8.3GB); v2 bf16 ~12h ETA (no
# AVX512-BF16 on this CPU); v4 batch-16 died when TWO evals ran concurrently
# (13:29 earlyoom). flock guard now hard-prevents concurrent instances.
set -x
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec 9>/root/research/abliteration/qwen2.5-0.5b-004/consolidation/tq.lock
flock -n 9 || { echo "ANOTHER TQ EVAL IS RUNNING - ABORT"; exit 99; }
cd /root/research/abliteration/qwen2.5-0.5b-004/consolidation
V=/root/research/venvs/ablate/bin/python
VAR=/root/research/abliteration/qwen2.5-0.5b-004/variant_wd_ML_BN

echo "=== BASE LEG START $(date -Is) ==="
$V -m lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,revision=7ae557604adf67be50417f59c2c2f167def9a775,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 8 --seed 0 \
  --output_path tq_base
echo "=== BASE LEG EXIT $? $(date -Is) ==="

echo "=== VARIANT LEG START $(date -Is) ==="
$V -m lm_eval --model hf \
  --model_args pretrained=$VAR,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 8 --seed 0 \
  --output_path tq_r2
echo "=== VARIANT LEG EXIT $? $(date -Is) ==="

echo TQ_DONE