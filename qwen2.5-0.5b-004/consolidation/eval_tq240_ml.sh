#!/bin/bash
# Consolidation evals v6: TQ240 + multilingual panel, sequential in one job.
# TQ: paired 240-question subsample (--limit 240 = first 240 docs, identical
# across arms) with --log_samples so a paired bootstrap CI on the mc2 delta
# is computable. fp32 (native fast CPU path) + batch 8 + 8 threads, RSS ~3.7GB.
# History: v1 fp32/auto earlyoom-killed (RSS 8.3GB); v2 bf16 ~12h ETA (no
# AVX512-BF16); v4/v5 full-set = ~2h per leg (too slow) -> v6 subsample.
# flock guard hard-prevents concurrent instances (13:29 incident).
set -u
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec 9>/root/research/abliteration/qwen2.5-0.5b-004/consolidation/tq240.lock
flock -n 9 || { echo "ANOTHER EVAL IS RUNNING - ABORT"; exit 99; }
cd /root/research/abliteration/qwen2.5-0.5b-004/consolidation
V=/root/research/venvs/ablate/bin/python
VAR=/root/research/abliteration/qwen2.5-0.5b-004/variant_wd_ML_BN

echo "=== TQ240 BASE LEG START $(date -Is) ==="
$V -m lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,revision=7ae557604adf67be50417f59c2c2f167def9a775,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 8 --seed 0 --limit 240 --log_samples \
  --output_path tq240_base
echo "=== TQ240 BASE LEG EXIT $? $(date -Is) ==="

echo "=== TQ240 VARIANT LEG START $(date -Is) ==="
$V -m lm_eval --model hf \
  --model_args pretrained=$VAR,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 8 --seed 0 --limit 240 --log_samples \
  --output_path tq240_r2
echo "=== TQ240 VARIANT LEG EXIT $? $(date -Is) ==="

echo "=== MULTILINGUAL START $(date -Is) ==="
$V probes_multilingual.py
echo "=== MULTILINGUAL EXIT $? $(date -Is) ==="

echo EVALS_DONE