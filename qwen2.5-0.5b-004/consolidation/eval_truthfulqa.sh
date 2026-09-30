#!/bin/bash
# TruthfulQA (MC2, the standard informational-curve task) on base + r2-variant, identical config.
set -x
cd /root/research/abliteration/qwen2.5-0.5b-004/consolidation
V=/root/research/venvs/ablate/bin/python
CACHE=/root/.cache/huggingface/hub

$V -m lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,revision=7ae557604adf67be50417f59c2c2f167def9a775,dtype=float32 \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size auto --seed 0 \
  --output_path tq_base 2>&1 | tail -20

$V -m lm_eval --model hf \
  --model_args pretrained=$CACHE/models--sbussiso--Qwen2.5-0.5B-abliterated-r2/snapshots/6117fb79bf7520674ed9eed71a5cba43eef50443,dtype=float32,trust_remote_code=False \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size auto --seed 0 \
  --output_path tq_r2 2>&1 | tail -20

echo TQ_DONE
