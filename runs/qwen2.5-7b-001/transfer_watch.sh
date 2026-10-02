#!/usr/bin/env bash
# Retry-transfer wd_ML weights whenever the colab proxy recovers; logs to /tmp/m7b/transfer.log
S=m006-7b-rw-1
for i in $(seq 1 480); do
  printf 'x' > /tmp/probe.bin
  if sudo -u sbussiso /home/sbussiso/.local/bin/colab upload -s $S /tmp/probe.bin /content/probe.bin 2>&1 | grep -q Uploaded; then
    echo "$(date -u +%FT%TZ) proxy UP; downloading weights" >> /tmp/m7b/transfer.log
    for f in config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja; do
      sudo -u sbussiso /home/sbussiso/.local/bin/colab download -s $S "/content/wd_ML/$f" "/tmp/m7b/wd_ML/$f" 2>&1 | tail -1 >> /tmp/m7b/transfer.log
    done
    if [ -s /tmp/m7b/wd_ML/model.safetensors ]; then
      echo "$(date -u +%FT%TZ) WEIGHTS LANDED" >> /tmp/m7b/transfer.log
      break
    fi
  else
    echo "$(date -u +%FT%TZ) proxy down ($i)" >> /tmp/m7b/transfer.log
  fi
  sleep 30
done
