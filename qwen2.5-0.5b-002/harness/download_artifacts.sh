#!/bin/bash
# Mission 003 artifact download: variant model + evidence trail.
# Usage: download_artifacts.sh /content/wd_A
# Downloads as sbussiso into /tmp/m003/dl (chmod 777), caller copies as root.
set -x
COLAB="sudo -u sbussiso /home/sbussiso/.local/bin/colab"
S=m003-abl-rw-1
VAR="$1"
[ -z "$VAR" ] && { echo "usage: $0 /content/wd_A"; exit 1; }

rm -rf /tmp/m003/dl
mkdir -p /tmp/m003/dl
chmod 777 /tmp/m003/dl

# single tarball with the evidence trail
echo 'import shutil; shutil.make_archive("/content/m003_artifacts", "tar", root_dir="/content", base_dir="abliteration_out"); shutil.make_archive("/content/m003_mmlu", "tar", root_dir="/content", base_dir="mmlu_results"); import shutil as s; s.copy("/content/mmlu_summary.json", "/content/m003_mmlu_summary.json") if __import__("os").path.exists("/content/mmlu_summary.json") else None' | $COLAB exec -s $S --timeout 300

$COLAB download -s $S "$VAR/model.safetensors" /tmp/m003/dl/model.safetensors
$COLAB download -s $S "$VAR/config.json" /tmp/m003/dl/config.json
$COLAB download -s $S "$VAR/generation_config.json" /tmp/m003/dl/generation_config.json
$COLAB download -s $S "$VAR/tokenizer.json" /tmp/m003/dl/tokenizer.json
$COLAB download -s $S "$VAR/tokenizer_config.json" /tmp/m003/dl/tokenizer_config.json
$COLAB download -s $S "$VAR/chat_template.jinja" /tmp/m003/dl/chat_template.jinja
$COLAB download -s $S /content/m003_artifacts.tar /tmp/m003/dl/m003_artifacts.tar
$COLAB download -s $S /content/m003_mmlu.tar /tmp/m003/dl/m003_mmlu.tar
$COLAB download -s $S /content/mmlu_log.txt /tmp/m003/dl/mmlu_log.txt
$COLAB download -s $S /content/run_002.py /tmp/m003/dl/run_002.py
$COLAB download -s $S /content/prompt_sets.py /tmp/m003/dl/prompt_sets.py
$COLAB download -s $S /content/m003_mmlu_summary.json /tmp/m003/dl/mmlu_summary.json

echo "=== downloaded ==="
ls -la /tmp/m003/dl/