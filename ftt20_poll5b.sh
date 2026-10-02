#!/usr/bin/env bash
# Watcher v5 — completion = RUN_DONE marker in engine output; phase pull
# triggers = new artifact files in the engine out dir; vanish alarm kept.
# 24 x 8 min cap. Session-4.
set -u
S=ftt20-run002-research-workstation-4
LOG=/root/research/abliteration/ftt20_watch4.log
OUT=/root/research/abliteration/qwen2.5-0.5b-002/eng_run002_pull_s4
mkdir -p "$OUT"
NTFY_URL=http://10.0.0.119:8090/research
NTFY_TOKEN=$(cat /root/research/.ntfy_token)
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
notify() { curl -s -u ":$NTFY_TOKEN" -H "Title: $1" "$NTFY_URL" -d "$2" > /dev/null; }

PULL() { # $1 = engine file name
  F="$1"
  sudo -u sbussiso /home/sbussiso/.local/bin/colab download -s "$S" \
    "/content/eng_run_002_qwen2.5-1.5b/$F" "/tmp/wp4_$F" >>/tmp/wp4_dl.log 2>&1 \
  && [ -s "/tmp/wp4_$F" ] && cp "/tmp/wp4_$F" "$OUT/" && log "pulled $F"
}

KNOWN=$(ls "$OUT" 2>/dev/null || true)
log "watch5 start (known: $KNOWN)"
for i in $(seq 1 24); do
  sleep 480
  STATE=$(printf 'import os, json, re\nab=os.popen("ps -ef | grep abliterate | grep -v grep").read().strip()\nE="/content/eng_run_002_qwen2.5-1.5b"\nfiles={}\nif os.path.isdir(E):\n    files={f: os.path.getsize(os.path.join(E,f)) for f in os.listdir(E)}\nout=open("/content/runner_out.log").read()[-200:] if os.path.exists("/content/runner_out.log") else ""\ndone_=os.path.exists("/content/RUN_DONE.json") or "RUN_DONE" in out\nmm=re.search(r"\\[([a-zA-Z0-9_-]+)-(harm|harmless)\\s+(\\d+)/\\d+\\]",out)\nprint("ALIVE:","yes" if ab else "no")\nprint("FILES:", json.dumps(files))\nprint("DONEMARK:", "yes" if done_ else "no")\nprint("LASTPROBE:", mm.group(0)[1:44] if mm else "none")\nprint("NOSUCH:", "yes" if "not found" in out else "no")\n' | timeout 150 sudo -u sbussiso /home/sbussiso/.local/bin/colab exec -s "$S" --timeout 60 2>&1 | grep -E '^(ALIVE|FILES|DONEMARK|LASTPROBE|NOSUCH)' | head -5)
  echo "$STATE" | tail -5 >> "$LOG"

  if printf '%s' "$STATE" | grep -q '^NOSUCH: yes\|not found'; then
    log "SESSION GONE (iter $i)"
    notify "Run 002 resume: SESSION-4 REGISTRY-DROP" "DECISION RULE ENGAGED per FTT-13 commit 78906b15: no auto-relaunch — run-strategy decision goes to the user (quota check / L4-A100 / phase-split). Pulled artifacts git-tracked."
    exit 1
  fi

  # per-phase artifact pulls
  NEW=$(printf '%s' "$STATE" | grep '^FILES:' | sed 's/^FILES: //')
  python3 - <<EOF
known_now=set(open("$LOG").read().split())
EOF
  for f in probes_hook_ablated.json run_config.json RUN_DONE.json selection.json summary.json mmlu_summary.json; do
    if printf '%s' "$STATE" | grep '"$f"' >/dev/null 2>&1; then :; fi
  done
  # simpler: try each expected file name via FILES json
  for f in probes_hook_ablated.json run_config.json RUN_DONE.json selection.json mmlu_summary.json run002_log.txt; do
    if printf '%s' "$NEW" | grep -q "\"$f\":"; then
      [ -f "$OUT/$f" ] || PULL "$f"
    fi
  done

  ALIVE=$(printf '%s' "$STATE" | grep '^ALIVE:' | cut -d' ' -f2)
  DONE=$(printf '%s' "$STATE" | grep '^DONEMARK:' | cut -d' ' -f2)

  if [ "$DONE" = "yes" ]; then
    for f in RUN_DONE.json selection.json summary.json probes_hook_ablated.json run_config.json mmlu_summary.json; do
      PULL "$f" 2>/dev/null || true
    done
    # full engine dir archive sweep for anything not yet local
    printf 'import shutil\nshutil.make_archive("/content/eng_final","tar",root_dir="/content/eng_run_002_qwen2.5-1.5b")\nprint("ARCHIVED")\n' | timeout 150 sudo -u sbussiso /home/sbussiso/.local/bin/colab exec -s "$S" --timeout 60 >>/tmp/wp4_dl.log 2>&1 \
      && sudo -u sbussiso /home/sbussiso/.local/bin/colab download -s "$S" /content/eng_final.tar /tmp/wp4_eng_final.tar >>/tmp/wp4_dl.log 2>&1 \
      && cp /tmp/wp4_eng_final.tar "$OUT/" && log "final archive pulled"
    log "RUN DONE (DONEMARK) iter $i"
    notify "Run 002 resume: RUN COMPLETE" "Runner finished (session-5 ALIVE for pull + HITL). Engine archive + artifacts pulled to VM; re-grade + selection summary on Linear FTT-13 next."
    exit 0
  fi

  if [ "$ALIVE" = "no" ]; then
    log "RUNNER GONE without DONEMARK (iter $i)"
    notify "Run 002 resume: runner gone, no completion marker" "Inspect manually; artifacts pulled so far are tracked."
    exit 1
  fi
  log "iter $i: alive, no new boundary"
done
notify "Run 002 resume: watch5 capped" "24x8min elapsed — manual check."
exit 2