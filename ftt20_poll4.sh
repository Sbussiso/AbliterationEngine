#!/usr/bin/env bash
# Watcher v4: runner liveness + phase-boundary + completion + vanish alarm.
# 24 x 8 min cap (~3.2h completion cap). Session-4.
set -u
S=ftt20-run002-research-workstation-4
LOG=/root/research/abliteration/ftt20_watch4.log
NTFY_URL=http://10.0.0.119:8090/research
NTFY_TOKEN=$(cat /root/research/.ntfy_token)
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
notify() { curl -s -u ":$NTFY_TOKEN" -H "Title: $1" "$NTFY_URL" -d "$2" > /dev/null; }

LAST_PHASE="none"
log "watch4 start (session $S)"
for i in $(seq 1 24); do
  sleep 480
  STATE=$(printf 'import os\nab=os.popen("ps -ef | grep abliterate | grep -v grep").read().strip()\nsrc=None\nfor p in ["/content/run002_log.txt","/content/runner_out.log"]:\n    if os.path.exists(p) and os.path.getsize(p)>100: src=p\nlog=open(src).read() if src else ""\nimport re\nm6=[l for l in log.splitlines() if re.match(r"^\\[\\d/6\\]",l)]\nm2=re.findall(r"^\\[\\d/2\\][^\\n]*",log,re.M)\nexitv=None\nif os.path.exists("/content/exit_code.txt"):\n    t=open("/content/exit_code.txt").read().strip()\n    if t.isdigit(): exitv=t\nmm=re.search(r"\\[([a-zA-Z0-9_-]+)-(harm|harmless)\\s+(\\d+)/\\d+\\]",log)\nprint("ALIVE:","yes" if ab else "no")\nprint("LASTPH:",(m6[-1] if m6 else (m2[-1] if m2 else "none"))[:80])\nprint("RUNN:",mm.group(0)[1:40] if mm else "none")\nprint("EXITV:",exitv)\n' | timeout 150 sudo -u sbussiso /home/sbussiso/.local/bin/colab exec -s "$S" --timeout 60 2>&1 | grep -E '^(ALIVE|LASTPH|RUNN|EXITV)|not found' | head -5)
  echo "$STATE" | tail -5 >> "$LOG"

  if printf '%s' "$STATE" | grep -q 'not found'; then
    log "SESSION GONE (iter $i)"
    notify "Run 002 resume: SESSION-4 REGISTRY-DROP" "Third drop in the pattern (~90 min pattern-clock). DECISION RULE ENGAGED: no auto-relaunch — bringing run-strategy decision to the user (quota check / L4-A100 / phase-split). Artifacts pulled so far remain git-tracked."
    exit 1
  fi

  ALIVE=$(printf '%s' "$STATE" | grep '^ALIVE:' | cut -d' ' -f2)
  PH=$(printf '%s' "$STATE" | grep '^LASTPH:' | sed 's/^LASTPH: //')
  EX=$(printf '%s' "$STATE" | grep '^EXITV:' | cut -d' ' -f2 | awk '$1 ~ /^[0-9]+$/ {print $1}')

  if [ -n "$EX" ]; then
    log "RUN DONE exit=$EX (iter $i)"
    notify "Run 002 resume: RUN DONE (exit=$EX)" "Runner finished (session LEFT ALIVE for artifact pull + HITL). Pulling artifacts now; details on Linear FTT-13."
    exit 0
  fi

  if [ "$ALIVE" = "no" ]; then
    log "RUNNER GONE without exit sentinel (iter $i)"
    notify "Run 002 resume: runner gone, no sentinel" "Runner process vanished without exit_code.txt (session still registered). Inspect kernel/output manually."
    exit 1
  fi

  if [ -n "$PH" ] && [ "$PH" != "$LAST_PHASE" ]; then
    log "boundary: $PH"
    notify "Run 002 resume: $PH" "Phase advanced on session-4."
    LAST_PHASE="$PH"
  else
    log "iter $i: alive=yes phase unchanged"
  fi
done
notify "Run 002 resume: watch4 capped" "24x8min elapsed without completion — manual check."
exit 2