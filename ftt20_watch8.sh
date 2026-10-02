#!/usr/bin/env bash
# watch8 = watch7 parametrized: USAGE: ftt20_watch8.sh <session-name> [outdir-basename]
# + hard deadline alarm: T+FIT*3/4 (FIT env, default 3600 = declared GPU lease
#   -> 45min; FIT=7200 for CPU sessions -> 90min); 5-min-remaining notice.
# + FTT-27 3-attempt PULL retries w/ honest FAIL logs
# + registry-drop alarm (sessions.json absence) — proven on watch7/s5
# + DONE marker / runner-gone alarms; exits by design after any alarm.
set -u
S="${1:-ftt20-rs2-1}"
OUTBASE="${2:-rs2}"
FIT="${FIT:-3600}"   # per-session lease seconds (declared on the assignment wire)
LOG=/root/research/abliteration/ftt20_watch8_${OUTBASE}.log
OUT=/root/research/abliteration/qwen2.5-0.5b-002/eng_run002_pull_${OUTBASE}
STATE_PY=/tmp/wp6_state.py
mkdir -p "$OUT"
NTFY_URL=http://10.0.0.119:8090/research
NTFY_TOKEN=$(cat /root/research/.ntfy_token)
DEADLINE=$(( $(date +%s) + FIT * 3 / 4 ))
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
notify() { curl -s -u ":$NTFY_TOKEN" -H "Title: $1" "$NTFY_URL" -d "$2" > /dev/null; }
PULL() {
  F="$1"; ATTEMPT=0; OK=0
  while [ "$ATTEMPT" -lt 3 ]; do
    ATTEMPT=$((ATTEMPT+1))
    rm -f "/tmp/wp8_$F"
    sudo -u sbussiso /home/sbussiso/.local/bin/colab download -s "$S" \
      "/content/eng_run_002_qwen2.5-1.5b/$F" "/tmp/wp8_$F" >>/tmp/wp8_dl.log 2>&1
    if [ -s "/tmp/wp8_$F" ]; then
      cp "/tmp/wp8_$F" "$OUT/$F"; log "pulled $F (attempt $ATTEMPT)"; OK=1; break
    fi
    log "pull-FAIL $F attempt $ATTEMPT (empty/missing)"
    sleep 5
  done
  [ "$OK" -eq 0 ] && log "pull-GIVEUP $F after 3 attempts"
  return 0
}
COLAB="sudo -u sbussiso /home/sbussiso/.local/bin/colab exec -s $S --timeout 60 -f $STATE_PY"
log "watch8 start (session $S, FIT=$FIT deadline $((FIT*3/4/60))min from ARM (not session-create; expect +1-2min skew), out=eng_run002_pull_${OUTBASE})"
for i in $(seq 1 44); do
  sleep 240
  NOW=$(date +%s); LEFT=$(( (DEADLINE-NOW)/60 ))
  STATE=$($COLAB 2>&1 | grep -E '^(ALIVE|FILES|LASTPROBE|NOSEEK|DONE|TAIL)' | head -6)
  log "$STATE"
  printf '%s\n' "$STATE" > /tmp/wp6_state_out.txt
  python3 - "$OUT" /tmp/wp6_state_out.txt <<'PYEOF'
import sys, json, os
out = sys.argv[1]
state = open(sys.argv[2]).read()
files = {}
for line in state.splitlines():
    if line.startswith("FILES: "):
        files = json.loads(line[7:])
kf = os.path.join(out, "_sizes.json")
known = json.load(open(kf)) if os.path.exists(kf) else {}
todo = [f for f, sz in files.items() if known.get(f) != sz]
open(kf, "w").write(json.dumps(files))
with open("/tmp/wp8_pulist.txt", "w") as fh:
    fh.write("\n".join(todo))
PYEOF
  N=$(grep -c . /tmp/wp8_pulist.txt || true)
  log "changed files: $N (deadline in ${LEFT}min)"
  while read -r f; do [ -n "$f" ] && PULL "$f"; done < /tmp/wp8_pulist.txt
  ALIVE=$(printf '%s' "$STATE" | grep '^ALIVE:' | cut -d' ' -f2)
  DONE=$(printf '%s' "$STATE" | grep '^DONE:' | cut -d' ' -f2)
  if [ "$DONE" = "yes" ]; then
    log "DONE MARKER SEEN (iter $i)"
    notify "Run 002 rs: LADDER/RUN DONE marker" "watch8 pulled artifacts; next: MMLU phase."
    exit 0
  fi
  if [ "$LEFT" -le 5 ] && [ "$LEFT" -gt 0 ]; then
    log "deadline-5min notice (iter $i)"
    notify "Run 002 rs: deadline in ~5min" "banked data safe; expect reaper-window stop."
  fi
  if [ "$LEFT" -le 0 ]; then
    log "DEADLINE HIT (iter $i) — banked; stopping per REAPERSPLIT discipline"
    notify "Run 002 rs: 45-min deadline" "watch8 stopped pulling; reaper window; banked data safe."
    exit 0
  fi
  if ! grep -q "$S" /home/sbussiso/.config/colab-cli/sessions.json 2>/dev/null; then
    log "REGISTRY-DROP: $S absent from sessions.json (iter $i)"
    notify "Run 002 rs: registry drop" "forensics kit pulls partials; banked data safe."
    exit 1
  fi
  if printf '%s' "$STATE" | grep -q 'NOSEEK: yes'; then
    log "SESSION GONE (iter $i)"
    notify "Run 002 rs: session gone path" "Inspect; banked files safe."
    exit 1
  fi
  if [ "$ALIVE" = "no" ]; then
    log "RUNNER GONE without DONE (iter $i)"
    notify "Run 002 rs: runner gone w/o marker" "Inspect; banked files safe."
    exit 1
  fi
done
notify "watch8 capped" "44x4min elapsed; manual check."
exit 2