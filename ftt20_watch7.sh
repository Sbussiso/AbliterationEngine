#!/usr/bin/env bash
# Watcher v7 = v6b + FTT-27 fix: 3-attempt PULL retries with honest FAIL logs,
# + registry-drop detection (sessions.json absence) + same DONE/vanish alarms.
set -u
S=ftt20-run002-research-workstation-5
LOG=/root/research/abliteration/ftt20_watch7.log
OUT=/root/research/abliteration/qwen2.5-0.5b-002/eng_run002_pull_s5
STATE_PY=/tmp/wp6_state.py
mkdir -p "$OUT"
NTFY_URL=http://10.0.0.119:8090/research
NTFY_TOKEN=$(cat /root/research/.ntfy_token)
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
notify() { curl -s -u ":$NTFY_TOKEN" -H "Title: $1" "$NTFY_URL" -d "$2" > /dev/null; }
PULL() {
  F="$1"; ATTEMPT=0; OK=0
  while [ "$ATTEMPT" -lt 3 ]; do
    ATTEMPT=$((ATTEMPT+1))
    rm -f "/tmp/wp7_$F"
    sudo -u sbussiso /home/sbussiso/.local/bin/colab download -s "$S" \
      "/content/eng_run_002_qwen2.5-1.5b/$F" "/tmp/wp7_$F" >>/tmp/wp7_dl.log 2>&1
    if [ -s "/tmp/wp7_$F" ]; then
      cp "/tmp/wp7_$F" "$OUT/$F"; log "pulled $F (attempt $ATTEMPT)"; OK=1; break
    fi
    log "pull-FAIL $F attempt $ATTEMPT (empty/missing)"
    sleep 5
  done
  if [ "$OK" -eq 0 ]; then log "pull-GIVEUP $F after 3 attempts"; fi
}
COLAB="sudo -u sbussiso /home/sbussiso/.local/bin/colab exec -s $S --timeout 60 -f $STATE_PY"
log "watch7 start (v7: FTT-27 retry pulls + registry-drop detection)"
for i in $(seq 1 44); do
  sleep 240
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
with open("/tmp/wp7_pulist.txt", "w") as fh:
    fh.write("\n".join(todo))
PYEOF
  N=$(grep -c . /tmp/wp7_pulist.txt || true)
  log "changed files: $N"
  while read -r f; do [ -n "$f" ] && PULL "$f"; done < /tmp/wp7_pulist.txt
  if ! grep -q "$S" /home/sbussiso/.config/colab-cli/sessions.json 2>/dev/null; then
    log "REGISTRY-DROP: $S absent from sessions.json (iter $i)"
    notify "Run 002 s5: registry drop" "probe-by-kernel next per FTT-13; banked artifacts in eng_run002_pull_s5/"
    exit 1
  fi
  ALIVE=$(printf '%s' "$STATE" | grep '^ALIVE:' | cut -d' ' -f2)
  DONE=$(printf '%s' "$STATE" | grep '^DONE:' | cut -d' ' -f2)
  if [ "$DONE" = "yes" ]; then
    log "DONE MARKER SEEN (iter $i)"
    notify "Run 002: LADDER/RUN DONE marker" "watch7 pulled final artifacts; next: MMLU then publish."
    exit 0
  fi
  if [ "$ALIVE" = "no" ]; then
    log "RUNNER GONE without DONE (iter $i)"
    notify "Run 002: runner gone w/o marker" "Inspect; banked files safe."
    exit 1
  fi
done
notify "watch7 capped" "44x4min elapsed; manual check."
exit 2