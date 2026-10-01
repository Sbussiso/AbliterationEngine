#!/usr/bin/env bash
# Watcher v6b — staged .py state poll (the heredoc/printf version was broken:
# bash -x showed empty STATE). Pulls every changed engine file; honest logs.
set -u
S=ftt20-run002-research-workstation-5
LOG=/root/research/abliteration/ftt20_watch6.log
OUT=/root/research/abliteration/qwen2.5-0.5b-002/eng_run002_pull_s5
STATE_PY=/tmp/wp6_state.py
mkdir -p "$OUT"
NTFY_URL=http://10.0.0.119:8090/research
NTFY_TOKEN=$(cat /root/research/.ntfy_token)
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
notify() { curl -s -u ":$NTFY_TOKEN" -H "Title: $1" "$NTFY_URL" -d "$2" > /dev/null; }
PULL() {
  F="$1"
  sudo -u sbussiso /home/sbussiso/.local/bin/colab download -s "$S" \
    "/content/eng_run_002_qwen2.5-1.5b/$F" "/tmp/wp6_$F" >>/tmp/wp6_dl.log 2>&1 \
  && [ -s "/tmp/wp6_$F" ] && cp "/tmp/wp6_$F" "$OUT/$F" && log "pulled $F"
}
COLAB="sudo -u sbussiso /home/sbussiso/.local/bin/colab exec -s $S --timeout 60 -f $STATE_PY"
log "watch6b start"
for i in $(seq 1 40); do
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
with open("/tmp/wp6_pulist.txt", "w") as fh:
    fh.write("\n".join(todo))
PYEOF
  N=$(grep -c . /tmp/wp6_pulist.txt || true)
  log "changed files: $N"
  while read -r f; do [ -n "$f" ] && PULL "$f"; done < /tmp/wp6_pulist.txt
  ALIVE=$(printf '%s' "$STATE" | grep '^ALIVE:' | cut -d' ' -f2)
  DONE=$(printf '%s' "$STATE" | grep '^DONE:' | cut -d' ' -f2)
  if [ "$DONE" = "yes" ]; then
    log "DONE MARKER SEEN (iter $i)"
    notify "Run 002: LADDER/RUN DONE marker" "watch6 pulled final artifacts; next: MMLU then publish."
    exit 0
  fi
  if printf '%s' "$STATE" | grep -q 'NOSEEK: yes'; then
    log "SESSION GONE (iter $i)"
    notify "Run 002 s5 registry drop?" "Banked artifacts safe; relaunch protocol ready."
    exit 1
  fi
  if [ "$ALIVE" = "no" ]; then
    log "RUNNER GONE without DONE (iter $i)"
    notify "Run 002: runner gone w/o marker" "Inspect; banked files safe."
    exit 1
  fi
done
notify "watch6 capped" "40x4min elapsed; manual check."
exit 2
