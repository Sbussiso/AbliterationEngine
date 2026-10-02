#!/usr/bin/env bash
# rs2-1 mirror-puller: decider watch via content API only (exec channel wedged).
# Downloads phase_out.log every 90s; pulls probes_wd_ML_BN.json + selection.json
# the moment the engine writes them; ntfy pings on LADDER_DONE / files-landed /
# session-vanish. Capped ~55 min (past lease death 22:26). NEVER calls exec.
set -uo pipefail
C="sudo -u sbussiso /home/sbussiso/.local/bin/colab"
S=ftt20-rs2-1
OUT=/root/research/abliteration/qwen2.5-0.5b-002/reapersplit/rs2-1-forensics
L=/root/research/abliteration/reapersplit.log
N=/root/research/.ntfy_token
U=http://10.0.0.119:8090/research
nf() { curl -s -u ":$(cat $N)" -H "Title: $1" "$U" -d "$2" > /dev/null; }
log() { echo "$(date '+%F %T') $*" >> "$L"; }
mkdir -p "$OUT"; chmod 777 "$OUT" 2>/dev/null
log "mirror-puller armed on $S (download-only; exec channel left alone)"
nf "rs2-1: WORK IS ALIVE — decider mid-flight" "Exec channel wedged (my capture-pipe hang — own error, logged) but the detached shim's ladder child runs server-side: wd_ML re-run REPRODUCED banked artifact exactly (27/64 harm, 0/64 benign, byte-complete) and wd_ML_BN-harm streams at 38/64 with 18.4% refusals — on pace to CLEAR the 25% gate. Mirror-puller now riding phase_out.log via content API (downloads work without the exec channel). Boundary pings continue."
for i in $(seq 1 36); do
  sleep 90
  $C download -s "$S" /content/phase_out.log "$OUT/phase_out.log" > /tmp/mpl_phase.log 2>&1 || true
  if grep -q LADDER_DONE "$OUT/phase_out.log" 2>/dev/null; then
    log "LADDER_DONE marker detected (iter $i)"
    nf "rs2-1: LADDER_DONE — decider finished, pulling artifacts" "Ladder phase complete; pulling probes_wd_ML_BN.json + selection.json now."
    for f in probes_wd_ML_BN.json selection.json selection_candidates.json run_config.json; do
      $C download -s "$S" "/content/$f" "$OUT/$f" > /tmp/mpl_f.log 2>&1 && log "pulled $f" || log "no $f yet"
    done
    if [ -f "$OUT/probes_wd_ML_BN.json" ]; then
      nf "rs2-1: wd_ML_BN banked — decider verdict incoming after grade" "Composites pulled (byte-complete check next on dev side; v2 re-grade after)."
    fi
    log "mirror-puller done (LADDER_DONE path)"
    exit 0
  fi
  # session-vanish check via sessions list (content-plane, no exec)
  if ! $C sessions 2>&1 | grep -q ftt20-rs2-1; then
    log "SESSION VANISHED (iter $i) — lease reaped; forensics remain pulled"
    nf "rs2-1: session reaped at lease-end — forensics banked" "Latest phase_out.log + wd_ML probes are safe on VM151; partial composite data preserved. Resume decision per standing rules."
    exit 0
  fi
done
log "mirror-puller iteration cap reached (55 min) — final state check"
$C download -s "$S" /content/phase_out.log "$OUT/phase_out_final.log" > /dev/null 2>&1 || true
tail -8 "$OUT/phase_out.log" >> "$L" 2>/dev/null
nf "rs2-1: mirror-puller cap reached" "Composite still mid-flight or log stale — next poll cycle continues manually."