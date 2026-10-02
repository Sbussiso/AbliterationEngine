#!/usr/bin/env bash
# FTT-20 phase-split resume kit — REAPERSPLIT (2026-09-30).
# Runs the REMAINING mission tail across fresh ~50-min Colab sessions,
# engineered around the arch-invariant ~60-min server-side kernel reaper
# (4x T4 + 1x L4 instances confirm busy-unsafe, keep-alive-blind deaths).
#
# Design (all verified against engine source this session):
#   - runner.sh PHASE= ladder|mmlu — phase verbs exist (pipeline.py)
#   - ladder phase loads ONLY stage-A artifacts (core._out_dir) and skips
#     banked variants (edits._banked_variant_summary: complete
#     probes_<name>.json in out_dir w/ full grader fields -> reused)
#   - run_variant never uses ctx["model"]; ladder is artifact-driven
#   - wd_ML_BN auto-triggers when best-of-V1..3 >= 0.25 (it is: 0.4375)
#   - ENG_VARBASE puts per-variant disk dirs anywhere writable
#
# Stages (each a fresh session ~= 20-45 min GPU):
#   1. stage-bn : upload banked artifacts (s5 pull) -> ladder -> sel + probe JSONs
#   2. mmlu     : mmlu phase on selected variant -> mmlu_summary.json
#   3. publish-prep (HITL stop; NO publish until user go/no-go)
#
# Env: S=<session base name> (default ftt20-rs2), BUNDLE=<bundle path>,
#      BANKED=<s5 pulled artifacts dir>.
set -euo pipefail
S="${S:-ftt20-rs2}"
GPU="${GPU:-T4}"           # A (T4) or A+C (A100) — lease is fit-proportional either way
BUNDLE="${BUNDLE:-/root/research/abliteration/bundles/eng_run_002_qwen2.5-1.5b_20260930T222244Z.tar.gz}"
BANKED="${BANKED:-/root/research/abliteration/qwen2.5-0.5b-002/eng_run002_pull_s5}"
LOG=/root/research/abliteration/reapersplit.log
NTFY_TOKEN=$(cat /root/research/.ntfy_token)
NTFY_URL=http://10.0.0.119:8090/research
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
notify() { curl -s -u ":$NTFY_TOKEN" -H "Title: $1" "$NTFY_URL" -d "$2" > /dev/null; }
COLAB="sudo -u sbussiso /home/sbussiso/.local/bin/colab"

STAGE="$1"; shift || true
N="${N:-1}"
SN="${S}-${N}"
OUT_ROOT="${OUT_ROOT:-/root/research/abliteration/qwen2.5-0.5b-002/reapersplit}"

log "=== STAGE $STAGE session $SN ==="
$COLAB new -s "$SN" --gpu "$GPU" 2>&1 | tee -a "$LOG"

# stage banked artifacts: tar banked probe artifacts locally, upload, extract
T=$(mktemp /tmp/banked.XXXX.tar)
tar -cf "$T" -C "$BANKED" probes_baseline.json probes_wd_B.json \
  probes_wd_BN.json probes_wd_ML.json layer_coherence.json \
  layer_directions.npz refusal_direction_A.npy refusal_direction_B.npy \
  run_config.json harness_sha256.json 2>/dev/null
chmod 644 "$T"
$COLAB upload -s "$SN" "$T" /content/banked.tar 2>&1 | tee -a "$LOG"

# bundle
B=$(basename "$BUNDLE"); cp "$BUNDLE" /tmp/; chmod 644 "/tmp/$B"
$COLAB upload -s "$SN" "/tmp/$B" "/content/$B" 2>&1 | tee -a "$LOG"

# one exec: verify sha, extract, place + VALIDATE banked, narrow spec to the
# remaining two variants (dev-workstation co-sign 2026-09-30: an invalid
# banked file must cap re-work at wd_ML/wd_ML_BN, not re-run the full 5-var
# ladder inside the reaper window)
cat > /tmp/prestage_s.py <<'PYEOF'
import tarfile, hashlib, json, glob, os, re, shutil
B = "__B__"
b = open("/content/" + B, "rb").read()
h = hashlib.sha256(b).hexdigest()
assert h.startswith("8ac5ebafb2f75a0c"), "bundle sha mismatch: " + h
t = tarfile.open("/content/" + B); t.extractall("/content"); t.close()
bt = tarfile.open("/content/banked.tar"); bt.extractall("/content/banked_s5"); bt.close()

# locate engine out dir (bundle-extracted) and copy banked artifacts in
cands = sorted(glob.glob("/content/eng_run_002*"))
assert cands, "no engine dir after extract: " + ", ".join(sorted(glob.glob('/content/*')))
OUT = cands[0]
os.makedirs(OUT, exist_ok=True)
kept, dropped = [], []
for f in sorted(glob.glob("/content/banked_s5/*")):
    name = os.path.basename(f)
    if name.startswith("probes_"):
        ok = False
        try:
            d = json.load(open(f))
            r_h, r_b = d["harmful"], d["harmless"]
            ok = (len(r_h) == len(r_b) == 64 and all(
                x.get("refused") is not None and x.get("output")
                for x in r_h + r_b))
        except Exception:
            ok = False
        if ok:
            shutil.copy(f, os.path.join(OUT, name)); kept.append(name)
        else:
            dropped.append(name + " (invalid -> re-runs)")
    else:
        shutil.copy(f, os.path.join(OUT, name)); kept.append(name)

# narrow the spec's ladder.variants to the remaining work: wd_ML + wd_ML_BN
specs = [p for p in glob.glob("/content/**/*.yaml", recursive=True)
         if os.path.isfile(p)
         and "wd_ML_BN" in open(p, errors="ignore").read()]
narrowed = []
for p in specs:
    s = open(p).read()
    s2, n = re.subn(r"(variants:\s*)\[[^\]]*?\]",
                    r"\1[wd_ML, wd_ML_BN]", s, count=1)
    if n == 1:
        open(p, "w").write(s2)
        narrowed.append(os.path.relpath(p, "/content"))

print("PRESTAGE_OK out=" + OUT)
print("KEPT:", ", ".join(kept))
print("DROPPED:", ", ".join(dropped) if dropped else "none")
print("NARROWED:", ", ".join(narrowed) if narrowed else "NO_SPEC_MATCH")
PYEOF
sed -i "s/__B__/$B/" /tmp/prestage_s.py
chmod 644 /tmp/prestage_s.py
$COLAB upload -s "$SN" /tmp/prestage_s.py /tmp/prestage_s.py 2>&1 | tail -1
printf 'exec(open("/tmp/prestage_s.py").read())\n' | $COLAB exec -s "$SN" --timeout 180 2>&1 | tee -a "$LOG" | head -6

# run the phase detached (setsid), with ENG_OUT_ROOT=/content, banked into place
PH="$STAGE" \
printf 'import subprocess\np=subprocess.run(["bash","-c","mkdir -p /content/eng_run_002_qwen2.5-1.5b && cp /content/banked_s5/* /content/eng_run_002_qwen2.5-1.5b/ && cd /content && UV_VENV_CLEAR=1 PHASE=%s setsid nohup bash runner.sh > /content/phase_out.log 2>&1 & echo DETACHED $!"],capture_output=True,text=True)\nprint(p.stdout, p.stderr[:200])\n' "$STAGE" > /tmp/launch_s.py
chmod 644 /tmp/launch_s.py
$COLAB upload -s "$SN" /tmp/launch_s.py /tmp/launch_s.py 2>&1 | tail -1
printf 'exec(open("/tmp/launch_s.py").read())\n' | $COLAB exec -s "$SN" --timeout 90 2>&1 | tee -a "$LOG" | head -3

# verify runner alive 30s later
sleep 30
printf 'import os\nab=os.popen("ps -ef | grep abliterate | grep -v grep").read().strip()\nprint("RUNNER:", "alive" if ab else "NOT_RUNNING")\nprint("out_tail:", (open("/content/phase_out.log").read()[-300:] if os.path.exists("/content/phase_out.log") else "-"))\n' > /tmp/check_s.py
chmod 644 /tmp/check_s.py
$COLAB upload -s "$SN" /tmp/check_s.py /tmp/check_s.py 2>&1 | tail -1
printf 'exec(open("/tmp/check_s.py").read())\n' | $COLAB exec -s "$SN" --timeout 90 2>&1 | tee -a "$LOG" | head -6

notify "REAPERSPLIT stage $SN launched" "phase=$STAGE — runner detached+verified; boundary pings from this watcher; artifact pulls on completion markers." > /dev/null
log "launched $STAGE on $SN"

# wait for completion markers (poll every 4 min, up to 12 iters = 48 min)
LAUNCH_TS=$(date +%s)
FIT="${FIT:-3600}"          # per-session lease seconds (declared on the
                            # assignment wire: fit:3600 GPU / fit:7200 CPU)
DEADLINE=$(( FIT * 3 / 4 )) # 45 min at fit=3600; 15-min reaper margin
LEASE_PROBE_DONE=0
for i in $(seq 1 12); do
  sleep 240
  # passive lease-expiry probe: one T+~last-window exec writes+reads back a
  # file through the SAME proxy — discriminates proxy-path death (write 200s
  # now, session 404s before lease end) from true VM death. Zero cost.
  NOW=$(( $(date +%s) - LAUNCH_TS ))
  if [ "$LEASE_PROBE_DONE" -eq 0 ] && [ "$NOW" -ge $(( DEADLINE + 180 )) ]; then
    printf 'import os, time\nt=int(time.time())\nopen("/content/LEASE_PROBE", "w").write(str(t))\nback=open("/content/LEASE_PROBE").read()\nprint("LEASE_PROBE_T", t, "READBACK_OK" if back == str(t) else "READBACK_FAIL")\n' > /tmp/lp_s.py
    chmod 644 /tmp/lp_s.py
    $COLAB upload -s "$SN" /tmp/lp_s.py /tmp/lp_s.py 2>&1 | tail -1
    printf 'exec(open("/tmp/lp_s.py").read())\n' | $COLAB exec -s "$SN" --timeout 90 2>&1 | grep LEASE_PROBE_T | tee -a "$LOG"
    LEASE_PROBE_DONE=1
    log "lease-probe receipt written (proxy liveness at T=$(( NOW / 60 ))min)"
  fi
  cat > /tmp/poll_s.py <<'PYEOF'
import os, re
out = open("/content/phase_out.log").read() if os.path.exists("/content/phase_out.log") else ""
mark = "pending"
if "LADDER_DONE" in out: mark = "LADDER_DONE"
elif "MMLU_DONE" in out: mark = "MMLU_DONE"
else:
    m = re.search(r"\bRUN_DONE\b", out)
    if m: mark = "RUN_DONE"
ab = os.popen("ps -ef | grep abliterate | grep -v grep").read().strip()
print("MARK:", mark)
print("ALIVE:", "yes" if ab else "no")
PYEOF
  chmod 644 /tmp/poll_s.py
  $COLAB upload -s "$SN" /tmp/poll_s.py /tmp/poll_s.py 2>&1 | tail -1
  printf 'exec(open("/tmp/poll_s.py").read())\n' | $COLAB exec -s "$SN" --timeout 90 2>&1 | tee -a "$LOG" | grep -E '^(MARK|ALIVE)' | head -2 > /tmp/mark_s.txt || true
  MARK=$(grep '^MARK:' /tmp/mark_s.txt | cut -d' ' -f2)
  ALIVE=$(grep '^ALIVE:' /tmp/mark_s.txt | cut -d' ' -f2)
  if [ "$MARK" != "pending" ]; then
    log "Phase complete ($MARK) on $SN"
    notify "REAPERSPLIT: $SN phase done ($MARK)" "Artifacts ready to pull."
    break
  fi
  if [ "$ALIVE" = "no" ]; then
    log "Runner died without marker on $SN (iter $i) — bank forensics (partial probes + log) then exit"
    notify "REAPERSPLIT: runner died without marker ($SN)" "Banking what streamed before inspecting; partial data pulled."
    mkdir -p "$OUT_ROOT/$SN"; chmod 777 "$OUT_ROOT/$SN" 2>/dev/null || true
    $COLAB download -s "$SN" /content/phase_out.log "$OUT_ROOT/$SN/phase_out_partial.log" >>/tmp/rs_dl.log 2>&1 && log "pulled phase_out_partial.log" || true
    for f in probes_wd_ML_BN.json probes_wd_ML.json run_config.json; do
      $COLAB download -s "$SN" "/content/eng_run_002_qwen2.5-1.5b/$f" "$OUT_ROOT/$SN/$f" >>/tmp/rs_dl.log 2>&1 && log "pulled partial $f (may be incomplete)" || true
    done
    exit 1
  fi
done

# pull phase artifacts
mkdir -p "$OUT_ROOT/$SN"; chmod 777 "$OUT_ROOT/$SN" 2>/dev/null || true
for f in probes_wd_ML_BN.json selection.json selection_candidates.json run_config.json mmlu_summary.json summary.json LADDER_DONE MMLU_DONE; do
  $COLAB download -s "$SN" "/content/eng_run_002_qwen2.5-1.5b/$f" "$OUT_ROOT/$SN/$f" >>/tmp/rs_dl.log 2>&1 && log "pulled $f" || true
done
$COLAB download -s "$SN" /content/phase_out.log "$OUT_ROOT/$SN/phase_out.log" >>/tmp/rs_dl.log 2>&1 && log "pulled phase_out.log" || true
ls "$OUT_ROOT/$SN" | tee -a "$LOG"
notify "REAPERSPLIT: $SN artifacts pulled" "$(ls "$OUT_ROOT/$SN" | tr '\n' ' ')"
log "=== stage $SN complete ==="