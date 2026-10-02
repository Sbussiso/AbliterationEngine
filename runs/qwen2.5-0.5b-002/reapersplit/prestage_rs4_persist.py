"""rs2-4 combined prestage: extract bundle + banked state + rebuild variant
to the ENGINE's canonical VARBASE path (/content/wd_ML_BN) in one step —
no cp-fix needed as a separate stage (the 550c41bc lesson applied at
source). Then chains PHASE=mmlu with lm-eval installed.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

# 1. verify + extract bundle
B = "eng_run_002_qwen2.5-1.5b_20260930T222244Z.tar.gz"
b = open("/content/" + B, "rb").read()
h = hashlib.sha256(b).hexdigest()
assert h.startswith("8ac5ebafb2f75a0c"), "bundle sha mismatch: " + h
t = tarfile.open("/content/" + B)
t.extractall("/content")
t.close()

# 2. extract banked state
bt = tarfile.open("/content/banked.tar")
bt.extractall("/content/banked_s5")
bt.close()

# 3. place banked artifacts into engine out-dir (validate probes)
OUT = "/content/eng_run_002_qwen2.5-1.5b"
os.makedirs(OUT, exist_ok=True)
kept, suspect = [], []
for f in sorted(glob.glob("/content/banked_s5/*")):
    n = os.path.basename(f)
    ok = True
    if n.startswith("probes_"):
        ok = False
        try:
            d = json.load(open(f))
            if (len(d["harmful"]) == len(d["harmless"]) == 64):
                ok = all(x.get("refused") is not None and x.get("output")
                         for x in d["harmful"] + d["harmless"])
        except Exception:
            pass
    shutil.copy(f, os.path.join(OUT, n))
    (kept if ok else suspect).append(n)
print("PRESTAGE_OK out=" + OUT)
print("KEPT:", ",".join(kept))
if suspect:
    print("SUSPECT:", ",".join(suspect))

# 4. rebuild variant DIRECTLY to engine's canonical VARBASE path
#    (edits.py:282-283: VARBASE = eng_base() = /content)
import sys
sys.path.insert(0, "/content/src")
from abliteration_engine import core  # noqa: E402
from abliteration_engine.spec import load_spec  # noqa: E402
spec = load_spec("/content/specs/qwen25_1p5b_run002_resume.yaml")
try:
    core.ensure_markers(spec)
except AttributeError:
    core.REFUSAL_MARKERS = None
import torch  # noqa: E402
import numpy as _np  # noqa: E402
from abliteration_engine.edits import (  # noqa: E402
    orthogonalize_layer_output, orthogonalize_lm_head,
    orthogonalize_final_norm, save_variant)

sel = json.load(open(os.path.join(OUT, "selection.json")))
variant = sel["selected"]
k_combo = sel["k_layers_combo"]
npz = _np.load(os.path.join(OUT, "layer_directions.npz"))
dirs_all = {int(l): npz["directions"][l]
            for l in npz["decoder_layers"]}
dir_B = torch.from_numpy(
    _np.load(os.path.join(OUT, "refusal_direction_B.npy"))).float()

tok, model = core.load_patient(spec)
info = {"layers": k_combo, "resids": {}}
for l in k_combo:
    info["resids"][f"L{l}"] = orthogonalize_layer_output(
        model, l, torch.from_numpy(dirs_all[l]).float())
mc, comp = orthogonalize_lm_head(model, dir_B)
_, resid = orthogonalize_final_norm(model, dir_B)
info["lm_head_max|W r_B|"] = mc
info["norm_wdB_before"] = comp
info["norm_resid_fp32"] = resid
print("[rebuild] edit applied: "
      + json.dumps(info, default=str)[:300], flush=True)

vdir = "/content/" + variant
save_variant(model, tok, vdir, expect_tied=False)
del model
torch.cuda.empty_cache()
missing = [f for f in ("config.json", "model.safetensors")
           if not os.path.exists(os.path.join(vdir, f))]
assert not missing, "save incomplete: missing " + str(missing)
print("REBUILD_OK dir=" + vdir, flush=True)

# 5. chain: lm-eval install + PHASE=mmlu
chain = (
    "pip install -q lm-eval > /tmp/lm4.log 2>&1; ",
    "echo LMEVAL_RC=$? > /content/lm4_rc.txt; ",
    "UV_VENV_CLEAR=1 PHASE=mmlu bash /content/runner.sh "
    ">> /content/phase_out.log 2>&1; ",
)
full = ("cd /content && setsid nohup bash -c '%s' > /dev/null 2>&1 "
        "& echo CHAINED $!" % "".join(chain).rstrip("; "))
p = subprocess.run(["bash", "-c", full], capture_output=True, timeout=25)
out = (p.stdout or b"").decode().strip()
err = (p.stderr or b"").decode().strip()[:160]
print("OUT:", out if out else err)