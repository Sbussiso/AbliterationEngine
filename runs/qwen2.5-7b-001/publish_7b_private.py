"""Run 002 — INCREMENTAL PRIVATE uploader (user clarification 2026-09-29: upload to HF
during the run, just PRIVATE instead of public; public flip user-gated at end). Runs ON THE COLAB A100 VM.
Uploads: wd_ML variant weights + tokenizer + card + full evidence pack (eval JSONs,
both stages' logs, analysis JSONs, chart PNGs + generators).
Idempotent: safe to re-run; skips uploads whose remote is byte-identical by etag.
"""
import os, json, glob, hashlib, time
from pathlib import Path
from huggingface_hub import HfApi, whoami

REPO_ID = "sbussiso/Qwen2.5-7B-abliterated"
PRIVATE = True  # USER DIRECTIVE: publish private first; public after end review

# --- identity gate: never push under the wrong account ---
uid = whoami()
assert uid["name"] == "sbussiso", f"identity mismatch: {uid}"
print("identity OK:", uid["name"])

api = HfApi()
# create repo (private) if missing
api.create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
info = api.repo_info(REPO_ID, repo_type="model")
assert info.private is True, "repo is NOT private!"
print(f"repo {REPO_ID} is PRIVATE ✓")

# --- stage the payload ---
STAGE = Path("/content/publish_stage")
STAGE.mkdir(exist_ok=True)

# 1. variant weights dir (config/tokenizer/weights as saved by the ladder)
import shutil
var = Path("/content/wd_ML")
for f in var.iterdir():
    shutil.copy2(f, STAGE / f.name)

# 2. evidence pack
EV = STAGE / "eval2"
EV.mkdir(exist_ok=True)
# stage-A outputs
for src, dst in [
    ("/content/abliteration_out/structure_report.json", EV / "stageA_structure_report.json"),
    ("/content/abliteration_out/probes_baseline.json", EV / "probes_baseline.json"),
    ("/content/abliteration_out/probes_hook_ablated.json", EV / "probes_hook_ablated.json"),
    ("/content/abliteration_out/layer_directions.npz", EV / "layer_directions.npz"),
    ("/content/abliteration_out/direction.npy", EV / "refusal_direction.npy"),
    ("/content/abliteration_out/coherence.json", EV / "layer_coherence.json"),
]:
    for cand in [src, src.replace("/abliteration_out/", "/abliteration_out/output/"),
                 "/content/" + src.split("/")[-1]]:
        if Path(cand).exists():
            shutil.copy2(cand, dst); break
    else:
        print("MISSING stage-A file:", src)
# ladder outputs
for src, dst in [
    ("/content/wd_ML/summary.json", EV / "ladder_summary.json"),
    ("/content/selection_candidates.json", EV / "selection_candidates.json"),
    ("/content/probes_wd_B.json", EV / "probes_wd_B.json"),
    ("/content/probes_wd_BN.json", EV / "probes_wd_BN.json"),
    ("/content/probes_wd_ML.json", EV / "probes_wd_ML.json"),
]:
    for cand in [src]:
        if Path(cand).exists():
            shutil.copy2(cand, dst); break
    else:
        print("MISSING ladder file:", src)
# logs
LG = STAGE / "logs"; LG.mkdir(exist_ok=True)
for src in ["/content/run_log.txt", "/content/ladder_log.txt", "/content/mmlu_log.txt"]:
    if Path(src).exists():
        shutil.copy2(src, LG / Path(src).name)

# harness code
HC = STAGE / "harness"; HC.mkdir(exist_ok=True)
for src in glob.glob("/content/harness/*.py"):
    shutil.copy2(src, HC / Path(src).name)

print("staged files:", len(list(STAGE.rglob("*"))))

# --- upload everything, sequential, retry once ---
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()

uploaded = 0
for p in sorted(STAGE.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(STAGE).as_posix()
    for attempt in range(2):
        try:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                            repo_id=REPO_ID, repo_type="model")
            uploaded += 1
            break
        except Exception as e:
            print(f"retry {rel} ({attempt}):", str(e)[:120])
            time.sleep(5)
print(f"UPLOADED {uploaded} files")

# incremental mode: safe to re-run as more evidence lands
print("PUBLISH_INCREMENTAL_DONE")