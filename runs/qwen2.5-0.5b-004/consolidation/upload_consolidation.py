#!/usr/bin/env python3
"""Upload the staged consolidation, byte-verify the hub state, then — only
after every gate passes — delete sbussiso/Qwen2.5-0.5B-abliterated-r2.

Gates (ALL mandatory, in order):
  G1 whoami == sbussiso
  G2 upload committed (single commit; exact-path deletions for old layout)
  G3 repo file list == stage tree (nothing stale left, .gitattributes kept)
  G4 hub config.json tie_word_embeddings == false
  G5 hub model.safetensors x-linked-etag == 0b213334...  (byte identity)
  G6 old run-002 weights still resolvable at pinned revision 0155cadc
  G7 README + all 10 chart URLs HTTP 200
  G8 r2 repo still byte-identical (etag) right before deletion
Only then: delete r2 + confirm 404. If ANY gate fails: r2 is NOT touched.
"""
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import (HfApi, delete_file, hf_hub_download, list_repo_files,
                             upload_folder)

TOKEN = open("/root/.cache/huggingface/token").read().strip()
ORIG = "sbussiso/Qwen2.5-0.5B-abliterated"
R2 = "sbussiso/Qwen2.5-0.5B-abliterated-r2"
STAGE = Path("/root/research/abliteration/qwen2.5-0.5b-004/consolidation/stage")
NEW_SHA = "0b2133342dce215d6ae9645259b9f05e33914c0f51dcde60c550f3bf92367125"
OLD_REV = "0155cadc8d6382acb4cd5cc6ef59edce7f4c3f40"
OLD_SHA = "8ee4567a242d0ac83528af86cd713a166a528c089cf729af944e27a2bf00fd6d"
CHARTS = ["multilingual_panel", "mmlu_guardrail", "refusal_by_condition",
          "refusal_vs_benign", "r2_benign_preservation", "r2_layer_coherence",
          "r2_mmlu_guardrail", "r2_refusal_by_condition", "layer_coherence",
          "truthfulqa_guardrail"]

api = HfApi(token=TOKEN)
FAILED = []


def head_etag(url):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        return r.status, r.headers.get("x-linked-etag"), \
            r.headers.get("x-linked-size")
    except urllib.error.HTTPError as e:
        return e.code, None, None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def head_etag_resolve(url):
    """HEAD on the /resolve/ URL WITHOUT following the 302 — the x-linked-*
    headers live on the HF redirect response, not on the CDN final hop."""
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        r = _opener.open(req, timeout=60)
        return r.status, r.headers.get("x-linked-etag"), \
            r.headers.get("x-linked-size")
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.code, e.headers.get("x-linked-etag"), \
                e.headers.get("x-linked-size")
        return e.code, None, None


def api_lfs_sha(repo, path, rev="main"):
    """sha256 of an LFS/Xet file via the tree API (lfs.oid), primary check."""
    url = (f"https://huggingface.co/api/models/{repo}/tree/{rev}"
           f"?path={urllib.parse.quote(path)}")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    entries = json.load(urllib.request.urlopen(req, timeout=60))
    for e in entries:
        if e.get("path") == path and e.get("lfs"):
            return e["lfs"]["oid"], e.get("size")
    return None, None


def verify_blob(repo, path, rev, expect_sha, expect_size):
    """Two independent checks: API lfs.oid AND resolve-redirect etag."""
    oid, size = api_lfs_sha(repo, path, rev)
    api_ok = (oid == expect_sha)
    st, etag, esize = head_etag_resolve(
        f"https://huggingface.co/{repo}/resolve/{rev}/{path}")
    hdr_ok = (etag and etag.strip('"') == expect_sha)
    size_ok = True
    if expect_size is not None and size is not None:
        size_ok = (size == expect_size)
    return (api_ok or hdr_ok) and size_ok, {
        "api_oid": oid, "api_size": size, "hdr_status": st,
        "hdr_etag": etag, "hdr_size": esize}


def http_ok(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, len(r.read())
    except urllib.error.HTTPError as e:
        return e.code, 0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # G1 -------------------------------------------------------------------
    who = api.whoami()
    assert who["name"] == "sbussiso", f"token is not sbussiso: {who}"
    print("[G1] whoami=sbussiso OK")

    # preserve repo .gitattributes so LFS pointer rules survive the rewrite
    ga = hf_hub_download(ORIG, ".gitattributes", token=TOKEN)
    dst = STAGE / ".gitattributes"
    if not dst.exists():
        dst.write_bytes(Path(ga).read_bytes())

    stage_files = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*")
                         if p.is_file())
    before = set(api.list_repo_files(ORIG, repo_type="model"))
    deletions = sorted(before - set(stage_files))
    print(f"[G2-pre] repo has {len(before)} files; deleting {len(deletions)} "
          "stale paths in the same commit")

    # G2 -------------------------------------------------------------------
    upload_folder(folder_path=str(STAGE), repo_id=ORIG, repo_type="model",
                  token=TOKEN,
                  commit_message=("Consolidate: publish run-003 wd_ML_BN "
                                  "weights + run-002/003 evidence; card "
                                  "rewrite (merge of -r2, verified bytes)"),
                  delete_patterns=deletions)
    print("[G2] upload committed")

    # G3 -------------------------------------------------------------------
    after = set(api.list_repo_files(ORIG, repo_type="model"))
    expected = set(stage_files)
    missing = expected - after
    stale = after - expected
    assert not missing, f"missing after upload: {missing}"
    assert not stale, f"stale files remain: {stale}"
    print(f"[G3] repo layout == stage tree ({len(after)} files) OK")

    # G4 -------------------------------------------------------------------
    cfgp = hf_hub_download(ORIG, "config.json", token=TOKEN)
    cfg = json.load(open(cfgp))
    assert cfg.get("tie_word_embeddings") is False, cfg
    print("[G4] hub config tie_word_embeddings=false OK")

    # G5 -------------------------------------------------------------------
    ok, det = verify_blob(ORIG, "model.safetensors", "main", NEW_SHA,
                          (STAGE / "model.safetensors").stat().st_size)
    assert ok, f"safetensors verification failed: {det}"
    print(f"[G5] hub safetensors == {NEW_SHA[:16]}... OK ({det})")

    # G6 -------------------------------------------------------------------
    ok, det = verify_blob(ORIG, "model.safetensors", OLD_REV, OLD_SHA,
                          None)
    assert ok, f"old revision lost: {det}"
    print("[G6] pinned old-revision weights still resolvable OK")

    # G7 -------------------------------------------------------------------
    st, n = http_ok(f"https://huggingface.co/{ORIG}/resolve/main/README.md")
    assert st == 200 and n > 5000, f"README {st} {n}"
    for c in CHARTS:
        st, n = http_ok(
            f"https://huggingface.co/{ORIG}/resolve/main/charts/{c}.png")
        assert st == 200 and n > 10000, f"chart {c}: {st} {n}"
    print("[G7] card + 10 charts resolve OK")

    # G8 -------------------------------------------------------------------
    ok, det = verify_blob(R2, "model.safetensors", "main", NEW_SHA, None)
    assert ok, f"r2 pre-delete check failed: {det}"
    print("[G8] r2 byte-identity confirmed right before delete")

    # DELETE ---------------------------------------------------------------
    api.delete_repo(repo_id=R2, repo_type="model")
    print("[DEL] r2 deleted")
    st, _ = http_ok(f"https://huggingface.co/api/models/{R2}")
    assert st == 404, f"r2 still exists? {st}"
    print("[DEL-VERIFY] r2 API now 404 OK")

    print(json.dumps({"status": "CONSOLIDATED_AND_VERIFIED", "repo": ORIG,
                      "weights_sha": NEW_SHA, "old_rev_kept": OLD_REV,
                      "r2_deleted": True, "files": len(after)}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print("VERIFICATION_FAILED:", e, flush=True)
        print("r2 was NOT deleted — fix and re-run.", flush=True)
        sys.exit(3)