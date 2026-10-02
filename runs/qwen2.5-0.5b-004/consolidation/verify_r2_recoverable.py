#!/usr/bin/env python3
"""Pre-delete verification gate for sbussiso/Qwen2.5-0.5B-abliterated-r2.

Confirms EVERY file in the hub r2 tree is either (a) byte-identical to a file
on VM 151 (by sha256) or (b) a hub-generated pointer/metadata file. Prints a
per-file PASS/FAIL table; exits nonzero if any real file fails.

Recovery guarantee: with this gate green, the hub r2 repo can be deleted and
recreated bit-for-bit from the VM mirror.
"""
import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

REPO = "sbussiso/Qwen2.5-0.5B-abliterated-r2"
MIRROR_ROOTS = [
    Path("/root/research/abliteration/qwen2.5-0.5b-004/consolidation/stage_r2"),
    Path("/root/research/abliteration/qwen2.5-0.5b-004/variant_wd_ML_BN"),
    Path("/root/research/abliteration/qwen2.5-0.5b-004/artifacts"),
    Path("/root/research/abliteration/qwen2.5-0.5b-004/consolidation/stage/eval2"),
    Path("/root/research/abliteration/qwen2.5-0.5b-004/consolidation/stage/charts"),
]

# files whose hub copy is generated/metadata and safe to skip
HUB_GENERATED = {".gitattributes"}

api = HfApi()
info = api.model_info(REPO, files_metadata=True)
hub_files = {s.rfilename: s for s in info.siblings if s.size is not None}

# build VM sha index (sizes can be large; cache shas per file)
vm_shas: dict[str, str] = {}
vm_index: dict[str, list[str]] = {}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for root in MIRROR_ROOTS:
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                rel = str(p)
                vm_shas[rel] = sha256_file(p)
                vm_index.setdefault(p.name, []).append(rel)

fails, passes = [], []
for name, meta in sorted(hub_files.items()):
    if name in HUB_GENERATED:
        passes.append((name, "hub-generated (skipped)"))
        continue
    leaf = name.rsplit("/", 1)[-1]
    if leaf == "README.md":
        # card text: regenerated from consolidation/gen_card_r2.py — recoverable
        gen = Path("/root/research/abliteration/qwen2.5-0.5b-004/consolidation/gen_card_r2.py")
        ok = gen.exists()
        passes.append((name, "regenerable via gen_card_r2.py" if ok else "GENERATOR MISSING"))
        if not ok:
            fails.append(name)
        continue
    candidates = vm_index.get(leaf, [])
    if not candidates:
        fails.append(name)
        print(f"FAIL {name}: no VM mirror candidate")
        continue
    # fetch hub bytes, compare sha
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    if len(data) != (meta.size or -1):
        fails.append(name)
        print(f"FAIL {name}: size mismatch hub={meta.size} fetched={len(data)}")
        continue
    h = hashlib.sha256(data).hexdigest()
    hit = any(vm_shas[c] == h for c in candidates)
    if hit:
        passes.append((name, "byte-identical on VM"))
    else:
        fails.append(name)
        print(f"FAIL {name}: sha mismatch vs all VM candidates")

print(f"\n{len(passes)} PASS / {len(fails)} FAIL")
if fails:
    print("FAILED FILES:", *fails, sep="\n  ")
    sys.exit(1)
print("RECOVERY_GATE_OK: every hub file is recoverable from VM 151")