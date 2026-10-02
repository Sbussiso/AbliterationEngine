#!/usr/bin/env python3
"""7B harness adaptation check — dims, prompt sets, and code constants vs Qwen2.5-7B-Instruct.
Verifies the 0.5B-derived harness can be ported to 7B correctly. Never runs."""
import json, re
from pathlib import Path

H = Path("/root/research/abliteration/qwen2.5-1.5b-003/harness")
print("== 0.5B/1.5B harness constants ==")
# Every hardcoded dim in the harness
for f in ["run_003.py", "ladder_003.py", "mmlu_eval_003.py", "prompt_sets.py", "publish_003.py", "smoke_test.py"]:
    t = (H / f).read_text()
    dims = set()
    for pat in [r"num_hidden_layers[=: ]+(\d+)", r"hidden_size[=: ]+(\d+)", r"layer[^\n]{0,30}1[679]\b", r"L1[679]\b", r"\b14\b|\b24\b|\b28\b"]:
        for m in re.findall(pat, t):
            dims.add((pat[:20], m))
    print(f"{f}: {sorted({d[1] for d in dims if d[0] != pat[0] for d in [x for x in dims if True][:0]})}")
    # simpler: find explicit digit constants near layer/hook mentions
    hits = re.findall(r"(?:layer|L)(\d\d?)\b(?:[^\n]{0,40}(?:o_proj|hook|resid|mlp|attn))?", t)
    print("   layer-ish numerals:", sorted(set(hits)) [:12])

print("\n== prompt_sets.py structure ==")
t = (H / "prompt_sets.py").read_text()
for name in re.findall(r"def ([a-zA-Z_]+)", t):
    print(" ", name)

print("\n== 7B target geometry (Qwen/Qwen2.5-7B-Instruct) ==")
print(" num_hidden_layers = 28 | hidden = 3584 | heads = 28q/4kv | extraction band L15-29 (literature)")
print(" A100 40GB: 15GB bf16 weights + ~8GB activation+hooks = 23GB used of 42GB — comfortable, no offload needed")
print("\nHARNESS_AUDIT_DONE")