#!/usr/bin/env python3
"""Run 004 — RefusalBench-NQ paired analysis: base vs abliterated Qwen2.5-0.5B.

Reads results_base.jsonl + results_abl.jsonl (same 360 instances, paired).
Computes: per-arm per-stratum metrics, primary endpoint (category-matched
refusal accuracy delta), McNemar test, bootstrap CI, MRR/FRR, intensity
degradation. Writes analysis JSON + markdown table snippets.
"""
import json, math, random
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
random.seed(0)

def load(name):
    rows = {}
    for l in open(HERE / name):
        d = json.loads(l)
        rows[d["id"]] = d
    return rows

base, abl = load("results_base.jsonl"), load("results_abl.jsonl")
assert set(base) == set(abl), "arm sets differ!"
ids = sorted(base)
n = len(ids)

def agg(rows):
    """Arm-level metrics."""
    refused = sum(rows[i]["refused"] for i in ids)
    # refusal-expected subset
    ref_ids = [i for i in ids if rows[i]["expected"] != "ANSWER_CORRECTLY"]
    ans_ids = [i for i in ids if rows[i]["expected"] == "ANSWER_CORRECTLY"]
    cat_ok = sum(rows[i]["acceptable_match"] for i in ref_ids)
    # missed refusal = refusing-row answered
    missed = sum(not rows[i]["refused"] for i in ref_ids)
    frr = sum(rows[i]["refused"] for i in ans_ids) / max(len(ans_ids), 1)
    out = dict(
        n=n,
        refusal_rate_all=refused / n,
        # primary: category-matched refusal accuracy over refusal-expected
        n_ref= len(ref_ids),
        catmatch= sum(rows[i]["acceptable_match"] for i in ref_ids) / len(ref_ids),
        binary_refusal_on_flawed= sum(rows[i]["refused"] for i in ref_ids) / len(ref_ids),
        MRR= missed / len(ref_ids),
        n_ans= len(ans_ids),
        FRR= sum(rows[i]["refused"] for i in ans_ids) / max(len(ans_ids), 1),
    )
    return out, ref_ids, ans_ids

b_agg, ref_ids, ans_ids = agg(base)
a_agg, _, _ = agg(abl)

# paired deltas per instance (category-match on flawed subset)
delta_pairs = [(base[i]["acceptable_match"], abl[i]["acceptable_match"]) for i in ref_ids]
b_c = sum(x for x, _ in delta_pairs); a_c = sum(y for _, y in delta_pairs)
both = sum(x and y for x, y in delta_pairs)
only_b = b_c - both; only_a = a_c - both
# McNemar exact-ish (binomial on discordant pairs)
d = only_a + only_b
if d > 0:
    # two-sided exact binomial p
    from math import comb
    k = min(only_a, only_b)
    p = sum(comb(d, i) for i in range(0, k+1)) / 2**d * 2
    p = min(1.0, p)
else:
    p = 1.0

# bootstrap CI on paired delta (acceptable_match proportion)
B = 10000
diffs = []
ref_ids_list = list(ref_ids)
for _ in range(B):
    samp = [ref_ids_list[random.randrange(len(ref_ids_list))] for _ in range(len(ref_ids_list))]
    db = sum(base[i]["acceptable_match"] for i in samp) / len(samp)
    da = sum(abl[i]["acceptable_match"] for i in samp) / len(samp)
    diffs.append(da - db)
diffs.sort()
ci = (diffs[int(0.025*B)], diffs[int(0.975*B)-1])

# per-class x intensity
def strata(rows):
    s = defaultdict(lambda: [0,0,0])  # class,int -> [cat_ok, flaws, ref_ok]
    for i in ids:
        r = rows[i]
        if r["expected"] == "ANSWER_CORRECTLY": continue
        k = (r["perturbation_class"], r["intensity"])
        s[k][1] += 1
        s[k][0] += r["acceptable_match"]
        s[k][2] += r["refused"]
    return s

bs, as_ = strata(base), strata(abl)

# intensity degradation: cat-match by intensity
def by_intensity(rows):
    d = defaultdict(lambda: [0,0])
    for i in ids:
        r = rows[i]
        if r["expected"] == "ANSWER_CORRECTLY": continue
        d[r["intensity"]][1] += 1
        d[r["intensity"]][0] += r["acceptable_match"]
    return {k: v[0]/v[1] for k, v in d.items()}

# FRR by arm on answerable
out = {
    "n": n, "n_refusal_expected": len(ref_ids), "n_answerable": len(ans_ids),
    "base": b_agg, "abl": a_agg,
    "paired": {
        "both_correct": both, "only_base": only_b, "only_abl": only_a,
        "discordant": d, "mcnemar_exact_p": p,
        "delta_pp": (a_c - b_c) / len(ref_ids) * 100,
        "bootstrap_ci95_pp": [round(x*100, 2) for x in ci],
    },
    "strata": {
        "base": {f"{k[0]}|{k[1]}": {"cat_ok": v[0], "ref_ok": v[2], "n": v[1]} for k, v in bs.items()},
        "abl": {f"{k[0]}|{k[1]}": {"cat_ok": v[0], "ref_ok": v[2], "n": v[1]} for k, v in as_.items()},
    },
    "intensity_degradation": {"base": by_intensity(base), "abl": by_intensity(abl)},
}
json.dump(out, open(HERE / "analysis.json", "w"), indent=1)
print("=== PRIMARY: category-matched refusal accuracy (flawed instances, n=%d) ===" % len(ref_ids))
print(f"base: {b_agg['catmatch']*100:.1f}%  ablated: {a_agg['catmatch']*100:.1f}%  delta: {(a_c-b_c)/len(ref_ids)*100:+.2f}pp  CI95 {ci}")
print(f"McNemar discordant pairs: only_abl={only_a} only_base={only_b} exact p={p:.4f}")
print(f"binary refusal on flawed: base {b_agg['binary_refusal_on_flawed']*100:.1f}% → abl {a_agg['binary_refusal_on_flawed']*100:.1f}%")
print(f"MRR: base {b_agg['MRR']*100:.1f}% → abl {a_agg['MRR']*100:.1f}%   FRR(answerable): base {b_agg['FRR']*100:.1f}% → abl {a_agg['FRR']*100:.1f}%")
print("\n=== per class x intensity (cat-match %) ===")
for k in sorted(bs):
    kb = f"{k[0]}|{k[1]}"
    print(f"{k[0]:24s} {k[1]:7s} base {bs[k][0]}/{bs[k][1]} = {100*bs[k][0]/bs[k][1]:.0f}%  abl {as_[k][0]}/{as_[k][1]} = {100*as_[k][0]/as_[k][1]:.0f}%")
print("ANALYSIS_DONE")