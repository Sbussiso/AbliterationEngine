#!/usr/bin/env python3
"""Paired analysis of the TQ240 subsample: base vs wd_ML_BN.

Recomputes per-doc mc2 from lm-eval's logged samples (resps = raw
loglikelihoods of true/false reference answers), then:
  - paired per-doc delta d_i = variant - base (identical 240 docs both arms)
  - 10k-resample paired bootstrap percentile CI on mean delta
  - exact two-sided sign test (math.comb binomial)
Prints a compact JSON verdict block for the card.
"""
import json
import math
import random
from pathlib import Path

CONS = Path("/root/research/abliteration/qwen2.5-0.5b-004/consolidation")


def load_docs(side):
    hits = sorted((CONS / f"tq240_{side}").rglob("samples_*.jsonl"))
    assert hits, f"no samples jsonl for {side}"
    lines = [json.loads(l) for l in open(hits[-1], encoding="utf-8")]
    # recompute per-doc mc2 from raw loglikelihoods. lm-eval sample layout:
    # resps = one entry per candidate, each = [[ll_str, is_greedy], ...];
    # candidates aligned with doc['mc2_targets']['labels'] (1=true, 0=false)
    out = {}
    for r in lines:
        doc_id = r["doc_id"]
        labels = r["doc"]["mc2_targets"]["labels"]
        lls = [float(cand[0][0]) for cand in r["resps"]]
        assert len(lls) == len(labels), (doc_id, len(lls), len(labels))
        p_true = sum(math.exp(l) for l, b in zip(lls, labels) if b == 1)
        p_false = sum(math.exp(l) for l, b in zip(lls, labels) if b == 0)
        out[doc_id] = p_true / (p_true + p_false)
    return out


def main():
    base = load_docs("base")
    var = load_docs("r2")
    ids = sorted(set(base) & set(var))
    assert len(ids) == 240, f"paired docs: {len(ids)} (expected 240)"
    d = [var[i] - base[i] for i in ids]
    n = len(d)
    mean = sum(d) / n
    base_mean = sum(base[i] for i in ids) / n
    var_mean = sum(var[i] for i in ids) / n

    rng = random.Random(0)
    B = 10000
    boots = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B) - 1]

    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    n_eff = pos + neg
    two_sided = min(pos, neg)
    # exact binomial two-sided p
    total = sum(math.comb(n_eff, k) for k in range(0, two_sided + 1)) * 2 / \
        2 ** n_eff
    p = min(1.0, total)

    print(json.dumps({
        "n_docs": n,
        "base_mc2_pct": round(100 * base_mean, 2),
        "variant_mc2_pct": round(100 * var_mean, 2),
        "delta_pp": round(100 * mean, 2),
        "bootstrap_ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
        "sign_pos_neg": [pos, neg],
        "sign_test_p": round(p, 4),
        "verdict": ("no significant truthfulness cost (CI contains 0)"
                    if lo <= 0 <= hi else
                    ("significant DECREASE" if mean < 0 else
                     "significant INCREASE")),
        "subset": "first 240 TruthfulQA docs (lm-eval --limit 240, seed 0), "
                  "identical both arms",
    }, indent=1))


if __name__ == "__main__":
    main()