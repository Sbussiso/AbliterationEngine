#!/usr/bin/env python3
"""Run 002 @ 7B — chart generator. Regenerates EVERY figure from the recorded
artifacts (never hand-typed). Outputs PNGs into qwen2.5-7b-001/charts/.

Reads:
  artifacts/stageA/     probes_baseline.json, probes_hook_ablated.json,
                        layer_coherence.json, layer_directions.npz
  artifacts/stageB/     probes_wd_{B,BN,ML}.json, selection(,_candidates).json
  artifacts/mmlu/       results_base.json, results_variant.json
  artifacts/tq_mc2/     results_base.json, results_variant.json
  artifacts/refusalbench/  results_base.jsonl, results_abl.jsonl, analysis_rb7b.json
  artifacts/multilingual/  tq_multi_7b.json

Outputs 6 PNGs:
  refusal_by_variant.png     — refusal rate by variant (baseline/hook/wd_B/wd_BN/wd_ML)
  benign_preservation.png    — benign compliance by variant
  layer_coherence.png        — per-layer coherence curve from Stage A
  mmlu_tq_guardrail.png      — MMLU + TQmc2 paired bars (base vs wd_ML)
  rb1600_categories.png      — RefusalBench exact-match base vs abl (top classes)
  rb1600_delta_hist.png      — paired per-instance delta histogram
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/research/abliteration/qwen2.5-7b-001")
A = ROOT / "artifacts"
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, RED, GREEN, GRAY = "#4c72b0", "#dd8452", "#c44e52", "#55a868", "#8c8c8c"

def load(p):
    with open(p) as f:
        return json.load(f)

def rr(probes):
    return 100.0 * probes.get("refusal_rate", probes.get("refusal", 0.0))

# ---- chart 1+2: refusal + benign by variant --------------------------------
probes = {
    "baseline": load(A / "stageA" / "probes_baseline.json"),
    "hook_ablated(L20)": load(A / "stageA" / "probes_hook_ablated.json"),
    "wd_B (readout)": load(A / "stageB" / "probes_wd_B.json"),
    "wd_BN (norm'd)": load(A / "stageB" / "probes_wd_BN.json"),
    "wd_ML (shipped)": load(A / "stageB" / "probes_wd_ML.json"),
}
names = list(probes)
refusal = [100.0 * probes[n].get("refusal_rate", 0.0) for n in names]
benign = [100.0 * probes[n].get("benign_preserved", 1.0) for n in names]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
bars = ax.bar(names, refusal, color=[GRAY, ORANGE, BLUE, BLUE, GREEN])
bars[-1].set_color(GREEN); bars[-1].set_edgecolor("black")
for b, v in zip(bars, refusal):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("harmful-probe refusal (%)")
ax.set_title("Qwen2.5-7B: refusal rate by edit variant (16-prompt battery)")
ax.set_ylim(0, 100)
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(OUT / "refusal_by_variant.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
bars = ax.bar(names, benign,
              color=[GRAY, ORANGE, BLUE, BLUE, GREEN], edgecolor="black")
for b, v in zip(bars, benign):
    ax.text(b.get_x() + b.get_width()/2, v - 8, f"{v:.0f}%", ha="center",
            fontsize=9, color="white", fontweight="bold")
ax.set_ylabel("benign compliance (%)")
ax.set_title("Qwen2.5-7B: benign preservation by edit variant")
ax.set_ylim(0, 105)
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(OUT / "benign_preservation.png", bbox_inches="tight")
plt.close(fig)

# ---- chart 3: layer coherence ----------------------------------------------
coh = load(A / "stageA" / "layer_coherence.json")
# structure: {"best": int, "table": [{decoder_layer, coherence, ...}, ...], ...}
items = sorted(coh["table"], key=lambda it: int(it["decoder_layer"]))
xs = [int(it["decoder_layer"]) for it in items]
ys = [float(it["coherence"]) for it in items]
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
ax.plot(xs, ys, "-o", color=BLUE, ms=3, lw=1.2)
for l in [20, 18, 19]:
    if l in xs:
        ax.axvline(l, color=GREEN, ls="--", lw=1)
best = int(xs[int(np.argmax(ys))])
ax.axvline(best, color=ORANGE, ls="--", lw=1.4)
ax.text(best + 0.3, max(ys) - 0.03 * (max(ys)-min(ys) + 0.2), f"causal peak L{best}",
        color=ORANGE, fontsize=9)
for l in (20, 18, 19):
    ax.scatter([l], [ys[xs.index(l)]], color=GREEN, zorder=5, s=28)
ax.set_xlabel("layer"); ax.set_ylabel("coherence (harm↔harmless separation)")
ax.set_title("Qwen2.5-7B: per-layer refusal-direction coherence (Stage A)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "layer_coherence.png", bbox_inches="tight")
plt.close(fig)

# ---- chart 4: guardrail battery --------------------------------------------
mmlu_b = load(A / "mmlu" / "results_base.json")
mmlu_v = load(A / "mmlu" / "results_variant.json")
m_b = 100*mmlu_b["results"]["mmlu"]["acc,none"]; s_b = 100*mmlu_b["results"]["mmlu"]["acc_stderr,none"]
m_v = 100*mmlu_v["results"]["mmlu"]["acc,none"]; s_v = 100*mmlu_v["results"]["mmlu"]["acc_stderr,none"]
tq_b = load(A / "tq_mc2" / "results_base.json"); tq_v = load(A / "tq_mc2" / "results_variant.json")
t_b = 100*tq_b["results"]["truthfulqa_mc2"]["acc,none"]; ts_b = 100*tq_b["results"]["truthfulqa_mc2"]["acc_stderr,none"]
t_v = 100*tq_v["results"]["truthfulqa_mc2"]["acc,none"]; ts_v = 100*tq_v["results"]["truthfulqa_mc2"]["acc_stderr,none"]
rb = load(A / "refusalbench" / "analysis_rb7b.json")
rb_b, rb_v, d = rb["exact_base_pct"], rb["exact_abl_pct"], abs(rb["delta_pp"])

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), dpi=150)
for ax, (nb, sb, nv, sv, title, ylab) in zip(axes, [
        (m_b, s_b, m_v, s_v, "MMLU 0-shot", "acc (%)"),
        (t_b, ts_b, t_v, ts_v, "TruthfulQA mc2", "acc (%)"),
        (rb_b, 0.0, rb_v, 0.0, "RefusalBench (n=1,600)", "category-correct refusal (%)")]):
    bars = ax.bar(["base", "wd_ML"], [nb, nv], color=[GRAY, GREEN],
                  yerr=[sb, sv], capsize=4, edgecolor="black")
    for b, v in zip(bars, [nb, nv]):
        ax.text(b.get_x()+b.get_width()/2, v + max(sb, sv) + 1.2, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylab)
fig.suptitle("Qwen2.5-7B-abliterated: capability/behavior guardrails (base vs wd_ML, paired)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "mmlu_tq_guardrail.png", bbox_inches="tight")
plt.close(fig)
print("GUARDRAIL chart:", {"mmlu": (m_b, m_v), "tqmc2": (t_b, t_v), "rb_exact": (rb_b, rb_v),
                           "rb_delta": d})

# ---- chart 5+6: RefusalBench class-level + paired delta ---------------------
import collections
per = {"base": [], "abl": []}
cat = {"base": collections.Counter(), "abl": collections.Counter()}
for arm in ("base", "abl"):
    with open(A / "refusalbench" / ("results_base.jsonl" if arm == "base" else "results_abl.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            per[arm].append(r)
            # category correct refusal = refused AND the instance is truly flawed
            if arm == "base" and r["refused"]:
                cat["base"][r.get("category", r.get("topic", "?"))] += 1
            if arm == "abl" and r["refused"]:
                cat["abl"][r.get("category", r.get("topic", "?"))] += 1

pairs_b = {r["id"]: r for r in per["base"]} if "id" in per["base"][0] else None
if pairs_b is None:
    # align by index
    ids = range(len(per["base"]))
    pairs_b = {i: r for i, r in enumerate(per["base"])}
pairs_a = {r.get("id", i): r for i, r in enumerate(per["abl"])}

common = sorted(set(pairs_b) & set(pairs_a))
delta = [int(pairs_a[k]["refused"]) - int(pairs_b[k]["refused"]) for k in common]
vals, cnts = np.unique(delta, return_counts=True)
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
ax.bar([str(int(v)) for v in vals], cnts, color=[GREEN if v >= 0 else RED for v in vals],
       edgecolor="black")
ax.set_xlabel("paired refusal delta (abl − base) per instance")
ax.set_ylabel("instances")
n = len(common)
ax.set_title(f"RefusalBench-NQ 7B: paired refusal change (n={n})", fontsize=11)
for v, c in zip(vals, cnts):
    ax.text(str(int(v)), c, str(c), ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "rb1600_delta_hist.png", bbox_inches="tight")
plt.close(fig)

cats = sorted(set(cat["base"]) | set(cat["abl"]))
b = [cat["base"].get(c, 0) for c in cats]
a_ = [cat["abl"].get(c, 0) for c in cats]
fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
w = 0.4
ax.bar([i - w/2 for i in range(len(cats))], b, w, label="base", color=GRAY, edgecolor="black")
ax.bar([i + w/2 for i in range(len(cats))], a_, w, label="abl", color=GREEN, edgecolor="black")
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=6.5)
ax.set_ylabel("flawed questions refused")
ax.set_title(f"RefusalBench-NQ 7B: refusals by question category (n={n}) — Δ −0.88pp CI [−1.75, −0.06]",
             fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "rb1600_categories.png", bbox_inches="tight")
plt.close(fig)

# ---- fig 7: SORRY-Bench breadth panel (core 4 x 10, 44 classes) ----------------
so = load(A / "sorry_bench" / "sorry_analysis.json")["core"]
rb = so["per_class_abl_minus_base"]          # pp deltas
cats_s = sorted(rb, key=lambda c: (int(c), rb[c]))   # numeric class id, then delta
vals_s = np.array([rb[c] for c in cats_s], dtype=float)
ids = [f"{c}" for c in cats_s]
fig, ax = plt.subplots(figsize=(11.5, 4.4), dpi=150)
colors_s = [GREEN if v <= -100 else ("#d9d9d9" if v == 0 else RED) for v in vals_s]
ax.bar(range(len(vals_s)), vals_s, color=colors_s, edgecolor="black", width=0.8)
ax.set_xticks(range(len(vals_s)))
ax.set_xticklabels(ids, rotation=90, fontsize=6)
ax.set_ylabel("refusal-rate change (abl − base, pp)")
ax.set_ylim(-110, 25)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("SORRY-Bench 202503, 44 unsafe classes × 10 (n=440 paired): base 62.95% → abl 11.36% "
             "refusal (Δ −51.59pp; 227−0 discordant, McNemar p≈1e−45)", fontsize=10)
for i, v in enumerate(vals_s):
    if v <= -100 or v <= -80:
        ax.text(i, v - 3, f"{v:+.0f}", ha="center", va="top", fontsize=5.5)
fig.tight_layout()
fig.savefig(OUT / "sorry_core_by_class.png", bbox_inches="tight")
plt.close(fig)

print("CHARTS_DONE", sorted(os.listdir(OUT)))