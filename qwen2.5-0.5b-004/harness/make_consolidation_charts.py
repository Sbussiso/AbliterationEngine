#!/usr/bin/env python3
"""Consolidation-update charts: TruthfulQA paired delta + multilingual panel.

Data sources (recorded artifacts, never hand-typed):
  - consolidation/tq240_base/**/results_*.json  (base leg, truthfulqa_mc2)
  - consolidation/tq240_r2/**/results_*.json    (variant leg, truthfulqa_mc2)
  - consolidation/probes_multilingual.json      (en/zh/ru/de panel, 3 conditions)

Run inside the repo it documents:
  python3 make_consolidation_charts.py  (expects the artifacts/ dir layout of
  /root/research/abliteration/qwen2.5-0.5b-004/)
Outputs two PNGs into charts_consolidation/ and copies nothing elsewhere.
"""
import json
import math
import os
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root = run dir
CONS = os.path.join(ROOT, "consolidation")
OUT = os.path.join(ROOT, "artifacts", "charts_consolidation")
os.makedirs(OUT, exist_ok=True)

FG = "#1a1a2e"
MUT = "#6b7280"
ACC = "#e11d48"
GRID = "#e5e7eb"

plt.rcParams.update({
    "font.size": 11, "text.color": FG, "axes.edgecolor": MUT,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


def load_mc2(results_dir):
    files = glob.glob(os.path.join(results_dir, "**", "results_*.json"), recursive=True)
    if not files:
        sys.exit(f"no results json under {results_dir}")
    with open(files[0]) as fh:
        d = json.load(fh)
    acc = d["results"]["truthfulqa_mc2"]["acc,none"]
    stderr = d["results"]["truthfulqa_mc2"]["acc_stderr,none"]
    return acc, stderr, files[0]


base_acc, base_se, base_file = load_mc2(os.path.join(CONS, "tq240_base"))
var_acc, var_se, var_file = load_mc2(os.path.join(CONS, "tq240_r2"))
n = 240
delta = base_acc - var_acc
se_pair = math.sqrt(base_se ** 2 + var_se ** 2)

# ---- Figure 1: TruthfulQA paired bars --------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
labels = ["Qwen2.5-0.5B-Instruct\n(base)", "Abliterated composite\n(wd_ML_BN, current main)"]
vals = [base_acc * 100, var_acc * 100]
errs = [base_se * 100, var_se * 100]
colors = ["#9ca3af", ACC]
bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, width=0.55,
              error_kw={"ecolor": FG, "elinewidth": 1.2})
for b, v, e in zip(bars, vals, errs):
    ax.text(b.get_x() + b.get_width() / 2, v + e + 1.0, f"{v:.2f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_ylabel("TruthfulQA mc2 accuracy (%)")
ax.set_ylim(0, 55)
ax.set_title(f"TruthfulQA (240-question paired subsample): delta = {delta*100:+.2f}pp\n"
             f"95% CI [{(delta-1.96*se_pair)*100:+.2f}, {(delta+1.96*se_pair)*100:+.2f}]  ·  "
             "Arditi et al. cross-model range: −1.0 to −3.5pp",
             fontsize=11, loc="left")
ax.text(0.0, -0.16,
        f"lm-eval truthfulqa_mc2 · fp16 · batch 8 · greedy · n={n} · sources: "
        f"{os.path.relpath(base_file, ROOT)} , {os.path.relpath(var_file, ROOT)}",
        transform=ax.transAxes, fontsize=7, color=MUT)
fig.tight_layout()
p1 = os.path.join(OUT, "consol_truthfulqa_delta.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: multilingual panel ------------------------------------------
with open(os.path.join(CONS, "probes_multilingual.json")) as fh:
    ml = json.load(fh)

LANGS = ["en", "zh", "ru", "de"]
CONDS = [("baseline", "Baseline", "#9ca3af"),
         ("hook", "Hook (L17)", "#2563eb"),
         ("variant", "Composite edit\n(wd_ML_BN)", ACC)]

def rate(cond, lang, cat):
    rows = [r for r in ml["conditions"][cond]["rows"]
            if r["lang"] == lang and r["category"] == cat]
    return sum(r["refused"] for r in rows), len(rows)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True,
                         gridspec_kw={"width_ratios": [3, 3]})
x = list(range(len(LANGS)))
w = 0.25

ax = axes[0]
for k, (cond, label, color) in enumerate(CONDS):
    vals, hts = [], []
    for lang in LANGS:
        r, n_ = rate(cond, lang, "harmful")
        vals.append(100 * r / n_)
    bars = ax.bar([i + (k - 1) * w for i in x], vals, width=w, color=color, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9, color=FG)
ax.set_xticks(x, [f"{l}" for l in LANGS])
ax.set_title("Harmful prompts refused (%)", fontsize=11, loc="left")
ax.set_ylabel("refusal rate (%)")
ax.set_ylim(0, 115)
ax.legend(frameon=False, fontsize=9, loc="upper right")

ax = axes[1]
for k, (cond, label, color) in enumerate(CONDS):
    vals = []
    for lang in LANGS:
        r, n_ = rate(cond, lang, "benign")
        vals.append(100 * r / n_)
    bars = ax.bar([i + (k - 1) * w for i in x], vals, width=w, color=color, label=label)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, color=FG)
ax.set_xticks(x, LANGS)
ax.set_title("Benign prompts refused (%) — over-refusal", fontsize=11, loc="left")

fig.suptitle("Multilingual mini-panel (en/zh/ru/de) — 2 harmful + 2 benign per language, paired translations",
             fontsize=12, x=0.02, ha="left")
fig.text(0.02, -0.02,
         "scorer: run-003 constant markers + per-language refusal markers · greedy, seed 0 · "
         "n=2/cell (directional only) · source: consolidation/probes_multilingual.json",
         fontsize=7, color=MUT)
# (tight_layout warns because suptitle/footnote overlap the shared axes —
#  acceptable: savefig(bbox_inches="tight") handles the final crop)
p2 = os.path.join(OUT, "consol_multilingual_panel.png")
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)

print("CHARTS_OK", p1, p2, sep="\n")