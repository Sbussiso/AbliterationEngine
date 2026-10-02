#!/usr/bin/env python3
"""Run 004 charts — generated programmatically from recorded artifacts.
1. Primary endpoint: cat-match refusal accuracy base vs abl (bar, with CI).
2. Refusal-rate on flawed + FRR on answerable, per arm.
3. Per-class x intensity heatmap-style grouped bars (cat-match %).
Output: charts_run004/*.png (300 dpi matplotlib default Agg).
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#131417",
    "axes.facecolor": "#131417",
    "savefig.facecolor": "#131417",
    "text.color": "#e8e4da",
    "axes.edgecolor": "#3a3d44",
    "axes.labelcolor": "#e8e4da",
    "xtick.color": "#e8e4da",
    "ytick.color": "#e8e4da",
    "grid.color": "#26282e",
    "legend.framealpha": 0.2,
})
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
OUT = HERE / "charts_run004"
OUT.mkdir(exist_ok=True)

A = json.load(open(HERE / "analysis.json"))
S = A["strata"]

# ---- Fig 1: primary + binary + MRR/FRR grouped bar ----
labels = ["Category-match\nrefusal acc (flawed)", "Binary refusal\n(flawed)", "Missed refusal\n(MRR)", "False refusal\n(answerable)"]
b_vals = [A["base"]["catmatch"]*100, A["base"]["binary_refusal_on_flawed"]*100, A["base"]["MRR"]*100, A["base"]["FRR"]*100]
a_vals = [A["abl"]["catmatch"]*100, A["abl"]["binary_refusal_on_flawed"]*100, A["abl"]["MRR"]*100, A["abl"]["FRR"]*100]
x = range(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.bar([i-w/2 for i in x], b_vals, w, label="Qwen2.5-0.5B-Instruct (base)", color="#6b6b6b")
ax.bar([i+w/2 for i in x], a_vals, w, label="sbussiso/Qwen2.5-0.5B-abliterated", color="#c8a24a")
for i, (bv, av) in enumerate(zip(b_vals, a_vals)):
    ax.text(i-w/2, bv+1.2, f"{bv:.1f}", ha="center", fontsize=9)
    ax.text(i+w/2, av+1.2, f"{av:.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("%")
ax.set_title("Run 004 — RefusalBench-NQ paired eval, n=360 (240 flawed + 120 answerable), greedy, rule-based scorer")
ax.set_ylim(0, 105); ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig1_primary_metrics.png", dpi=200); plt.close(fig)

# ---- Fig 2: per-class cat-match grouped bars (HIGH + MEDIUM) ----
strata = sorted(set(k.split("|")[0] for k in S["base"]))
short = {"P-Ambiguity": "Ambiguity", "P-Contradiction": "Contradiction", "P-EpistemicMismatch": "EpistemicMismatch",
         "P-FalsePremise": "FalsePremise", "P-GranularityMismatch": "GranularityMismatch", "P-MissingInfo": "MissingInfo"}
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
for ax, intensity in zip(axes, ["HIGH", "MEDIUM"]):
    keys = [f"{c}|{intensity}" for c in strata]
    b = [100*S["base"][k]["cat_ok"]/S["base"][k]["n"] for k in keys]
    a = [100*S["abl"][k]["cat_ok"]/S["abl"][k]["n"] for k in keys]
    xx = range(len(keys))
    ax.bar([i-0.19 for i in xx], b, 0.38, label="base", color="#6b6b6b")
    ax.bar([i+0.19 for i in xx], a, 0.38, label="abl", color="#c8a24a")
    ax.set_xticks(xx); ax.set_xticklabels([short[c] for c in strata], rotation=30, ha="right", fontsize=8)
    ax.set_title(f"intensity={intensity}"); ax.grid(axis="y", alpha=0.3)
    if intensity == "HIGH": ax.set_ylabel("category-matched refusal accuracy (%)")
    ax.set_ylim(0, 20)
axes[0].legend()
fig.suptitle("Run 004 — cat-match refusal accuracy by perturbation class (n=20/stratum)")
fig.tight_layout(); fig.savefig(OUT / "fig2_classes.png", dpi=200); plt.close(fig)

# ---- Fig 3: per-class refusal rate (binary) grouped bars ----
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
for ax, intensity in zip(axes, ["HIGH", "MEDIUM"]):
    keys = [f"{c}|{intensity}" for c in strata]
    b = [100*S["base"][k]["ref_ok"]/S["base"][k]["n"] for k in keys]
    a = [100*S["abl"][k]["ref_ok"]/S["abl"][k]["n"] for k in keys]
    x = list(range(len(keys)))
    ax.bar([i-0.19 for i in x], b, 0.38, label="base", color="#6b6b6b")
    ax.bar([i+0.19 for i in x], a, 0.38, label="abl", color="#c8a24a")
    ax.set_xticks(x); ax.set_xticklabels([short[c] for c in strata], rotation=30, ha="right", fontsize=8)
    ax.set_title(f"intensity={intensity}"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
axes[0].set_ylabel("binary refusal rate on flawed (%)"); axes[0].legend()
fig.suptitle("Run 004 — binary refusal rate by class (flawed groundings)")
fig.tight_layout(); fig.savefig(OUT / "fig3_refusal_rates.png", dpi=300); plt.close(fig)

# QA: sizes + nonwhite
from PIL import Image
import numpy as np
for p in sorted(OUT.glob("*.png")):
    im = Image.open(p).convert("RGB")
    arr = np.asarray(im)
    nonwhite = (arr < 245).any(axis=2).mean()
    print(f"{p.name}: {im.size} nonwhite={nonwhite:.3f}")
print("CHARTS_DONE")