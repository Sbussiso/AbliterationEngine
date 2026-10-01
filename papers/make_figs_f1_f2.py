"""F1+F2 figures for the FTT-13 paper — read every digit from committed
probe JSONs (user chart doctrine: never hand-typed numbers).

F1: harmful refusal rate by condition (baseline / hook / wd_B / wd_BN /
    wd_ML / wd_ML_BN) with the 25% publish gate line.
F2: benign preservation by condition, each drawn against its own
    baseline-10pp floor.

Outputs papers/figures/f1_ladder_refusal.png, f2_benign_preservation.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "qwen2.5-0.5b-002")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

PROBES = {
    "baseline": "eng_run002_pull_s5/probes_baseline.json",
    "hook": "eng_run002_pull_s5/probes_hook_ablated.json",
    "wd_B": "eng_run002_pull_s5/probes_wd_B.json",
    "wd_BN": "eng_run002_pull_s5/probes_wd_BN.json",
    "wd_ML": "eng_run002_pull_s5/probes_wd_ML.json",
    "wd_ML_BN": "reapersplit/rs2-1-forensics/probes_wd_ML_BN.json",
}
LABELS = {
    "hook": "hook\n(inference-time)",
    "wd_B": "wd_B\n(lm_head)",
    "wd_BN": "wd_BN\n(lm_head+norm)",
    "wd_ML": "wd_ML\n(K=3 multi-layer)",
    "wd_ML_BN": "wd_ML_BN\n(K=5 composite)",
}

rows = {}
for cond, rel in PROBES.items():
    d = json.load(open(os.path.join(ROOT, rel)))
    h, b = d["harmful"], d["harmless"]
    n_h, n_b = len(h), len(b)
    ref_h = sum(1 for x in h if x.get("refused"))
    rows[cond] = {
        "refusal_rate": ref_h / n_h,
        "benign_preserved": (n_b - sum(1 for x in b if x.get("refused")))
        / n_b,
        "n_h": n_h, "n_b": n_b,
    }

conds = list(PROBES)
base_pres = rows["baseline"]["benign_preserved"]
floor = base_pres - 0.10

C_BAR = "#c8a24a"
C_REF = "#3f6d8e"
C_BASE = "#6b6b6b"
colors = [C_BASE] + [C_REF] + [C_BAR] * 4

plt.rcParams.update({
    "figure.facecolor": "#131417", "axes.facecolor": "#131417",
    "savefig.facecolor": "#131417", "text.color": "#e8e4da",
    "axes.edgecolor": "#3a3d44", "axes.labelcolor": "#e8e4da",
    "xtick.color": "#e8e4da", "ytick.color": "#e8e4da",
    "grid.color": "#26282e", "font.family": "DejaVu Sans",
    "font.size": 10,
})

# ---- F1
fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=160)
xs = range(len(conds))
vals = [rows[c]["refusal_rate"] * 100 for c in conds]
bars = ax.bar(xs, vals, color=colors, width=0.62)
ax.axhline(25, color="#d96f5c", lw=1.6, ls="--")
ax.text(len(conds) - 0.45, 26.5, "publish gate, 25%",
        color="#d96f5c", fontsize=9, ha="right")
for x, v in zip(xs, vals):
    ax.text(x, v + 1.6, f"{v:.1f}%", ha="center", fontsize=9)
labels = ["baseline"] + [LABELS[c] for c in conds[1:]]
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, fontsize = 8.6)
ax.set_ylabel("harmful-prompt refusal rate (%)")
ax.set_title("Run 002 (Qwen2.5-1.5B): refusal rate by edit condition",
             fontsize=11, pad=10)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f1_ladder_refusal.png"))
plt.close(fig)

# ---- F2
fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=160)
vals2 = [rows[c]["benign_preserved"] * 100 for c in conds]
ax.bar(xs, vals2, color=colors, width=0.62)
ax.axhline(floor * 100, color="#d96f5c", lw=1.6, ls="--")
ax.text(len(conds) - 0.45, floor * 100 + 1.6,
        f"floor, baseline −10pp = {floor*100:.1f}%",
        color="#d96f5c", fontsize=9, ha="right")
for x, v in zip(xs, vals2):
    ax.text(x, v + 1.4, f"{v:.1f}%", ha="center", fontsize=9)
ax.axhline(base_pres * 100, color="#9aa0a6", lw=1, ls=":")
ax.text(0.02, base_pres * 100 + 1.2, "baseline",
        color="#9aa0a6", fontsize=8.6)
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, fontsize=8.6)
ax.set_ylabel("benign probes answered (%)")
ax.set_title("Run 002: benign preservation (utility guardrail)",
             fontsize=11, pad=10)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f2_benign_preservation.png"))
plt.close(fig)

print("wrote f1_ladder_refusal.png + f2_benign_preservation.png from",
      f"{len(conds)} committed probe files; floor={floor:.4f}")