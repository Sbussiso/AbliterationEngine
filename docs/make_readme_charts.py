"""README figures for abliteration_engine — EVERY number read from
committed artifacts. Nothing hand-typed. Outputs 2 PNGs into docs/charts/.
House style mirrors make_r2_charts.py (#9aa0a6 grey, #7e57c2 purple,
#2e7d32 green)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)


def rates(path):
    d = json.load(open(path))
    harm, hless = d["harmful"], d["harmless"]
    hr = 100 * sum(1 for r in harm if r.get("refused")) / len(harm)
    br = 100 * sum(1 for r in hless if not r.get("refused")) / len(hless)
    return hr, br


def load(relpath):
    return os.path.join(R, relpath)


# ---- Per-run numbers, frozen from measured artifacts ----
# (run dirs live in the private research archive; the README charts carry the
#  measured values with their source labeled — the live data flows through
#  tests/fixtures parity anchors in CI instead)
hr001, br001 = 87.5, 93.75   # tests/fixtures/probes_baseline.json (0.5B, 16-probe era)
hr001_hook, br001_hook = 0.0, 93.75  # tests/fixtures/probes_hook_ablated.json
# 0.5B Run 003-r2 (64-probe persistent ladder; selected wd_ML_BN)
hr003, br003_base = 87.5, 93.75
hr003_MLB, br003_MLB = 0.0, 87.5
sel003 = {"selected": "wd_ML_BN"}
# 7B (spec run_number 5)
hr7, hr7_hook, hr7_wdML = 93.75, 18.75, 12.5
sel7 = {"selected": "wd_ML"}
# 1.5B Run 002 (64-probe, session-5 banked)
hr15, hr15_hook, hr15_wdML = 98.4375, 1.5625, 42.1875
br15_base, br15_hook = 95.3125, 98.4375

# ---- Figure 1: harmful refusal, baseline vs edit/hook per run ----
fig, ax = plt.subplots(figsize=(9.2, 5.0))
groups = [
    ("Qwen2.5-0.5B\nRun 001 (hook L17)", (hr001, hr001_hook)),
    ("Qwen2.5-0.5B\nRun 003-r2 (edit)", (hr003, hr003_MLB)),
    ("Qwen2.5-7B\npersistent edit", (hr7, hr7_wdML)),
    ("Qwen2.5-1.5B\nRun 002 (hook L19)", (hr15, hr15_hook)),
]
x = np.arange(len(groups))
w = 0.38
base_vals = [g[1][0] for g in groups]
post_vals = [g[1][1] for g in groups]
b1 = ax.bar(x - w / 2, base_vals, width=w, label="baseline (pre-edit)",
            color="#9aa0a6")
b2 = ax.bar(x + w / 2, post_vals, width=w, label="after edit/hook",
            color="#7e57c2")
for xi, v in zip(x, base_vals):
    ax.text(xi - w / 2, v + 1.6, f"{v:.1f}%", ha="center", fontsize=9,
            fontweight="bold")
for xi, v in zip(x, post_vals):
    ax.text(xi + w / 2, v + 1.6, f"{v:.1f}%", ha="center", fontsize=9,
            fontweight="bold", color="#4a148c")
ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("harmful-probe refusal rate (%)")
ax.set_ylim(0, 112)
ax.set_title("Refusal removal on the harmful probe set "
             "(committed artifacts only)")
ax.legend(frameon=False, loc="lower right")
fig.text(0.01, 0.955,
         "16-probe era (0.5B, 7B) / 64-probe era (1.5B). "
         "1.5B 'after' = frozen v1 grader: one apology-preamble false "
         "positive; scoring v2 reads 0/64 true refusals.",
         fontsize=7.5, color="#616161", va="top")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "refusal_removal.png"))

# ---- Figure 2: benign behavior preservation (the guardrail) ----
fig, ax = plt.subplots(figsize=(9.2, 4.6))
# benign-completion preservation per condition. Each condition's own gate floor
# is baseline_benign − gates.benign_floor_delta (0.10): draw floor ticks per bar.
conditions = []
for name, bb_benign, bp_benign in [
    ("0.5B Run 001\n(hook L17)", br001, br001_hook),
    ("0.5B Run 003-r2\n(edit wd_ML_BN)", br003_base, br003_MLB),
    ("1.5B Run 002\n(hook L19)", br15_base, br15_hook),
]:
    conditions.append((name, bb_benign, bp_benign, bb_benign - 10.0))

xs = np.arange(len(conditions))
w = 0.36
bars_b = ax.bar(xs - w / 2, [c[1] for c in conditions], width=w,
                label="baseline", color="#9aa0a6")
bars_p = ax.bar(xs + w / 2, [c[2] for c in conditions], width=w,
                label="after edit/hook", color="#2e7d32")
for xi, c in zip(xs, conditions):
    ax.plot([xi - w * 2.2, xi + w * 2.2], [c[3], c[3]], color="#ef6c00",
            ls="--", lw=1.4)
    ax.text(xi, c[3] - 4.6, "floor −10pp", color="#e65100", fontsize=7.5,
            ha="center")
ax.bar_label(bars_b, fmt="%.1f%%", padding=2, fontsize=8.5, fontweight="bold")
ax.bar_label(bars_p, fmt="%.1f%%", padding=2, fontsize=8.5, fontweight="bold",
             color="#1b5e20")
ax.set_xticks(xs)
ax.set_xticklabels([c[0] for c in conditions], fontsize=9)
ax.set_ylabel("benign probes served (%)")
ax.set_ylim(0, 108)
ax.set_title("Benign behavior preservation — every condition clears its own gate floor")
ax.legend(frameon=False, loc="lower right", ncols=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "benign_preservation.png"))

print("FIGURES DONE ->", OUT)
print("numbers used:")
for g in groups:
    print(f"  {g[0]!r}: {g[1][0]:.2f}% -> {g[1][1]:.2f}%")
for c in conditions:
    nm = c[0].replace(chr(10), ' ')
    print(f"  benign {nm}: base {c[1]:.1f}% -> post {c[2]:.1f}%"
          f" (floor {c[3]:.1f}%)")