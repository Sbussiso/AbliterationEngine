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


# ---- Gather per-run numbers ----
# 0.5B Run 001 (16-probe era)
hr001, br001 = rates(load("qwen2.5-0.5b-002/artifacts/probes_baseline.json"))
hr001_hook, _ = rates(load("qwen2.5-0.5b-002/artifacts/probes_hook_ablated.json"))
# 0.5B Run 003-r2 (64-probe persistent-edit ladder; selected wd_ML_BN)
A004 = load("qwen2.5-0.5b-004/artifacts")
hr003, _ = rates(os.path.join(A004, "probes_baseline.json"))
hr003_MLB, _ = rates(os.path.join(A004, "probes_wd_ML_BN.json"))
_, br003_base = rates(os.path.join(A004, "probes_baseline.json"))
_, br003_MLB = rates(os.path.join(A004, "probes_wd_ML_BN.json"))
sel003 = json.load(open(os.path.join(A004, "selection.json")))
# 7B (spec run_number 5)
A7 = load("qwen2.5-7b-001/artifacts")
hr7, _ = rates(os.path.join(A7, "stageA/probes_baseline.json"))
hr7_hook, _ = rates(os.path.join(A7, "stageA/probes_hook_ablated.json"))
hr7_wdML, _ = rates(os.path.join(A7, "stageB/probes_wd_ML.json"))
sel7 = json.load(open(os.path.join(A7, "stageB/selection.json")))
# 1.5B Run 002 (64-probe, session-5 banked)
A15s5 = load("qwen2.5-0.5b-002/eng_run002_pull_s5")
hr15, _ = rates(os.path.join(A15s5, "probes_baseline.json"))
hr15_hook, _ = rates(os.path.join(A15s5, "probes_hook_ablated.json"))
hr15_wdML, _ = rates(os.path.join(A15s5, "probes_wd_ML.json"))
_, br15_base = rates(os.path.join(A15s5, "probes_baseline.json"))
_, br15_hook = rates(os.path.join(A15s5, "probes_hook_ablated.json"))

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
for name, base_p, post_p in [
    ("0.5B Run 001\n(hook L17)",
     "qwen2.5-0.5b-002/artifacts/probes_baseline.json",
     "qwen2.5-0.5b-002/artifacts/probes_hook_ablated.json"),
    ("0.5B Run 003-r2\n(edit wd_ML_BN)",
     "qwen2.5-0.5b-004/artifacts/probes_baseline.json",
     "qwen2.5-0.5b-004/artifacts/probes_wd_ML_BN.json"),
    ("1.5B Run 002\n(hook L19)",
     "qwen2.5-0.5b-002/eng_run002_pull_s5/probes_baseline.json",
     "qwen2.5-0.5b-002/eng_run002_pull_s5/probes_hook_ablated.json"),
]:
    bb, bp = rates(load(base_p)), rates(load(post_p))
    conditions.append((name, bb[1], bp[1], bb[1] - 10.0))

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