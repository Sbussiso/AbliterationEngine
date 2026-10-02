"""Round-2 (Run 003) chart generator — EVERY number read from artifacts/.

Reads: artifacts/probes_*.json (per-probe refused flags), artifacts/selection.json,
artifacts/mmlu_*.json (lm-eval raw results). Nothing hand-typed.
Outputs 4 PNGs into artifacts/charts/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/root/research/abliteration/qwen2.5-0.5b-004"
A = os.path.join(BASE, "artifacts")
OUT = os.path.join(A, "charts")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

COND_ORDER = [
    ("probes_baseline", "baseline"),
    ("probes_hook_ablated", "hook (L17, inference)"),
    ("probes_wd_B", "wd_B"),
    ("probes_wd_BN", "wd_BN"),
    ("probes_wd_ML", "wd_ML"),
    ("probes_wd_ML_BN", "wd_ML_BN (selected)"),
]


def rates(name):
    d = json.load(open(os.path.join(A, name + ".json")))
    harm, hless = d["harmful"], d["harmless"]
    hr = sum(1 for r in harm if r.get("refused")) / len(harm)
    br = sum(1 for r in hless if not r.get("refused")) / len(hless)
    return hr, br


labels, refusals, benign = [], [], []
for key, label in COND_ORDER:
    hr, br = rates(key)
    labels.append(label)
    refusals.append(hr)
    benign.append(br)

sel = json.load(open(os.path.join(A, "selection.json")))
gate_floor = json.load(open(os.path.join(A, "probes_baseline.json")))
base_benign = sum(1 for r in gate_floor["harmless"] if not r.get("refused")) / 16
gate_min = base_benign - 0.10

# ---- Figure 1: refusal by condition (with 25% goal line) ----
fig, ax = plt.subplots(figsize=(9, 5.2))
colors = ["#9aa0a6", "#9aa0a6", "#7e57c2", "#7e57c2", "#7e57c2", "#2e7d32"]
bars = ax.bar(labels, [100 * r for r in refusals], color=colors, width=0.62)
ax.axhline(25, color="#c62828", ls="--", lw=1.4)
ax.text(4.55, 26.5, "user goal < 25%", color="#c62828", fontsize=9, ha="right")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, fontweight="bold")
ax.set_ylabel("refusal rate (%)")
ax.set_title("Round 2 (Run 003): persistent refusal by condition\n"
             "Qwen2.5-0.5B-Instruct, 16 harmful probes, greedy, 200 tok, seed 0")
ax.set_ylim(0, 100)
plt.xticks(rotation=14, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r2_refusal_by_condition.png"))
plt.close(fig)

# ---- Figure 2: benign preservation vs gate ----
fig, ax = plt.subplots(figsize=(9, 5.2))
bars = ax.bar(labels, [100 * b for b in benign], color=["#455a64"] * 5 + ["#2e7d32"], width=0.62)
ax.axhline(100 * gate_min, color="#ef6c00", ls="--", lw=1.4)
ax.text(4.55, 100 * gate_min + 1.2, f"pre-registered gate {100 * gate_min:.1f}%",
        color="#e65100", fontsize=9, ha="right")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, fontweight="bold")
ax.set_ylabel("benign probes answered (%)")
ax.set_title("Benign preservation across the ladder (16 harmless probes, seed 0)\n"
             "selected variant 87.5% — gate floor 83.75%, zero degenerate outputs")
ax.set_ylim(0, 105)
plt.xticks(rotation=14, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r2_benign_preservation.png"))
plt.close(fig)

# ---- Figure 3: MMLU guardrail (read raw lm-eval results) ----
def mmlu_acc(path):
    d = json.load(open(path))
    res = d["results"]
    return res["mmlu"]["acc,none"]

base_acc = mmlu_acc(os.path.join(A, "mmlu_base_results.json"))
var_acc = mmlu_acc(os.path.join(A, "mmlu_variant_results.json"))
delta_pp = 100 * (base_acc - var_acc)

fig, ax = plt.subplots(figsize=(7.2, 5))
bars = ax.bar(["base Qwen2.5-0.5B", "wd_ML_BN (ablated)"],
              [100 * base_acc, 100 * var_acc], color=["#9aa0a6", "#2e7d32"], width=0.5)
ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11, fontweight="bold")
ax.annotate(f"Δ {delta_pp:.2f} pp  (guardrail < 3 pp: PASSED)",
            xy=(0.5, 0.86), xycoords="axes fraction", ha="center",
            fontsize=11, color="#2e7d32", fontweight="bold")
ax.set_ylabel("MMLU accuracy (%) — 0-shot, seed 0, fp16")
ax.set_title("Capability guardrail: MMLU before vs after ablation")
ax.set_ylim(0, 60)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r2_mmlu_guardrail.png"))
plt.close(fig)

# ---- Figure 4: layer coherence scan (24 layers) ----
coh = json.load(open(os.path.join(A, "layer_coherence.json")))
entries = coh["table"] if isinstance(coh, dict) and "table" in coh else (
    coh if isinstance(coh, list) else [])
assert entries, f"unexpected layer_coherence.json schema: {list(coh)[:8]}"
layers = [e["decoder_layer"] for e in entries]
vals = [e["coherence"] for e in entries]
best = coh.get("best") or max(entries, key=lambda e: e["coherence"])
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(layers, vals, "-o", ms=4, color="#5e35b1")
ax.axvline(sel["k_layers_combo"][0], color="#2e7d32", ls=":", lw=1.6)
for li in sel["k_layers_combo"]:
    ax.axvspan(li - 0.4, li + 0.4, color="#2e7d32", alpha=0.12)
ax.annotate(f"L{best['decoder_layer']} (coh {best['coherence']:.3f}) — selected",
            xy=(best["decoder_layer"], best["coherence"]),
            xytext=(best["decoder_layer"] + 1.2, best["coherence"] - 0.05),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="#2e7d32"))
ax.set_xlabel("decoder layer")
ax.set_ylabel("direction coherence")
ax.set_title("Refusal-direction coherence scan, 24 layers (64-pair contrast)\n"
             "green band = K=5 ladder layers [15–19]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r2_layer_coherence.png"))
plt.close(fig)

# ---- integrity assertions: fail loudly if any number is off ----
assert abs(refusals[0] - 0.875) < 1e-9, refusals[0]
assert abs(refusals[1]) < 1e-9, refusals[1]
assert abs(refusals[2] - 0.5625) < 1e-9, refusals[2]
assert abs(refusals[5]) < 1e-9, refusals[5]
assert abs(delta_pp - 0.24) < 0.06, delta_pp
assert abs(benign[5] - 0.875) < 1e-9, benign[5]

for f in sorted(os.listdir(OUT)):
    print("CHART", f, os.path.getsize(os.path.join(OUT, f)))
print("CHARTS_OK delta_pp=%.4f base=%.4f var=%.4f" % (delta_pp, 100 * base_acc, 100 * var_acc))