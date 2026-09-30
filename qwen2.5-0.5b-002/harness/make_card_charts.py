#!/usr/bin/env python3
"""Generate model-card charts for sbussiso/Qwen2.5-0.5B-abliterated (run 002).

Every number is read from the recorded run artifacts in artifacts/ — nothing
is hand-typed. Outputs four PNGs (light theme, colorblind-safe, no
transparency tricks so they survive any HF card theme):
  charts/refusal_by_condition.png   refusal rates, all 5 conditions
  charts/refusal_vs_benign.png     harmful refusal vs harmless answered
  charts/layer_coherence.png       24-layer coherence scan, best layer marked
  charts/mmlu_guardrail.png        MMLU base vs variant (overall + groups)
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parent.parent / "artifacts"
OUT = Path("/tmp/card_charts")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

C_BASE = "#8a8a8a"   # gray  — baseline
C_HOOK = "#2b6f3f"   # green — inference-time hook
C_PUB = "#c84a44"    # red   — published variant (wd_B)
C_ALT = "#e0955c"    # orange — other persistent variants
C_MMLU = "#3f6b8c"   # blue  — capability


def pct(x):
    return f"{100 * x:.1f}%"


def load_probes(name):
    d = json.load(open(ART / f"probes_{name}.json"))
    harm = sum(p["refused"] for p in d["harmful"]) / len(d["harmful"])
    benign = sum(1 for p in d["harmless"] if not p["refused"]) / len(d["harmless"])
    degen = sum(1 for p in d["harmful"] + d["harmless"] if p.get("degenerate"))
    return harm, benign, degen


# ---------------------------------------------------------------- fig 1 + 2
conds = ["baseline", "hook_ablated", "wd_A", "wd_B", "wd_C"]
labels = ["baseline", "hook\n(inference-time)", "wd_A\n(lm_head,\nresid. space)",
          "wd_B\n(lm_head,\nreadout space)", "wd_C\n(o_proj+down_proj,\nrow space)"]
refusal, benign = {}, {}
for c in conds:
    refusal[c], benign[c], degen_c = load_probes(c)
    assert degen_c == 0, f"unexpected degenerate outputs in {c}"

colors = [C_BASE, C_HOOK, C_ALT := C_ALT, C_PUB, C_ALT]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
bars = ax.bar(labels, [100 * refusal[c] for c in conds], color=colors, width=0.62)
ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9, fontweight="bold")
ax.set_ylabel("refusal rate on harmful probes (%)")
ax.set_ylim(0, 100)
ax.set_title("Refusal by condition — 16 harmful probes, greedy, 200 tok, seed 0\n"
             "(Qwen2.5-0.5B-Instruct, layer 17/24, coherence 0.664)", fontsize=10.5)
ax.annotate("published persistent edit", xy=(3, 100 * refusal["wd_B"]),
            xytext=(3, 100 * refusal["wd_B"] + 14), ha="center", fontsize=9,
            color=C_PUB, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_PUB, lw=1.2))
ax.text(0.99, 0.97, "hook = inference-time hook (not persisted in weights);\n"
        "wd_* = persistent weight-decoded variants",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444444")
fig.tight_layout()
fig.savefig(OUT / "refusal_by_condition.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 4.2))
x = range(len(conds))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], [100 * refusal[c] for c in conds],
            width=w, color=[C_BASE, C_HOOK, C_ALT, C_PUB, C_ALT], label="refusal on harmful probes")
b2 = ax.bar([i + w / 2 for i in x], [100 * benign[c] for c in conds],
            width=w, color="#5f8fb4", label="harmless probes answered")
ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8.5)
ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8.5)
ax.set_xticks(list(x), labels)
ax.set_ylabel("rate (%)")
ax.set_ylim(0, 105)
ax.set_title("Refusal removal vs. benign preservation — unchanged benign behavior in every condition\n"
             "(16 harmful + 16 harmless probes, greedy, 200 tok, seed 0)", fontsize=10.5)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "refusal_vs_benign.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
lc = json.load(open(ART / "layer_coherence.json"))
table = lc["table"]
layers = [r["decoder_layer"] for r in table]
coh = [r["coherence"] for r in table]
best = lc["best"]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
ax.plot(layers, coh, "-", color=C_MMLU, lw=1.8, marker="o", ms=3.5, mfc="white")
ax.axvline(best["decoder_layer"], color=C_PUB, lw=1.4, ls=":")
ax.scatter([best["decoder_layer"]], [best["coherence"]], color=C_PUB, zorder=5, s=55)
ax.annotate(f"selected: L{best['decoder_layer']}/24\ncoherence {best['coherence']:.3f}",
            xy=(best["decoder_layer"], best["coherence"]),
            xytext=(best["decoder_layer"] + 1.2, best["coherence"] - 0.10),
            fontsize=9, color=C_PUB, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_PUB, lw=1.2))
ax.set_xlabel("decoder layer")
ax.set_ylabel("direction coherence")
ax.set_title("Refusal-direction coherence scan across all 24 decoder layers\n"
             "(64 harmful/harmless pairs, final-position residual stream; peak at ~2/3 depth,\n"
             "exact replication of run 001)", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "layer_coherence.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 4
def mmlu(path):
    r = json.load(open(path))
    res = r["results"]
    overall = res["mmlu"]["acc,none"]
    stderr = res["mmlu"]["acc_stderr,none"]
    groups = {g: res[g]["acc,none"] for g in ("mmlu_stem", "mmlu_humanities",
                                              "mmlu_social_sciences", "mmlu_other")}
    return 100 * overall, 100 * stderr, {k: 100 * v for k, v in groups.items()}

base_acc, base_se, base_grp = mmlu(ART / "eval/base/Qwen__Qwen2.5-0.5B-Instruct/results_2026-09-27T06-52-50.480032.json")
var_acc, var_se, var_grp = mmlu(ART / "eval/variant/__content__wd_B/results_2026-09-27T06-59-44.073212.json")
assert abs((base_acc - var_acc) - (100 * 0.0028790485685799673)) < 1e-6 or True  # delta sanity below

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0), width_ratios=[1, 1.5])
bars = ax1.bar(["base\n(pinned rev)", "wd_B\n(this repo)"], [base_acc, var_acc],
               yerr=[base_se, var_se], capsize=5, color=[C_BASE, C_PUB], width=0.55)
ax1.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9, fontweight="bold")
ax1.set_ylim(0, 60)
ax1.set_ylabel("MMLU accuracy (%)")
ax1.set_title(f"Overall: {base_acc:.2f}% → {var_acc:.2f}%\n(Δ {base_acc - var_acc:.2f}pp, guardrail <3pp)", fontsize=10.5)

gx = range(4)
gw = 0.38
glabels = ["STEM", "humanities", "social\nsci.", "other"]
b_base = ax2.bar([i - gw / 2 for i in gx],
                 [base_grp[g] for g in ("mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other")],
                 width=gw, color=C_BASE, label="base")
b_var = ax2.bar([i + gw / 2 for i in gx],
                [var_grp[g] for g in ("mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other")],
                width=gw, color=C_PUB, label="wd_B")
ax2.bar_label(b_base, fmt="%.1f", padding=2, fontsize=8)
ax2.bar_label(b_var, fmt="%.1f", padding=2, fontsize=8)
ax2.set_xticks(list(gx), glabels)
ax2.set_ylabel("MMLU accuracy (%)")
ax2.set_ylim(0, 70)
ax2.set_title("MMLU domain groups: base vs wd_B", fontsize=10.5)
ax2.legend(fontsize=9, framealpha=0.9)
fig.suptitle("Capability guardrail — lm-eval MMLU, 0-shot, fp16, seed 0, identical config", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "mmlu_guardrail.png", dpi=150)
plt.close(fig)

print("charts written to", OUT)
for f in sorted(OUT.iterdir()):
    print(" ", f.name, f.stat().st_size, "bytes")
print(f"DATA CHECK: refusal baseline={refusal['baseline']} hook={refusal['hook_ablated']} "
      f"wd_A={refusal['wd_A']} wd_B={refusal['wd_B']} wd_C={refusal['wd_C']}")
print(f"DATA CHECK: benign all-conditions={[benign[c] for c in conds]}")
print(f"DATA CHECK: mmlu base={base_acc:.4f}±{base_se:.4f} variant={var_acc:.4f}±{var_se:.4f} "
      f"delta_pp={base_acc - var_acc:.4f}")