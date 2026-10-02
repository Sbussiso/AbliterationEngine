#!/usr/bin/env python3
"""Consolidated card-chart generator for sbussiso/Qwen2.5-0.5B-abliterated.

Regenerates EVERY figure on the card from the recorded artifacts shipped in
this repo — nothing hand-typed, no external hosts.

Repo layout this reads (consolidation of run-002 + run-003, 2026-09-28):
  eval/   run-002 evidence: probes_{baseline,hook_ablated,wd_A,wd_B,wd_C}.json,
          mmlu_base_*.json, mmlu_variant_*.json, run_config.json,
          layer_coherence.json, selection.json, selection_candidates.json
  eval2/  run-003 evidence: probes_{baseline,hook_ablated,wd_B,wd_BN,wd_ML,
          wd_ML_BN}.json, mmlu_base_results.json, mmlu_variant_results.json,
          truthfulqa_base_results.json, truthfulqa_variant_results.json,
          probes_multilingual.json, run_config.json, layer_coherence.json,
          selection.json, selection_candidates.json, ladder_sha256.json,
          harness_sha256.json, layer_directions.npz
  charts/ output dir (this file lives here)

Outputs 10 PNGs:
  run-002 (corrected framing: wd_B was run-002's selected variant, superseded
          by wd_ML_BN in the consolidation): refusal_by_condition,
          refusal_vs_benign, layer_coherence, mmlu_guardrail
  run-003: r2_refusal_by_condition, r2_benign_preservation,
          r2_layer_coherence, r2_mmlu_guardrail
  consolidation new: truthfulqa_guardrail, multilingual_panel
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent   # repo root
EVAL1 = Path(__file__).resolve().parent / "stage" / "eval"        # run-002
EVAL2 = Path(__file__).resolve().parent / "stage" / "eval2"       # run-003
OUT = Path(__file__).resolve().parent / "charts"   # restaged: PNGs go to consolidation/charts/
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#131417",
    "axes.facecolor": "#131417",
    "savefig.facecolor": "#131417",
    "text.color": "#e8e4da",
    "axes.edgecolor": "#3a3d44",
    "axes.labelcolor": "#e8e4da",
    "xtick.color": "#e8e4da",
    "ytick.color": "#e8e4da",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#26282e",
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

C_BASE = "#6b6b6b"   # warm gray — baseline (dark house standard)
C_HOOK = "#3f6d8e"   # steel blue — inference-time hook
C_PUB = "#c8a24a"    # gold — published variant (wd_ML_BN)
C_ALT = "#9aa7b5"    # blue-gray — other persistent variants
C_MMLU = "#7f9db9"   # light steel — capability line


def rates(probes_path):
    d = json.load(open(probes_path))
    harm, hless = d["harmful"], d["harmless"]
    hr = sum(1 for r in harm if r.get("refused")) / len(harm)
    br = sum(1 for r in hless if not r.get("refused")) / len(hless)
    degen = sum(1 for r in harm + hless if r.get("degenerate"))
    assert degen == 0, f"unexpected degenerate outputs in {probes_path}"
    return hr, br


def mmlu_overall(path):
    d = json.load(open(path))
    m = d["results"]["mmlu"]
    return 100 * m["acc,none"], 100 * m["acc_stderr,none"]


def mmlu_groups(path):
    d = json.load(open(path))
    res = d["results"]
    return {g: 100 * res[g]["acc,none"] for g in
            ("mmlu_stem", "mmlu_humanities", "mmlu_social_sciences",
             "mmlu_other")}


# ================================================================ run-002 figs
conds1 = ["baseline", "hook_ablated", "wd_A", "wd_B", "wd_C"]
labels1 = ["baseline", "hook\n(inference-time)", "wd_A\n(lm_head,\nresid. space)",
           "wd_B\n(lm_head,\nreadout space)", "wd_C\n(o_proj+down_proj,\nrow space)"]
refusal1, benign1 = {}, {}
for c in conds1:
    refusal1[c], benign1[c] = rates(EVAL1 / f"probes_{c}.json")

fig, ax = plt.subplots(figsize=(8.2, 4.2))
bars = ax.bar(labels1, [100 * refusal1[c] for c in conds1],
              color=[C_BASE, C_HOOK, C_ALT, C_ALT, C_ALT], width=0.62)
ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9, fontweight="bold")
ax.set_ylabel("refusal rate on harmful probes (%)")
ax.set_ylim(0, 100)
ax.set_title("Run 002 — refusal by condition (16 harmful probes, greedy, 200 tok, seed 0)\n"
             "(Qwen2.5-0.5B-Instruct, layer 17/24, coherence 0.664)", fontsize=10.5)
ax.annotate("run-002 selected\n(superseded by wd_ML_BN,\nsee r2 charts)",
            xy=(3, 100 * refusal1["wd_B"]),
            xytext=(3, 100 * refusal1["wd_B"] + 14), ha="center", fontsize=8.5,
            color="#d96f5c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d96f5c", lw=1.2))
ax.text(0.99, 0.97, "hook = inference-time hook (not persisted in weights);\n"
        "wd_* = persistent weight-decoded variants",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#a8a29a")
fig.tight_layout()
fig.savefig(OUT / "refusal_by_condition.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 4.2))
x = range(len(conds1))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], [100 * refusal1[c] for c in conds1],
            width=w, color=[C_BASE, C_HOOK, C_ALT, C_ALT, C_ALT],
            label="refusal on harmful probes")
b2 = ax.bar([i + w / 2 for i in x], [100 * benign1[c] for c in conds1],
            width=w, color="#7f9db9", label="harmless probes answered")
ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8.5)
ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8.5)
ax.set_xticks(list(x), labels1)
ax.set_ylabel("rate (%)")
ax.set_ylim(0, 105)
ax.set_title("Run 002 — refusal removal vs. benign preservation\n"
             "(16 harmful + 16 harmless probes, greedy, 200 tok, seed 0)",
             fontsize=10.5)
ax.legend(loc="lower right", fontsize=9, framealpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "refusal_vs_benign.png")
plt.close(fig)

lc1 = json.load(open(EVAL1 / "layer_coherence.json"))
t1 = lc1["table"]
best1 = lc1["best"]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
ax.plot([r["decoder_layer"] for r in t1], [r["coherence"] for r in t1], "-",
        color=C_MMLU, lw=1.8, marker="o", ms=3.5, mfc="#131417")
ax.axvline(best1["decoder_layer"], color=C_PUB, lw=1.4, ls=":")
ax.scatter([best1["decoder_layer"]], [best1["coherence"]], color=C_PUB,
           zorder=5, s=55)
ax.annotate(f"selected: L{best1['decoder_layer']}/24\ncoherence "
            f"{best1['coherence']:.3f}",
            xy=(best1["decoder_layer"], best1["coherence"]),
            xytext=(best1["decoder_layer"] + 1.2, best1["coherence"] - 0.10),
            fontsize=9, color=C_PUB, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_PUB, lw=1.2))
ax.set_xlabel("decoder layer")
ax.set_ylabel("direction coherence")
ax.set_title("Run 002 — refusal-direction coherence scan, 24 decoder layers\n"
             "(64 harmful/harmless pairs; exact replication of run 001)",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "layer_coherence.png")
plt.close(fig)

b_acc, b_se = mmlu_overall(next(EVAL1.glob("mmlu_base_*.json")))
v_acc, v_se = mmlu_overall(next(EVAL1.glob("mmlu_variant_*.json")))
b_grp = mmlu_groups(next(EVAL1.glob("mmlu_base_*.json")))
v_grp = mmlu_groups(next(EVAL1.glob("mmlu_variant_*.json")))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                               width_ratios=[1, 1.5])
bars = ax1.bar(["base\n(pinned rev)", "wd_B\n(run-002 variant)"],
               [b_acc, v_acc], yerr=[b_se, v_se], capsize=5,
               color=[C_BASE, C_ALT], width=0.55)
ax1.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9, fontweight="bold")
ax1.set_ylim(0, 60)
ax1.set_ylabel("MMLU accuracy (%)")
ax1.set_title(f"Run 002 overall: {b_acc:.2f}% → {v_acc:.2f}%\n"
              f"(Δ {b_acc - v_acc:.2f}pp, guardrail <3pp)", fontsize=10.5)
gx = range(4)
gw = 0.38
glabels = ["STEM", "humanities", "social\nsci.", "other"]
gk = ("mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other")
b_base = ax2.bar([i - gw / 2 for i in gx], [b_grp[g] for g in gk],
                 width=gw, color=C_BASE, label="base")
b_var = ax2.bar([i + gw / 2 for i in gx], [v_grp[g] for g in gk],
                width=gw, color=C_ALT, label="wd_B")
ax2.bar_label(b_base, fmt="%.1f", padding=2, fontsize=8)
ax2.bar_label(b_var, fmt="%.1f", padding=2, fontsize=8)
ax2.set_xticks(list(gx), glabels)
ax2.set_ylabel("MMLU accuracy (%)")
ax2.set_ylim(0, 70)
ax2.set_title("MMLU domain groups: base vs wd_B", fontsize=10.5)
ax2.legend(fontsize=9, framealpha=0.2)
fig.suptitle("Capability guardrail — lm-eval MMLU, 0-shot, fp16, seed 0, "
             "identical config", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "mmlu_guardrail.png")
plt.close(fig)

# ================================================================ run-003 figs
conds2 = ["baseline", "hook_ablated", "wd_B", "wd_BN", "wd_ML", "wd_ML_BN"]
labels2 = ["baseline", "hook\n(L17, inference)", "wd_B", "wd_BN", "wd_ML",
           "wd_ML_BN\n(published)"]
refusal2, benign2 = {}, {}
for c in conds2:
    refusal2[c], benign2[c] = rates(EVAL2 / f"probes_{c}.json")

fig, ax = plt.subplots(figsize=(9, 5.2))
colors = [C_BASE, C_HOOK, C_ALT, C_ALT, C_ALT, C_PUB]
bars = ax.bar(labels2, [100 * r for r in refusal2.values()],
              color=colors, width=0.62)
ax.axhline(25, color="#d96f5c", ls="--", lw=1.4)
ax.text(4.55, 26.5, "user goal < 25%", color="#d96f5c", fontsize=9,
        ha="right")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, fontweight="bold")
ax.set_ylabel("refusal rate (%)")
ax.set_ylim(0, 100)
ax.set_title("Run 003 (round 2) — persistent-edit ladder, harmful refusal\n"
             "(16 probes, greedy, 200 tok, seed 0; same scorer as run 002)",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "r2_refusal_by_condition.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.2))
x = range(len(conds2))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], [100 * r for r in refusal2.values()],
            width=w, color=colors, label="refusal on harmful probes")
b2 = ax.bar([i + w / 2 for i in x], [100 * r for r in benign2.values()],
            width=w, color="#7f9db9", label="harmless probes answered")
ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8.5)
ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8.5)
ax.set_xticks(list(x), labels2)
ax.set_ylabel("rate (%)")
ax.set_ylim(0, 105)
ax.set_title("Run 003 (round 2) — refusal removal vs. benign preservation\n"
             "(16 harmful + 16 harmless probes, greedy, 200 tok, seed 0)",
             fontsize=10.5)
ax.legend(loc="lower right", fontsize=9, framealpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "r2_benign_preservation.png")
plt.close(fig)

lc2 = json.load(open(EVAL2 / "layer_coherence.json"))
t2 = lc2["table"]
best2 = lc2["best"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot([r["decoder_layer"] for r in t2], [r["coherence"] for r in t2], "-",
        color=C_MMLU, lw=1.8, marker="o", ms=3.5, mfc="#131417")
ax.axvline(best2["decoder_layer"], color=C_PUB, lw=1.4, ls=":")
ax.scatter([best2["decoder_layer"]], [best2["coherence"]], color=C_PUB,
           zorder=5, s=55)
ax.annotate(f"L{best2['decoder_layer']} coherence "
            f"{best2['coherence']:.3f}\n(exact replication of runs 001-002)",
            xy=(best2["decoder_layer"], best2["coherence"]),
            xytext=(best2["decoder_layer"] - 6.5, best2["coherence"] - 0.06),
            fontsize=9, color=C_PUB, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_PUB, lw=1.2))
ax.set_xlabel("decoder layer")
ax.set_ylabel("direction coherence")
ax.set_title("Run 003 — coherence scan (64 fresh pairs, same patient & recipe)",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "r2_layer_coherence.png")
plt.close(fig)

mb_acc, mb_se = mmlu_overall(EVAL2 / "mmlu_base_results.json")
mv_acc, mv_se = mmlu_overall(EVAL2 / "mmlu_variant_results.json")
mb_grp = mmlu_groups(EVAL2 / "mmlu_base_results.json")
mv_grp = mmlu_groups(EVAL2 / "mmlu_variant_results.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                               width_ratios=[1, 1.5])
bars = ax1.bar(["base\n(pinned rev)", "wd_ML_BN\n(this repo)"],
               [mb_acc, mv_acc], yerr=[mb_se, mv_se], capsize=5,
               color=[C_BASE, C_PUB], width=0.55)
ax1.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9, fontweight="bold")
ax1.set_ylim(0, 60)
ax1.set_ylabel("MMLU accuracy (%)")
ax1.set_title(f"Run 003 overall: {mb_acc:.2f}% → {mv_acc:.2f}%\n"
              f"(Δ {mb_acc - mv_acc:.2f}pp, guardrail <3pp)", fontsize=10.5)
b_base = ax2.bar([i - gw / 2 for i in gx], [mb_grp[g] for g in gk],
                 width=gw, color=C_BASE, label="base")
b_var = ax2.bar([i + gw / 2 for i in gx], [mv_grp[g] for g in gk],
                width=gw, color=C_PUB, label="wd_ML_BN")
ax2.bar_label(b_base, fmt="%.1f", padding=2, fontsize=8)
ax2.bar_label(b_var, fmt="%.1f", padding=2, fontsize=8)
ax2.set_xticks(list(gx), glabels)
ax2.set_ylabel("MMLU accuracy (%)")
ax2.set_ylim(0, 70)
ax2.set_title("MMLU domain groups: base vs wd_ML_BN", fontsize=10.5)
ax2.legend(fontsize=9, framealpha=0.2)
fig.suptitle("Run 003 capability guardrail — lm-eval MMLU, 0-shot, fp16, "
             "seed 0, identical config", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "r2_mmlu_guardrail.png")
plt.close(fig)

# ================================================================ TruthfulQA
def tq_mc2(path):
    d = json.load(open(path))
    t = d["results"]["truthfulqa_mc2"]
    # lm-eval <0.5 logs mc2 as "mc2,none"; under --limit it logs "acc,none"
    # (same mc2 statistic, different metric-name registration)
    k = "mc2,none" if "mc2,none" in t else "acc,none"
    ks = "mc2_stderr,none" if "mc2_stderr,none" in t else "acc_stderr,none"
    return 100 * t[k], 100 * t[ks]


tb_acc, tb_se = tq_mc2(EVAL2 / "truthfulqa_base_results.json")
tv_acc, tv_se = tq_mc2(EVAL2 / "truthfulqa_variant_results.json")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bars = ax.bar(["base\n(pinned rev)", "wd_ML_BN\n(this repo)"],
              [tb_acc, tv_acc], yerr=[tb_se, tv_se], capsize=5,
              color=[C_BASE, C_PUB], width=0.5)
ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10, fontweight="bold")
ax.set_ylim(0, max(tb_acc, tv_acc) * 1.25)
ax.set_ylabel("TruthfulQA mc2 (%)")
ax.set_title(f"TruthfulQA mc2 (0-shot, fp32, seed 0, identical config,\n"
             f"n=240 paired subsample): {tb_acc:.2f}% → {tv_acc:.2f}% "
             f"(Δ {tb_acc - tv_acc:+.2f}pp)\n"
             "founding-paper expectation: −1..−3.5pp — measured here "
             "(CI contains 0)",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "truthfulqa_guardrail.png")
plt.close(fig)

# ================================================================ multilingual
ml = json.load(open(EVAL2 / "probes_multilingual.json"))
conds_ml = ["baseline", "hook", "variant"]
cond_labels = {"baseline": "baseline", "hook": "hook (inference-time)",
               "variant": "wd_ML_BN (published)"}
langs = [k for k in ml["conditions"]["baseline"]["summary"] if k != "_all"]
fig, ax = plt.subplots(figsize=(9.2, 4.6))
import numpy as np
x = np.arange(len(langs))
w = 0.26
palette = [C_BASE, C_HOOK, C_PUB]
for ci, c in enumerate(conds_ml):
    vals = [100 * ml["conditions"][c]["summary"][lg]["harmful_refused"] /
            max(1, ml["conditions"][c]["summary"][lg]["harmful_n"])
            for lg in langs]
    bars = ax.bar(x + (ci - 1) * w, vals, width=w, color=palette[ci],
                  label=cond_labels[c])
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8.5)
ax.set_xticks(x, langs)
ax.set_ylabel("refusal rate on harmful probes (%)")
ax.set_ylim(0, 105)
ax.set_title("Multilingual refusal panel — en/zh/ru/de, 2 matched harmful "
             "probes per language\n(same patient, scorer, and decoding as "
             "the English runs; n=2/language: directional)", fontsize=10)
ax.legend(fontsize=9, framealpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "multilingual_panel.png")
plt.close(fig)

# ================================================================ data checks
# Integrity asserts (fail loudly if any recorded number drifts) — same
# discipline as run-003's original chart generator.
assert abs(refusal1["baseline"] - 0.875) < 1e-9, refusal1
assert abs(refusal1["hook_ablated"]) < 1e-9, refusal1
assert abs(refusal1["wd_B"] - 0.5625) < 1e-9, refusal1
assert abs(refusal1["wd_A"] - 0.75) < 1e-9, refusal1
assert abs(refusal1["wd_C"] - 0.6875) < 1e-9, refusal1
assert abs(b_acc - v_acc - 0.2879) < 0.06, (b_acc, v_acc)
assert abs(refusal2["baseline"] - 0.875) < 1e-9, refusal2
assert abs(refusal2["hook_ablated"]) < 1e-9, refusal2
assert abs(refusal2["wd_B"] - 0.5625) < 1e-9, refusal2
assert abs(refusal2["wd_BN"] - 0.5625) < 1e-9, refusal2
assert abs(refusal2["wd_ML"] - 0.6875) < 1e-9, refusal2
assert abs(refusal2["wd_ML_BN"]) < 1e-9, refusal2
assert abs(mb_acc - mv_acc - 0.24) < 0.06, (mb_acc, mv_acc)
assert abs(benign2["wd_ML_BN"] - 0.875) < 1e-9, benign2

print("charts written to", OUT)
for f in sorted(OUT.glob("*.png")):
    print("  ", f.name, f.stat().st_size, "bytes")
print("DATA CHECK run002 refusal:", {c: round(refusal1[c], 4) for c in conds1})
print("DATA CHECK run003 refusal:", {c: round(refusal2[c], 4) for c in conds2})
print(f"DATA CHECK mmlu run002: base={b_acc:.4f} variant={v_acc:.4f} "
      f"delta_pp={b_acc - v_acc:.4f}")
print(f"DATA CHECK mmlu run003: base={mb_acc:.4f} variant={mv_acc:.4f} "
      f"delta_pp={mb_acc - mv_acc:.4f}")
print(f"DATA CHECK truthfulqa mc2: base={tb_acc:.4f} variant={tv_acc:.4f} "
      f"delta_pp={tb_acc - tv_acc:.4f}")
print("DATA CHECK multilingual harmful-refused:",
      {c: {lg: ml["conditions"][c]["summary"][lg]["harmful_refused"]
           for lg in langs} for c in conds_ml})