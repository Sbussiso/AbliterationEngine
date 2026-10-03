"""Model-card charts, rendered from the same artifacts as the card tables.

Three PNGs, written into <variant dir>/charts/ so they upload with the
weights and the card embeds them by relative path:

  refusal_by_condition.png   harmful-prompt refusal: baseline, hook, every
                             ladder variant; publish gate as a reference line
  benign_by_condition.png    benign probes answered, same conditions; gate
                             floor (baseline - benign_floor_delta) marked
  mmlu_guardrail.png         MMLU accuracy base vs variant (dots + stderr)
                             and the observed loss vs the guardrail limit

Every number is computed by the caller from probe/MMLU files — nothing here
is typed by hand. matplotlib is optional (`pip install
abliteration-engine[charts]`); without it publish ships the text-only card.

Palette: the house dark palette (surface #131417, ink #e8e4da), re-stepped so
it passes the categorical checks on that surface: steel #4e8fd6 vs neutral
gray #6b6b6b is 16.8 dE (normal vision), gold #b08a38 sits in the dark
lightness band. Identity is never color-alone: every bar carries its
condition name and value label. Bars always start at zero; the MMLU accuracy
panel uses dots, so its zoomed axis does not exaggerate a bar length.
PNG metadata is stripped so a re-render of the same artifacts is
byte-identical.
"""
import os

SURFACE, INK, MUTED = "#131417", "#e8e4da", "#9aa0a6"
GRID, EDGE = "#26282e", "#3a3d44"
C_BASELINE, C_HOOK, C_VARIANT = "#6b6b6b", "#4e8fd6", "#b08a38"
C_THRESHOLD = "#d96f5c"

_RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": EDGE, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
}
_COLOR = {"baseline": C_BASELINE, "hook": C_HOOK, "variant": C_VARIANT}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})  # deterministic bytes


def _style(ax):
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _condition_bars(plt, conditions, key, title, ylabel, path, ref_lines):
    """conditions: [{label, kind, refusal_rate, benign_preserved,
    published?}]. Width grows with the bar count so 2-3 line tick labels
    never run together."""
    width = max(8.6, 1.3 * (len(conditions) + 1.85) + 1.5)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(width, 4.9), dpi=160)
        xs = list(range(len(conditions)))
        vals = [c[key] * 100 for c in conditions]
        ax.bar(xs, vals, color=[_COLOR[c["kind"]] for c in conditions],
               width=0.62)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.6, f"{v:.1f}%", ha="center", fontsize=9,
                    color=INK)
        # reference lines are labelled in a reserved right margin, never
        # on top of a bar
        n = len(xs)
        for y, text, color, ls, va in ref_lines:
            ax.axhline(y, color=color, lw=1.6 if ls == "--" else 1, ls=ls)
            ax.text(n - 0.3, y + (1.2 if va == "bottom" else -1.2), text,
                    color=color, fontsize=8.6, ha="left", va=va)
        ax.set_xlim(-0.6, n + 1.25)
        ax.set_xticks(xs)
        ticks = ax.set_xticklabels([c["label"] for c in conditions],
                                   fontsize=8.6)
        for t, c in zip(ticks, conditions):
            if c.get("published"):
                t.set_fontweight("bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_ylim(0, 108)
        _style(ax)
        _save(fig, path)
        plt.close(fig)


def _mmlu_chart(plt, mmlu, variant, base_name, path):
    base = mmlu["base"]["acc"] * 100
    var = mmlu["variant_model"]["acc"] * 100
    sb = (mmlu["base"].get("acc_stderr") or 0) * 100
    sv = (mmlu["variant_model"].get("acc_stderr") or 0) * 100
    delta = base - var
    limit = float(mmlu["guardrail_loss_pp_limit"])
    with plt.rc_context(_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=160,
                                     gridspec_kw={"width_ratios": [1.15, 1]})
        # accuracy: dots + stderr (a zoomed axis is honest for dots, not bars)
        a1.errorbar([0], [base], yerr=[sb], fmt="o", color=C_BASELINE,
                    ecolor=INK, ms=9, capsize=4, lw=1.2)
        a1.errorbar([1], [var], yerr=[sv], fmt="o", color=C_VARIANT,
                    ecolor=INK, ms=9, capsize=4, lw=1.2)
        for x, v, s in ((0, base, sb), (1, var, sv)):
            a1.text(x + 0.08, v, f"{v:.2f}%", va="center", fontsize=10)
        lo, hi = min(base - sb, var - sv), max(base + sb, var + sv)
        pad = max(1.0, (hi - lo) * 0.8)
        a1.set_xlim(-0.5, 1.6)
        a1.set_ylim(lo - pad, hi + pad)
        a1.set_xticks([0, 1])
        a1.set_xticklabels([f"base\n{base_name}", f"variant\n{variant}"],
                           fontsize=8.6)
        a1.set_ylabel("MMLU accuracy (%, 0-shot)")
        a1.set_title("capability retention", fontsize=11, pad=10)
        _style(a1)
        # loss vs limit: one bar from zero against the dashed limit line
        a2.bar([0], [max(delta, 0)], width=0.45, color=C_VARIANT)
        a2.axhline(limit, color=C_THRESHOLD, lw=1.6, ls="--")
        a2.text(0, max(delta, 0) + limit * 0.03, f"{delta:.2f}pp",
                ha="center", va="bottom", fontsize=10)
        verdict = "PASSED" if delta < limit else "FAILED"
        a2.text(0.85, limit * 1.02, f"{limit:.1f}pp limit\n{verdict}",
                ha="right", va="bottom", fontsize=9.5, color=C_THRESHOLD)
        a2.set_xlim(-0.9, 0.9)
        a2.set_xticks([0])
        a2.set_xticklabels([f"observed loss\n({variant})"], fontsize=8.6)
        a2.set_ylim(0, max(limit, delta) * 1.35)
        a2.set_ylabel("MMLU loss (pp)")
        a2.set_title("loss vs the designed limit", fontsize=11, pad=10)
        _style(a2)
        _save(fig, path)
        plt.close(fig)
    return delta, limit


def render_card_charts(out_dir, conditions, base_preserved, floor_delta,
                       publish_refusal, mmlu, variant, model_name):
    """Write the three PNGs into out_dir/charts. Returns
    [(relative_path, alt_text)] in card order. Raises ImportError when
    matplotlib is not installed (caller ships the text-only card)."""
    plt = _plt()
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    out = []

    p = os.path.join(charts_dir, "refusal_by_condition.png")
    _condition_bars(
        plt, conditions, "refusal_rate",
        f"{model_name}: harmful-prompt refusal by condition",
        "harmful-prompt refusal rate (%)", p,
        [(publish_refusal * 100,
          f"publish gate\n< {publish_refusal * 100:g}%", C_THRESHOLD, "--",
          "bottom")])
    summary = ", ".join(f"{c['label'].splitlines()[0]} "
                        f"{c['refusal_rate'] * 100:.1f}%" for c in conditions)
    out.append(("charts/refusal_by_condition.png",
                f"Harmful-prompt refusal by condition: {summary}"))

    floor = (base_preserved - floor_delta) * 100
    p = os.path.join(charts_dir, "benign_by_condition.png")
    _condition_bars(
        plt, conditions, "benign_preserved",
        f"{model_name}: benign prompts answered by condition",
        "benign probes answered (%)", p,
        [(floor, f"gate floor {floor:.1f}%\n(baseline − "
                 f"{floor_delta * 100:g}pp)", C_THRESHOLD, "--", "top"),
         (base_preserved * 100, f"baseline {base_preserved * 100:.1f}%",
          MUTED, ":", "bottom")])
    summary = ", ".join(f"{c['label'].splitlines()[0]} "
                        f"{c['benign_preserved'] * 100:.1f}%"
                        for c in conditions)
    out.append(("charts/benign_by_condition.png",
                f"Benign prompts answered by condition: {summary}"))

    p = os.path.join(charts_dir, "mmlu_guardrail.png")
    delta, limit = _mmlu_chart(plt, mmlu, variant, model_name, p)
    out.append(("charts/mmlu_guardrail.png",
                f"MMLU guardrail: base {mmlu['base']['acc'] * 100:.2f}% vs "
                f"{variant} {mmlu['variant_model']['acc'] * 100:.2f}%, loss "
                f"{delta:.2f}pp against a {limit:.1f}pp limit"))
    return out
