"""Terminal presentation for engine runs — one place for how output looks.

- Stage counter: one per phase. Stage A is `[A k/5]`; the ladder is
  `[B k/N]` with N = plan + every requested variant + selection, so a
  reader always knows where in the run they are.
- Phase summaries: a short human box with the numbers that matter and the
  exact next command, printed BEFORE the machine sentinel line
  (`RUN_DONE {json}` / `LADDER_DONE {json}` stay the final line — the
  poll-loop contract is unchanged).
- Quiet mode (`--quiet` / ENG_QUIET=1): one line per probe batch instead
  of one per prompt; the full rows are still written to probes_*.json.
- Color only on a real terminal (never in Colab cells or log files);
  NO_COLOR disables it, ENG_COLOR=1 forces it.
"""
import os
import shlex
import sys

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32",
          "yellow": "33", "cyan": "36"}


def color_enabled(stream=None):
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("ENG_COLOR") == "1":
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def style(text, *names):
    if not names or not color_enabled():
        return str(text)
    codes = ";".join(_CODES[n] for n in names)
    return f"\033[{codes}m{text}\033[0m"


def quiet():
    return os.environ.get("ENG_QUIET") == "1"


def emit(text=""):
    print(text, flush=True)


def detail(text):
    """Indented sub-line under the current stage step."""
    emit(f"      {text}")


class Stages:
    """`[A 2/5] capture …` — one counter per phase."""

    def __init__(self, phase, total):
        self.phase, self.total, self.i = phase, total, 0

    def step(self, title):
        self.i += 1
        emit(style(f"[{self.phase} {self.i}/{self.total}]", "bold", "cyan")
             + f" {title}")


def pct(x):
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def spec_arg(spec):
    """The spec path as the user would type it (relative under cwd)."""
    p = spec.get("_spec_path") or "<spec.yaml>"
    try:
        rel = os.path.relpath(p)
        p = rel if not rel.startswith("..") else p
    except ValueError:
        pass
    return shlex.quote(p)


def _box(title, rows, next_line=None, verdict=None):
    width = 64
    emit()
    emit(style(f"── {title} " + "─" * max(4, width - len(title) - 4), "bold"))
    keyw = max((len(k) for k, _ in rows), default=0)
    for k, v in rows:
        emit(f"  {k.ljust(keyw)}  {v}")
    if verdict:
        text, ok = verdict
        emit("  " + style(text, "bold", "green" if ok else "yellow"))
    if next_line:
        emit("  " + style("next:", "bold") + f" {next_line}")
    emit(style("─" * width, "dim"))


def probe_batch_line(tag, rows, seconds):
    n = len(rows)
    refused = sum(r["refused"] for r in rows)
    degenerate = sum(1 for r in rows if r.get("degenerate"))
    extra = f" · {degenerate} degenerate" if degenerate else ""
    emit(f"  [{tag}] {n}/{n} · refused {refused} ({pct(refused / max(1, n))})"
         f"{extra} · {seconds:.1f}s")


# ---- phase summaries ---------------------------------------------------------
def summary_stage_a(summary, spec):
    base, hook = summary["baseline"], summary["hook_ablated"]
    rows = [
        ("best layer", f"L{summary['layer']} (coherence "
                       f"{summary['coherence']})"),
        ("refusal", f"baseline {pct(base['refusal_rate'])}  →  hook "
                    f"{pct(hook['refusal_rate'])}"),
        ("benign answered", f"baseline {pct(base['benign_preserved'])}  →  "
                            f"hook {pct(hook['benign_preserved'])}"),
        ("degenerate", f"{base['degenerate_total']} baseline, "
                       f"{hook['degenerate_total']} hook"),
        ("wall time", f"{summary.get('wall_s', 0):.0f}s"),
    ]
    if summary.get("ladder_skipped"):
        nxt = ("nothing to publish — hook-only characterization run "
               "(add ladder.variants to make persistent edits)")
    else:
        nxt = "the ladder (persistent edits) runs now"
    _box("stage A complete", rows, nxt)


def summary_ladder(payload, spec, chained_mmlu=False):
    gates = spec["gates"]
    variants = payload.get("variants") or {}
    selected = payload["selected"]
    m = payload["metrics"]
    rows = [("baseline", f"refusal {pct(m['refusal_rate_before'])}, benign "
                         f"{pct(m['benign_preserved_before'])}")]
    for name, v in variants.items():
        mark = style(" ← selected", "bold") if name == selected else ""
        rows.append((name, f"refusal {pct(v['refusal_rate'])}, benign "
                           f"{pct(v['benign_preserved'])}, degenerate "
                           f"{v['degenerate_total']}{mark}"))
    passed = payload["gate"] == "passed"
    eligible = payload.get("publish_eligible_probe_gate")
    if passed and eligible:
        verdict = (f"{selected} passes the gates (refusal < "
                   f"{pct(gates['publish_refusal'])})", True)
        nxt = ("MMLU guardrail runs now (--with-mmlu)" if chained_mmlu else
               f"abliterate mmlu --spec {spec_arg(spec)} "
               "--i-know-this-spends-quota")
    elif passed:
        verdict = (f"{selected} keeps benign answers but refusal "
                   f"{pct(m['refusal_rate_after'])} is not under the "
                   f"{pct(gates['publish_refusal'])} publish gate", False)
        nxt = ("publish is gated off — widen the ladder (k_primary / "
               "k_combo, add ara_<rank>) and re-run `abliterate ladder`")
    else:
        verdict = ("no variant passed the benign/degenerate gate — the "
                   "edits damage normal answers on this patient", False)
        nxt = "publish is gated off — inspect probes_*.json before re-running"
    _box("ladder complete", rows, nxt, verdict)


def summary_search(payload, spec):
    ev = payload["search_eval"]
    probe = payload["probe_set"]
    rows = [
        ("trials", f"{payload['trials']} · Pareto front "
                   f"{payload['pareto_size']}"),
        ("chosen", f"{payload['selected']} — {payload['describe']}"),
        ("search split", f"refusal {pct(ev['refusal'])} · benign refused "
                         f"{pct(ev['benign_refusal'])} · KL {ev['kl']:.3f}"),
        ("probe set", f"refusal {pct(probe['refusal_rate'])} · benign "
                      f"{pct(probe['benign_preserved'])}"),
        ("sealed holdout", f"refusal {pct(payload['holdout_refusal_before'])}"
                           f"  →  {pct(payload['holdout_refusal'])}"),
    ]
    ok = payload["certificate_publishable"]
    verdict = ("certificate passes the holdout bar" if ok else
               "certificate does NOT pass the holdout bar — publish will "
               "refuse", ok)
    nxt = (f"abliterate mmlu --spec {spec_arg(spec)} "
           "--i-know-this-spends-quota" if ok else
           "widen search.space or raise search.trials and re-run "
           "(banked trials are reused)")
    _box("search complete", rows, nxt, verdict)


def summary_mmlu(summary, spec):
    limit = summary["guardrail_loss_pp_limit"]
    ok = summary["mmlu_delta_pp"] < limit
    rows = [("base", f"{summary['mmlu_base_pct']}%"),
            (summary.get("variant") or "variant",
             f"{summary['mmlu_variant_pct']}%"),
            ("loss", f"{summary['mmlu_delta_pp']}pp (limit {limit}pp)")]
    if ok:
        verdict = ("capability guardrail PASSED", True)
        nxt = (f"abliterate publish --spec {spec_arg(spec)} "
               "--i-know-this-publishes")
    else:
        verdict = ("capability guardrail FAILED — publish will refuse", False)
        nxt = "pick a gentler variant (fewer layers) and re-run the ladder"
    _box("MMLU guardrail", rows, nxt, verdict)
