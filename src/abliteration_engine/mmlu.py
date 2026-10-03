"""MMLU guardrail stage (mmlu_eval_003.py faithful).

Identical lm-eval config both sides (task=mmlu full 57-subject,
num_fewshot=0, batch_size=auto, dtype=float16, seed=0); delta guardrail
base-variant < gates.mmlu_max_loss_pp else STOP-and-report. Variant =
selection.json selected_variant_dir (ladder runs) — guarded-off for
hook-only specs.

Colab-side only; sentinel <eng_base>/mmlu_exit_code.txt + final line
MMLU_DONE {summary} in the log (v2 contract preserved for the poll loop).

Exit codes: 0 guardrail passed; 3 parse failure; 4 lm_eval missing;
5 no selection.json; 6 guardrail FAILED (summary still written, publish
refuses); otherwise lm_eval's own non-zero exit code.

Banked resume: a side's prior results are reused only when they provably
came from the same model — the results file's recorded model_args must
match, and for the variant side the eng_provenance.json sidecar must match
selection.json's selected variant + provenance fingerprint. Anything else
is moved aside (<dir>.stale-<ts>) and re-evaluated.
"""
import glob
import json
import os
import subprocess
import sys
import time


def _parse_lm_eval(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "**",
                                          "results_*.json"),
                             recursive=True))
    if not files:
        return None
    d = json.load(open(files[-1]))
    res = d.get("results", {})
    mmlu = res.get("mmlu") or res.get("all") or {}
    acc = mmlu.get("acc,none")
    stderr = mmlu.get("acc_stderr,none")
    if acc is None:
        vals = [v.get("acc,none") for k, v in res.items()
                if k not in ("all",) and isinstance(v, dict)
                and v.get("acc,none") is not None]
        acc = sum(vals) / len(vals) if vals else None
        stderr = None
    n_subj = len([k for k in res if k not in ("all", "mmlu")])
    return {"acc": acc, "acc_stderr": stderr, "subjects_seen": n_subj,
            "results_file": files[-1],
            "version": d.get("config", {}).get("lm_eval_version")}


PROVENANCE_FILE = "eng_provenance.json"


def _model_args_dict(ma):
    if isinstance(ma, dict):
        return {str(k): str(v) for k, v in ma.items()}
    out = {}
    for part in str(ma or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _reusable(odir, model_args, provenance):
    """True iff odir's banked lm-eval results belong to this exact model."""
    prior = _parse_lm_eval(odir)
    if not prior or prior["acc"] is None:
        return False
    try:
        cfg = json.load(open(prior["results_file"])).get("config", {})
    except Exception:
        return False
    want, got = _model_args_dict(model_args), _model_args_dict(
        cfg.get("model_args"))
    if any(got.get(k) != want.get(k) for k in ("pretrained", "revision")):
        return False
    if provenance is None:
        return True
    try:
        side = json.load(open(os.path.join(odir, PROVENANCE_FILE)))
    except Exception:
        return False
    return side == provenance


def mmlu_phase(spec_path):
    """`abliterate mmlu` — base vs variant, writes mmlu_summary.json to the
    run out dir; sentinel <eng_base>/mmlu_exit_code.txt."""
    from . import core
    from .spec import load_spec

    spec = load_spec(spec_path)
    out_dir = core._out_dir(spec, create=True)
    log = os.path.join(out_dir, "mmlu_log.txt")
    exit_f = os.environ.get("ENG_MMLU_EXIT_FILE") \
        or os.path.join(core.eng_base(), "mmlu_exit_code.txt")

    def logw(m):
        with open(log, "a") as f:
            f.write(m + "\n")

    sel_path = os.path.join(out_dir, "selection.json")
    if not os.path.exists(sel_path):
        logw("REFUSING: no selection.json - hook-only runs have no "
             "ladder variant to evaluate (publish gated off by spec), "
             "MMLU stage not applicable")
        with open(exit_f, "w") as f:
            f.write("5")
        return 5
    sel = json.load(open(sel_path))

    # smoke-run finding 2026-10-01: lm-eval is NOT a pip-install dep
    # (it ships via the gpu extra / preinstalled Colab stack). A fresh
    # clone + `pip install -q .` session died here with empty stdout and
    # rc=3 — indistinguishable from a hung cell to the person watching.
    # Fail loud with the exact fix instead; rc=3 stays reserved for
    # genuine parse failures (v2 contract).
    try:
        import lm_eval  # noqa: F401
    except ModuleNotFoundError:
        print("MMLU BLOCKED: lm_eval is not installed in this session.",
              flush=True)
        print("Fix:  !pip install -q lm-eval    (then re-run this cell)",
              flush=True)
        logw("REFUSING: lm_eval not importable - install lm-eval "
             "(see pyproject gpu extra) and re-run the mmlu phase")
        with open(exit_f, "w") as f:
            f.write("4")
        return 4

    with open(exit_f, "w") as f:
        f.write("running")
    open(log, "w").close()
    # a summary from an earlier run must never outlive a re-run that fails
    # or is abandoned — publish would otherwise trust the old pass
    summary_path = os.path.join(out_dir, "mmlu_summary.json")
    if os.path.exists(summary_path):
        os.remove(summary_path)
    logw(f"mmlu phase start "
         f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    pat = spec["patient"]
    base_args = (f"pretrained={pat['model_id']},"
                 f"revision={pat['revision']},dtype=float16")
    var_args = f"pretrained={sel['selected_variant_dir']},dtype=float16"
    out_base = os.path.join(out_dir, "mmlu_results", "base")
    out_var = os.path.join(out_dir, "mmlu_results", "variant")

    def run_lm_eval(tag, model_args, odir, provenance=None):
        # banked-MMLU resume: a prior session may have already banked this
        # side; reuse its results_*.json instead of re-burning 40+ min T4 —
        # but ONLY if they came from this exact model (see module doc).
        if _reusable(odir, model_args, provenance):
            prior = _parse_lm_eval(odir)
            logw(f"=== {tag}: BANKED RESUME (prior-session results reused, "
                 f"acc={prior['acc']:.4f})")
            return 0
        if os.path.isdir(odir) and os.listdir(odir):
            stale = f"{odir}.stale-{int(time.time())}"
            os.rename(odir, stale)
            logw(f"=== {tag}: prior results not provably from this model -> "
                 f"moved aside to {stale}")
        cmd = [sys.executable, "-m", "lm_eval", "--model", "hf",
               "--model_args", model_args, "--tasks", "mmlu",
               "--num_fewshot", "0", "--batch_size", "auto",
               "--seed", "0", "--output_path", odir]
        logw(f"=== {tag}: {' '.join(cmd)}")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        logw(f"=== {tag}: exit={r.returncode} "
             f"wall={time.time() - t0:.0f}s")
        if r.stdout:
            logw("--- stdout (tail 4000) ---")
            logw(r.stdout[-4000:])
        if r.stderr:
            logw("--- stderr (tail 4000) ---")
            logw(r.stderr[-4000:])
        if r.returncode == 0 and provenance is not None:
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, PROVENANCE_FILE), "w") as f:
                json.dump(provenance, f, indent=2)
        return r.returncode

    rc1 = run_lm_eval("base", base_args, out_base)
    var_prov = {"selected": sel["selected"],
                "selected_variant_dir": sel["selected_variant_dir"],
                # fingerprint only: cosmetic spec edits (repo_id, card text)
                # change the spec sha but not the variant's weights
                "fingerprint": (sel.get("selected_provenance") or {})
                .get("fingerprint")}
    rc2 = run_lm_eval("variant", var_args, out_var, provenance=var_prov)

    b, v = _parse_lm_eval(out_base), _parse_lm_eval(out_var)
    if not b or not v or b["acc"] is None or v["acc"] is None:
        logw("PARSE_FAILED: " + json.dumps({"base": b, "variant": v}))
        with open(exit_f, "w") as f:
            f.write("3")
        return 3

    delta_pp = (b["acc"] - v["acc"]) * 100.0
    max_loss = spec["gates"]["mmlu_max_loss_pp"]
    summary = {
        "base": b, "variant_model": v,
        "mmlu_base_pct": round(b["acc"] * 100, 2),
        "mmlu_variant_pct": round(v["acc"] * 100, 2),
        "mmlu_delta_pp": round(delta_pp, 2),
        f"guardrail_{str(max_loss).replace('.', '')}pp": delta_pp < max_loss,
        "guardrail_loss_pp_limit": max_loss,
        "variant": sel["selected"],
        # ties the guardrail result to the exact weights it evaluated;
        # publish compares this to selection.json
        "variant_fingerprint": var_prov["fingerprint"],
    }
    json.dump(summary, open(os.path.join(out_dir, "mmlu_summary.json"),
                            "w"), indent=2)
    logw("MMLU_DONE " + json.dumps(summary))
    rc = rc1 if rc1 else rc2 if rc2 else 0
    if rc == 0 and not delta_pp < max_loss:
        logw(f"GUARDRAIL FAILED: MMLU loss {delta_pp:.2f}pp >= "
             f"{max_loss}pp limit - STOP (publish will refuse)")
        print(f"MMLU GUARDRAIL FAILED: loss {delta_pp:.2f}pp >= "
              f"{max_loss}pp", flush=True)
        rc = 6
    with open(exit_f, "w") as f:
        f.write(str(rc))
    return rc


if __name__ == "__main__":
    sys.exit(mmlu_phase(sys.argv[1]))