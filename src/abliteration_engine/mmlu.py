"""MMLU guardrail stage — FTT-20 port (mmlu_eval_003.py faithful).

Identical lm-eval config both sides (task=mmlu full 57-subject,
num_fewshot=0, batch_size=auto, dtype=float16, seed=0); delta guardrail
base-variant < gates.mmlu_max_loss_pp else STOP-and-report. Variant =
selection.json selected_variant_dir (ladder runs) — guarded-off for
hook-only specs.

Colab-side only; sentinel <eng_base>/mmlu_exit_code.txt + final line
MMLU_DONE {summary} in the log (v2 contract preserved for the poll loop).
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

    with open(exit_f, "w") as f:
        f.write("running")
    open(log, "w").close()
    logw(f"mmlu phase start "
         f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    pat = spec["patient"]
    base_args = (f"pretrained={pat['model_id']},"
                 f"revision={pat['revision']},dtype=float16")
    var_args = f"pretrained={sel['selected_variant_dir']},dtype=float16"
    out_base = os.path.join(out_dir, "mmlu_results", "base")
    out_var = os.path.join(out_dir, "mmlu_results", "variant")

    def run_lm_eval(tag, model_args, odir):
        # banked-MMLU resume: a prior session may have already banked this
        # side; reuse its results_*.json instead of re-burning 40+ min T4.
        prior = _parse_lm_eval(odir)
        if prior and prior["acc"] is not None:
            logw(f"=== {tag}: BANKED RESUME (prior-session results reused, "
                 f"acc={prior['acc']:.4f})")
            return 0
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
        return r.returncode

    rc1 = run_lm_eval("base", base_args, out_base)
    rc2 = run_lm_eval("variant", var_args, out_var)

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
    }
    json.dump(summary, open(os.path.join(out_dir, "mmlu_summary.json"),
                            "w"), indent=2)
    logw("MMLU_DONE " + json.dumps(summary))
    rc = rc1 if rc1 else rc2 if rc2 else 0
    with open(exit_f, "w") as f:
        f.write(str(rc))
    return rc


if __name__ == "__main__":
    sys.exit(mmlu_phase(sys.argv[1]))