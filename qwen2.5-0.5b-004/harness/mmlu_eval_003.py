#!/usr/bin/env python3
"""Mission 004 stage: MMLU base-vs-ablated via lm-eval, identical config.

Runs on the Colab VM as a DETACHED subprocess (survives CLI disconnects).
Sentinels:
  /content/mmlu_exit_code.txt : "running" | "0" | "<exit code>"
  /content/mmlu_log.txt       : full stdout/stderr
Final line: MMLU_DONE {json summary} in the log.

Identical config both sides: task=mmlu (full 57-subject), num_fewshot=0,
batch_size=auto, dtype=float16, seed=0. Delta guardrail: base - variant
must be < 3.0pp or the mission says STOP-and-report (checked by caller).
Variant = the ladder-selected dir from selection.json.
"""
import glob
import json
import os
import subprocess
import sys
import time

LOG = "/content/mmlu_log.txt"
EXIT = "/content/mmlu_exit_code.txt"
OUT_BASE = "/content/mmlu_results/base"
OUT_VAR = "/content/mmlu_results/variant"
BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def run_lm_eval(tag, model_args, out_dir):
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "mmlu",
        "--num_fewshot", "0",
        "--batch_size", "auto",
        "--seed", "0",
        "--output_path", out_dir,
    ]
    log(f"=== {tag}: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"=== {tag}: exit={r.returncode} wall={time.time() - t0:.0f}s")
    if r.stdout:
        log("--- stdout (tail 4000) ---")
        log(r.stdout[-4000:])
    if r.stderr:
        log("--- stderr (tail 4000) ---")
        log(r.stderr[-4000:])
    return r.returncode


def parse(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "**", "results_*.json"),
                             recursive=True))
    if not files:
        return None
    d = json.load(open(files[-1]))
    res = d.get("results", {})
    # task group "mmlu" aggregates acc,none across 57 subjects
    mmlu = res.get("mmlu") or res.get("all") or {}
    acc = mmlu.get("acc,none")
    stderr = mmlu.get("acc_stderr,none")
    if acc is None:
        # fall back: mean over subject tasks
        vals = [v.get("acc,none") for k, v in res.items()
                if k not in ("all",) and isinstance(v, dict)
                and v.get("acc,none") is not None]
        acc = sum(vals) / len(vals) if vals else None
        stderr = None
    n_subj = len([k for k in res if k not in ("all", "mmlu")])
    return {"acc": acc, "acc_stderr": stderr, "subjects_seen": n_subj,
            "results_file": files[-1],
            "version": d.get("config", {}).get("lm_eval_version")}


def main():
    os.system(f"echo running > {EXIT}")
    open(LOG, "w").close()
    log(f"mmlu stage start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "lm-eval"], capture_output=True, text=True)
    log(f"pip install lm-eval rc={r.returncode}")
    if r.returncode != 0:
        log("pip stderr tail: " + r.stderr[-3000:])
        os.system(f"echo 4 > {EXIT}")
        return
    import lm_eval  # noqa: F401  (verify import works before the long runs)
    log("lm_eval import OK")

    sel = json.load(open("/content/abliteration_out/selection.json"))
    variant_dir = sel["selected_variant_dir"]
    variant_name = sel["selected"]
    log(f"selected variant: {variant_name} at {variant_dir} "
        f"(gate={sel['gate']})")
    assert os.path.isdir(variant_dir), f"missing {variant_dir}"

    import torch  # ensure GPU is free of the ablation run
    log(f"cuda available: {torch.cuda.is_available()} "
        f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    base_args = f"pretrained={BASE_ID},revision={REVISION},dtype=float16"
    var_args = f"pretrained={variant_dir},dtype=float16"

    rc1 = run_lm_eval("base", base_args, OUT_BASE)
    rc2 = run_lm_eval("variant", var_args, OUT_VAR)
    log(f"exit codes: base={rc1} variant={rc2}")

    b = parse(OUT_BASE)
    v = parse(OUT_VAR)
    if not b or not v or b["acc"] is None or v["acc"] is None:
        log("PARSE_FAILED: " + json.dumps({"base": b, "variant": v}))
        os.system(f"echo 3 > {EXIT}")
        return
    delta_pp = (b["acc"] - v["acc"]) * 100.0
    summary = {
        "base": b, "variant": v,
        "mmlu_base_pct": round(b["acc"] * 100, 2),
        "mmlu_variant_pct": round(v["acc"] * 100, 2),
        "mmlu_delta_pp": round(delta_pp, 2),
        "guardrail_3pp": delta_pp < 3.0,  # loss < 3pp; negative = improvement
        "variant": variant_name,
    }
    json.dump(summary, open("/content/mmlu_summary.json", "w"), indent=2)
    log("MMLU_DONE " + json.dumps(summary))
    os.system(f"echo {rc1 if rc1 else rc2 if rc2 else 0} > {EXIT}")


if __name__ == "__main__":
    main()