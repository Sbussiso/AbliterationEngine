"""Artifact parity comparison — FTT-20 merge gate (Run 001 baseline).

CPU-only. Compares an eng-v3 run's artifact dir against a baseline run's
dir (Run 001 known-good from harness v2). Tolerances: exact for
deterministic JSON metrics; cosine+L1 for fp16 nondeterministic
directions; declared-eps per artifact in parity.yaml overrides.
"""
import json
import os

import numpy as np


def _cos_l1(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    l1 = float(np.abs(a - b).sum())
    return cos, l1


def _rates(probes_path):
    if not os.path.exists(probes_path):
        return None
    p = json.load(open(probes_path))
    rh, rb = p["harmful"], p["harmless"]
    return {
        "refusal_rate": sum(r["refused"] for r in rh) / max(1, len(rh)),
        "benign_preserved": 1 - sum(r["refused"] for r in rb)
        / max(1, len(rb)),
        "degenerate_total": sum(r["degenerate"] for r in rh + rb),
    }


def parity_check(spec, baseline_dir, run_dir=None, cos_tol=0.999,
                 l1_tol=0.01):
    """Run-dir defaults to the spec's expected out dir. Returns a result
    dict; parity_ok False on ANY mismatch beyond tolerance."""
    run_dir = run_dir or os.path.join(
        os.environ.get("ENG_OUT_ROOT", "/content"),
        f"eng_run_{spec['run_card']['run_number']:03d}_"
        f"{spec['run_card']['patient']}")
    checks = []
    ok = True

    # deterministic JSON metrics: exact
    for name in ("baseline", "hook_ablated"):
        a = _rates(os.path.join(baseline_dir, f"probes_{name}.json"))
        b = _rates(os.path.join(run_dir, f"probes_{name}.json"))
        exact = (a == b)
        checks.append({"artifact": f"probes_{name}.json",
                       "exact": exact, "baseline": a, "v3": b})
        ok &= exact

    # directions: fp16 nondeterminism headroom (cos >= tol, L1 <= tol)
    for f in ("refusal_direction_A.npy", "refusal_direction_B.npy"):
        pa = os.path.join(baseline_dir, f)
        pb = os.path.join(run_dir, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            checks.append({"artifact": f, "missing": True})
            ok = False
            continue
        cos, l1 = _cos_l1(np.load(pa), np.load(pb))
        passed = cos >= cos_tol and l1 <= l1_tol
        checks.append({"artifact": f, "cosine": round(cos, 6), "l1": round(l1, 6),
                       "passed": passed})
        ok &= passed

    # selection-level: selected variant + gate metrics must match exactly
    for f in ("selection.json",):
        if not (os.path.exists(os.path.join(baseline_dir, f))
                and os.path.exists(os.path.join(run_dir, f))):
            checks.append({"artifact": f, "missing": True})
            ok = False
            continue
        s_b = json.load(open(os.path.join(baseline_dir, f)))
        s_v = json.load(open(os.path.join(run_dir, f)))
        keys = ("selected", "gate", "publish_eligible_probe_gate")
        d = {k: (s_b.get(k), s_v.get(k)) for k in keys
             if s_b.get(k) != s_v.get(k)}
        exact = not d
        checks.append({"artifact": f, "exact": exact, "diff": d})
        ok &= exact

    return {"parity_ok": bool(ok), "run_dir": run_dir,
            "baseline_dir": baseline_dir, "checks": checks}