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

    # selection-level: the SHARED-candidates rule (established on the first
    # real cross-version parity, 2026-09-30). Ladder lists legitimately differ
    # between spec generations (v2 run-001 ran wd_A/wd_B/wd_C; the v3 spec
    # declares the mission-004 list wd_B/BN/ML/ML_BN), so selection.json is
    # only comparable WHERE THE CANDIDATE SETS OVERLAP:
    #   - every candidate present in BOTH must have identical
    #     refusal_rate/benign_preserved/degenerate/passes_gate;
    #   - the baseline's selected variant, when present in the v3 run's
    #     candidates, must reproduce its v2 metrics exactly.
    # A pure selected-variant equality check would fail on legitimate spec
    # differences and pass on nothing, so it is not the gate.
    for f in ("selection.json",):
        if not (os.path.exists(os.path.join(baseline_dir, f))
                and os.path.exists(os.path.join(run_dir, f))):
            checks.append({"artifact": f, "missing": True})
            ok = False
            continue
        cands_b = {c["variant"]: c for c in json.load(
            open(os.path.join(baseline_dir, "selection_candidates.json")))}
        cands_v = {c["variant"]: c for c in json.load(
            open(os.path.join(run_dir, "selection_candidates.json")))}
        shared = sorted(set(cands_b) & set(cands_v))
        m = {}
        for name in shared:
            for k in ("refusal_rate", "benign_preserved",
                      "degenerate_total", "passes_gate"):
                vb, vv = cands_b[name].get(k), cands_v[name].get(k)
                if vb != vv:
                    m[f"{name}.{k}"] = (vb, vv)
        s_b = json.load(open(os.path.join(baseline_dir, f)))
        sel_b = s_b.get("selected")
        if sel_b in cands_v:
            for k in ("refusal_rate", "benign_preserved",
                      "degenerate_total"):
                vb, vv = cands_b[sel_b].get(k), cands_v[sel_b].get(k)
                if vb != vv:
                    m[f"selected.{sel_b}.{k}"] = (vb, vv)
        else:
            m[f"baseline-selected-{sel_b}-absent-from-v3"] = "info-only"
        exact = not {k: v for k, v in m.items() if v != "info-only"}
        checks.append({"artifact": f, "shared_candidates": shared,
                       "exact": exact, "diff": m})
        ok &= exact

    return {"parity_ok": bool(ok), "run_dir": run_dir,
            "baseline_dir": baseline_dir, "checks": checks}