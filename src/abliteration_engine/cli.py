#!/usr/bin/env python3
"""abliterate — single CLI for eng v3 (dev-workstation owns the
packaged form, this is the engine-side reference CLI).

Verbs (spec'd; GPU-bound ones implemented):
  plan      print the full stage plan for a spec, NO model load (CPU-safe)
  validate  spec-load + artifact-contract pre-checks, CPU-safe
  run       execute the pipeline for a spec (GPU; Colab-side)
  ladder    execute stage B only (GPU)
  mmlu      execute the MMLU guardrail stage only (GPU)
  publish   execute the publish stage only (local VM 151; HITL-gated)
  parity    diff a run's artifacts against a baseline run dir (CPU-safe)

Every verb takes --spec <yaml>. GPU verbs refuse to run without
--i-know-this-spends-quota (HITL preserved in the CLI too).
"""
import argparse
import json
import os
import sys

from abliteration_engine.bundle import build_bundle
from abliteration_engine.data import resolve_markers, resolve_probe_set
from abliteration_engine.spec import SpecError, load_spec, spec_hash


def plan(spec_path):
    """Full stage plan with NO model load. CPU-safe. The dry-run."""
    spec = load_spec(spec_path)
    ps = spec["probe_sets"]
    harmful = resolve_probe_set(ps["harmful"])
    harmless = resolve_probe_set(ps["harmless"])
    markers = resolve_markers(ps["refusal_markers"])
    pat = spec["patient"]
    rc = spec["run_card"]
    print(f"=== abliterate plan — run {rc['run_number']} "
          f"({rc['patient']}) spec {os.path.basename(spec_path)} "
          f"(sha {spec_hash(spec)[:12]})")
    print(f"patient: {pat['model_id']} @ {pat['revision']}")
    se = pat.get("structure_expect") or {}
    if se:
        print(f"structure_expect: {json.dumps(se, sort_keys=True)}")
    print(f"probe sets: harmful={ps['harmful']} ({len(harmful)} prompts), "
          f"harmless={ps['harmless']} ({len(harmless)} prompts)")
    print(f"markers: builtin={ps['refusal_markers']} "
          f"({len(markers)} markers)")
    print(f"decoding: {spec['decoding']}")
    lad_keys = {k: v for k, v in spec["ladder"].items()
                if k != "variants"} if spec["ladder"].get("variants") \
        else {"note": "empty ladder = hook-only characterization "
                      "(stage 5 + publish gated off)"}
    print(f"ladder: {spec['ladder']['variants']} {lad_keys}")
    if spec["ladder"].get("ara"):
        a = spec["ladder"]["ara"]
        print(f"  ARA: rank={a['rank']} layers={a.get('layers') or 'ALL'} "
              f"w_pg={a['preserve_good_weight']} w_sb={a['steer_bad_weight']} "
              f"w_oc={a['overcorrect_weight']} k={a['neighbor_count']} "
              f"steps={a['steps']}x{a['max_iter']} lr={a['lr']} "
              f"pools {a['good']}/{a['bad']}")
    if spec.get("directions"):
        print(f"directions: {spec['directions']}")
    print(f"gates: {spec['gates']}")
    pub = spec.get("publish") or {}
    if pub:
        print(f"publish: {pub.get('repo_id')} "
              f"license={pub.get('license')} "
              f"HITL before_publish={spec.get('hitl', {}).get('before_publish')}")
    print("stage plan:")
    stages = [
        ("1  load_patient", f"{pat['model_id']} @ {pat['revision'][:8]}"),
        ("2  capture", f"{ps['n_pairs']}+{ps['n_pairs']} prompts, "
                       "all-layer final-position residuals"),
        ("3  directions", "coherence scan -> L*, dir_A (residual), "
                          "dir_B (readout), layer_directions.npz"),
        ("4  probe", f"baseline + hook-ablated, {ps['n_probes']}+"
                     f"{ps['n_probes']} probes each, greedy"),
        ("5  ladder", f"variants {spec['ladder']['variants']} "
                      "(edit->save->reload->verify->probe each)"),
        ("5.5 mmlu", "base vs variant, "
                     f"delta <= {spec['gates']['mmlu_max_loss_pp']}pp gate"),
        ("6  publish", f"{pub.get('repo_id', '<unset: stage gated off>')} "
                       f"(HITL after_selection="
                       f"{spec.get('hitl', {}).get('after_selection')})"),
    ]
    for name, desc in stages:
        print(f"  {name:14s} {desc}")
    return 0


def validate(spec_path):
    spec = load_spec(spec_path)
    ps = spec["probe_sets"]
    harmful = resolve_probe_set(ps["harmful"])
    harmless = resolve_probe_set(ps["harmless"])
    markers = resolve_markers(ps["refusal_markers"])
    # explicit raises, not asserts: `python -O` must not skip validation
    if not (harmful and harmless and markers):
        raise SpecError("probe sets and refusal markers must be non-empty")
    for label, prompts in (("harmful", harmful), ("harmless", harmless)):
        need = max(ps["n_pairs"], ps["n_probes"])
        if len(prompts) < need:
            raise SpecError(f"probe_sets.{label} has {len(prompts)} prompts,"
                            f" n_pairs/n_probes need {need}")
    spec_hash(spec)
    return 0


def parity(spec_path, baseline_dir, run_dir=None, cos_tol=0.999,
           l1_tol=0.01):
    """Compare a v3 run's artifacts against a baseline (Run 001) dir.
    CPU-safe; exit 0 iff parity within tolerance."""
    from abliteration_engine.parity import parity_check
    spec = load_spec(spec_path)
    result = parity_check(spec, baseline_dir, run_dir,
                          cos_tol=cos_tol, l1_tol=l1_tol)
    ok = result["parity_ok"]
    print(json.dumps(result, indent=2, default=str))
    return 0 if ok else 1


def _quota_refusal(verb):
    print(f"REFUSING: `abliterate {verb}` spends GPU quota. Re-invoke with "
          "--i-know-this-spends-quota (HITL preserved).")
    return 2


def run(spec_path, assume_yes=False):
    if not assume_yes:
        return _quota_refusal("run")
    from abliteration_engine.pipeline import run_pipeline
    return run_pipeline(load_spec(spec_path))


def bundle(spec_path, out_dir="bundles"):
    """Build the one-upload Colab artifact (CPU-safe, no model load)."""
    try:
        result = build_bundle(spec_path, out_dir=out_dir)
    except SpecError as e:
        print(f"REFUSING: spec invalid — {e}", file=sys.stderr)
        return 2
    print(f"=== abliterate bundle — {result['run_dir']} "
          f"(spec sha {result['spec_sha'][:12]}, engine-verified)")
    print(f"tarball: {result['tar']}")
    print(f"sha256:  {result['sha256']}")
    print("runner contract: uv sync --frozen (system-site-packages venv), "
          "sentinels engine-owned, ENG_OUT_ROOT=/content")
    return 0


def main(argv=None):
    # Shared flags on a parent parser so `--spec` works before OR after
    # the verb (both `abliterate --spec X plan` and `abliterate plan
    # --spec X` parse identically).
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS so a flag given on the main parser (pre-verb) is not
    # clobbered by the subparser's absent-default behavior.
    common.add_argument("--spec", default=argparse.SUPPRESS)
    common.add_argument("--baseline", default=argparse.SUPPRESS,
                        help="[parity] baseline artifacts dir")
    common.add_argument("--run-dir", default=argparse.SUPPRESS,
                        help="[parity] v3 artifacts dir")
    common.add_argument("--i-know-this-spends-quota", action="store_true",
                        default=argparse.SUPPRESS)
    common.add_argument("--variant-dir", default=argparse.SUPPRESS,
                        help="[publish] local dir holding selected variant")
    common.add_argument("--mmlu", default=argparse.SUPPRESS,
                        help="[publish] mmlu_summary.json path")
    common.add_argument("--i-know-this-publishes", action="store_true",
                        default=argparse.SUPPRESS)
    common.add_argument("--out-dir", default=argparse.SUPPRESS,
                        help="[bundle] output dir for the tarball")

    ap = argparse.ArgumentParser(prog="abliterate", parents=[common])
    verbs = ap.add_subparsers(dest="verb", required=True)
    for v in ("plan", "validate", "run", "ladder", "mmlu", "publish",
              "parity", "bundle"):
        verbs.add_parser(v, parents=[common])
    args = ap.parse_args(argv)
    if not getattr(args, "spec", None):
        ap.error("--spec is required")

    if args.verb == "plan":
        return plan(args.spec)
    if args.verb == "validate":
        return validate(args.spec)
    if args.verb == "bundle":
        return bundle(args.spec, out_dir=getattr(args, "out_dir", "bundles"))
    if args.verb == "parity":
        return parity(args.spec, getattr(args, "baseline", None),
                      getattr(args, "run_dir", None))
    if args.verb == "run":
        return run(args.spec,
                   getattr(args, "i_know_this_spends_quota", False))
    if args.verb in ("ladder", "mmlu") and \
            not getattr(args, "i_know_this_spends_quota", False):
        return _quota_refusal(args.verb)
    if args.verb == "ladder":
        from abliteration_engine.pipeline import ladder_phase
        return ladder_phase(args.spec)
    if args.verb == "mmlu":
        from abliteration_engine.mmlu import mmlu_phase
        return mmlu_phase(args.spec)
    if args.verb == "publish":
        from abliteration_engine.publish import publish_phase
        if not (getattr(args, "variant_dir", None)
                and getattr(args, "mmlu", None)):
            ap.error("publish needs --variant-dir and --mmlu")
        return publish_phase(args.spec, args.variant_dir, args.mmlu,
                             assume_publish=getattr(
                                 args, "i_know_this_publishes", False))
    raise SystemExit(f"verb '{args.verb}' not implemented yet")


if __name__ == "__main__":
    sys.exit(main())