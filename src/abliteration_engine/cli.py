#!/usr/bin/env python3
"""abliterate — single CLI for eng v3 (dev-workstation owns the
packaged form, this is the engine-side reference CLI).

Verbs:
  init      write a starter spec for a HF model (CPU; reads the hub)
  plan      print the full stage plan for a spec, NO model load (CPU-safe)
  validate  spec-load + artifact-contract pre-checks, CPU-safe
  run       execute the pipeline for a spec (GPU; Colab-side)
  ladder    execute stage B only (GPU)
  mmlu      execute the MMLU guardrail stage only (GPU)
  publish   execute the publish stage only (local VM 151; HITL-gated)
  parity    diff a run's artifacts against a baseline run dir (CPU-safe)

Every verb but init takes --spec <yaml>. GPU verbs refuse to run without
--i-know-this-spends-quota (HITL preserved in the CLI too).
"""
import argparse
import json
import os
import sys

from abliteration_engine.bundle import build_bundle
from abliteration_engine.data import resolve_markers, resolve_probe_set
from abliteration_engine.publish import PublishGateError
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


def run(spec_path, assume_yes=False, with_mmlu=False):
    if not assume_yes:
        return _quota_refusal("run")
    from abliteration_engine import core
    from abliteration_engine.pipeline import _write_sentinel_exit, run_pipeline
    spec = load_spec(spec_path)
    chain = with_mmlu and bool(spec["ladder"]["variants"])
    # chained: exit_code.txt must stay "running" through MMLU, not read "0"
    # the moment the ladder finishes
    rc = run_pipeline(spec, final=not chain)
    if with_mmlu and not chain:
        print("--with-mmlu: skipped (hook-only spec, no variant to "
              "evaluate)", flush=True)
    if rc != 0 or not chain:
        return rc
    from abliteration_engine.mmlu import mmlu_phase
    rc = mmlu_phase(spec_path)
    _write_sentinel_exit(core.sentinel_exit(), rc)
    return rc


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


VERBS = {
    # verb: (one-line help, needs --spec, GPU, verb-specific flags)
    "init": ("write a starter spec for a Hugging Face model (pins the "
             "revision, fills structure_expect)", False, False,
             ("model", "revision", "out", "run_number", "patient_name")),
    "plan": ("print the full stage plan for a spec — no model load, no side "
             "effects", True, False, ()),
    "validate": ("fail-fast spec check (pins, probe-set sizes, gates)",
                 True, False, ()),
    "run": ("stage A (capture, directions, probes) then the ladder; add "
            "--with-mmlu to chain the MMLU guardrail", True, True,
            ("with_mmlu",)),
    "ladder": ("stage B only: persistent-edit variants against existing "
               "stage-A artifacts", True, True, ()),
    "mmlu": ("MMLU guardrail: base vs selected variant (exits 6 on a failed "
             "guardrail)", True, True, ()),
    "publish": ("verify every gate, generate the model card, push to "
                "Hugging Face", True, False,
                ("variant_dir", "mmlu", "publishes")),
    "parity": ("diff a run's artifacts against a baseline run dir", True,
               False, ("baseline", "run_dir")),
    "bundle": ("freeze engine + spec + runner into a sha256'd tarball for "
               "Colab", True, False, ("out_dir",)),
}

_FLAGS = {
    # dest: (flags, kwargs)
    "spec": (("--spec",), {"metavar": "YAML", "help": "run spec file"}),
    "quota": (("--i-know-this-spends-quota",),
              {"action": "store_true", "dest": "i_know_this_spends_quota",
               "help": "confirm this verb may spend GPU quota"}),
    "with_mmlu": (("--with-mmlu",), {"action": "store_true",
                                     "help": "run the MMLU guardrail after "
                                             "the ladder (same session)"}),
    "baseline": (("--baseline",), {"metavar": "DIR",
                                   "help": "baseline artifacts dir"}),
    "run_dir": (("--run-dir",), {"metavar": "DIR",
                                 "help": "run artifacts dir (default: the "
                                         "spec's run dir)"}),
    "variant_dir": (("--variant-dir",),
                    {"metavar": "DIR",
                     "help": "selected variant's weights (default: "
                             "selection.json selected_variant_dir)"}),
    "mmlu": (("--mmlu",), {"metavar": "JSON",
                           "help": "mmlu_summary.json (default: the run "
                                   "dir's)"}),
    "publishes": (("--i-know-this-publishes",),
                  {"action": "store_true", "dest": "i_know_this_publishes",
                   "help": "confirm the push (required while "
                           "hitl.before_publish is true)"}),
    "out_dir": (("--out-dir",), {"metavar": "DIR",
                                 "help": "output dir for the tarball "
                                         "(default: bundles)"}),
    "model": (("--model",), {"metavar": "HF_ID",
                             "help": "model id, e.g. Qwen/Qwen2.5-1.5B-"
                                     "Instruct"}),
    "revision": (("--revision",), {"metavar": "SHA_OR_REF",
                                   "help": "branch/tag/sha to pin (default: "
                                           "main, resolved to its sha)"}),
    "out": (("--out",), {"metavar": "YAML",
                         "help": "where to write the spec (default: "
                                 "specs/<model>.yaml)"}),
    "run_number": (("--run-number",), {"type": int, "metavar": "N",
                                       "help": "run_card.run_number "
                                               "(default: 1)"}),
    "patient_name": (("--patient-name",),
                     {"metavar": "NAME",
                      "help": "run_card.patient short name (default: from "
                              "the model id)"}),
}


def _add(parser, dest, hidden=False):
    flags, kw = _FLAGS[dest]
    kw = dict(kw, default=argparse.SUPPRESS)
    if hidden:
        kw["help"] = argparse.SUPPRESS
    parser.add_argument(*flags, **kw)


def _build_parser():
    ap = argparse.ArgumentParser(
        prog="abliterate",
        description="Config-driven abliteration engine. Every run is a YAML "
                    "spec; `abliterate <verb> -h` shows a verb's options.",
        epilog="GPU verbs (run, ladder, mmlu) refuse without "
               "--i-know-this-spends-quota.")
    # every flag is also accepted BEFORE the verb (`abliterate --spec X
    # plan`); hidden here so `abliterate -h` stays readable. SUPPRESS
    # defaults keep a pre-verb flag from being clobbered by the subparser.
    for dest in _FLAGS:
        _add(ap, dest, hidden=dest != "spec")
    verbs = ap.add_subparsers(dest="verb", required=True, metavar="VERB")
    for verb, (desc, needs_spec, gpu, extra) in VERBS.items():
        sp = verbs.add_parser(verb, help=desc, description=desc)
        if needs_spec:
            _add(sp, "spec")
        # accepted (hidden) on CPU verbs too: runner.sh and habit pass it
        # to every verb, and it must not be an argparse error there
        _add(sp, "quota", hidden=not gpu)
        for dest in extra:
            _add(sp, dest)
    return ap


def _refuse(msg):
    print(f"REFUSING: {msg}", file=sys.stderr)
    return 2


def main(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)
    a = vars(args)
    if VERBS[args.verb][1] and not a.get("spec"):
        ap.error("--spec is required")
    if VERBS[args.verb][2] and not a.get("i_know_this_spends_quota"):
        return _quota_refusal(args.verb)
    try:
        return _dispatch(args.verb, a)
    except SpecError as e:
        return _refuse(f"spec invalid — {e}")
    except FileNotFoundError as e:
        return _refuse(f"file not found — {e.filename or e}")
    except PublishGateError as e:
        print(f"PUBLISH BLOCKED: {e}", file=sys.stderr)
        return 3


def _dispatch(verb, a):
    spec = a.get("spec")
    if verb == "init":
        if not a.get("model"):
            return _refuse("init needs --model <hf model id>")
        from abliteration_engine.init_spec import init_spec
        return init_spec(a["model"], revision=a.get("revision"),
                         out=a.get("out"), run_number=a.get("run_number", 1),
                         patient_name=a.get("patient_name"))
    if verb == "plan":
        return plan(spec)
    if verb == "validate":
        return validate(spec)
    if verb == "bundle":
        return bundle(spec, out_dir=a.get("out_dir", "bundles"))
    if verb == "parity":
        if not a.get("baseline"):
            return _refuse("parity needs --baseline <dir>")
        return parity(spec, a["baseline"], a.get("run_dir"))
    if verb == "run":
        return run(spec, True, with_mmlu=a.get("with_mmlu", False))
    if verb == "ladder":
        from abliteration_engine.pipeline import ladder_phase
        return ladder_phase(spec)
    if verb == "mmlu":
        from abliteration_engine.mmlu import mmlu_phase
        return mmlu_phase(spec)
    if verb == "publish":
        from abliteration_engine.publish import publish_phase
        return publish_phase(spec, a.get("variant_dir"), a.get("mmlu"),
                             assume_publish=a.get("i_know_this_publishes",
                                                  False))
    raise SystemExit(f"verb '{verb}' not implemented")


if __name__ == "__main__":
    sys.exit(main())