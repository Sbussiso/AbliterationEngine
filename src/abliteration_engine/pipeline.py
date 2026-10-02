"""GPU-stage pipeline (research-workstation, 2026-09-30).

`run` phase execution against the packaged engine. Faithful port of the
v2 mission-004 stage-A flow (run_003.py + ladder_003.py semantics,
already ported into core.from_spec / edits.run_ladder) with three
packaged-engine obligations added:

1. stage-5 gating ENFORCED (dev freeze review): empty ladder.variants ->
   stage 5 skipped, no selection.json, publish gated off;
2. sentinel contract 100% engine-owned (bundle runner never writes
   markers): writes /content/exit_code.txt, RUN/LADDER_DONE stdout
   markers, stage error files;
3. hook_scope spec field consumed by stage 4 (Run 000 recreation path).

Phase functions here are GPU-bound and Colab-side only; the local
CPU-safe surface stays plan/validate/parity/bundle per the standing
harness-testing division (user directive 2026-09-30).
"""
import json
import os
import sys

from . import core
from .spec import load_spec


def _write_sentinel_exit(path, code):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(code))


def _run_phase(fn, spec, exit_file, done_key):
    """One sentinel-wrapped engine stage. Writes exit_code; returns rc."""
    out_dir = core._out_dir(spec, create=True)
    os.makedirs(out_dir, exist_ok=True)
    rc = 1
    try:
        result = fn(spec)
        rc = 0
        out_dir = core._out_dir(spec)
        summary = result if isinstance(result, dict) else {"ok": rc}
        print(f"{done_key} " + json.dumps(summary, default=str), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            stage = done_key.split("_")[0].lower()
            err_f = f"{stage}_error.txt"
            with open(os.path.join(core._out_dir(spec), err_f), "w") as f:
                f.write(traceback.format_exc()[-8000:])
        except Exception:
            pass
    _write_sentinel_exit(exit_file, rc)
    return rc


def run_pipeline(spec):
    """`abliterate run` — stage A, then stage B WHEN the spec has ladder
    variants. Hook-only specs stop after stage A (stage-5 gating)."""
    exit_file = core.sentinel_exit()
    rc_a = _run_phase(core.from_spec, spec, exit_file, "RUN_DONE")
    if rc_a != 0:
        return rc_a
    if not spec["ladder"]["variants"]:
        print("stage B skipped: empty ladder (hook-only run)", flush=True)
        return 0
    from . import edits  # torch-bound module; deferred for CPU CLI paths
    return _run_phase(lambda s: edits.run_ladder(s, {"out_dir": core._out_dir(s),
                                                     "model": None}),
                      spec, exit_file, "LADDER_DONE")


def ladder_phase(spec_path):
    """`abliterate ladder` — stage B only against existing artifacts."""
    spec = load_spec(spec_path)
    if not spec["ladder"]["variants"]:
        print("REFUSING: ladder phase selected but spec has no variants "
              "(hook-only spec)", flush=True)
        _write_sentinel_exit(
            core.sentinel_exit(), 2)
        return 2
    from . import edits  # torch-bound module; deferred for CPU CLI paths
    core.ensure_markers(spec)  # standalone phase: from_spec never ran (rs2-1)
    exit_file = core.sentinel_exit()
    return _run_phase(lambda s: edits.run_ladder(s, {"out_dir": core._out_dir(s),
                                                     "model": None}),
                      spec, exit_file, "LADDER_DONE")


if __name__ == "__main__":
    sys.exit(run_pipeline(load_spec(sys.argv[1])))