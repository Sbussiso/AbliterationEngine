#!/usr/bin/env bash
# rs2 resume shim: standalone PHASE=ladder never executes core.from_spec, so
# core.REFUSAL_MARKERS (module global set only inside from_spec) stays None
# and run_probes crashes. Set it from the spec before invoking the CLI.
set -euo pipefail
BUNDLE_DIR=/content
cd "$BUNDLE_DIR"
mkdir -p .venv
cd "$BUNDLE_DIR"
exec python3 - <<PYEOF
import sys, json
sys.path.insert(0, "/content/src")
from abliteration_engine import spec as specmod, data, core
spec = specmod.load_spec("/content/specs/qwen25_1p5b_run002_resume.yaml")
core.REFUSAL_MARKERS = data.resolve_markers(spec["probe_sets"]["refusal_markers"])
from abliteration_engine.pipeline import ladder_phase
sys.exit(ladder_phase("/content/specs/qwen25_1p5b_run002_resume.yaml"))
PYEOF
