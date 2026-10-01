"""REFUSAL_MARKERS standalone-phase regression (2026-09-30, rs2-1).

core.REFUSAL_MARKERS is a module global populated ONLY inside from_spec()
(core.py:401-402, the full-run path). core.run_probes() falls back to it via
`markers = markers or REFUSAL_MARKERS` (core.py:228). The rs2-1 incident:
the first-ever standalone PHASE=ladder ran pipeline.ladder_phase() without
from_spec ever executing, so markers stayed None and the first probe call
raised TypeError: 'NoneType' object is not iterable.

Fix contract (core.ensure_markers): any runtime phase that can reach
run_probes without from_spec MUST populate the global from its loaded spec
first. ladder_phase and mmlu_phase call it; the shim (rsladder.sh,
1856ac3) becomes redundant once this lands.

CPU-CI: no torch needed — ensure_markers only touches spec + data module.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from abliteration_engine import core, data  # noqa: E402


@pytest.fixture
def spec_dict():
    return {
        "run": {"name": "qwen2.5-0.5b-test"},
        "patient": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
        "probe_sets": {"refusal_markers": "builtin:fp_explicit_v1",
                       "n_probes": 2},
    }


def test_ensure_markers_populates_from_spec(spec_dict):
    core.REFUSAL_MARKERS = None
    core.ensure_markers(spec_dict)
    expected = data.resolve_markers("builtin:fp_explicit_v1")
    assert core.REFUSAL_MARKERS == expected
    assert len(core.REFUSAL_MARKERS) > 0


def test_ensure_markers_idempotent(spec_dict):
    core.REFUSAL_MARKERS = None
    core.ensure_markers(spec_dict)
    first = core.REFUSAL_MARKERS
    core.ensure_markers(spec_dict)
    assert core.REFUSAL_MARKERS is first  # same list object, not rebuilt


def test_ensure_markers_missing_key_raises_actionable(spec_dict):
    core.REFUSAL_MARKERS = None
    bad = {k: v for k, v in spec_dict.items() if k != "probe_sets"}
    core.ensure_markers.__doc__ and core.ensure_markers  # ensure present
    with pytest.raises((KeyError, ValueError)):
        core.ensure_markers(bad)
    # global untouched on failure: run_probes fall-back stays explicit
    assert core.REFUSAL_MARKERS is None


def test_file_ref_markers_resolved_via_helper(tmp_path, spec_dict):
    p = tmp_path / "custom.txt"
    p.write_text("I'm sorry, but I can't\n")
    spec_dict["probe_sets"]["refusal_markers"] = f"file:{p}"
    core.REFUSAL_MARKERS = None
    core.ensure_markers(spec_dict)
    assert core.REFUSAL_MARKERS == ["I'm sorry, but I can't"]