# Regression test: refusAL_MARKERS cross-spec leak in a single persistent kernel.
# Scenario (this incident class): a process runs spec A (run_pipeline/from_spec
# sets the module global), then runs spec B whose markers differ. ensure_markers
# must NOT short-circuit on the stale global — every phase that starts with a
# NEW spec must re-resolve from THAT spec.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from abliteration_engine import core, data  # noqa: E402


def _spec(markers):
    return {
        "run": {"name": "cross-spec-test"},
        "patient": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
        "probe_sets": {"refusal_markers": markers, "n_probes": 2},
    }


def test_ensure_markers_reresolves_when_spec_markers_change():
    core.REFUSAL_MARKERS = None
    spec_a = _spec(["i'm sorry"])
    core.ensure_markers(spec_a)
    assert core.REFUSAL_MARKERS == ["i'm sorry"]

    # spec B in the same kernel wants a DIFFERENT marker list: the stale
    # spec-A global must not survive.
    spec_b = _spec(["absolutely not"])
    core.ensure_markers(spec_b)
    assert core.REFUSAL_MARKERS == ["absolutely not"], (
        "ensure_markers kept the previous spec's markers across specs")


def test_from_spec_marker_source_of_truth():
    # from_spec (full-run path) also re-resolves unconditionally — pin it.
    import inspect
    src = inspect.getsource(core.from_spec)
    assert "refusal_markers" in src and "global REFUSAL_MARKERS" in src, (
        "from_spec must resolve markers from ITS spec, not a global short-circuit")


def test_builtin_change_detected():
    core.REFUSAL_MARKERS = None
    core.ensure_markers(_spec("builtin:fp_explicit_v1"))
    assert core.REFUSAL_MARKERS == data.resolve_markers("builtin:fp_explicit_v1")
    other = [m for m in data.BUILTIN_MARKERS
             if data.resolve_markers(f"builtin:{m}") !=
             core.REFUSAL_MARKERS]
    if other:  # a second builtin exists: switching specs must switch markers
        core.ensure_markers(_spec(f"builtin:{other[0]}"))
        assert core.REFUSAL_MARKERS == data.resolve_markers(f"builtin:{other[0]}")