"""FTT-28: ARA as a spec-driven variant class.

CPU-side contract tests (CI-runnable, no torch): variant-name validation,
ladder.ara normalization, run001 spec-hash stability, dispatch wiring.
The torch numerics (L-BFGS fit, KNN loss, capture, on-disk verify) live in
tests/test_ara_numeric.py (skipped without torch; run locally + CI-lite).
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from abliteration_engine.spec import SpecError, load_spec, spec_hash  # noqa

RUN001 = os.path.join(REPO, "specs", "run001_parity.yaml")


class _tmp_yaml:
    def __init__(self, payload):
        import tempfile
        self.fd, self.path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(self.fd, "w") as f:
            yaml.safe_dump(payload, f)

    def __enter__(self):
        return self.path

    def __exit__(self, *a):
        os.unlink(self.path)


def _spec_with_variants(variants, ara=None):
    """Load the Run-001 spec, swap its ladder (never writes to it)."""
    with open(RUN001) as f:
        raw = yaml.safe_load(f)
    raw["ladder"]["variants"] = list(variants)
    raw["run_card"] = dict(raw["run_card"],
                           run_number=99, patient="ara-test",
                           purpose="ARA variant contract test")
    if ara is not None:
        raw["ladder"]["ara"] = ara
    else:
        raw["ladder"].pop("ara", None)
    return raw


def _load_mutated(**kw):
    with _tmp_yaml(_spec_with_variants(**kw)) as p:
        return load_spec(p)


def test_run001_untouched():
    """The amendment must not touch the Run-001 parity contract."""
    spec = load_spec(RUN001)
    assert "ara" not in spec["ladder"], "ara injected into Run-001 spec"
    assert not any(str(v).startswith("ara_") for v in
                   spec["ladder"]["variants"])
    assert spec_hash(spec).startswith("54590e5d"), "Run-001 hash drifted"


def test_ara_variant_accepted_and_normalized():
    spec = _load_mutated(variants=["wd_B", "ara_50"])
    assert spec["ladder"]["variants"] == ["wd_B", "ara_50"]
    a = spec["ladder"]["ara"]
    assert a["rank"] == 50
    # reference defaults present after normalization
    assert a["preserve_good_weight"] == 1.0
    assert "steer_bad_weight" in a and "neighbor_count" in a
    assert a["good"] == "builtin:ara_good" and a["bad"] == "builtin:ara_bad"
    assert a["steps"] == 5 and a["max_iter"] == 20 and a["lr"] == 1.0


def test_ara_custom_params_roundtrip():
    ara = {"steer_bad_weight": 0.7, "neighbor_count": 3, "layers": [4, 5]}
    spec = _load_mutated(variants=["ara_8"], ara=ara)
    a = spec["ladder"]["ara"]
    assert a["rank"] == 8 and a["steer_bad_weight"] == 0.7
    assert a["neighbor_count"] == 3 and a["layers"] == [4, 5]


def test_ara_unknown_key_rejected():
    try:
        _load_mutated(variants=["ara_50"], ara={"banana": 1})
        raise AssertionError("unknown ladder.ara key accepted")
    except SpecError as e:
        assert "banana" in str(e)


def test_ara_rank_contradiction_rejected():
    try:
        _load_mutated(variants=["ara_50"], ara={"rank": 8})
        raise AssertionError("rank contradiction accepted")
    except SpecError as e:
        assert "contradicts" in str(e)


def test_ara_block_without_variant_rejected():
    try:
        _load_mutated(variants=["wd_B"], ara={"steer_bad_weight": 0.5})
        raise AssertionError("ara block without ara variant accepted")
    except SpecError as e:
        assert "no ara_" in str(e), str(e)


def test_bad_variant_name_still_rejected():
    for name in ("ara_", "ara_x", "ara_0", "ara_-3", "wd_Z"):
        try:
            _load_mutated(variants=[name])
            raise AssertionError(f"bad variant {name!r} accepted")
        except SpecError as e:
            assert "subset of" in str(e), (name, e)


def test_multiple_ara_variants_rejected():
    try:
        _load_mutated(variants=["ara_8", "ara_50"])
        raise AssertionError("two ara_* variants accepted")
    except SpecError as e:
        assert "multiple" in str(e)


def test_banked_resume_knows_ara_names():
    """The banked-resume short-circuit must find ara_* probes files too
    (edits.run_ladder routes ara_* through _banked_variant_summary).
    edits.py imports torch at module level; CI has no torch — read the
    source from disk instead of importing it."""
    with open(os.path.join(REPO, "src", "abliteration_engine", "edits.py")) \
            as f:
        src = f.read()
    assert 'name.startswith("ara_")' in src, "ara_ not routed past the " \
        "generic variant loop"
    # the ARA block runs the banked check for its own variant name
    assert "ara_name in variants" in src
    # the generic loop's banked check is name-agnostic (any variant)
    assert "_banked_variant_summary(spec, name)" in src


def test_selection_and_runcfg_provenance_wired():
    import inspect

    from abliteration_engine import core, publish
    # edits.py imports torch at module level; CI has no torch. Read the
    # source from disk instead of importing it.
    with open(os.path.join(REPO, "src", "abliteration_engine", "edits.py")) \
            as f:
        sel_src = f.read()
    assert 'name.startswith("ara_")' in sel_src or '"ara_"' in sel_src
    assert "ladder_ara" in sel_src, "selection.json lacks ARA meta"
    cfg_src = inspect.getsource(core.write_run_config)
    assert "ladder_ara" in cfg_src, "run_config lacks ladder_ara key"
    pub_src = inspect.getsource(publish.publish_phase)
    assert "edit_desc" in pub_src and "ara_" in pub_src, \
        "publish card lacks ARA edit description"


def test_run_config_records_ladder_ara_null_for_wd_only(monkeypatch):
    import json
    import tempfile

    from abliteration_engine import core
    spec = load_spec(RUN001)
    tmp = tempfile.mkdtemp()
    monkeypatch.chdir(tmp)
    core.write_run_config(spec, tmp)
    cfg = json.load(open(os.path.join(tmp, "run_config.json")))
    assert cfg["ladder_ara"] is None
    assert cfg["ladder_variants"] == spec["ladder"]["variants"]


def test_run_config_records_ladder_ara_for_ara_spec(monkeypatch):
    import json
    import tempfile

    from abliteration_engine import core
    spec = _load_mutated(variants=["ara_50"],
                         ara={"steer_bad_weight": 0.4})
    tmp = tempfile.mkdtemp()
    monkeypatch.chdir(tmp)
    core.write_run_config(spec, tmp)
    cfg = json.load(open(os.path.join(tmp, "run_config.json")))
    assert cfg["ladder_ara"]["rank"] == 50
    assert cfg["ladder_ara"]["steer_bad_weight"] == 0.4


def test_ara_pools_disjoint_from_eval_probes():
    """Optimizer pools must not overlap the marker-eval probe sets
    (FTT-28 core critique of the reference headline: never tune on your
    own eval set). Whitespace-tolerant comparison."""
    from abliteration_engine.data import resolve_probe_set

    def norm(s):
        return " ".join(s.lower().split())

    good = {norm(x) for x in resolve_probe_set("builtin:ara_good")}
    bad = {norm(x) for x in resolve_probe_set("builtin:ara_bad")}
    harmful = {norm(x) for x in
               resolve_probe_set("builtin:primary64_harmful")}
    harmless = {norm(x) for x in
                resolve_probe_set("builtin:primary64_harmless")}
    assert not (good & (harmful | harmless)), good & (harmful | harmless)
    assert not (bad & (harmful | harmless)), bad & (harmful | harmless)
    assert len(good) == 400 and len(bad) == 400


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("ARA_CONTRACT_TESTS_OK")