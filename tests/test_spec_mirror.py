"""Spec-mirror integrity: harness_v3/specs/ is the freeze-review mirror of
the machine-read specs/ dir (bundle verb + CI consume specs/). Any file
present in BOTH dirs must be byte-identical - a copy that edits one side
silently = the exact drift class this harness exists to kill. Files in
only one dir are allowed (freeze-only or machine-only manifests).
"""
import filecmp
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "specs")
MIRROR = os.path.join(REPO, "harness_v3", "specs")


def _shared_names():
    pkg = {n for n in os.listdir(PKG) if n.endswith(".yaml")}
    mir = {n for n in os.listdir(MIRROR) if n.endswith(".yaml")}
    return sorted(pkg & mir), pkg ^ mir


def test_shared_specs_byte_identical():
    shared, _ = _shared_names()
    assert shared, "expected at least one mirrored spec (run001_parity)"
    for name in shared:
        assert filecmp.cmp(os.path.join(PKG, name),
                           os.path.join(MIRROR, name), shallow=False), (
            f"SPEC MIRROR DRIFT: {name} — copy the file, never hand-edit "
            "one side")


def test_run001_hash_identical_from_both_dirs():
    """The stronger contract: not just byte-identical files but identical
    NORMALIZED spec_hash from either location (catches yaml-formatting
    churn that survives byte-compare only by luck)."""
    from abliteration_engine.spec import load_spec, spec_hash
    h = {}
    for d in (PKG, MIRROR):
        p = os.path.join(d, "run001_parity.yaml")
        if os.path.exists(p):
            h[d] = spec_hash(load_spec(p))
    assert len(h) == 2
    assert h[PKG] == h[MIRROR] == (
        "54590e5d8e4511a7d579c0c8410f34c6138531f69dd7cc117488044799293257")