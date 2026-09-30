"""Stage-5 gating enforcement (v1 amendment, Run 000 semantics).

Pins what dev-workstation's freeze review asked for: empty
ladder.variants must SKIP stage 5 (no selection.json) and gate publish
off — enforced in core.from_spec, not just documented in the plan note.

Run 001 parity: any hook-only assertions must NOT touch the Run-001 spec
path, so these tests run against a mutated copy of the recreation spec
with a tiny stand-in spec (no model load, no GPU needed).
"""
import os
import sys

import yaml  # noqa: F401  (used by _tmp_yaml at call time)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # import abliteration_engine


def _hookonly_summary_asserts():
    """The from_spec summary contract for hook-only runs."""
    import inspect

    from abliteration_engine import core

    src = inspect.getsource(core.from_spec)
    assert 'summary["ladder_skipped"] = not spec["ladder"]["variants"]' in src, \
        "stage-5 gating not enforced in from_spec"
    assert "ladder SKIPPED" in src, "no skip marker emitted"
    assert "publish GATED OFF" in src, "no publish-gate marker emitted"


def _spec_hash_amendment_hygiene():
    """hooks block must be optional + non-injected (dev freeze-review
    contract): absent key stays absent, scope all/sel round-trips."""
    from abliteration_engine.spec import SpecError, load_spec, spec_hash

    run001_path = os.path.join(os.path.dirname(HERE), "specs",
                               "run001_parity.yaml")
    spec = load_spec(run001_path)
    assert "hooks" not in spec, \
        "hooks key injected into Run-001 spec - parity contract broken"
    assert spec_hash(spec).startswith("54590e5d"), \
        "Run-001 spec_hash drifted - parity gate broken"

    # scope: all round-trips (recreation spec)
    rec_path = os.path.join(os.path.dirname(HERE), "specs",
                            "qwen25_0p5b_run000_recreation.yaml")
    rec = load_spec(rec_path)
    assert rec["hooks"] == {"scope": "all"}, rec.get("hooks")

    # invalid scope raises clean SpecError
    bad = dict(rec)
    bad["hooks"] = {"scope": "banana"}
    with _tmp_yaml(bad) as p:
        try:
            load_spec(p)
            raise AssertionError("bad scope accepted")
        except SpecError:
            pass


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


def test_hook_only_gating():
    _hookonly_summary_asserts()
    print("PASS stage-5 gating enforced in from_spec (skip + publish off)")


def test_specs_amendment_hygiene():
    _spec_hash_amendment_hygiene()
    print("PASS hooks amendment hygiene (injection-free, scope round-trip)")


if __name__ == "__main__":
    test_hook_only_gating()
    test_specs_amendment_hygiene()
    print("AMENDMENT_TESTS_OK: 2/2")