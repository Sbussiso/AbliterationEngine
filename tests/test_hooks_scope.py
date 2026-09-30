"""hooks.scope amendment contract (v1 amendment, commit cbbeeca).

Guarantees pinned here so the Run-001 parity gate cannot drift:
- hooks ABSENT in YAML -> normalized spec carries NO hooks key at all
  (never default-injected) -> spec_hash unchanged -> byte-identical
  contract for Run-001 parity;
- hooks.scope validated strictly (selected|all) and injected ONLY when
  present;
- empty ladder.variants validates (hook-only characterization shape,
  Run 000 semantics) instead of demanding a non-empty subset;
- stage-plan note for empty ladder (misleading 'variants [] edit->...'
  line is a known cosmetic; pipeline must skip stage 5 in port).
"""
import os
import tempfile

import pytest
import yaml

from abliteration_engine.spec import SpecError, load_spec, spec_hash

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN001 = os.path.join(REPO, "harness_v3", "specs", "run001_parity.yaml")


def _load_dict(mutate=None):
    spec = yaml.safe_load(open(RUN001))
    if mutate:
        mutate(spec)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(spec, f)
    return load_spec(path)


def test_absent_hooks_never_injected():
    s = _load_dict()
    assert "hooks" not in s, (
        "hooks default-injected -> Run-001 spec_hash would change and "
        "byte-identical parity contract breaks")


def test_absent_hooks_hash_matches_freeze():
    """Full 64-hex spec_hash pinned: v1 amendment must not shift Run-001
    provenance. If this fails, either the amendment default-injected
    hooks or run001_parity.yaml changed - both need a parity review."""
    s = _load_dict()
    assert spec_hash(s) == ("54590e5d8e4511a7d579c0c8410f34c6138531f69d"
                            "d7cc117488044799293257")


def test_hooks_scope_all_roundtrip():
    s = _load_dict(lambda d: d.update({"hooks": {"scope": "all"}}))
    assert s["hooks"] == {"scope": "all"}


@pytest.mark.parametrize("bad", ["layers_1_to_5", "", None, ["all"]])
def test_hooks_scope_invalid_rejected(bad):
    with pytest.raises(SpecError):
        _load_dict(lambda d: d.update({"hooks": {"scope": bad}}))


def test_empty_ladder_validates():
    s = _load_dict(lambda d: d.update({"ladder": {"variants": []}}))
    assert s["ladder"]["variants"] == []


def test_wd_ML_BN_rule_still_enforced_with_empty_allowed():
    with pytest.raises(SpecError):
        _load_dict(lambda d: d.update(
            {"ladder": {"variants": ["wd_ML_BN"]}}))