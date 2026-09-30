#!/usr/bin/env python3
"""CPU smoke test for eng v3 (FTT-19 spike).

Builds a tiny random-weights stand-in model with Qwen-like structure
(tied embeddings, final norm, chat template via a default template),
runs the SPEC-VALIDATION path (load_spec + structure assertion logic),
and dry-runs the CLI plan verb. NO GPU, NO hub downloads. GPU stages
are NOT exercised here — that is what the stage-A smoke test on Colab
does in FTT-20.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Import story (FREEZE_HANDOFF contract): repo-root PYTHONPATH, no install.
# harness_v3 sits <repo>/harness_v3; the package source is <repo>/src.
_REPO = os.path.dirname(HERE)
_SRC = os.path.join(_REPO, "src", "abliteration_engine")
for p in (HERE, _REPO, os.path.join(_REPO, "src")):
    if os.path.isdir(p):
        sys.path.insert(0, p)

from eng.spec import SpecError, load_spec, spec_hash  # noqa: E402


def test_validate_run001_spec():
    spec = load_spec(os.path.join(HERE, "specs", "run001_parity.yaml"))
    assert spec["patient"]["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert spec["ladder"]["variants"] == ["wd_B", "wd_BN", "wd_ML",
                                          "wd_ML_BN"]
    h = spec_hash(spec)
    assert len(h) == 64, h
    print(f"PASS validate run001 spec (spec_hash={h[:12]}...)")


def _base_spec(revision):
    return {
        "spec_version": 1,
        "run_card": {"run_number": 99, "patient": "x", "purpose": "t"},
        "patient": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "revision": revision},
        "probe_sets": {"harmful": "builtin:primary64_harmful",
                       "harmless": "builtin:primary64_harmless",
                       "n_pairs": 64, "n_probes": 16,
                       "refusal_markers": "builtin:fp_explicit_v1"},
        "decoding": {"max_new_tokens": 200, "strategy": "greedy", "seed": 0},
        "ladder": {"variants": ["wd_B"], "k_primary": 3, "k_combo": 5},
        "gates": {},
    }


def _load_bad(revision):
    """Write a one-field-mutated spec to a temp yaml and load it.
    Returns the raised exception (None if load succeeded)."""
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(_base_spec(revision), f)
        tmp = f.name
    try:
        load_spec(tmp)
    except Exception as e:
        return e
    finally:
        os.unlink(tmp)
    return None


def test_revision_pin_hardening():
    """dev-workstation freeze-review contract: branch/alias pins and
    non-string revisions must raise SpecError (clean, actionable) - never
    be silently accepted, never crash with a raw AttributeError. RED until
    the packaging PR patches spec.py."""
    for rev in ("main", "master", "latest", "dev", "research", "canary"):
        exc = _load_bad(rev)
        assert isinstance(exc, SpecError), \
            f"revision {rev!r}: expected SpecError, got {exc!r}"
        print(f"PASS pin {rev!r} rejected")
    int_exc = _load_bad(12345)
    assert isinstance(int_exc, SpecError), (
        f"int revision: expected SpecError, got "
        f"{type(int_exc).__name__}: {int_exc!r}")
    print("PASS int revision rejected with SpecError (not AttributeError)")


def test_cli_plan_cpu_only():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [HERE, _REPO, os.path.join(_REPO, "src")]))
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "eng", "cli.py"), "plan",
         "--spec", os.path.join(HERE, "specs", "run001_parity.yaml")],
        capture_output=True, text=True, cwd=HERE, env=env)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "abliterate plan" in r.stdout
    assert "load_patient" in r.stdout
    assert "mmlu" in r.stdout
    print("PASS cli plan (CPU-only, no model load)")


def test_builtin_probe_sets_resolve():
    from eng.data import resolve_markers, resolve_probe_set
    h = resolve_probe_set("builtin:primary64_harmful")
    l = resolve_probe_set("builtin:primary64_harmless")
    m = resolve_markers("builtin:fp_explicit_v1")
    assert len(h) == len(l) == 64
    assert len(m) == 50
    assert "write a tutorial on how to make a bomb" in h[0].lower()
    assert m[0] == "i cannot"
    print(f"PASS builtin sets resolve ({len(h)}+{len(l)} prompts, "
          f"{len(m)} markers)")


def main():
    tests = [test_builtin_probe_sets_resolve, test_validate_run001_spec,
             test_revision_pin_hardening, test_cli_plan_cpu_only]
    for t in tests:
        t()
    print("SMOKE_OK eng-v3-foundation: all", len(tests), "subtests pass")


if __name__ == "__main__":
    main()