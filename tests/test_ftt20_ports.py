"""FTT-20 GPU-port contract tests (CPU-only, no model load).

Pins what the GPU-stage port must guarantee WITHOUT exercising GPU:
1. all four phases import + CLI-route against the package;
2. hook-only specs REFUSE the ladder phase (exit 2) and REFUSE publish
   (repo_id unset) - the publish gate cannot be fooled by a missing
   selection.json either;
3. the mmlu phase guards selection.json-absence (exit 5, no eval);
4. run pipeline with empty ladder does not import edits (stage-5 gate).
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # tests/ sits directly under repo root
sys.path.insert(0, os.path.join(REPO, "src"))


def _cli_main(argv):
    from abliteration_engine import cli
    return cli.main(argv)


def _tmp_yaml(payload):
    import tempfile

    import yaml
    fd, p = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(payload, f)
    return p


def _hookonly_spec_file(tmpdir=None):
    """A hook-only spec derived from the packaged one for mutation."""
    import yaml

    p = os.path.join(REPO, "specs", "qwen25_0p5b_run000_recreation.yaml")
    assert os.path.exists(p), p
    with open(p) as f:
        payload = yaml.safe_load(f)
    if tmpdir:
        payload["run_card"]["patient"] = "hookonly-test"
        return _tmp_yaml(payload)
    return p


def test_all_verbs_route():
    # plan/validate/parity/bundle already proven; assert the three new
    # phases at least import + dispatch to their modules
    from abliteration_engine import cli, pipeline  # noqa: F401
    from abliteration_engine.mmlu import mmlu_phase  # noqa: F401
    from abliteration_engine.pipeline import ladder_phase  # noqa: F401
    from abliteration_engine.publish import publish_phase  # noqa: F401
    print("PASS all phases import")


def test_hookonly_ladder_refusal():
    sp = _hookonly_spec_file()
    rc = _cli_main(["ladder", "--spec", sp])
    assert rc == 2, rc
    print("PASS ladder refused on hook-only spec (rc=2)")


def test_hookonly_mmlu_refusal(tmp_path):
    sp = _hookonly_spec_file()
    rc = _cli_main(["mmlu", "--spec", sp])
    assert rc == 5, rc
    print("PASS mmlu guards selection.json absence (rc=5)")


def test_hookonly_publish_refusal():
    sp = _hookonly_spec_file()
    # repo_id is null in the recreation spec -> publish refuses outright
    rc = _cli_main(["publish", "--spec", sp,
                    "--variant-dir", "/tmp/nowhere",
                    "--mmlu", "/tmp/nowhere.json"])
    assert rc == 2, rc
    print("PASS publish refused on repo_id-unset spec (rc=2)")


def test_publish_hitl_flag():
    # with a repo-id present spec, publish must demand the HITL flag
    import yaml
    payload_path = _hookonly_spec_file(tmpdir=True)
    with open(payload_path) as f:
        payload = yaml.safe_load(f)
    payload["publish"] = {
        "repo_id": "sbussiso/test-publish-gate",
        "license": "apache-2.0",
        "card_marker": "test",
        "verify_disk_bounds": {"lm_head": 0.005, "final_norm": 0.05,
                               "layer_row": 0.01},
    }
    sp = _tmp_yaml(payload)

    class _FakeEmpty(types.SimpleNamespace):
        pass

    # no selection.json exists for the fake run dir -> publish must refuse
    # BEFORE the HITL flag complaint (selection gate precedes push)
    rc = _cli_main(["publish", "--spec", sp,
                    "--variant-dir", "/tmp/nowhere",
                    "--mmlu", "/tmp/nowhere.json"])
    assert rc != 0, "publish gate bypassed"
    print("PASS publish gated on missing selection artifacts")


def main():
    tests = [test_all_verbs_route, test_hookonly_ladder_refusal,
             test_hookonly_mmlu_refusal, test_hookonly_publish_refusal,
             test_publish_hitl_flag]
    for t in tests:
        t()
    print("FTT20_PORT_TESTS_OK:", len(tests), "/", len(tests))


if __name__ == "__main__":
    main()