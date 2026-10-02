"""Bundle-verb contract tests (CPU-only).

The bundle is the one-upload Colab artifact; the operator contract is:
- contains the package source, pyproject.toml, uv.lock, the run spec,
  runner.sh, bundle_meta.json (hash manifest);
- runner installs via `uv sync --frozen` in a --system-site-packages venv
  (Colab CUDA torch reused, versions exactly as CI validated);
- sentinel contract stays engine-owned (no runner-side sentinel logic);
- manifest carries spec_sha256, uv.lock sha256, source git commit;
- tarball is reproducible per-spec-content and named with the canonical
  run dir name (no two runs share a dir).
"""
import json
import os
import tarfile

import pytest

from abliteration_engine import cli

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(REPO, "specs", "run001_parity.yaml")
REQUIRED_MEMBERS = {
    "runner.sh", "pyproject.toml", "uv.lock", "bundle_meta.json",
    "specs/run001_parity.yaml",
    "src/abliteration_engine/__init__.py",
    "src/abliteration_engine/core.py",
    "src/abliteration_engine/edits.py",
    "src/abliteration_engine/spec.py",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("bundles")
    rc = cli.main(["bundle", "--spec", SPEC, "--out-dir", str(out)])
    assert rc == 0
    tars = sorted(p for p in os.listdir(out) if p.endswith(".tar.gz"))
    assert len(tars) == 1, tars
    tar_path = os.path.join(out, tars[0])
    return tar_path, out


def test_bundle_members_complete(built):
    tar_path, _ = built
    with tarfile.open(tar_path) as tf:
        names = set(tf.getnames())
    missing = REQUIRED_MEMBERS - names
    assert not missing, f"missing from bundle: {sorted(missing)}"


def test_runner_contract(built):
    tar_path, _ = built
    with tarfile.open(tar_path) as tf:
        runner = tf.extractfile("runner.sh").read().decode()
    assert "uv sync --frozen" in runner
    assert "--system-site-packages" in runner
    assert "--i-know-this-spends-quota" in runner
    assert "PHASE" in runner          # run|ladder|mmlu|publish selector
    assert "exit_code.txt" not in runner   # sentinel stays engine-owned
    assert "ENG_OUT_ROOT" in runner


def test_manifest_hashes(built):
    tar_path, _ = built
    with tarfile.open(tar_path) as tf:
        meta = json.loads(tf.extractfile("bundle_meta.json").read())
    assert len(meta["spec_sha256"]) == 64
    assert len(meta["uv_lock_sha256"]) == 64
    assert meta["engine_version"]
    assert meta["run_dir"].startswith("eng_run_")
    names = meta["files"]
    assert "runner.sh" in names and "specs/run001_parity.yaml" in names
    # spec content inside bundle matches the manifest's spec hash
    with tarfile.open(tar_path) as tf:
        import hashlib
        blob = tf.extractfile("specs/run001_parity.yaml").read()
    assert hashlib.sha256(blob).hexdigest() == meta["files"][
        "specs/run001_parity.yaml"]


def test_sidecar_sha256_matches(built):
    tar_path, out = built
    sidecar = tar_path + ".sha256"
    assert os.path.exists(sidecar)
    digest = open(sidecar).read().split()[0]
    import hashlib
    assert hashlib.sha256(open(tar_path, "rb").read()).hexdigest() == digest


def test_bundle_spec_rejects_bad(built, tmp_path):
    """A bundle for an invalid spec never builds (fail-fast preserved)."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("spec_version: 9\n")
    rc = cli.main(["bundle", "--spec", str(bad), "--out-dir",
                   str(tmp_path / "o")])
    assert rc != 0


def test_bundle_carries_readme(tmp_path):
    """pyproject declares readme='README.md'; builder must stage it.
    Caught live on Colab 2026-09-30: uv sync --frozen failed at the
    wheel-build step ('failed to open README.md') without this."""
    import tarfile

    from abliteration_engine.bundle import build_bundle

    spec = os.path.join(REPO, "specs", "run001_parity.yaml")
    out = build_bundle(spec, out_dir=str(tmp_path / "bundles"))
    with tarfile.open(out["tar"]) as tf:
        names = tf.getnames()
    assert "README.md" in names, names
    print("PASS bundle carries README.md (wheel build can read it)")