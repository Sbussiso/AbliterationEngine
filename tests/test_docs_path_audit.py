"""Tutorials + README path audit (v0.2.0): every file path that the docs
claim exists must exist in the tree (the c5e7410 audit, now CI-enforced —
the doc set grew with the FTT-28 ARA amendment and nothing re-audited it).

Scope: backticked repo-relative paths in tutorials/*.md + README.md.
Deliberately allowed-missing: specs/my_first_run.yaml (reader-created per
Tutorial 2 instructions) and any path in fenced code the reader edits.
Markdown links are NOT checked here (they were verified live at c5e7410;
path resolution is the drifting part).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DOCS = [
    os.path.join(REPO, "README.md"),
    *sorted(os.path.join(REPO, "tutorials", name)
            for name in os.listdir(os.path.join(REPO, "tutorials"))
            if name.endswith(".md")),
]

# explicit allowlist: reader-created or illustrative-only paths
ALLOWED_MISSING = {
    "specs/my_first_run.yaml",
}

# paths that live under a directory the reader creates/copies (prefix forms)
PREFIX_OK = ("eng_run_", "<results-folder>", "<the-winner's-model-folder>",
             "/content/", "bundles/")


def _candidate_paths(md_text):
    """Backticked tokens that look like repo paths."""
    out = set()
    for tick in re.findall(r"`([^`\n]+)`", md_text):
        tok = tick.strip()
        if tok.startswith("!") or " " in tok or "(" in tok:
            continue
        if not tok or "/" not in tok:
            continue
        if tok.endswith((".yaml", ".json", ".py", ".md", ".txt", ".png",
                         ".npy", ".npz")) or tok.startswith(("src/", "specs/",
                                                            "tests/", "docs/",
                                                            "tutorials/")):
            out.add(tok)
    return out


def test_documented_paths_resolve():
    missing = []
    checked = 0
    for path in DOCS:
        with open(path) as f:
            text = f.read()
        for tok in _candidate_paths(text):
            checked += 1
            if tok in ALLOWED_MISSING:
                continue
            if tok.startswith(PREFIX_OK):
                continue
            if tok.startswith(("http://", "https://")):
                continue
            # strip optional trailing sentence punctuation
            target = os.path.join(REPO, tok)
            if not os.path.exists(target):
                # allow directory-qualified globs the reader expands
                # (e.g. probes_wd_<NAME>.json is a pattern, not a file)
                if "<" in tok or "*" in tok:
                    continue
                missing.append((os.path.relpath(path, REPO), tok))
    assert not missing, \
        f"documented paths missing from tree ({checked} checked): {missing}"
    print(f"PASS: {checked} documented paths resolve")


if __name__ == "__main__":
    test_documented_paths_resolve()
    print("PATH_AUDIT_OK")