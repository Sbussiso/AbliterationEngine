"""Run 010 regression: file:-based probe pools must strip comment lines.

The file: branch of resolve_probe_set used to return every non-blank line
verbatim, so a pool file carrying `#` documentation headers shipped its own
header as a probe prompt (found while wiring the code_harm64/code_benign64
pools; 66/67 resolved entries instead of 64). resolve_markers already
stripped comments on file: refs — the branches now agree.
"""
import os
import textwrap

from abliteration_engine.data import resolve_probe_set


def test_file_probe_set_strips_comments(tmp_path):
    p = tmp_path / "pool.txt"
    p.write_text(textwrap.dedent("""\
        # header documentation line
        prompt one
        prompt two
        # selection receipts below
        prompt three
        """))
    got = resolve_probe_set("file:" + str(p))
    assert got == ["prompt one", "prompt two", "prompt three"]


def test_file_probe_set_no_comments_unchanged(tmp_path):
    p = tmp_path / "plain.txt"
    p.write_text("alpha\nbeta\n")
    assert resolve_probe_set("file:" + str(p)) == ["alpha", "beta"]


def test_builtin_probe_sets_have_no_comment_lines():
    # comment-stripping on the file: branch must never change what the
    # shipped builtin pools resolve to (they carry no # lines today)
    root = os.path.join(os.path.dirname(
        __import__("abliteration_engine").__file__), "sets")
    for name in ("primary64_harmful", "primary64_harmless",
                 "ara_good", "ara_bad"):
        path = os.path.join(root, f"{name}.txt")
        if not os.path.exists(path):
            continue
        raw = [ln.strip() for ln in open(path) if ln.strip()]
        assert not any(ln.startswith("#") for ln in raw), name
        assert resolve_probe_set(f"builtin:{name}") == raw


def test_run010_code_pools_resolve_to_64():
    # workspace pools of this run, when present in the environment
    ws = os.environ.get("WS")
    if not ws:
        return
    for name in ("code_harm64.txt", "code_benign64.txt"):
        path = os.path.join(ws, name)
        if not os.path.exists(path):
            continue
        got = resolve_probe_set("file:" + path)
        assert len(got) == 64, (name, len(got))
        assert not any(x.startswith("#") for x in got), name