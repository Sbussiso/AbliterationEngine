"""BUG-2 regression pins (tie-state assumptions on untied patients).

The developer-reported bug (run-010, Qwen2.5-Coder-7B ships untied):
save_variant asserted cfg.tie_word_embeddings == expect_tied with
expect_tied hardcoded from the tied-Qwen chat-patient era — the assert
killed the leg before any write. The fix shape:
  - save_variant asserts POST-APPLICATION consistency (config flag ==
    pointer-level tie state of the model being saved; saved file ==
    memory); the ladder expectation, when given, is a SECOND check.
  - expect_tied_for(name, model) DERIVES per-variant expectations from
    the model's base state (ara_*/wd_ML: base state untouched; wd_B/
    wd_BN/wd_ML_BN: untied-by-design, true on every base).
  - ara.py derives the expectation from the loaded base instead of
    hardcoded True.
"""
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "nn") or not hasattr(torch.nn, "Module"):
    # test_banked_resume installs a CPU-side stub torch into sys.modules
    # (its summary/filename tests never touch tensors). A module-gated file
    # like this one must treat the stub as absence (regression-file
    # convention, same guard as test_review_regressions).
    pytest.skip("torch stub in sys.modules (banked-resume fixture)",
                allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from abliteration_engine import edits  # noqa: E402


class _Tok:
    """Tokenizer stand-in with a no-op save (weights saving is the save_
    variant contract under test, not serialization)."""

    def save_pretrained(self, path):
        return None


def _tiny(d=8, vocab=16, tied=False):
    """Minimal CausalLM-ish model with controllable tie state.

    tied=True: lm_head shares the embedding weight (pointer-identical).
    tied=False: independent lm_head (Linear weight).
    Also carries a no-op save_pretrained writing a truthy config.json
    (mimicking HF serialization of model.config)."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            emb = torch.nn.Embedding(vocab, d)
            self.embed_tokens = emb
            if tied:
                head_w = emb.weight
            else:
                head_w = torch.nn.Parameter(
                    torch.randn(vocab, d) * 0.02)
            self.lm_head = torch.nn.Linear(d, vocab, bias=False)
            self.lm_head.weight = head_w
            self.config = type("C", (), {
                "tie_word_embeddings": tied,
                "save_pretrained": lambda cfg, path, **kw: json.dump(
                    {"tie_word_embeddings": tied},
                    open(os.path.join(path, "config.json"), "w")),
            })()

        def get_output_embeddings(self):
            return self.lm_head

        def get_input_embeddings(self):
            return self.embed_tokens

        def save_pretrained(self, path, safe_serialization=True):
            self.config.save_pretrained(path)
    return M()


def test_save_variant_untied_patient_no_ladder_assumption(tmp_path):
    """An untied base + untie-by-design variant: expect_tied=False is the
    derived truth; save succeeds (the BUG-2 crash was False != True)."""
    m = _tiny(tied=False)
    out = edits.save_variant(m, _Tok(), str(tmp_path / "v"),
                             expect_tied=False)
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["tie_word_embeddings"] is False


def test_save_variant_tied_patient_still_pinned(tmp_path):
    """Tied base + lm_head variant: the historical contract still verifies
    (post-state untied, expectation untied)."""
    m = _tiny(tied=True)
    # the untie-by-design edit: give it an untied head first (what
    # orthogonalize_lm_head does) + flip the config (exactly as that fn)
    m.lm_head.weight = torch.nn.Parameter(torch.randn(16, 8) * 0.02)
    m.config = type("C", (), {
        "tie_word_embeddings": False,
        "save_pretrained": lambda cfg, path, **kw: json.dump(
            {"tie_word_embeddings": False},
            open(os.path.join(path, "config.json"), "w")),
    })()
    m.save_pretrained = m.config.save_pretrained  # type: ignore[method-assign]
    out = edits.save_variant(m, _Tok(), str(tmp_path / "v"),
                             expect_tied=False)
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["tie_word_embeddings"] is False


def test_save_variant_config_must_not_lie(tmp_path):
    """A model whose config claims tied but whose weights are NOT shared
    fails loudly BEFORE the save (config that lies about its own head)."""
    m = _tiny(tied=False)
    m.config.tie_word_embeddings = True   # the lie
    with pytest.raises(AssertionError, match="tie-state inconsistency"):
        edits.save_variant(m, _Tok(), str(tmp_path / "v"))


def test_save_variant_saved_file_tamper_detected(tmp_path):
    """A save that writes a config contradicting the in-memory state is
    caught after the write (the pre-fix code trusted the disk file only)."""
    m = _tiny(tied=False)

    def tampered_save(path, safe_serialization=True):
        os.makedirs(path, exist_ok=True)
        json.dump({"tie_word_embeddings": True},          # the lie
                  open(os.path.join(path, "config.json"), "w"))
    m.save_pretrained = tampered_save                      # type: ignore[method-assign]
    with pytest.raises(AssertionError, match="!= in-memory"):
        edits.save_variant(m, _Tok(), str(tmp_path / "v"))


def test_expect_tied_for_derives_from_model():
    tied = _tiny(tied=True)
    untied = _tiny(tied=False)
    # ara_* / wd_ML: base state passes through (bug class: hardcoded True)
    for name in ("ara_50", "wd_ML"):
        assert edits.expect_tied_for(name, tied) is True
        assert edits.expect_tied_for(name, untied) is False
    # lm_head flavors: untied-by-design on EVERY base
    for name in ("wd_B", "wd_BN", "wd_ML_BN"):
        assert edits.expect_tied_for(name, tied) is False
        assert edits.expect_tied_for(name, untied) is False
    # name-only calls stay back-compat for lm_head flavors
    assert edits.expect_tied_for("wd_B") is False
    # ...but layer-flavored names REQUIRE the model (the BUG-2 class)
    with pytest.raises(ValueError, match="BUG-2"):
        edits.expect_tied_for("wd_ML")
    with pytest.raises(ValueError, match="BUG-2"):
        edits.expect_tied_for("ara_50")


def test_coder7b_shape_end_to_end(tmp_path):
    """The reported scenario: an untied base + the full ladder list must
    produce post-state-consistent expectations (no name-carried True
    anywhere) and every save must succeed."""
    m = _tiny(tied=False)
    for name in ("wd_B", "wd_BN", "wd_ML", "wd_ML_BN", "ara_50"):
        exp = edits.expect_tied_for(name, m)
        assert exp is False, (name, exp)   # all five: post state untied
        edits.save_variant(m, _Tok(), str(tmp_path / name), expect_tied=exp)


if __name__ == "__main__":
    raise SystemExit("run via pytest")