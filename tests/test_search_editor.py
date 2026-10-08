"""search.Editor: in-memory trial edits must restore the base EXACTLY
(bit-identical weights, tied head re-tied) — otherwise every trial after
the first is scored on residue from the previous one. Needs torch."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from abliteration_engine import arch, search  # noqa: E402

BASE = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)


def _model(family, tied):
    torch.manual_seed(0)
    cls = {"qwen2": (tf.Qwen2Config, tf.Qwen2ForCausalLM),
           "llama": (tf.LlamaConfig, tf.LlamaForCausalLM)}[family]
    return cls[1](cls[0](tie_word_embeddings=tied, **BASE)).eval()


@pytest.mark.parametrize("family,tied", [("qwen2", True), ("llama", False)])
def test_apply_then_restore_is_bit_exact(family, tied):
    m = _model(family, tied)
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    rng = torch.Generator().manual_seed(1)
    dirs = {li: torch.randn(32, generator=rng) for li in (1, 2, 3)}
    dirs = {k: (v / v.norm()).numpy() for k, v in dirs.items()}
    point = {"start": 1, "end": 4, "alpha": 0.8, "components": "both",
             "direction_mode": "own", "readout": True}
    ed = search.Editor(m)
    ed.apply(point, dirs, torch.randn(32).numpy(), arch.edit_labels(m))
    changed = [k for k, v in m.state_dict().items()
               if not torch.equal(v, before[k])]
    assert any("o_proj" in k for k in changed)
    assert any("lm_head" in k for k in changed)
    if tied:
        # the input embeddings must NOT move when the tied head is edited
        assert torch.equal(m.get_input_embeddings().weight,
                           before["model.embed_tokens.weight"])
    ed.restore()
    after = m.state_dict()
    assert all(torch.equal(after[k], before[k]) for k in before)
    lm, emb = m.get_output_embeddings(), m.get_input_embeddings()
    assert (lm.weight.data_ptr() == emb.weight.data_ptr()) is tied
    assert bool(m.config.tie_word_embeddings) is tied


def test_partial_projection_algebra():
    """r·W_new == (1 − α) r·W_base — the identity verify_on_disk checks."""
    m = _model("llama", False)
    r = torch.randn(32)
    r = r / r.norm()
    W0 = arch.edit_linear(m, 2, "self_attn.o_proj").weight.detach().clone()
    ed = search.Editor(m)
    ed.project_rows(2, "self_attn.o_proj", r.numpy(), 0.6)
    W1 = arch.edit_linear(m, 2, "self_attn.o_proj").weight.detach()
    assert torch.allclose(r @ W1, 0.4 * (r @ W0), atol=1e-6)
