"""Arch-registry fixes (issue sweep 2026-10-08):

- routed-MoE families (qwen3_moe, qwen2_moe, mixtral, granitemoe, olmoe)
  have no single mlp.down_proj — qwen3_moe was mapped to the dense row, so
  stage 1 refused it; they now use the attention-only edit layout.
- Gemma-lineage RMSNorm scales by (1 + weight): the wd_BN final-norm edit
  must act on the effective scale or it silently does ~nothing.
Mapping checks are CPU-only; real-class checks need torch + transformers.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from abliteration_engine import arch  # noqa: E402

MOE = {"qwen3_moe": dict(num_experts=4, num_experts_per_tok=2,
                         moe_intermediate_size=16),
       "qwen2_moe": dict(num_experts=4, num_experts_per_tok=2,
                         moe_intermediate_size=16,
                         shared_expert_intermediate_size=16),
       "mixtral": dict(num_local_experts=4, num_experts_per_tok=2),
       "granitemoe": dict(num_local_experts=4, num_experts_per_tok=2)}
BASE = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8)


@pytest.mark.parametrize("mt", sorted(MOE) + ["olmoe"])
def test_moe_families_map_to_attention_only(mt):
    assert arch._LAYOUTS[arch.MODEL_TYPE_LAYOUT[mt]]["mlp_out"] is None
    assert mt in arch.MOE_EXPERT_MLP


def test_norm_offset_registry():
    class M:
        class config:
            model_type = "gemma2"
    assert arch.norm_weight_offset(M) == 1.0
    M.config.model_type = "qwen2"
    assert arch.norm_weight_offset(M) == 0.0


def _real(mt, **kw):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    cfg = tf.AutoConfig.for_model(mt, **BASE, **kw)
    return torch, tf.AutoModelForCausalLM.from_config(cfg)


@pytest.mark.parametrize("mt", sorted(MOE))
def test_real_moe_models_pass_compat_and_edit(mt):
    from abliteration_engine import edits
    torch, m = _real(mt, **MOE[mt])

    class Tok:
        chat_template = "x"
    assert arch.compat_problems(m, Tok()) == []
    resid = edits.orthogonalize_layer_output(m, 1, torch.randn(32))
    assert set(resid) == {"self_attn.o_proj"}


def test_real_gemma_final_norm_edit_removes_direction():
    from abliteration_engine import edits
    torch, m = _real("gemma2")
    torch.manual_seed(0)
    with torch.no_grad():
        m.model.norm.weight.copy_(torch.randn(32) * 0.3)
    d = torch.randn(32)
    edits.orthogonalize_final_norm(m, d)
    w_eff = 1.0 + m.model.norm.weight.data
    assert abs(float(w_eff @ (d / d.norm()))) < 1e-5
    assert edits.verify_final_norm_disk(m, d) < 1e-5
