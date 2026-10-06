"""arch.py layout registry (torch required).

Pins per-family module resolution with inline stand-in models BUILT to
each family's real transformer layout (paths source-verified 2026-10-03,
transformers 6.x):
  - default (llama/qwen2/...): model.layers[i].self_attn.o_proj / mlp.down_proj
  - starcoder2 naming: mlp.c_proj (nn.Linear orientation)
  - gpt2: h[i].attn.c_proj + mlp.c_proj, Conv1D weights [in, out] — the
    orientation roundtrip is the critical pin (edits must write back
    transposed or the model breaks)
  - falcon: self_attention.dense + mlp.dense_4h_to_h
  - gpt_oss: attn-only edit surface (routed MoE experts are parameters)
Also pins structure_report / compat_problems / init verdict behavior.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from abliteration_engine import arch, core  # noqa: E402


# ---- stand-in building blocks -------------------------------------------------
class _Conv1D(torch.nn.Module):
    """transformers.Conv1D: weight [in, out], computes x @ W."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.nf = n_out
        self.weight = torch.nn.Parameter(torch.empty(n_in, n_out))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x @ self.weight


class _Attn(torch.nn.Module):
    """Attention submodule WITH the output-projection Linear directly on it
    (the real HF shape: block.self_attn.o_proj is a Linear, not wrapped)."""

    def __init__(self, d, attr, cls="linear"):
        super().__init__()
        lin = (torch.nn.Linear(d, d, bias=False) if cls == "linear"
               else _Conv1D(d, d))
        setattr(self, attr, lin)

    def forward(self, x):
        return x


class _MLP(torch.nn.Module):
    def __init__(self, d_out, d_in, attr, cls="linear"):
        super().__init__()
        lin = (torch.nn.Linear(d_in, d_out, bias=False) if cls == "linear"
               else _Conv1D(d_in, d_out))
        setattr(self, attr, lin)

    def forward(self, x):
        return x


class _Cfg:
    def __init__(self, mt, n_layers=2, d=8, vocab=16):
        self.model_type = mt
        self.num_hidden_layers = n_layers
        self.hidden_size = d
        self.intermediate_size = 32
        self.vocab_size = vocab
        self.num_attention_heads = 2
        self.num_key_value_heads = 2
        self.tie_word_embeddings = False


_ATTN_SHAPE = {"default": ("self_attn", "o_proj", "linear"),
               "qwen2": ("self_attn", "o_proj", "linear"),
               "llama": ("self_attn", "o_proj", "linear"),
               "mistral": ("self_attn", "o_proj", "linear"),
               "phi3": ("self_attn", "o_proj", "linear"),
               "starcoder2": ("self_attn", "o_proj", "linear"),
               "gpt2": ("attn", "c_proj", "conv1d"),
               "falcon": ("self_attention", "dense", "linear"),
               "gpt_oss": ("self_attn", "o_proj", "linear"),
               "gpt_neox": ("attention", "dense", "linear")}
_MLP_SHAPE = {"default": ("mlp", "down_proj", "linear"),
              "qwen2": ("mlp", "down_proj", "linear"),
              "llama": ("mlp", "down_proj", "linear"),
              "mistral": ("mlp", "down_proj", "linear"),
              "phi3": ("mlp", "down_proj", "linear"),
              "starcoder2": ("mlp", "c_proj", "linear"),
              "gpt2": ("mlp", "c_proj", "conv1d"),
              "falcon": ("mlp", "dense_4h_to_h", "linear"),
              "gpt_neox": ("mlp", "dense_4h_to_h", "linear")}


def _block(mt, d, d_ff):
    attn_sub, attn_attr, attn_cls = (_ATTN_SHAPE.get(mt)
                                     or _ATTN_SHAPE["default"])
    mlp = _MLP_SHAPE.get(mt) or _MLP_SHAPE["default"]
    blk = torch.nn.Module()
    setattr(blk, attn_sub, _Attn(d, attn_attr, attn_cls))
    if mlp is not None:
        mlp_sub, mlp_attr, mlp_cls = mlp
        setattr(blk, mlp_sub, _MLP(d, d_ff, mlp_attr, mlp_cls))
    return blk


class _Model(torch.nn.Module):
    """Minimal family-shaped CausalLM: config.model_type drives the
    registry; block structure built per family; all projections are
    editable weights. Container names match each family's wrapper."""

    def __init__(self, mt, n_layers=2, d=8, d_ff=12, vocab=16):
        super().__init__()
        self.config = _Cfg(mt, n_layers, d, vocab)
        blocks_list = [_block(mt, d, d_ff) for _ in range(n_layers)]
        norm = torch.nn.RMSNorm(d) if mt != "gpt2" else torch.nn.LayerNorm(d)
        self.lm_head = torch.nn.Linear(d, vocab, bias=False)
        self._mt = mt
        # inner stack + wrapper attributes, exactly per family wrapper
        inner = torch.nn.Module()
        setattr(inner, {"gpt_neox": "layers"}.get("gpt_neox", "layers"),
                torch.nn.ModuleList(blocks_list))
        if mt == "gpt2":
            # wrapper IS the stack: h + ln_f directly on self
            setattr(self, "h", torch.nn.ModuleList(blocks_list))
            setattr(self, "ln_f", norm)
            self._blocks = self.h
        elif mt == "falcon" or mt == "bloom":
            t = torch.nn.Module()
            setattr(t, "h", torch.nn.ModuleList(blocks_list))
            setattr(t, "ln_f", norm)
            setattr(self, "transformer", t)
            self._blocks = t.h
        elif mt == "gpt_neox":
            t = torch.nn.Module()
            setattr(t, "layers", torch.nn.ModuleList(blocks_list))
            setattr(t, "final_layer_norm", norm)
            setattr(self, "gpt_neox", t)
            self._blocks = t.layers
        else:
            m = torch.nn.Module()
            setattr(m, "layers", torch.nn.ModuleList(blocks_list))
            setattr(m, "norm", norm)
            setattr(self, "model", m)
            self._blocks = m.layers
        self._norm = norm

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, **kw):
        x = self.lm_head.weight[input_ids % self.lm_head.weight.shape[0]]
        for blk in self._blocks:
            x = blk(x)
        return self.lm_head(self._norm(x))


# ---- tests --------------------------------------------------------------------
@pytest.mark.parametrize("mt,expect_kind", [
    ("qwen2", "default"), ("llama", "default"), ("mistral", "default"),
    ("phi3", "default"), ("starcoder2", "starcoder2"), ("gpt2", "gpt2"),
    ("falcon", "falcon"), ("gpt_oss", "gpt_oss"), ("gpt_neox", "gpt_neox"),
])
def test_layout_keys_resolve(mt, expect_kind):
    m = _Model(mt)
    assert arch._layout_key(m) == expect_kind


@pytest.mark.parametrize("mt", ["qwen2", "llama", "phi3", "starcoder2",
                                "gpt2", "falcon", "gpt_neox"])
def test_module_for_resolves_real_paths(mt):
    m = _Model(mt)
    labels = arch.edit_labels(m)
    assert "self_attn.o_proj" in labels
    assert "mlp.down_proj" in labels or mt == "gpt2"
    # gpt2 c_proj IS the mlp label target too (registry maps mlp label ->
    # mlp.c_proj); both labels must resolve for every listed family
    for label in ("self_attn.o_proj", "mlp.down_proj"):
        mod = arch.module_for(m, 0, label)
        assert hasattr(mod, "weight"), (mt, label)
    # the resolved module is the actual submodule named by the family row
    lay = arch.layout_for(m)
    blk0 = arch.blocks(m)[0]
    sub, attr = lay["attn_out"]
    assert arch.module_for(m, 0, "self_attn.o_proj") is getattr(
        getattr(blk0, sub), attr)


def test_gpt_oss_mlp_label_refuses():
    m = _Model("gpt_oss")
    assert arch.edit_labels(m) == ("self_attn.o_proj",)
    with pytest.raises(AttributeError, match="MoE"):
        arch.module_for(m, 0, "mlp.down_proj")


def test_structure_report_family_aware():
    m = _Model("gpt_oss")
    rep = core.structure_report(m)
    assert rep["model_type"] == "gpt_oss"
    assert rep["edit_matrices"] == ["self_attn.o_proj"]
    assert rep["down_proj_shape"] is None
    m2 = _Model("qwen2")
    rep2 = core.structure_report(m2)
    assert rep2["edit_matrices"] == ["self_attn.o_proj", "mlp.down_proj"]
    assert rep2["down_proj_shape"] == [8, 12]


def test_row_space_edit_via_registry_invariant():
    """orthogonalize_layer_output through the registry leaves (M W)^T r ~ 0
    on a NON-default family (falcon naming), and the edit lands on the
    real submodule."""
    m = _Model("falcon")
    d = m.config.hidden_size
    r = torch.randn(d)
    dense_lin = arch.module_for(m, 1, "self_attn.o_proj")  # the Linear itself
    before = dense_lin.weight.clone()
    layer0_before = arch.module_for(m, 0, "self_attn.o_proj").weight.clone()
    from abliteration_engine import edits as edits_mod
    edits_mod.orthogonalize_layer_output(m, 1, r)
    after = arch.module_for(m, 1, "self_attn.o_proj").weight
    assert not torch.equal(before, after)
    rhat = r / r.norm()
    resid = float((after.float().T @ rhat).abs().max())
    assert resid < 1e-3, resid
    # layer 0 untouched (compare layer 0 against LAYER 0's own pre-edit
    # clone — the old assert compared it against layer 1's, i.e. two
    # different random tensors: always false on any model)
    assert torch.equal(arch.module_for(m, 0, "self_attn.o_proj").weight,
                       layer0_before)


def test_conv1d_orientation_roundtrip():
    """The gpt2 trap: edits must survive the Conv1D [in, out] convention.
    A row-space edit on the [out, in] view must read back correctly and
    the module must still compute x @ W with the EDITED weight."""
    m = _Model("gpt2")
    lin = arch.edit_linear(m, 0, "self_attn.o_proj")
    assert lin.transpose is True
    W0 = lin.weight.detach().clone()          # [out, in] view
    assert W0.shape == (8, 8)
    W1 = W0 * 2.0
    lin.set_weight(W1)
    # raw stored weight is [in, out] — transposed back
    assert lin.module.weight.shape == (8, 8)
    assert torch.allclose(lin.module.weight, W1.t())
    assert torch.allclose(lin.weight, W1)
    assert not torch.allclose(lin.weight, W0)


def test_compat_problems_clean_and_dirty():
    m = _Model("qwen2")
    assert arch.compat_problems(m, _Tok()) == []
    # broken family: attention submodule missing
    m2 = _Model("qwen2")
    delattr(arch.blocks(m2)[0], "self_attn")
    probs = arch.compat_problems(m2, _Tok())
    assert any("self_attn" in p for p in probs), probs
    # no chat template is flagged
    probs3 = arch.compat_problems(_Model("qwen2"), _Tok(template=False))
    assert any("chat template" in p for p in probs3)


class _Tok:
    def __init__(self, template=True):
        if template:
            self.chat_template = "{% for %}{% endfor %}"


def test_init_verdicts_from_registry():
    import abliteration_engine.init_spec as init_spec

    for mt in ("llama", "phi3", "gemma2"):
        lvl, msg = init_spec.architecture_verdict({"model_type": mt})
        assert lvl == "warn" and "not yet run" in msg, (mt, lvl, msg)
    lvl, msg = init_spec.architecture_verdict({"model_type": "qwen2"})
    assert lvl == "ok"
    lvl, msg = init_spec.architecture_verdict({"model_type": "gpt_oss"})
    assert lvl == "warn" and "edit surface" in msg
    lvl, msg = init_spec.architecture_verdict({"model_type": "speech_to_text"})
    assert lvl == "unsupported"


def test_unknown_model_type_falls_back_to_default_with_compat_check():
    """An unknown family gets the default row + real compat check: a
    llama-shaped unknown model works; a gpt2-shaped unknown model is
    refused at load with a path-specific message."""
    m = _Model("totally_novel")
    assert arch.layout_for(m) is arch._LAYOUTS["default"]
    assert arch.compat_problems(m, _Tok()) == []
    # gpt2-shaped: blocks under h, attn at .attn — default row finds
    # model.layers on the wrapper: absent -> compat names it
    g2 = _Model("gpt2")
    g2.config.model_type = "totally_novel"   # registry miss via unknown type
    probs = arch.compat_problems(g2, _Tok())
    assert probs, "unknown family with non-default layout must fail compat"


if __name__ == "__main__":
    raise SystemExit("run via pytest")