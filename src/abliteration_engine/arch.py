"""Per-architecture module-layout adapters.

The engine's abstract surface — decoder layers, two residual-stream output
matrices per layer ("self_attn.o_proj" label = attention output projection,
"mlp.down_proj" label = MLP output projection), and the final norm — maps
onto different concrete module paths per Hugging Face architecture family.
This registry holds the maps (paths verified against the transformers
modeling sources each config.model_type dispatches to) so core/edits/ara
stay architecture-agnostic.

Layout rows:
  layers      ModuleList of decoder blocks (path from the INNER stack)
  root        attribute of the CausalLM wrapper holding the inner stack
              (None = the wrapper IS the stack, e.g. GPT-2)
  final_norm  final-norm attribute paths on the INNER stack (first hit)
  attn_out    (block-submodule, proj-attr) for the attention output matrix
  mlp_out     same for the MLP output matrix; None = not editable (MoE
              experts are parameter tensors — edited matrices = attn only)
  transpose   weight orientation: False = nn.Linear [out, in]; True =
              Conv1D-style [in, out], normalized to [out, in] views by _Lin

Verified per family (transformers 6.x modeling sources):
  llama/mistral/qwen2/qwen3/gemma2/gemma3/granite/olmo2/stablelm/smolLM3/
  exaone4/glm/cohere2 (and their moe variants): self_attn.o_proj +
  mlp.down_proj, inner norm path model.norm — the historical hardcoded
  paths, unchanged for these families.
  phi3: gate+up fused (gate_up_proj) but down_proj + o_proj intact — same
  default row.
  starcoder2: MLP down-projection is mlp.c_proj (Conv1D-STYLE name on a
  real nn.Linear — orientation not transposed).
  gpt_oss: attention.o_proj exists; the MLP is a routed MoE whose fused
  per-expert gate_up/down are nn.Parameters on one Experts module — the
  row-space edit applies to o_proj only (mmlu + publish verify honestly).
  gpt2: blocks under model.h (wrapper IS the stack — root None); attention
  output proj attn.c_proj; mlp.c_proj; both Conv1D = weight [in, out],
  so edits normalize via transpose=True; final norm ln_f.
  falcon: block.self_attention.dense + mlp.dense_4h_to_h (FalconMLP is
  nn.Linear-based); inner stack transformer.h; final norm transformer.ln_f.
  bloom: block.self_attention.dense + mlp.dense_4h_to_h; transformer.h;
  final norm transformer.ln_f.
  gpt_neox: block.attention.dense + mlp.dense_4h_to_h; gpt_neox.layers;
  final norm gpt_neox.final_layer_norm.
"""
# NOTE: torch imports are deferred into _Lin (CI runs CPU-only; the layout
# registry + compat checks are torch-free).


class _Lin:
    """Normalized view over nn.Linear / Conv1D weight orientation.

    The engine's edit math assumes x @ W.T with W of shape [out, in]
    (nn.Linear). Conv1D stores weight [in, out] and computes x @ W — so
    edits operate on weight.t() and write the transpose back. Everything
    else (row norms, verification, ARA fit) is identical on the view.
    """

    def __init__(self, module):
        self.module = module
        self.transpose = not hasattr(module, "out_features")

    @property
    def weight(self):
        # always the [out, in] view
        return self.module.weight.t() if self.transpose \
            else self.module.weight

    def set_weight(self, W_out_in):
        import torch

        t = W_out_in.t() if self.transpose else W_out_in
        self.module.weight = torch.nn.Parameter(
            t.to(self.module.weight.dtype).to(self.module.weight.device),
            requires_grad=False)


def _get(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


_LAYOUTS = {
    "default": {
        "layers": ("model", "layers"),
        "final_norm": (("model", "norm"),),
        "attn_out": ("self_attn", "o_proj"),
        "mlp_out": ("mlp", "down_proj"),
        "transpose": False,
    },
    "starcoder2": {
        "layers": ("model", "layers"),
        "final_norm": (("model", "norm"),),
        "attn_out": ("self_attn", "o_proj"),
        "mlp_out": ("mlp", "c_proj"),
        "transpose": False,
    },
    "gpt_oss": {
        "layers": ("model", "layers"),
        "final_norm": (("model", "norm"),),
        "attn_out": ("self_attn", "o_proj"),
        "mlp_out": None,
        "transpose": False,
    },
    "gpt2": {
        "layers": (None, "h"),        # wrapper IS the inner stack
        "final_norm": ((None, "ln_f"),),
        "attn_out": ("attn", "c_proj"),
        "mlp_out": ("mlp", "c_proj"),
        "transpose": True,
    },
    "falcon": {
        "layers": ("transformer", "h"),
        "final_norm": (("transformer", "ln_f"),),
        "attn_out": ("self_attention", "dense"),
        "mlp_out": ("mlp", "dense_4h_to_h"),
        "transpose": False,
    },
    "bloom": {
        "layers": ("transformer", "h"),
        "final_norm": (("transformer", "ln_f"),),
        "attn_out": ("self_attention", "dense"),
        "mlp_out": ("mlp", "dense_4h_to_h"),
        "transpose": False,
    },
    "gpt_neox": {
        "layers": ("gpt_neox", "layers"),
        "final_norm": (("gpt_neox", "final_layer_norm"),),
        "attn_out": ("attention", "dense"),
        "mlp_out": ("mlp", "dense_4h_to_h"),
        "transpose": False,
    },
}

_ATTN_LABEL = "self_attn.o_proj"
_MLP_LABEL = "mlp.down_proj"


def _layout_key(model):
    mt = getattr(model.config, "model_type", None)
    return MODEL_TYPE_LAYOUT.get(mt, "default")


def layout_for(model):
    """The layout row for a loaded model (from config.model_type).
    Unknown model_type falls back to 'default' — check_patient_compat
    still verifies the module paths concretely against that row."""
    return _LAYOUTS.get(_layout_key(model), _LAYOUTS["default"])


def _inner_stack(model, lay):
    """The inner stack (decoder container's parent). The layers path is
    (parent_attr, layers_attr); parent None on wrapper-is-stack families
    (gpt2)."""
    parent_attr, _layers_attr = lay["layers"]
    return model if parent_attr is None else getattr(model, parent_attr)


def blocks(model, lay=None):
    """The decoder-block ModuleList (layout-aware)."""
    lay = lay or layout_for(model)
    parent_attr, layers_attr = lay["layers"]
    parent = model if parent_attr is None else getattr(model, parent_attr)
    return getattr(parent, layers_attr)


def n_layers(model):
    """Decoder layer count (layout-aware; raises AttributeError when the
    stack is absent)."""
    return len(blocks(model))


def edit_labels(model):
    """Active edit-matrix labels for this model: ('self_attn.o_proj',
    'mlp.down_proj'); gpt_oss-style MoE experts shrink to attn-only."""
    lay = layout_for(model)
    return (_ATTN_LABEL,) if lay["mlp_out"] is None else (_ATTN_LABEL,
                                                          _MLP_LABEL)


def module_for(model, layer_idx, label):
    """The weight-holding projection module for `label` at one decoder
    layer. Raises AttributeError naming the missing path (check Patient-
   _compat surfaces this before GPU spend)."""
    lay = layout_for(model)
    if label == _ATTN_LABEL:
        sub, attr = lay["attn_out"]
    elif label == _MLP_LABEL:
        if lay["mlp_out"] is None:
            raise AttributeError(
                f"model_type {getattr(model.config, 'model_type', '?')}: "
                "no MLP output projection to edit (fused/routed MoE experts "
                "are parameter tensors — this engine edits the attention "
                "output projection only on that family)")
        sub, attr = lay["mlp_out"]
    else:
        raise ValueError(f"unknown component label {label!r} (have "
                         f"{_ATTN_LABEL!r} / {_MLP_LABEL!r})")
    block = blocks(model)[layer_idx]
    mod = getattr(block, sub, None)
    if mod is None:
        raise AttributeError(
            f"decoder layer {layer_idx} has no {sub!r} (model_type "
            f"{getattr(model.config, 'model_type', '?')})")
    lin = getattr(mod, attr, None)
    if lin is None:
        raise AttributeError(
            f"decoder layer {layer_idx} has no {sub}.{attr}")
    return lin


def edit_linear(model, layer_idx, label):
    """The normalized _Lin view for editing (orientation-normalized)."""
    return _Lin(module_for(model, layer_idx, label))


def final_norm_module(model):
    """The final-norm module (layout-aware, with the historical fallbacks
    kept as a safety net)."""
    lay = layout_for(model)
    inner = _inner_stack(model, lay)
    for parent_attr, norm_attr in lay["final_norm"]:
        parent = inner if parent_attr is None else getattr(inner, parent_attr,
                                                           None)
        n = getattr(parent, norm_attr, None)
        if n is not None and hasattr(n, "weight"):
            return n
    # historical fallbacks (exotic-but-similar layouts still work)
    m = getattr(model, "model", None)
    n = getattr(m, "norm", None) if m is not None else None
    if n is None:
        n = getattr(model, "norm", None) or getattr(model,
                                                   "final_layer_norm", None)
    if n is None:
        raise RuntimeError("final norm module not found")
    return n


def compat_problems(model, tok):
    """Structural problems for THIS architecture (list of strings; empty =
    engine-compatible). The layout registry decides the expected paths;
    every expectation is checked CONCRETELY against the loaded model."""
    problems = []
    # chat template: every prompt is chat-wrapped
    if not getattr(tok, "chat_template", None):
        problems.append("tokenizer has no chat template (use the model's "
                        "-Instruct/-Chat variant)")
    try:
        bs = blocks(model)
    except AttributeError as e:
        return problems + [f"no decoder layer stack the engine can walk "
                           f"({e})"]
    if not bs:
        return problems + ["decoder layer stack is empty"]
    b0 = bs[0]
    for label in (_ATTN_LABEL,):
        try:
            if not hasattr(module_for(model, 0, label), "weight"):
                problems.append("attention output projection has no weight")
        except AttributeError as e:
            problems.append(str(e))
    lay = layout_for(model)
    if lay["mlp_out"] is not None:
        sub, attr = lay["mlp_out"]
        mod = getattr(b0, sub, None)
        lin = getattr(mod, attr, None) if mod is not None else None
        if getattr(lin, "weight", None) is None:
            problems.append(f"decoder layers have no {sub}.{attr}")
    try:
        final_norm_module(model)
    except RuntimeError:
        problems.append("no final norm module")
    if model.get_output_embeddings() is None:
        problems.append("no output embeddings (lm_head)")
    return problems


# config.model_type -> layout key (modeling-source verified, transformers 6.x)
_MODEL_FAMILIES_DEFAULT = (
    "llama", "mistral", "gemma", "gemma2", "gemma3", "gemma3_text",
    "granite", "granitemoe", "olmo2", "stablelm", "phi", "phi3",
    "glm", "smollm3", "exaone4", "cohere2", "minicpm", "internlm2",
    "baichuan", "zamba2",
)
MODEL_TYPE_LAYOUT = {mt: "default" for mt in _MODEL_FAMILIES_DEFAULT}
MODEL_TYPE_LAYOUT.update({
    "qwen2": "default",
    "qwen3": "default",
    "qwen3_moe": "default",
    "gpt_oss": "gpt_oss",
    "starcoder2": "starcoder2",
    "gpt2": "gpt2",
    "falcon": "falcon",
    "bloom": "bloom",
    "gpt_neox": "gpt_neox",
})

# init's honest status classes, driven from this registry
TESTED_MODEL_TYPES = {"qwen2", "llama"}  # GPU-proven by this program
# (llama promoted 2026-10-06: run-009 Llama-3.2-1B-Instruct end-to-end —
# full ladder artifact-backed, gate passed, MMLU clean; windows 2/4 banks)
SAME_LAYOUT_UNTESTED = {"mistral", "qwen3", "gemma2", "gemma3",
                        "phi3", "granite", "olmo2", "stablelm", "glm",
                        "smollm3", "cohere2", "starcoder2", "gpt2",
                        "gpt_neox"}

# model_type values whose routed-MoE experts make MLP edits inapplicable
MOE_EXPERT_MLP = {"gpt_oss"}