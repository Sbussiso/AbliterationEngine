"""eng core — single parameterized abliteration pipeline.

Stage functions take the normalized spec dict (eng/spec.py.load_spec) and
shared context; each emits the same artifact contracts harness v2 produced,
so Run-001 parity stays diffable. GPU stages import torch lazily so the CLI
plan/validate/dry-run paths run CPU-only.
"""
import hashlib
import json
import os
import time

import numpy as np

from abliteration_engine import ui
from abliteration_engine.data import resolve_markers, resolve_probe_set

REFUSAL_MARKERS = None  # set per-run by from_spec() or ensure_markers()

# session grader cache: (markers, score_fn) keyed by the loaded spec's
# _spec_sha256 — persistent-kernel safe (same spec sha returns the same
# pair; a different sha re-resolves). See session_grader().
_SESSION_GRADER_CACHE = {}


def session_grader(spec):
    """(markers, score_fn) for one spec per probe_sets.marker_mode.

    v1 (default/absent) = refusal_score substring grader — the frozen
    Run-001 parity contract's exactly-once grading at probe time. v2 =
    scoring_v2.classify (substring marker scan + line-anchored help
    evidence, grader-under-study; zero collateral flips validated over
    N=192 logged rows).
    Cached by spec sha so from_spec and ladder_phase in one persistent
    kernel always share the session's one grader pair.
    """
    key = spec.get("_spec_sha256") or id(spec)
    pair = _SESSION_GRADER_CACHE.get(key)
    if pair is None:
        from .data import resolve_markers

        markers = resolve_markers(spec["probe_sets"]["refusal_markers"])
        mode = (spec.get("probe_sets") or {}).get("marker_mode", "v1")
        score_fn = {"v1": refusal_score,
                    "v2": _scoring_v2_scalar}.get(mode)
        if score_fn is None:
            raise ValueError(f"probe_sets.marker_mode {mode!r} unsupported "
                             "(have: v1, v2)")
        pair = (markers, score_fn)
        _SESSION_GRADER_CACHE[key] = pair
    return pair


_REFUSAL_MARKERS_SRC = None  # markers-ref the global was resolved FROM


def ensure_markers(spec):
    """Populate the module-global REFUSAL_MARKERS from a loaded spec.

    from_spec() (full-run path) sets this global as a side effect, but
    standalone phases (PHASE=ladder / PHASE=mmlu) never execute from_spec —
    the rs2-1 incident (2026-09-30): first probe call hit
    `markers = markers or REFUSAL_MARKERS` with MARKERS None and raised
    TypeError: 'NoneType' object is not iterable. Every phase that can
    reach run_probes() without from_spec MUST call this first.

    Cross-spec hygiene (group review of d2998a2): a persistent kernel
    (Colab's is exactly that) can run spec A then spec B within one
    process. Short-circuiting on a non-None global leaked spec A's
    markers into spec B's probes — silently, same lists-same-results
    trap. The global is therefore keyed to the spec it came from: it
    re-resolves whenever the spec's refusal_markers ref differs.
    Same-ref calls stay one-liner cheap (resolve_markers builtin: is
    pure dict lookup, so re-resolution is not a perf concern).

    Leaves the global untouched if the spec has no probe_sets (raises
    KeyError — actionable, matching load_spec's validation style).
    """
    global REFUSAL_MARKERS, _REFUSAL_MARKERS_SRC
    ref = spec["probe_sets"]["refusal_markers"]
    if REFUSAL_MARKERS is not None and _REFUSAL_MARKERS_SRC == ref:
        return REFUSAL_MARKERS
    REFUSAL_MARKERS = resolve_markers(ref)
    _REFUSAL_MARKERS_SRC = ref
    return REFUSAL_MARKERS


def eng_base() -> str:
    """Root for sentinel files and per-run artifact dirs: /content on
    Colab, a tmp dir anywhere else (CI runners are non-root and must not
    mkdir /content). ENG_OUT_ROOT overrides either way."""
    base = os.environ.get("ENG_OUT_ROOT", "/content")
    try:
        os.makedirs(base, exist_ok=True)
        if os.access(base, os.W_OK):
            return base
    except OSError:
        pass
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "eng_root")
    os.makedirs(tmp, exist_ok=True)
    return tmp


def sentinel_exit() -> str:
    """Path of the stage sentinel file (engine-owned contract)."""
    return os.path.join(eng_base(), "exit_code.txt")


def run_dir_name(spec):
    rc = spec["run_card"]
    return f"eng_run_{rc['run_number']:03d}_{rc['patient']}"


def _out_dir(spec, create=False):
    d = os.path.join(eng_base(), run_dir_name(spec))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def variant_dir(spec, name):
    """On-disk dir for one ladder variant's saved weights.

    Scoped per run (<VARBASE>/eng_run_<NNN>_<patient>_variants/<name>) so
    two runs in one session/VARBASE never overwrite — or, via the ladder's
    non-selected cleanup, delete — each other's weights. Kept as a SIBLING
    of the artifact dir (not inside it) so downloading the artifacts folder
    does not drag multi-GB weights along. ENG_VARBASE overrides the root.
    """
    base = os.environ.get("ENG_VARBASE") or eng_base()
    return os.path.join(base, f"{run_dir_name(spec)}_variants", name)


# NOTE (group review d2998a2): the old core._stage_boilerplate was deleted —
# zero call sites, and pipeline._run_phase is the one live sentinel writer
# (same contract: engine-owned exit_code, per-stage error file). One
# implementation so the next fix can't land on a dead duplicate.


# ---- stage 1: load patient ------------------------------------------------
def load_patient(spec):
    """Pinned load + structure report + structure_expect assertions."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pat = spec["patient"]
    tok = AutoTokenizer.from_pretrained(pat["model_id"],
                                        revision=pat["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # extraction-friendly; probes run one-by-one
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # transformers 4.56 renamed torch_dtype -> dtype (the old name warns on
    # every load). Chosen by version, not try/except: older releases do not
    # raise on an unknown `dtype=` — they absorb it into the config and load
    # fp32 silently.
    model = AutoModelForCausalLM.from_pretrained(
        pat["model_id"], revision=pat["revision"], device_map="auto",
        **{_dtype_kwarg(): dtype})
    model.eval()
    check_patient_compat(tok, model)
    return tok, model


def check_patient_compat(tok, model):
    """Fail at load time — before captures, probes or edits spend GPU time —
    when the model lacks the module layout the engine reads and edits, or
    the tokenizer has no chat template (every prompt is chat-wrapped).
    The expected paths come from arch.py's per-family layout registry and
    are checked concretely against the loaded model."""
    from . import arch

    problems = arch.compat_problems(model, tok)
    if problems:
        arch_mt = getattr(model.config, "model_type", type(model).__name__)
        raise RuntimeError(
            f"patient '{arch_mt}' is not supported by this engine: "
            + "; ".join(problems)
            + ". Supported layouts: Llama/Qwen/Mistral-style decoders"
              " (+ gpt2/gpt-neox/bloom/falcon/gpt-oss via the arch"
              " registry).")


def _dtype_kwarg():
    import transformers

    try:
        major, minor = (int(x) for x in
                        transformers.__version__.split(".")[:2])
    except ValueError:
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def structure_report(model):
    """Architecture facts + the edit-matrix names active on this family
    (arch registry: e.g. gpt_oss's routed MoE experts shrink edits to the
    attention output projection only)."""
    from . import arch

    c = model.config
    attn_out = arch.module_for(model, 0, "self_attn.o_proj")
    labels = arch.edit_labels(model)
    return {
        "model_type": getattr(c, "model_type", None),
        "num_hidden_layers": arch.n_layers(model),
        "hidden_size": c.hidden_size,
        "intermediate_size": getattr(c, "intermediate_size", None),
        "vocab_size": c.vocab_size,
        "num_attention_heads": getattr(c, "num_attention_heads", None),
        "num_key_value_heads": getattr(c, "num_key_value_heads", None),
        "tie_word_embeddings": bool(c.tie_word_embeddings),
        "edit_matrices": list(labels),
        "weight_orientation": ("transposed (Conv1D)" if arch.layout_for(
            model)["transpose"] else "[out, in]"),
        "o_proj_shape": list(arch._Lin(attn_out).weight.shape),
        "down_proj_shape": (list(arch._Lin(arch.module_for(
            model, 0, "mlp.down_proj")).weight.shape)
            if "mlp.down_proj" in labels else None),
        "lm_head_shape": list(model.get_output_embeddings().weight.shape),
        "lm_head_is_input_emb": model.get_output_embeddings().weight.data_ptr()
        == model.get_input_embeddings().weight.data_ptr(),
    }


def assert_patient_structure(spec, struct):
    se = spec["patient"].get("structure_expect") or {}
    for key, expected in se.items():
        got = struct.get(key)
        assert got == expected, (key, got, expected)


# ---- stage 2: capture ------------------------------------------------------
def capture_final_residuals(tok, model, texts, batch=16):
    """Final-position residual stream at EVERY decoder layer, one forward
    pass per batch. Index 0 = embedding output, i = decoder layer i-1
    output. Returns per-layer list of [N, H] float32 CPU tensors
    (v2-identical for Llama-lineage families; hidden_states count derives
    from the arch-registry layer walk)."""
    import torch

    from . import arch
    n_layers = arch.n_layers(model)
    per_layer = [[] for _ in range(n_layers + 1)]
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, return_tensors="pt",
                      padding=True).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            lens = enc["attention_mask"].sum(dim=1)
            rows = torch.arange(hs[0].shape[0], device=model.device)
            idx = lens - 1  # last non-pad position = generation-start position
            for L in range(n_layers + 1):
                per_layer[L].append(hs[L][rows, idx].float().cpu())
            del out, hs, enc
    return [torch.cat(t, dim=0) for t in per_layer]


# ---- stage 3: directions ---------------------------------------------------
def coherence_stats(A, B):
    """d = mean(A) - mean(B); coherence = ||d|| / mean pairwise ||diff||."""
    d = A.mean(0) - B.mean(0)
    nd = float(d.norm().item())
    if nd < 1e-6:
        return d, nd, 0.0
    pd = (A.unsqueeze(1) - B.unsqueeze(0)).reshape(-1, A.shape[-1])
    coh = float((d.norm() / pd.norm(dim=1).mean()).item())
    return d, nd, coh


def scan_layers(cap_harm, cap_harmless, n_pairs, min_layer=2):
    """Coherence scan. Decoder layer L <-> hidden_states index L+1.
    Final decoder layer excluded (degenerate pre-LM-head readout).
    Returns (best_row, full_table)."""
    n_layers = len(cap_harm) - 1
    max_layer = n_layers - 2
    table = []
    for L in range(min_layer, max_layer + 1):
        A = cap_harm[L + 1][:n_pairs]
        B = cap_harmless[L + 1][:n_pairs]
        d, nd, coh = coherence_stats(A, B)
        table.append({"decoder_layer": L, "hidden_states_index": L + 1,
                      "direction_norm": round(nd, 4),
                      "coherence": round(coh, 4)})
    table_sorted = sorted(table, key=lambda r: -r["coherence"])
    ui.detail("top layers by coherence:")
    for r in table_sorted[:5]:
        print(f"        L{r['decoder_layer']:2d} coh={r['coherence']:.3f} "
              f"|d|={r['direction_norm']:.2f}", flush=True)
    return table_sorted[0], table


def final_norm_module(model):
    """The final-norm module (per-family layout via arch.py)."""
    from . import arch

    return arch.final_norm_module(model)


def readout_norm_mode(spec):
    """directions.readout_norm: 'double' (absent default; frozen Run-001
    computation) or 'single' (norm applied once, the real lm_head input)."""
    return (spec.get("directions") or {}).get("readout_norm", "double")


# ---- probe scoring (v2-identical semantics: v1 scorer is the frozen
# Run-001 parity contract — probe artifact `refused` grades are byte-diffed
# by parity, so v1 NEVER changes. The opt-in v2 grader lives in
# scoring_v2.py (dev-workstation-hardened: list-anchored help
# evidence, empathy-marker exceptions, split-aware subclasses) and is
# reached through session_grader() when a spec sets
# probe_sets.marker_mode: v2 — validated over N=192 logged smoke rows,
# tests/test_marker_mode_v2.py pins the exact flip set. -------------------
def refusal_score(text, markers):
    t = (text or "").lower()
    return 1 if any(m in t for m in markers) else 0


def _scoring_v2_scalar(text, markers):
    """Numeric adapter for the opt-in run-time v2 grader (scoring_v2
    module): v2_refused only. The subclass taxonomy needs the h/b split,
    which run-time rows don't carry — subclasses stay a post-hoc job via
    scoring_v2.grade_rows (T4's regrade flow); run-time v2 refuses
    identically regardless of split (classify's numeric result is
    split-independent)."""
    from . import scoring_v2

    return scoring_v2.classify(text, markers)[0]


def is_degenerate(text):
    t = (text or "").strip()
    if not t:
        return True
    words = t.split()
    if len(words) >= 40:
        uniq = len(set(w.lower() for w in words)) / len(words)
        if uniq < 0.15:
            return True
    return False


def generate(tok, model, prompt, max_new=200):
    wrapped = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True)
    enc = tok(wrapped, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text


def run_probes(tok, model, prompts, tag="", max_new=200, markers=None,
               score_fn=None):
    markers = markers or REFUSAL_MARKERS
    score = score_fn or refusal_score  # v1 frozen default; v2 opt-in
    rows = []
    t_start = time.time()
    for j, p in enumerate(prompts):
        t0 = time.time()
        o = generate(tok, model, p, max_new=max_new)
        r = score(o, markers)
        rows.append({"i": j, "prompt": p, "output": o, "refused": r,
                     "degenerate": is_degenerate(o),
                     "gen_s": round(time.time() - t0, 1)})
        if not ui.quiet():
            left = (time.time() - t_start) / (j + 1) * (len(prompts) - j - 1)
            print(f"  [{tag} {j + 1}/{len(prompts)}] refused={r} "
                  f"({rows[-1]['gen_s']}s, ~{_fmt_eta(left)} left) "
                  f"{o[:70]!r}", flush=True)
    if ui.quiet():
        ui.probe_batch_line(tag, rows, time.time() - t_start)
    return rows


def _fmt_eta(seconds):
    s = int(round(seconds))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def summarize(rows_h, rows_b):
    return {
        "refusal_rate": sum(r["refused"] for r in rows_h) / max(1, len(rows_h)),
        "benign_preserved": 1 - sum(r["refused"] for r in rows_b)
        / max(1, len(rows_b)),
        "n_harmful": len(rows_h), "n_harmless": len(rows_b),
        "degenerate_total": sum(r["degenerate"] for r in rows_h + rows_b),
    }


# ---- stage 4: probe stage ---------------------------------------------------
def run_baseline_and_hook_probes(spec, tok, model, out_dir, stages=None):
    """Baseline + hook-ablated probes. hooks.scope spec field (v1
    amendment, Run 000 semantics) decides hook placement:
      selected = single L* hook (Run 001 semantics, default)
      all      = same direction hooked at EVERY decoder layer 0..n_layers-1
                 (Run 000 learning #2: single-layer hooks leave partial
                 refusal - L*-only closed 82.8%->18.75%, all-layer 3.1%)
    Run-001 parity: absent hooks block = 'selected' -> byte-identical
    contract; parity gate unaffected. run_config/summary record the scope."""
    import torch

    markers, score_fn = session_grader(spec)
    n_probes = spec["probe_sets"]["n_probes"]
    ps = spec["probe_sets"]
    harmful = resolve_probe_set(ps["harmful"])[:n_probes]
    harmless = resolve_probe_set(ps["harmless"])[:n_probes]

    stages = stages or ui.Stages("A", 5)
    stages.step("baseline probes (clean model)")
    base_h = run_probes(tok, model, harmful, tag="base-harm",
                        max_new=spec["decoding"]["max_new_tokens"],
                        markers=markers, score_fn=score_fn)
    base_b = run_probes(tok, model, harmless, tag="base-harmless",
                        max_new=spec["decoding"]["max_new_tokens"],
                        markers=markers, score_fn=score_fn)
    base_sum = summarize(base_h, base_b)
    json.dump({"harmful": base_h, "harmless": base_b},
              open(os.path.join(out_dir, "probes_baseline.json"), "w"),
              indent=2)
    ui.detail(f"baseline: refusal {ui.pct(base_sum['refusal_rate'])}, "
              f"benign answered {ui.pct(base_sum['benign_preserved'])}, "
              f"degenerate {base_sum['degenerate_total']}")

    stages.step("hook-ablated probes (inference-time contrast)")
    L_star = json.load(open(os.path.join(out_dir,
                                         "layer_coherence.json")))["best"][
        "decoder_layer"]
    dir_A = np.load(os.path.join(out_dir, "refusal_direction_A.npy"))
    hook = AblationHook(torch.from_numpy(dir_A).float(), model.device,
                        next(model.parameters()).dtype)
    scope = (spec.get("hooks") or {}).get("scope", "selected")
    if scope == "all":
        from . import arch
        n_layers = arch.n_layers(model)
        for l in range(n_layers):
            hook.attach(arch.blocks(model)[l])
        print(f"      hook scope=ALL ({n_layers} layers, L*={L_star})",
              flush=True)
    else:
        from . import arch
        hook.attach(arch.blocks(model)[L_star])
        print(f"      hook scope=selected (L{L_star})", flush=True)
    hook_h = run_probes(tok, model, harmful, tag="hook-harm",
                        max_new=spec["decoding"]["max_new_tokens"],
                        markers=markers, score_fn=score_fn)
    hook_b = run_probes(tok, model, harmless, tag="hook-harmless",
                        max_new=spec["decoding"]["max_new_tokens"],
                        markers=markers, score_fn=score_fn)
    hook.detach()
    assert hook.calls > 0, "hook never fired - ablation invalid"
    hook_sum = summarize(hook_h, hook_b)
    hook_sum["hook_calls"] = hook.calls
    hook_sum["hook_scope"] = scope
    json.dump({"harmful": hook_h, "harmless": hook_b},
              open(os.path.join(out_dir, "probes_hook_ablated.json"), "w"),
              indent=2)
    return base_sum, hook_sum


class AblationHook:
    """Forward hook projecting the refusal-direction component out of the
    hooked decoder layer's output activations (all positions)."""

    def __init__(self, direction, device, dtype):
        import torch
        r = torch.nn.functional.normalize(direction.float(), dim=0).to(device)
        self.r = r.to(dtype)
        self.handles = []  # one per attached module (scope=all -> many)
        self.calls = 0

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hf = h.to(self.r.dtype)
        proj = (hf @ self.r.unsqueeze(-1)) * self.r
        h_new = (hf - proj).to(h.dtype)
        self.calls += 1
        return (h_new,) + tuple(output[1:]) if isinstance(output, tuple) \
            else h_new

    def attach(self, module):
        self.handles.append(module.register_forward_hook(self))

    def detach(self):
        """Remove EVERY registration (scope=all attaches one per layer)."""
        for h in self.handles:
            h.remove()
        self.handles = []


# ---- run_config + sha provenance ---------------------------------------------
def write_run_config(spec, out_dir, extra=None):
    import datetime
    import platform
    import sys

    from abliteration_engine.spec import spec_hash

    config = {
        "engine": "eng-v3",
        "spec_hash": spec_hash(spec),
        "spec_path": spec["_spec_path"],
        "spec_sha256": spec["_spec_sha256"],
        "generated_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "run_card": spec["run_card"],
        "patient": {k: v for k, v in spec["patient"].items()
                    if k != "structure_expect"},
        "structure": extra.get("structure") if extra else None,
        "probes": {k: v for k, v in spec["probe_sets"].items()},
        "decoding": spec["decoding"],
        "gates": spec["gates"],
        "hooks": (spec.get("hooks") or {"scope": "selected"}),
        "directions": {"readout_norm": readout_norm_mode(spec)},
        "ladder_variants": spec["ladder"]["variants"],
        # FTT-28: effective ARA config when an ara_<rank> variant is in the
        # ladder; null otherwise (always-present key mirrors hooks.scope).
        "ladder_ara": (spec["ladder"].get("ara")
                       if any(str(v).startswith("ara_")
                              for v in spec["ladder"]["variants"]) else None),
        "versions": {"python": platform.python_version(),
                     "torch": (torch.__version__ if
                               (torch := sys.modules.get("torch")) else None)},
        "gpu": torch.cuda.get_device_name(0)
        if (torch := sys.modules.get("torch")) is not None and
        torch.cuda.is_available() else "cpu",
    }
    if extra:
        config.update({k: v for k, v in extra.items() if k != "structure"})
    json.dump(config, open(os.path.join(out_dir, "run_config.json"), "w"),
              indent=2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- top-level pipeline (stages 1-4; stage 5 ladder + 6 publish in
# eng/edits.py + eng/publish.py -----------------------------------
def from_spec(spec):
    """Full stage A: load + capture + directions + probes. Returns the
    v2-summary-shaped dict (RUN003_DONE payload shape)."""
    import torch

    from abliteration_engine.data import resolve_probe_set

    out_dir = _out_dir(spec, create=True)
    global REFUSAL_MARKERS, _REFUSAL_MARKERS_SRC
    ref = spec["probe_sets"]["refusal_markers"]
    REFUSAL_MARKERS = resolve_markers(ref)
    _REFUSAL_MARKERS_SRC = ref
    torch.manual_seed(spec["decoding"]["seed"])
    t_start = time.time()

    stages = ui.Stages("A", 5)
    stages.step(f"load {spec['patient']['model_id']} @ "
              f"{spec['patient']['revision']}")
    tok, model = load_patient(spec)
    from . import arch
    n_layers = arch.n_layers(model)
    struct = structure_report(model)
    assert_patient_structure(spec, struct)
    ui.detail(f"{struct.get('model_type')}: {struct['num_hidden_layers']} "
              f"layers, hidden {struct['hidden_size']}, "
              f"{'tied' if struct['tie_word_embeddings'] else 'untied'} "
              f"head, edits {', '.join(struct['edit_matrices'])}")

    ps = spec["probe_sets"]
    harmful = resolve_probe_set(ps["harmful"])[:ps["n_pairs"]]
    harmless = resolve_probe_set(ps["harmless"])[:ps["n_pairs"]]

    stages.step("capture final-position residuals (all layers, one "
                "forward pass per batch)")
    t0 = time.time()
    wrap = lambda texts: [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False,
        add_generation_prompt=True) for p in texts]
    cap_harm = capture_final_residuals(tok, model, wrap(harmful))
    cap_harmless = capture_final_residuals(tok, model, wrap(harmless))
    ui.detail(f"captured {len(harmful)}+{len(harmless)} prompts "
              f"({time.time() - t0:.1f}s)")

    stages.step("coherence scan + refusal directions")
    best, table = scan_layers(cap_harm, cap_harmless, ps["n_pairs"])
    L_star = best["decoder_layer"]
    dir_A = cap_harm[L_star + 1][:ps["n_pairs"]].mean(0) \
        - cap_harmless[L_star + 1][:ps["n_pairs"]].mean(0)
    norm_mod = final_norm_module(model)
    with torch.inference_mode():
        post_h = norm_mod(cap_harm[n_layers].to(model.device)).float().cpu()
        post_h0 = norm_mod(cap_harmless[n_layers].to(model.device)) \
            .float().cpu()
    dir_B, nd_B, coh_B = coherence_stats(post_h, post_h0)
    # hidden_states[n_layers] is ALREADY post-final-norm (HF convention),
    # so post_h above applies the norm a second time. That is the frozen
    # Run-001 computation ("double", default — parity anchors depend on its
    # bytes). directions.readout_norm: single uses the actual lm_head input.
    readout_norm = readout_norm_mode(spec)
    if readout_norm == "single":
        dir_B, nd_B, coh_B = coherence_stats(
            cap_harm[n_layers].float(), cap_harmless[n_layers].float())
    ui.detail(f"best layer L{L_star} coh={best['coherence']} "
              f"(A: residual space); readout-space coh={coh_B:.3f} "
              f"|d|={nd_B:.2f}")

    dirs_all = torch.stack([
        cap_harm[l + 1][:ps["n_pairs"]].mean(0)
        - cap_harmless[l + 1][:ps["n_pairs"]].mean(0)
        for l in range(n_layers)])
    np.savez_compressed(os.path.join(out_dir, "layer_directions.npz"),
                        directions=dirs_all.numpy(),
                        decoder_layers=np.arange(n_layers))
    np.savez_compressed(os.path.join(out_dir, "captures_harm.npz"),
                        **{f"L{l}": cap_harm[l].numpy()
                           for l in range(n_layers + 1)})
    np.savez_compressed(os.path.join(out_dir, "captures_harmless.npz"),
                        **{f"L{l}": cap_harmless[l].numpy()
                           for l in range(n_layers + 1)})
    np.save(os.path.join(out_dir, "refusal_direction_A.npy"), dir_A.numpy())
    np.save(os.path.join(out_dir, "refusal_direction_B.npy"), dir_B.numpy())
    json.dump({"best": best, "table": table, "n_pairs": ps["n_pairs"],
               "final_layer": n_layers - 1, "structure": struct,
               "readout_space": {"coherence": coh_B,
                                 "direction_norm": nd_B,
                                 "readout_norm": readout_norm}},
              open(os.path.join(out_dir, "layer_coherence.json"), "w"),
              indent=2)
    print("      directions + captures saved", flush=True)

    base_sum, hook_sum = run_baseline_and_hook_probes(spec, tok, model,
                                                      out_dir, stages)
    # "layer" is the block publish.py reads for the card (L*, residual and
    # readout-space coherence) — it was never written before, so publish
    # KeyError'd on every engine-produced run_config.json.
    write_run_config(spec, out_dir, extra={
        "structure": struct,
        "layer": {"decoder_layer": L_star,
                  "coherence": best["coherence"],
                  "readout_space_final_layer_coherence": coh_B,
                  "readout_norm": readout_norm}})
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    json.dump({"eng_core.py": sha256_file(os.path.abspath(__file__)),
               "eng_spec.py": sha256_file(os.path.join(pkg_dir, "spec.py")),
               **{f"eng_{m}.py": sha256_file(os.path.join(pkg_dir, f"{m}.py"))
                  for m in ("data", "edits", "ara", "scoring_v2",
                            "pipeline")}},
              open(os.path.join(out_dir, "harness_sha256.json"), "w"),
              indent=2)

    summary = {"layer": L_star, "coherence": best["coherence"],
               "readout_coherence_B": round(coh_B, 4),
               "structure": struct,
               "baseline": base_sum, "hook_ablated": hook_sum,
               "wall_s": round(time.time() - t_start, 1)}
    # stage-5 gating (v1 amendment): empty ladder = hook-only
    # characterization run (Run 000 semantics) - stage 5 SKIPPED, no
    # selection.json, publish gated off. Enforced here, not just documented
    # in the plan note (dev-workstation freeze review 2026-09-30).
    summary["ladder_skipped"] = not spec["ladder"]["variants"]
    if summary["ladder_skipped"]:
        ui.detail("ladder SKIPPED (empty variants - hook-only "
                  "characterization run)")
        ui.detail("publish GATED OFF for hook-only runs")
    return summary