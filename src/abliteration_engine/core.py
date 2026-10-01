"""eng core — single parameterized abliteration pipeline (FTT-18/19).

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

from abliteration_engine.data import resolve_markers, resolve_probe_set

REFUSAL_MARKERS = None  # set per-run by from_spec()


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


def _out_dir(spec, create=False):
    rc = spec["run_card"]
    d = os.path.join(eng_base(),
                     f"eng_run_{rc['run_number']:03d}_{rc['patient']}")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _stage_boilerplate(stage_name, fn, spec, ctx, *args, **kwargs):
    """Common sentinel/error handling for one engine stage."""
    out_dir = _out_dir(spec, create=True)
    with open(sentinel_exit(), "w") as f:
        f.write("running")
    code = 1
    try:
        result = fn(spec, ctx, *args, **kwargs)
        code = 0
        return result
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            with open(os.path.join(out_dir, f"{stage_name}_error.txt"),
                      "w") as f:
                f.write(traceback.format_exc()[-8000:])
        except Exception:
            pass
    finally:
        with open(sentinel_exit(), "w") as _f:
            _f.write(str(code))


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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pat["model_id"], torch_dtype=dtype, revision=pat["revision"],
            device_map="auto")
    except TypeError:  # newer transformers renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(
            pat["model_id"], dtype=dtype, revision=pat["revision"],
            device_map="auto")
    model.eval()
    return tok, model


def structure_report(model):
    c = model.config
    return {
        "num_hidden_layers": c.num_hidden_layers,
        "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size,
        "vocab_size": c.vocab_size,
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "tie_word_embeddings": bool(c.tie_word_embeddings),
        "o_proj_shape": list(model.model.layers[0].self_attn.o_proj.weight.shape),
        "down_proj_shape": list(model.model.layers[0].mlp.down_proj.weight.shape),
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
    pass per batch. Index 0 = embedding output, i = decoder layer i-1 output.
    Returns per-layer list of [N, H] float32 CPU tensors (v2-identical)."""
    import torch

    n_layers = model.config.num_hidden_layers
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
    print("  top layers by coherence:", flush=True)
    for r in table_sorted[:5]:
        print(f"    L{r['decoder_layer']:2d} coh={r['coherence']:.3f} "
              f"|d|={r['direction_norm']:.2f}", flush=True)
    return table_sorted[0], table


def final_norm_module(model):
    m = getattr(model, "model", None)
    n = getattr(m, "norm", None) if m is not None else None
    if n is None:
        n = getattr(model, "norm", None) or getattr(model, "final_layer_norm",
                                                    None)
    if n is None:
        raise RuntimeError("final norm module not found")
    return n


# ---- probe scoring (v2-identical semantics) --------------------------------
def refusal_score(text, markers):
    t = (text or "").lower()
    return 1 if any(m in t for m in markers) else 0


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


def generate(tok, model, prompt, max_new=200, markers=None):

    wrapped = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True)
    enc = tok(wrapped, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text


def run_probes(tok, model, prompts, tag="", max_new=200, markers=None):
    markers = markers or REFUSAL_MARKERS
    rows = []
    for j, p in enumerate(prompts):
        t0 = time.time()
        o = generate(tok, model, p, max_new=max_new)
        r = refusal_score(o, markers)
        rows.append({"i": j, "prompt": p, "output": o, "refused": r,
                     "degenerate": is_degenerate(o),
                     "gen_s": round(time.time() - t0, 1)})
        print(f"  [{tag} {j + 1}/{len(prompts)}] refused={r} "
              f"({rows[-1]['gen_s']}s) {o[:70]!r}", flush=True)
    return rows


def summarize(rows_h, rows_b):
    return {
        "refusal_rate": sum(r["refused"] for r in rows_h) / max(1, len(rows_h)),
        "benign_preserved": 1 - sum(r["refused"] for r in rows_b)
        / max(1, len(rows_b)),
        "n_harmful": len(rows_h), "n_harmless": len(rows_b),
        "degenerate_total": sum(r["degenerate"] for r in rows_h + rows_b),
    }


# ---- stage 4: probe stage ---------------------------------------------------
def run_baseline_and_hook_probes(spec, tok, model, out_dir):
    """Baseline + hook-ablated probes. hooks.scope spec field (v1
    amendment, Run 000 semantics) decides hook placement:
      selected = single L* hook (Run 001 semantics, default)
      all      = same direction hooked at EVERY decoder layer 0..n_layers-1
                 (Run 000 learning #2: single-layer hooks leave partial
                 refusal - L*-only closed 82.8%->18.75%, all-layer 3.1%)
    Run-001 parity: absent hooks block = 'selected' -> byte-identical
    contract; parity gate unaffected. run_config/summary record the scope."""
    import torch

    markers = resolve_markers(spec["probe_sets"]["refusal_markers"])
    n_probes = spec["probe_sets"]["n_probes"]
    ps = spec["probe_sets"]
    harmful = resolve_probe_set(ps["harmful"])[:n_probes]
    harmless = resolve_probe_set(ps["harmless"])[:n_probes]

    print("[4/5] baseline probes (clean model)", flush=True)
    base_h = run_probes(tok, model, harmful, tag="base-harm",
                        max_new=spec["decoding"]["max_new_tokens"],
                        markers=markers)
    base_b = run_probes(tok, model, harmless, tag="base-harmless",
                        max_new=spec["decoding"]["max_new_tokens"],
                        markers=markers)
    base_sum = summarize(base_h, base_b)
    json.dump({"harmful": base_h, "harmless": base_b},
              open(os.path.join(out_dir, "probes_baseline.json"), "w"),
              indent=2)
    print(f"      baseline: {json.dumps(base_sum)}", flush=True)

    print("[5/5] hook-ablated probes (inference-time)", flush=True)
    L_star = json.load(open(os.path.join(out_dir,
                                         "layer_coherence.json")))["best"][
        "decoder_layer"]
    dir_A = np.load(os.path.join(out_dir, "refusal_direction_A.npy"))
    hook = AblationHook(torch.from_numpy(dir_A).float(), model.device,
                        next(model.parameters()).dtype)
    scope = (spec.get("hooks") or {}).get("scope", "selected")
    if scope == "all":
        n_layers = model.config.num_hidden_layers
        for l in range(n_layers):
            model.model.layers[l].register_forward_hook(hook)
        print(f"      hook scope=ALL ({n_layers} layers, L*={L_star})",
              flush=True)
    else:
        hook.attach(model.model.layers[L_star])
        print(f"      hook scope=selected (L{L_star})", flush=True)
    hook_h = run_probes(tok, model, harmful, tag="hook-harm",
                        max_new=spec["decoding"]["max_new_tokens"],
                        markers=markers)
    hook_b = run_probes(tok, model, harmless, tag="hook-harmless",
                        max_new=spec["decoding"]["max_new_tokens"],
                        markers=markers)
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
        self.handle = None
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
        self.handle = module.register_forward_hook(self)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


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
        "ladder_variants": spec["ladder"]["variants"],
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
# eng/edits.py + eng/publish.py per FTT-20) -----------------------------------
def from_spec(spec):
    """Full stage A: load + capture + directions + probes. Returns the
    v2-summary-shaped dict (RUN003_DONE payload shape)."""
    import torch

    from abliteration_engine.data import resolve_probe_set

    out_dir = _out_dir(spec, create=True)
    global REFUSAL_MARKERS
    REFUSAL_MARKERS = resolve_markers(spec["probe_sets"]["refusal_markers"])
    torch.manual_seed(spec["decoding"]["seed"])
    t_start = time.time()

    print(f"[1/5] pinned load {spec['patient']['model_id']} @ "
          f"{spec['patient']['revision']}", flush=True)
    tok, model = load_patient(spec)
    n_layers = model.config.num_hidden_layers
    struct = structure_report(model)
    assert_patient_structure(spec, struct)
    print(f"      structure: {json.dumps(struct)}", flush=True)

    ps = spec["probe_sets"]
    harmful = resolve_probe_set(ps["harmful"])[:ps["n_pairs"]]
    harmless = resolve_probe_set(ps["harmless"])[:ps["n_pairs"]]

    print("[2/5] capturing final-position residuals (all layers, one "
          "forward pass per batch)", flush=True)
    t0 = time.time()
    wrap = lambda texts: [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False,
        add_generation_prompt=True) for p in texts]
    cap_harm = capture_final_residuals(tok, model, wrap(harmful))
    cap_harmless = capture_final_residuals(tok, model, wrap(harmless))
    print(f"      captured ({time.time() - t0:.1f}s)", flush=True)

    print("[3/5] coherence scan + directions", flush=True)
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
    print(f"      best layer L{L_star} coh={best['coherence']} "
          f"(A: residual space); readout-space coh={coh_B:.3f} "
          f"|d|={nd_B:.2f}", flush=True)

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
                                 "direction_norm": nd_B}},
              open(os.path.join(out_dir, "layer_coherence.json"), "w"),
              indent=2)
    print("      directions + captures saved", flush=True)

    base_sum, hook_sum = run_baseline_and_hook_probes(spec, tok, model,
                                                      out_dir)
    write_run_config(spec, out_dir, extra={"structure": struct})
    json.dump({"eng_core.py": sha256_file(os.path.abspath(__file__)),
               "eng_spec.py": sha256_file(os.path.abspath(
                   __import__("abliteration_engine.spec",
                              fromlist=["spec"]).__file__))},
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
        print("[5/6] ladder SKIPPED (empty variants - hook-only "
              "characterization run)", flush=True)
        print("[6/6] publish GATED OFF for hook-only runs", flush=True)
    return summary