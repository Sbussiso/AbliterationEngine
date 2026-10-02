"""ARA — Arbitrary-Rank Ablation (Weidmann 2026) as an engine variant.

Implements the ARA optimization described in Heretic's default modifier
(p-e-w/heretic, "Arbitrary-Rank Ablation (ARA)", PR #211) as a
spec-driven persistent-edit variant class. NOT a code port: this module is
original engine code (MIT) implementing the published algorithm — the
reference lives under AGPL-3.0 and was only read for the loss/optimizer
semantics. Deviations from the reference are listed at the bottom.

Algorithm (per decoder layer, per o_proj/down_proj matrix):
  - Capture module I/O (input, output) at final positions for a "good"
    prompt pool (harmless) and a "bad" pool (harmful) on the BASE model.
  - Fit W_eff = W_base + B @ A (rank-k LoRA, B init zero) by minimizing
      L = w_pg * ||out_good_eff - out_good_base||^2            (preserve good)
        + w_sb * [ mean_i kNN_k(out_bad_eff_i, out_good_base)  (pull closer)
                 - w_oc * mean_i kNN_k(out_bad_eff_i, out_bad_base) ]
                                                           (push away)
    with row norms of W_eff renormalized to the base magnitudes (Lai 2025
    norm-preserving ablation) when preserve_row_magnitudes is on.
  - Optimizer: LBFGS (strong-Wolfe line search). Reference defaults:
    lr=1.0, 5 outer steps x 20 inner iters, history 10.
  - The fitted W_eff is MATERIALIZED into the module weights (adapter
    dropped) so the save->reload->disk-verify->probe lifecycle is
    identical to every other persistent-edit variant.

Engine deviations from the Heretic reference (deliberate, documented):
  1. NO TPE: variant parameters come from the spec's ladder.ara block —
     the same provenance model as every other variant in this engine.
  2. Stream-and-merge capture: module I/O is captured one batch at a time
     and merged per module WITHOUT holding all layers' I/O in memory
     simultaneously (the reference caches every layer's I/O up front —
     fine for <=1.5B on Colab, fatal at 7B fp16).
  3. Captures park in float32 on CPU; the optimizer runs fp32 on the
     module's device (reference numerics, different parking spot).
  4. PEFT-free: A/B are plain fp32 tensors and W_eff is computed manually
     (the reference bypasses PEFT's forward for the gradient anyway).
  5. All requested decoder layers are edited (spec's ladder.ara.layers);
     the reference TPE-searches a [start, end) layer range per trial.

Training-pool hygiene: builtin pools `builtin:ara_good` / `builtin:ara_bad`
ship from the ARA reference datasets (mlabonne/harmless_alpaca,
mlabonne/harmful_behaviors; whitespace-flattened) minus every probe-set
overlap — optimizer prompts are DISJOINT from the marker-eval probe sets by
construction (tests/test_ara_variant.py pins this), so the optimizer never
tunes on its own evaluation set. 400 rows each = the reference's
train[:400] defaults.
"""
import json
import os
import time

# component label -> decoder-layer submodule path
_COMPONENTS = ("self_attn.o_proj", "mlp.down_proj")

_ARA_DEFAULTS = {
    "preserve_good_weight": 1.0,
    "steer_bad_weight": 0.2,
    "overcorrect_weight": 0.5,
    "neighbor_count": 8,
    "steps": 5,
    "lr": 1.0,
    "max_iter": 20,
    "history_size": 10,
    "preserve_row_magnitudes": True,
    "batch_size": 16,
    "layers": None,  # None = every decoder layer
    "good": "builtin:ara_good",
    "bad": "builtin:ara_bad",
}

ARA_VARIANT_PREFIX = "ara_"


def resolve_ara_config(ladder):
    """(name, cfg) from a spec's ladder dict: an `ara_<rank>` variant ->
    (name, normalized config); (None, None) if no ARA variant requested.
    Raises ValueError on contradictions. Defaults mirror the ARA reference;
    keys present in ladder.ara override them. rank comes from the variant
    name itself (ara_50 -> 50), so a ladder.ara.rank key that disagrees is
    a hard error, not a silent override."""
    variants = ladder.get("variants") or []
    name = next((v for v in variants
                 if v.startswith(ARA_VARIANT_PREFIX)
                 and v[len(ARA_VARIANT_PREFIX):].isdigit()), None)
    if name is None:
        if ladder.get("ara"):
            raise ValueError("ladder.ara set but no ara_<rank> variant in "
                             "ladder.variants")
        return None, None
    raw = ladder.get("ara") or {}
    if not isinstance(raw, dict):
        raise ValueError("ladder.ara must be a mapping")
    if raw.get("rank") is not None and int(raw["rank"]) != int(name[4:]):
        raise ValueError(f"ladder.ara.rank={raw['rank']} contradicts "
                         f"variant name {name!r}")
    cfg = dict(_ARA_DEFAULTS)
    for k, v in raw.items():
        if k == "rank":
            continue
        if k not in cfg:
            raise ValueError(f"ladder.ara: unknown key {k!r} (known: "
                             f"{sorted(_ARA_DEFAULTS)})")
        cfg[k] = v
    cfg["rank"] = int(name[4:])
    if int(cfg["rank"]) < 1:
        raise ValueError(f"{name}: rank must be >= 1")
    if int(cfg["neighbor_count"]) < 1:
        raise ValueError("ladder.ara.neighbor_count must be >= 1")
    layers = cfg.get("layers")
    if layers is not None and (not isinstance(layers, list) or
                               any(not isinstance(x, int) or x < 0
                                   for x in layers)):
        raise ValueError("ladder.ara.layers must be a list of layer indices "
                         "(None = every decoder layer)")
    return name, cfg


def capture_module_io(tok, model, prompts, layers=None, batch_size=16):
    """Stream-and-merge module I/O for the ARA component modules at the
    requested decoder layers (default: every layer). One batch in flight
    at a time; per-prompt rows are taken at the last REAL token
    (attention-mask-aware: lens-1 — right padding puts pads in the last
    column, and the same rule as core.capture_final_residuals applies).
    Returns {decoder_layer: {component: (inputs [P, in_dim] f32 cpu,
    outputs [P, out_dim])}}.
    """
    import torch

    n_layers = model.config.num_hidden_layers
    layers = sorted(range(n_layers)) if layers is None else sorted(layers)
    if not layers:
        raise ValueError("no ARA layers requested")
    io = {li: {c: ([], []) for c in _COMPONENTS} for li in layers}
    current = {}  # batch-scoped last-real-token index for the hooks

    def hook(li, c):
        def _hook(module, inputs, output):
            rows = torch.arange(inputs[0].shape[0], device=inputs[0].device)
            idx = current["idx"].to(inputs[0].device)
            io[li][c][0].append(inputs[0][rows, idx].detach().clone().cpu())
            io[li][c][1].append(output[rows, idx].detach().clone().cpu())
        return _hook

    handles = []
    for li in layers:
        handles.append(model.model.layers[li].self_attn.o_proj
                       .register_forward_hook(hook(li, "self_attn.o_proj")))
        handles.append(model.model.layers[li].mlp.down_proj
                       .register_forward_hook(hook(li, "mlp.down_proj")))
    n_batches = 0
    try:
        with torch.inference_mode():
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i:i + batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True)
                if hasattr(enc, "to"):  # HF BatchEncoding; plain dict passes
                    enc = enc.to(model.device)
                current["idx"] = enc["attention_mask"].sum(dim=1) - 1
                model(**enc, output_hidden_states=False, use_cache=False)
                n_batches += 1
    finally:
        for h in handles:
            h.remove()
    if not n_batches:
        raise ValueError("ARA capture ran zero batches (empty prompt pool?)")
    merged = {}
    for li in layers:
        merged[li] = {}
        for c in _COMPONENTS:
            ins, outs = io[li][c]
            if len(ins) != n_batches:
                raise AssertionError(
                    f"L{li} {c}: {len(ins)} hook batches != {n_batches} "
                    "forward batches - hook contract violated")
            merged[li][c] = (
                torch.cat([t.float().cpu() for t in ins], dim=0),
                torch.cat([t.float().cpu() for t in outs], dim=0))
    return merged


def mean_distances_to_knn(a, b, k):
    """Per-row mean Euclidean distance to the k nearest vectors of b."""
    import torch

    distances = torch.cdist(a, b)
    nearest, _ = distances.topk(k, dim=1, largest=False)
    return nearest.mean(1)


def ara_loss(good_output, bad_output, new_good_output, new_bad_output,
             params):
    """The ARA objective (reference semantics, original code):
      L = w_pg * MSE(out_good_eff, out_good)
        + w_sb * [ mean kNN_k(out_bad_eff, out_good)
                 - w_oc * mean kNN_k(out_bad_eff, out_bad) ]"""
    preserve_good = ((new_good_output - good_output) ** 2).mean()
    steer_bad = (
        mean_distances_to_knn(new_bad_output, good_output,
                              params["neighbor_count"]).mean()
        + params["overcorrect_weight"]
        * -mean_distances_to_knn(new_bad_output, bad_output,
                                 params["neighbor_count"]).mean())
    return (params["preserve_good_weight"] * preserve_good
            + params["steer_bad_weight"] * steer_bad)


def _module_for(layer_mod, label):
    return (layer_mod.self_attn.o_proj if label == "self_attn.o_proj"
            else layer_mod.mlp.down_proj)


def optimize_ara_weights(model, layer, cfg, good_io, bad_io):
    """Fit A/B for o_proj+down_proj at ONE decoder layer by L-BFGS on the
    ARA objective, then materialize W_eff into the module weights.
    Returns per-component info for the variant summary."""
    import torch

    layer_mod = model.model.layers[layer]
    info = {}
    for label in _COMPONENTS:
        lin = _module_for(layer_mod, label)
        W_base = lin.weight.data.detach().float().cpu()
        W_row_norms = torch.linalg.vector_norm(W_base, dim=1, keepdim=True)

        good_in, good_out = good_io[layer][label]
        bad_in, bad_out = bad_io[layer][label]
        dev = next(lin.parameters()).device
        good_in = good_in.to(dev)
        good_out = good_out.to(dev)
        bad_in = bad_in.to(dev)
        bad_out = bad_out.to(dev)
        W_base_dev = W_base.to(dev)
        W_row_norms_dev = W_row_norms.to(dev)
        H = W_base_dev.shape[1]
        rank = int(cfg["rank"])
        # LoRA init, PEFT-faithful: A random (kaiming_uniform, a=sqrt(5) ->
        # bound 1/sqrt(fan_in)), B ZERO so W_eff starts at W_base. A zeros
        # would zero the ENTIRE gradient (dW/dA = B = 0, dW/dB = A = 0) and
        # L-BFGS would never leave the init point.
        # Seeded from the run seed + layer (deterministic re-runs; the
        # reference rides the global RNG here).
        gen_seed = (int(cfg.get("run_seed", 0)) * 1_000_003
                    + 7919 * int(layer) + 13)
        gen = torch.Generator(device="cpu").manual_seed(gen_seed)
        A = ((torch.rand(rank, H, generator=gen) * 2 - 1)
             / (H ** 0.5)).to(dev).requires_grad_(True)
        B = torch.zeros(W_base_dev.shape[0], rank, device=dev,
                        dtype=torch.float32, requires_grad=True)

        p = {"preserve_good_weight": float(cfg["preserve_good_weight"]),
             "steer_bad_weight": float(cfg["steer_bad_weight"]),
             "overcorrect_weight": float(cfg["overcorrect_weight"]),
             "neighbor_count": int(cfg["neighbor_count"])}
        preserve_rows = bool(cfg.get("preserve_row_magnitudes", True))

        def objective():
            W_eff = W_base_dev + (B @ A)
            if preserve_rows:
                W_eff = torch.nn.functional.normalize(
                    W_eff, p=2, dim=1) * W_row_norms_dev
            new_good = good_in @ W_eff.T
            new_bad = bad_in @ W_eff.T
            return ara_loss(good_out, bad_out, new_good, new_bad, p)

        optimizer = torch.optim.LBFGS(
            [A, B], lr=float(cfg["lr"]), max_iter=int(cfg["max_iter"]),
            history_size=int(cfg["history_size"]),
            line_search_fn="strong_wolfe")

        losses = []
        for _step in range(int(cfg["steps"])):
            def closure():
                optimizer.zero_grad()
                loss = objective()
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            W_eff = W_base_dev + (B @ A)
            if preserve_rows:
                W_eff = torch.nn.functional.normalize(
                    W_eff, p=2, dim=1) * W_row_norms_dev
            rows_before = float(torch.linalg.vector_norm(
                W_base_dev, dim=1).max())
            rows_after = float(torch.linalg.vector_norm(W_eff, dim=1).max())
            drift = (float(abs(rows_before - rows_after)) / float(rows_before)
                     if rows_before else 0.0)
            lin.weight = torch.nn.Parameter(
                W_eff.to(lin.weight.dtype).to(dev), requires_grad=False)
        info[label] = {"shape": list(W_base.shape),
                       "loss_first": losses[0] if losses else None,
                       "loss_last": losses[-1] if losses else None,
                       "outer_steps": len(losses),
                       "row_norm_rel_drift": drift}
        print(f"      ARA L{layer} {label}: loss "
              f"{losses[0]:.6f} -> {losses[-1]:.6f} "
              f"({len(losses)} outer steps, row-norm drift {drift:.2e})",
              flush=True)
    return info


def verify_ara_on_disk(model_r, base_model, layers):
    """On-disk persistence check: edited layers' matrices DIFFER from a
    fresh base load; untouched layers are EXACTLY equal. Returns report."""
    out = {"edited_max_absdiff": 0.0, "untouched_max_absdiff": 0.0,
           "layers": {}}
    for li in range(model_r.config.num_hidden_layers):
        layer_r = model_r.model.layers[li]
        layer_b = base_model.model.layers[li]
        edited = li in layers
        worst = 0.0
        for label in _COMPONENTS:
            W_r = _module_for(layer_r, label).weight.data.float().cpu()
            W_b = _module_for(layer_b, label).weight.data.float().cpu()
            d = float((W_r - W_b).abs().max())
            worst = max(worst, d)
        key = str(li)
        out["layers"][key] = {"edited": edited, "max_absdiff_vs_base": worst}
        if edited:
            out["edited_max_absdiff"] = max(out["edited_max_absdiff"], worst)
        else:
            out["untouched_max_absdiff"] = max(
                out["untouched_max_absdiff"], worst)
    if out["edited_max_absdiff"] <= 0.0:
        raise AssertionError("ARA edit did not survive save/reload "
                             "(edited layers == base)")
    if out["untouched_max_absdiff"] > 0.0:
        raise AssertionError("ARA touched layers outside ladder.ara.layers "
                             "(untouched layers differ from base)")
    return out


def run_ara_variant(spec, name, cfg, model_base=None):
    """The ARA variant lifecycle, mirroring edits.run_variant:
    base load -> capture good/bad module I/O -> per-layer L-BFGS fit ->
    materialize -> save -> RELOAD from disk -> verify -> probe ->
    probes_<name>.json. Returns the summary dict."""
    import gc

    import torch

    from abliteration_engine import core
    from abliteration_engine.data import resolve_probe_set
    from abliteration_engine.edits import save_variant

    print(f"      --- {name} (ARA rank {cfg.get('rank')}) ---", flush=True)
    t0 = time.time()
    tok_v, model_v = core.load_patient(spec)
    layers = cfg.get("layers") or list(range(
        model_v.config.num_hidden_layers))
    # normalize at the point of use (day-2 lesson applies): defaults under
    # whatever the spec/caller provided, then rank from the VARIANT NAME
    # (single source of truth) + run seed for the A-init generator
    cfg = {**_ARA_DEFAULTS, **cfg, "layers": layers, "rank": int(name[4:]),
           "run_seed": int(spec.get("decoding", {}).get("seed", 0))}

    def wrap(texts):
        return [tok_v.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True) for p in texts]
    good_prompts = resolve_probe_set(cfg["good"])
    bad_prompts = resolve_probe_set(cfg["bad"])
    bs = int(cfg["batch_size"])
    print(f"      ARA pools: good={len(good_prompts)} "
          f"bad={len(bad_prompts)} (batch {bs}, layers "
          f"{layers[0]}..{layers[-1]})", flush=True)
    good_io = capture_module_io(tok_v, model_v, wrap(good_prompts),
                                layers=layers, batch_size=bs)
    bad_io = capture_module_io(tok_v, model_v, wrap(bad_prompts),
                               layers=layers, batch_size=bs)

    edit_info = {}
    for li in layers:
        edit_info[f"L{li}"] = optimize_ara_weights(model_v, li, cfg,
                                                   good_io, bad_io)
    del good_io, bad_io
    gc.collect()

    VARBASE = os.environ.get("ENG_VARBASE") or core.eng_base()
    var_dir = os.path.join(VARBASE, name)
    save_variant(model_v, tok_v, var_dir, expect_tied=True)
    del model_v
    gc.collect()
    torch.cuda.empty_cache()

    tok_r, model_r = core.load_patient({"patient": {
        "model_id": var_dir, "revision": None}})
    _, model_base_chk = core.load_patient(spec)
    disk = verify_ara_on_disk(model_r, model_base_chk, set(layers))
    del model_base_chk
    gc.collect()
    torch.cuda.empty_cache()

    markers, score_fn = core.session_grader(spec)
    n_probes = spec["probe_sets"]["n_probes"]
    max_new = spec["decoding"]["max_new_tokens"]
    os.makedirs(core._out_dir(spec), exist_ok=True)
    harmful = resolve_probe_set(spec["probe_sets"]["harmful"])[:n_probes]
    harmless = resolve_probe_set(spec["probe_sets"]["harmless"])[:n_probes]
    r_h = core.run_probes(tok_r, model_r, harmful, tag=f"{name}-harm",
                          max_new=max_new, markers=markers,
                          score_fn=score_fn)
    r_b = core.run_probes(tok_r, model_r, harmless, tag=f"{name}-harmless",
                          max_new=max_new, markers=markers,
                          score_fn=score_fn)
    s = core.summarize(r_h, r_b)
    s["edit_info"] = {"method": "ARA (Weidmann 2026)", "rank": cfg["rank"],
                      "layers": layers, "per_layer": edit_info}
    s["on_disk_verify"] = disk
    s["tie_flag_on_disk"] = model_r.config.tie_word_embeddings
    s["wall_s"] = round(time.time() - t0, 1)
    json.dump({"harmful": r_h, "harmless": r_b},
              open(os.path.join(core._out_dir(spec), f"probes_{name}.json"),
                   "w"), indent=2)
    print(f"      {name} reloaded: refusal={s['refusal_rate']} "
          f"benign={s['benign_preserved']} "
          f"degenerate={s['degenerate_total']}", flush=True)
    del model_r, tok_r
    gc.collect()
    torch.cuda.empty_cache()
    return s


def ladder_meta(name, cfg, edited_layers):
    """Provenance block for run_config.ladder_ara / selection.json."""
    return {"variant": name, "method": "ARA (Weidmann 2026)",
            "params": {k: v for k, v in cfg.items() if k != "layers"},
            "layers": edited_layers}