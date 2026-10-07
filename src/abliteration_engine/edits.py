"""eng edits — persistent-edit ladder (stage B).

Faithful port of ladder_003.py into spec-driven form. The verified mechanics
documented there carry over unchanged:

- lm_head edits go to a CLONED UNTIED parameter; config.tie_word_embeddings
  flipped to false before save (Qwen2.5 0.5B/1.5B ship tied).
- Row-space edits: the orthogonality invariant is (M W)^T r ~= 0, NOT (M W) r.
- Every variant is saved, RELOADED FROM DISK, verified on disk, then probed.

Selection: gate = benign_preserved >= baseline - gates.benign_floor_delta AND
degenerate == 0; pick lowest refusal among gate-passers (tie-break: ladder
order). publish_eligible = selected passed gate AND refusal <
gates.publish_refusal. MMLU guardrail is enforced later by eng/mmlu.py
before any publish.
"""
import gc
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch

from abliteration_engine import core


def orthogonalize_lm_head(model, direction):
    """W <- W - (W r) r^T on a CLONED, UNTIED parameter; flips
    config.tie_word_embeddings. Returns (max|W r| after edit, |r_hat|)."""
    lm = model.get_output_embeddings()
    assert lm is not None, "no output embeddings"
    r = direction.detach().float().cpu()
    r = r / r.norm()
    W = lm.weight.data
    W32 = W.float().cpu()
    proj = (W32 @ r).unsqueeze(1) * r.unsqueeze(0)
    W_new = (W32 - proj).to(W.dtype)
    max_comp = float((W_new.float() @ r).abs().max().item())
    lm.weight = torch.nn.Parameter(W_new.to(W.device), requires_grad=False)
    model.config.tie_word_embeddings = False
    return max_comp, float(r.norm().item())


def orthogonalize_final_norm(model, direction):
    """w <- w - (w.d) d on the final RMSNorm weight (readout-space
    direction). Returns (w.d before, |w_new . d| after edit-in-fp32).

    Heuristic, not an exact projection: the norm output is w * x_hat, whose
    component along d is sum_i w_i x_hat_i d_i — zero only for every input
    if w * d = 0, which w.d = 0 does not imply. Kept as-is (frozen ladder
    semantics; wd_BN/wd_ML_BN results and parity depend on it); the exact
    readout removal is the lm_head edit that always accompanies it."""
    n = core.final_norm_module(model)
    d = direction.detach().float().cpu()
    d = d / d.norm()
    w = n.weight.data
    w32 = w.float().cpu()
    comp = float(w32 @ d)
    w_new = w32 - comp * d
    resid = float(abs(w_new @ d))
    n.weight = torch.nn.Parameter(w_new.to(w.dtype).to(w.device),
                                  requires_grad=False)
    return comp, resid


def orthogonalize_layer_output(model, layer_idx, direction):
    """Row-space edit at one decoder layer: W <- M W, M = I - r r^T for the
    attention output-proj and MLP output-proj (residual-stream output
    matrices; module paths resolved per family by arch.py). Invariant:
    (M W)^T r ~= 0 (row-space, not column-space — run-002 gotcha).
    Conv1D-style transposed weights (gpt2 etc.) are handled by arch._Lin:
    the math below always sees the [out, in] view."""
    from . import arch

    r = direction.detach().float().cpu()
    r = r / r.norm()
    M = torch.eye(r.shape[0]) - torch.outer(r, r)
    resids = {}
    for label in arch.edit_labels(model):
        lin = arch.edit_linear(model, layer_idx, label)
        # capture on CPU: M lives on CPU (direction is flopped to CPU above),
        # and .float() alone would keep a CUDA tensor on cuda -> CPU(M)@CUDA(W)
        # raises on every untied-family edit (run-010 crash, commit 0763282).
        # set_weight routes the result back to the module's dtype/device.
        W = lin.weight.detach().float().cpu()
        W_new = (M @ W).to(torch.float32)
        resid = float((W_new.T @ r).abs().max().item())
        assert resid < 1e-3, (layer_idx, label, resid)
        lin.set_weight(W_new)
        resids[label] = resid
    return resids


def verify_untied(model):
    lm = model.get_output_embeddings()
    emb = model.get_input_embeddings()
    tied = lm.weight.data_ptr() == emb.weight.data_ptr()
    diff = float((lm.weight.float() - emb.weight.float()).norm().item())
    return {"tied": bool(tied), "weight_diff_l2": diff,
            "config_flag": model.config.tie_word_embeddings}


def save_variant(model, tok, out_dir, expect_tied=None):
    """Save a variant. Tie-state contract (bug BUG-2, run-010 coder-7B):
    the saved config must match the ACTUAL weight-sharing state of the
    model being saved (post-application truth), never a ladder-carried
    assumption.

    - Always: config.tie_word_embeddings must equal the real pointer-level
      tie state of the in-memory model (a config that lies about its own
      head is a bug upstream of the save — fail here, loudly).
    - expect_tied (optional, ladder-carried): when given, must also match;
      pass None for variants that never flip the tie (ara_* derives its
      state from the base).
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg_flag = bool(model.config.tie_word_embeddings)
    actual_tied = (model.get_output_embeddings().weight.data_ptr()
                   == model.get_input_embeddings().weight.data_ptr())
    assert cfg_flag is actual_tied, \
        (f"tie-state inconsistency before save: "
         f"config.tie_word_embeddings={cfg_flag} but lm_head/input "
         f"embeddings are {'tied' if actual_tied else 'NOT tied'} "
         f"(pointers {'match' if actual_tied else 'differ'})")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    saved = json.load(open(os.path.join(out_dir, "config.json")))
    assert bool(saved["tie_word_embeddings"]) is cfg_flag, \
        (f"saved config.json tie_word_embeddings={saved['tie_word_embeddings']}"
         f" != in-memory {cfg_flag}")
    if expect_tied is not None:
        assert cfg_flag is expect_tied, \
            (f"variant produced tie state {cfg_flag}, ladder expected "
             f"{expect_tied} (BUG-2 class: tied-base assumption on an "
             "untied patient?)")
    return out_dir


def verify_lm_head_disk(model_r, dir_vec, bound=5e-3):
    """Bound tolerates fp16 reload quantization noise (~3e-4 scale for a
    1536-dim dot with ~0.03-magnitude entries); a real incomplete edit
    leaves |W r| orders of magnitude higher."""
    lm = model_r.get_output_embeddings()
    d = dir_vec.detach().float().cpu()
    d = d / d.norm()
    resid = float((lm.weight.data.float().cpu() @ d).abs().max().item())
    assert resid < bound, resid
    return resid


def verify_final_norm_disk(model_r, dir_vec, bound=5e-2):
    """fp16 reload noise on a hidden-size (1536) dot of ~1.0-magnitude
    RMSNorm weights is ~1e-2; pre-edit alignment |w.d| is O(0.1+) so 5e-2
    cleanly separates 'edit survived' from 'edit lost'."""
    n = core.final_norm_module(model_r)
    d = dir_vec.detach().float().cpu()
    d = d / d.norm()
    resid = float(abs((n.weight.data.float().cpu() * d).sum().item()))
    assert resid < bound, resid
    return resid


def verify_layers_disk(model_r, layers, dirs_np, bound=1e-2, labels=None):
    """fp16 reload noise ~3e-4 scale; 1e-2 leaves 30x headroom. labels
    default to the family's edit matrices (arch registry)."""
    import torch

    from . import arch
    labels = labels or arch.edit_labels(model_r)
    out = {}
    for l in layers:
        r = torch.from_numpy(np.asarray(dirs_np[l])).float()
        r = r / r.norm()
        worst = {}
        for label in labels:
            lin = arch.edit_linear(model_r, l, label)
            ro = float((lin.weight.float().cpu().T @ r).abs().max().item())
            assert ro < bound, (l, label, ro)
            worst[label] = ro
        out[str(l)] = worst
    return out


def run_variant(name, edit_fn, out_dir, expect_tied, verify_fn, spec,
                tok_source_model=None, provenance=None):
    """Fresh base load -> edit -> save -> RELOAD from disk -> verify ->
    probe. Returns summary dict; dumps probes_<name>.json.
    (tok_source_model retained for call compatibility; the reloaded model
    never shares state with the source load.)

    expect_tied=None derives the tie expectation from the freshly loaded
    BASE (modifiers.flavor_for) — the ladder passes None because it never
    holds a model of its own (ctx["model"] is None on every pipeline path;
    deriving from it crashed every wd_ML ladder)."""
    print(f"      --- {name} ---", flush=True)
    t0 = time.time()
    tok_v, model_v = core.load_patient(spec)
    if expect_tied is None:
        expect_tied = expect_tied_for(name, model_v)  # base, pre-edit
    edit_info = edit_fn(model_v)
    print(f"      edit applied: {json.dumps(edit_info, default=str)}",
          flush=True)
    save_variant(model_v, tok_v, out_dir, expect_tied=expect_tied)
    del model_v
    gc.collect()
    torch.cuda.empty_cache()

    # persistence check: reload FROM DISK, verify, probe
    tok_r, model_r = core.load_patient({"patient": {
        "model_id": out_dir, "revision": None}})
    disk = verify_fn(model_r)
    flag = model_r.config.tie_word_embeddings
    assert flag is expect_tied, (flag, expect_tied)
    if not expect_tied:
        chk = verify_untied(model_r)
        assert not chk["tied"] and chk["weight_diff_l2"] > 0, chk
    n_probes = spec["probe_sets"]["n_probes"]
    max_new = spec["decoding"]["max_new_tokens"]
    from abliteration_engine.data import resolve_probe_set
    markers, score_fn = core.session_grader(spec)
    harmful = resolve_probe_set(spec["probe_sets"]["harmful"])[:n_probes]
    harmless = resolve_probe_set(spec["probe_sets"]["harmless"])[:n_probes]
    r_h = core.run_probes(tok_r, model_r, harmful, tag=f"{name}-harm",
                          max_new=max_new, markers=markers,
                          score_fn=score_fn)
    r_b = core.run_probes(tok_r, model_r, harmless, tag=f"{name}-harmless",
                          max_new=max_new, markers=markers,
                          score_fn=score_fn)
    s = core.summarize(r_h, r_b)
    s["edit_info"] = edit_info
    s["on_disk_verify"] = disk
    s["tie_flag_on_disk"] = flag
    s["wall_s"] = round(time.time() - t0, 1)
    write_variant_probes(spec, name, r_h, r_b, provenance)
    print(f"      {name} reloaded: refusal={s['refusal_rate']} "
          f"benign={s['benign_preserved']} "
          f"degenerate={s['degenerate_total']}", flush=True)
    del model_r
    gc.collect()
    torch.cuda.empty_cache()
    return s


def write_variant_probes(spec, name, r_h, r_b, provenance=None):
    """probes_<name>.json; `_provenance` (when given) is what banked resume
    checks before trusting the file in a later session."""
    payload = {"harmful": r_h, "harmless": r_b}
    if provenance is not None:
        payload["_provenance"] = provenance
    with open(os.path.join(core._out_dir(spec), f"probes_{name}.json"),
              "w") as f:
        json.dump(payload, f, indent=2)


def _arr_sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode()
                          + a.tobytes()).hexdigest()


def variant_provenance(spec, name, k_layers, dir_arrays):
    """Everything a variant's probe results depend on: pinned patient,
    probe/grader/decoding config, the stage-A arrays it edits with, and
    its own edit parameters. Banked results whose fingerprint differs are
    NOT reused (a re-run stage A, a marker_mode change, new k-layers or a
    new ARA config would otherwise silently reuse stale probes)."""
    ps = spec["probe_sets"]
    basis = {
        "variant": name,
        "patient": [spec["patient"]["model_id"], spec["patient"]["revision"]],
        "probes": [ps["harmful"], ps["harmless"], ps["n_probes"],
                   ps["refusal_markers"], ps.get("marker_mode", "v1")],
        "decoding": spec["decoding"],
        "k_layers": k_layers,
        "ara": (spec["ladder"].get("ara") if name.startswith("ara_")
                else None),
        "arrays": {k: _arr_sha(v) for k, v in sorted(dir_arrays.items())},
    }
    blob = json.dumps(basis, sort_keys=True, default=str).encode()
    return {"fingerprint": hashlib.sha256(blob).hexdigest(),
            "spec_sha256": spec.get("_spec_sha256")}


def select_variant(candidates, base_preserved, publish_refusal,
                   floor_delta=0.10, degenerate_max=0):
    """Deterministic selection over gate-passers with tie-break ladder
    order; publish_eligible only when the gate passed AND refusal <
    publish_refusal. Gate thresholds come from spec gates.benign_floor_delta
    and gates.degenerate_max."""
    for c in candidates:
        c["passes_gate"] = (c["benign_preserved"] >= base_preserved - floor_delta
                            and c["degenerate_total"] <= degenerate_max)
    gated = [c for c in candidates if c["passes_gate"]]
    if gated:
        selected = min(gated, key=lambda c: (c["refusal_rate"],
                                             c["ladder_index"]))
        eligible = selected["refusal_rate"] < publish_refusal
    else:
        selected = min(candidates, key=lambda c: (c["refusal_rate"],
                                                  c["degenerate_total"],
                                                  c["ladder_index"]))
        eligible = False
    return selected, eligible


def _banked_variant_summary(spec, name, provenance=None):
    """Resume short-circuit (2026-09-30, 3x Colab registry drops): if a
    COMPLETE probes_<name>.json from a prior session exists in the run
    out_dir, reuse it instead of re-editing + re-probing. Complete = both
    sides have exactly n_probes rows and every row has the grader fields
    (the variant's weights dir is NOT required — see the NOTE below).
    Returns summary or None (never raises -> corrupt files re-run normally).

    Provenance: when the caller passes the variant's expected provenance
    and the banked file carries a different fingerprint, the file is stale
    (stage A re-ran, grader/probe config or edit params changed) and is
    NOT reused. Files banked before provenance existed carry none: they are
    reused, flagged `banked_provenance: unverified` in the summary."""
    import json as _json
    import os as _os
    out_dir = core._out_dir(spec)
    p = _os.path.join(out_dir, f"probes_{name}.json")
    if not _os.path.exists(p):
        return None
    n_probes = spec["probe_sets"]["n_probes"]
    try:
        d = _json.load(open(p))
        r_h, r_b = d["harmful"], d["harmless"]
        if len(r_h) != n_probes or len(r_b) != n_probes:
            return None
        if not all(x.get("refused") is not None and x.get("output")
                   for x in r_h + r_b):
            return None
        banked_prov = d.get("_provenance")
        if provenance is not None and banked_prov is not None and \
                banked_prov.get("fingerprint") != provenance["fingerprint"]:
            print(f"      banked_resume skip {name}: provenance mismatch "
                  "(stale probes from a different stage A / config)",
                  flush=True)
            return None
        # NOTE: variant dir on disk intentionally NOT required — probes-only
        # banked resume never fabricates weights: if the SELECTED variant
        # was banked and its dir is gone, mmlu fails to load it and publish
        # refuses ("missing variant dir"). Rebuild by deleting that
        # variant's probes file and re-running the ladder (Tutorial 3).
        base = _json.load(open(_os.path.join(out_dir, "probes_baseline.json")))
        core.summarize(base["harmful"], base["harmless"])  # baseline sanity: file parses + rows keyed
        s = core.summarize(r_h, r_b)
        s["banked_resume"] = True
        s["banked_provenance"] = ("verified" if banked_prov is not None
                                  and provenance is not None
                                  else "unverified")
        s["edit_info"] = {"banked_resume": "probes restored from prior "
                          "session; edit+save+verify skipped"}
        s["on_disk_verify"] = {"banked_resume": "variant dir from disk"
                               if _os.path.isdir(core.variant_dir(spec, name))
                               else "PROBES_ONLY (variant dir absent)"}
        s["tie_flag_on_disk"] = None
        return s
    except Exception as e:  # corrupt/incomplete -> re-run normally
        print(f"      banked_resume skip {name}: {e}", flush=True)
        return None


_VARIANT_WD_FLAVOR = {"wd_B": "untie_head", "wd_BN": "untie_head",
                      "wd_ML": "keep_tie", "wd_ML_BN": "untie_head"}


def expect_tied_for(name, model=None):
    """Tie expectation for a variant ON THIS PATIENT (BUG-2 fix shape):
    delegated to the modifier registry's flavor_for (single source of
    truth — the registry table is authoritative; this wrapper keeps the
    historical call-site signature)."""
    from .modifiers import flavor_for
    return flavor_for(name, model)


def run_ladder(spec, ctx):
    """Stage B: the ladder from spec.ladder.variants over v2 stage-A
    artifacts. Returns the LADDER_DONE payload dict."""
    out_dir = ctx["out_dir"]
    lad = spec["ladder"]
    variants = lad["variants"]

    lc = json.load(open(os.path.join(out_dir, "layer_coherence.json")))
    dirs_np = np.load(os.path.join(out_dir, "layer_directions.npz"))
    dirs_all = dirs_np["directions"]
    dir_A = torch.from_numpy(
        np.load(os.path.join(out_dir, "refusal_direction_A.npy"))).float()
    dir_B = torch.from_numpy(
        np.load(os.path.join(out_dir, "refusal_direction_B.npy"))).float()
    base = json.load(open(os.path.join(out_dir, "probes_baseline.json")))
    base_sum = core.summarize(base["harmful"], base["harmless"])
    base_pres = base_sum["benign_preserved"]
    table = sorted(lc["table"], key=lambda r: -r["coherence"])
    k_layers_primary = [r["decoder_layer"] for r in table[:lad["k_primary"]]]
    k_layers_combo = [r["decoder_layer"] for r in table[:lad["k_combo"]]]
    print(f"[1/6] ladder plan: L*={lc['best']['decoder_layer']} "
          f"coh={lc['best']['coherence']} K_primary={k_layers_primary} "
          f"K_combo={k_layers_combo} "
          f"baseline_refusal={base_sum['refusal_rate']} "
          f"benign={base_pres}", flush=True)

    VAR_DIRS = {v: core.variant_dir(spec, v) for v in variants}

    # v1 amendment (FTT-28): ara_<rank> variants route to the ARA optimizer
    # (src/abliteration_engine/ara.py). Resolved here so the ladder log
    # carries the config; runs on the standard lifecycle AFTER wd_ML_BN
    # (the combo stays the max-intervention closer).
    ara_name, ara_cfg = None, None
    if any(v.startswith("ara_") for v in variants):
        from . import ara as ara_mod
        from .spec import SpecError
        try:
            ara_name, ara_cfg = ara_mod.resolve_ara_config(lad)
        except ValueError as e:
            raise SpecError(f"ladder.ara: {e}") from e
        print(f"      ARA variant {ara_name}: rank={ara_cfg['rank']} "
              f"layers={ara_cfg.get('layers') or 'ALL'} "
              f"(L-BFGS x{ara_cfg['steps']} @ lr {ara_cfg['lr']}, "
              f"pools {ara_cfg['good']} / {ara_cfg['bad']})", flush=True)

    def edit_wd_B(m):
        mc, rn = orthogonalize_lm_head(m, dir_B)
        return {"lm_head_max|W r_B|": mc, "r_norm": rn}

    def edit_wd_BN(m):
        mc, rn = orthogonalize_lm_head(m, dir_B)
        comp, resid = orthogonalize_final_norm(m, dir_B)
        return {"lm_head_max|W r_B|": mc, "norm_wdB_before": comp,
                "norm_resid_fp32": resid}

    def make_edit_wd_ML(k_layers):
        def _edit(m):
            info = {}
            for l in k_layers:
                info[f"L{l}"] = orthogonalize_layer_output(
                    m, l, torch.from_numpy(dirs_all[l]).float())
            return {"layers": k_layers, "resids": info}
        return _edit

    def edit_wd_ML_BN(m):
        info = {"layers": k_layers_combo, "resids": {}}
        for l in k_layers_combo:
            info["resids"][f"L{l}"] = orthogonalize_layer_output(
                m, l, torch.from_numpy(dirs_all[l]).float())
        mc, rn = orthogonalize_lm_head(m, dir_B)
        comp, resid = orthogonalize_final_norm(m, dir_B)
        info["lm_head_max|W r_B|"] = mc
        info["norm_wdB_before"] = comp
        info["norm_resid_fp32"] = resid
        return info

    bounds = spec["publish"].get("verify_disk_bounds", {})
    b_lm = bounds.get("lm_head", 5e-3)
    b_norm = bounds.get("final_norm", 5e-2)
    b_row = bounds.get("layer_row", 1e-2)

    def vfy_wd_B(mr):
        return {"lm_head": verify_lm_head_disk(mr, dir_B, b_lm)}

    def vfy_wd_BN(mr):
        return {"lm_head": verify_lm_head_disk(mr, dir_B, b_lm),
                "final_norm": verify_final_norm_disk(mr, dir_B, b_norm)}

    def vfy_wd_ML(mr):
        return {"layers": verify_layers_disk(mr, k_layers_primary, dirs_all,
                                             b_row)}

    def vfy_wd_ML_BN(mr):
        return {"lm_head": verify_lm_head_disk(mr, dir_B, b_lm),
                "final_norm": verify_final_norm_disk(mr, dir_B, b_norm),
                "layers": verify_layers_disk(mr, k_layers_combo, dirs_all,
                                             b_row)}

    edit_fns = {"wd_B": edit_wd_B, "wd_BN": edit_wd_BN}
    edit_fns["wd_ML"] = make_edit_wd_ML(k_layers_primary)
    edit_fns["wd_ML_BN"] = edit_wd_ML_BN
    verify_fns = {"wd_B": vfy_wd_B, "wd_BN": vfy_wd_BN, "wd_ML": vfy_wd_ML,
                  "wd_ML_BN": vfy_wd_ML_BN}
    # (tie expectations are name-keyed via expect_tied_for(); ara_* keeps
    # the tie — decoder-layer edits only)

    # per-variant provenance: what each variant's probes depend on
    _k = {"wd_B": [], "wd_BN": [], "wd_ML": k_layers_primary,
          "wd_ML_BN": k_layers_combo}
    _arrs = {"wd_B": {"dir_B": dir_B.numpy()},
             "wd_BN": {"dir_B": dir_B.numpy()},
             "wd_ML": {"dirs": dirs_all},
             "wd_ML_BN": {"dirs": dirs_all, "dir_B": dir_B.numpy()}}
    prov = {v: variant_provenance(spec, v, _k.get(v, []), _arrs.get(v, {}))
            for v in variants}

    summ = {}
    step = 2
    for name in variants:
        if name == "wd_ML_BN" or name.startswith("ara_"):
            continue  # conditional combos + ARA, after the others
        print(f"[{step}/6] {name}", flush=True)
        banked = _banked_variant_summary(spec, name, prov[name])
        if banked is not None:
            print(f"      {name}: BANKED RESUME (complete prior-session "
                  f"probes reused; refusal={banked['refusal_rate']} "
                  f"benign={banked['benign_preserved']} "
                  f"degenerate={banked['degenerate_total']})", flush=True)
            summ[name] = banked
            step += 1
            continue
        summ[name] = run_variant(name, edit_fns[name], VAR_DIRS[name],
                                 None,  # derived from the loaded base
                                 verify_fns[name], spec, ctx["model"],
                                 provenance=prov[name])
        step += 1

    # conditional max-intervention combo per v2 semantics
    if "wd_ML_BN" in variants:
        others = [v for v in summ if v != "wd_ML_BN"]
        assert others, ("wd_ML_BN combo with no preceding variant results "
                        "(spec validation guarantees wd_ML precedes it)")
        best3 = min(summ[v]["refusal_rate"] for v in others)
        if best3 >= spec["gates"]["publish_refusal"]:
            print(f"[{step}/6] wd_ML_BN: best of V1..V3 = {best3} >= "
                  f"{spec['gates']['publish_refusal']} -> running combo",
                  flush=True)
            summ["wd_ML_BN"] = run_variant(
                "wd_ML_BN", edit_fns["wd_ML_BN"], VAR_DIRS["wd_ML_BN"],
                None,  # derived from the loaded base
                verify_fns["wd_ML_BN"], spec, ctx["model"],
                provenance=prov["wd_ML_BN"])
        else:
            print(f"[{step}/6] wd_ML_BN skipped: best of V1..V3 = {best3} "
                  f"< {spec['gates']['publish_refusal']}", flush=True)
        step += 1

    # ara_<rank> variant (FTT-28): the ARA optimizer runs AFTER the wd_*
    # variants (and the conditional wd_ML_BN above) — every other lifecycle
    # (banked resume, tie handling, selection) treats it like any variant.
    if ara_name is not None and ara_name in variants:
        print(f"[{step}/6] {ara_name} (ARA optimizer)", flush=True)
        banked = _banked_variant_summary(spec, ara_name, prov[ara_name])
        if banked is not None:
            print(f"      {ara_name}: BANKED RESUME (complete prior-session "
                  f"probes reused; refusal={banked['refusal_rate']} "
                  f"benign={banked['benign_preserved']} "
                  f"degenerate={banked['degenerate_total']})", flush=True)
            summ[ara_name] = banked
        else:
            summ[ara_name] = ara_mod.run_ara_variant(
                spec, ara_name, ara_cfg, provenance=prov[ara_name])
        step += 1

    print(f"[{step + 1}/6] selection + artifacts", flush=True)
    cands = []
    for idx, name in enumerate(variants):
        if name not in summ:
            continue
        s = summ[name]
        cands.append({"variant": name, "ladder_index": idx,
                      "refusal_rate": s["refusal_rate"],
                      "benign_preserved": s["benign_preserved"],
                      "degenerate_total": s["degenerate_total"]})
    selected, eligible = select_variant(
        cands, base_pres, spec["gates"]["publish_refusal"],
        floor_delta=spec["gates"]["benign_floor_delta"],
        degenerate_max=spec["gates"].get("degenerate_max", 0))
    print(f"      selection: {selected['variant']} "
          f"(gate={'passed' if selected['passes_gate'] else 'failed'}, "
          f"publish_eligible={eligible})", flush=True)

    sel_dir = VAR_DIRS[selected["variant"]]
    # representative direction for hub aux file: readout-space dir_B for
    # lm_head-bearing variants, residual-space dir_A otherwise (ARA edits
    # decoder-layer matrices only -> residual-space dir_A)
    lm_head_variants = ("wd_B", "wd_BN", "wd_ML_BN")
    rep = dir_B if selected["variant"] in lm_head_variants else dir_A
    np.save(os.path.join(out_dir, "refusal_direction.npy"), rep.numpy())
    json.dump(cands, open(os.path.join(out_dir, "selection_candidates.json"),
                          "w"), indent=2)
    # selection payload built as a dict so amendment blocks can attach
    # provenance before the single json.dump (FTT-28 ladder_ara below).
    sel_payload = {"selected": selected["variant"],
                   "gate": "passed" if selected["passes_gate"] else "failed",
                   "selected_variant_dir": sel_dir,
                   "selected_variant_banked_resume":
                       bool(selected["variant"] in summ
                            and summ[selected["variant"]].get("banked_resume")),
                   "banked_resume_variants":
                       [v for v, s in summ.items() if s.get("banked_resume")],
                   # what produced the selected variant's weights/probes;
                   # mmlu.py keys its banked-results reuse on this
                   "selected_provenance": {
                       **prov[selected["variant"]],
                       # False when the winner's probes were reused from a
                       # pre-fingerprint banked file: the fingerprint is
                       # what the run expected, not what produced them
                       "verified": summ[selected["variant"]].get(
                           "banked_provenance", "verified") == "verified"},
                   "publish_eligible_probe_gate": eligible,
                   "publish_refusal_threshold":
                       spec["gates"]["publish_refusal"],
                   "k_layers_primary": k_layers_primary,
                   "k_layers_combo": k_layers_combo,
                   "metrics": {
                       "refusal_rate_before": base_sum["refusal_rate"],
                       "refusal_rate_after": selected["refusal_rate"],
                       "benign_preserved_before": base_pres,
                       "benign_preserved_after":
                           selected["benign_preserved"]}}
    # ARA provenance into selection.json (FTT-28): the effective optimizer
    # config + the layers actually edited. edited-layers list comes from the
    # run summary unless the variant was banked-resumed (then the spec's
    # configured list is all we have — recorded verbatim).
    if ara_name is not None and ara_name in summ:
        from . import ara as ara_mod
        per_layer = (summ[ara_name].get("edit_info") or {}).get("per_layer")
        edited = (sorted(int(k[1:]) for k in per_layer
                         if k.startswith("L"))
                  if per_layer
                  else (ara_cfg.get("layers") or []))
        sel_payload["ladder_ara"] = ara_mod.ladder_meta(ara_name, ara_cfg,
                                                        edited)
    json.dump(sel_payload, open(os.path.join(out_dir, "selection.json"),
                                "w"), indent=2)

    for name, d in VAR_DIRS.items():
        if name != selected["variant"] and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"      removed non-selected {name} dir", flush=True)

    return {"selected": selected["variant"],
            "gate": "passed" if selected["passes_gate"] else "failed",
            "publish_eligible_probe_gate": eligible,
            "metrics": {
                "refusal_rate_before": base_sum["refusal_rate"],
                "refusal_rate_after": selected["refusal_rate"],
                "benign_preserved_before": base_pres,
                "benign_preserved_after": selected["benign_preserved"]},
            "variants": {k: {"refusal_rate": v["refusal_rate"],
                             "benign_preserved": v["benign_preserved"],
                             "degenerate_total": v["degenerate_total"]}
                         for k, v in summ.items()}}