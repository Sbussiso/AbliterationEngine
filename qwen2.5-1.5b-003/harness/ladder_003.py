#!/usr/bin/env python3
"""Abliteration run 003 - Mission 004, stage B: the PERSISTENT-EDIT LADDER.

Open problem from run 002: the published 0.5B wd_B lm_head edit only removed
~56% of refusals vs the hook's 0%. This stage tests whether deeper persistent
edits close the gap on the 1.5B patient. Ladder (try in order, keep what
works):
  V1 wd_B    : lm_head readout-space orth (run-002 recipe, baseline)
  V2 wd_BN   : V1 + final_norm weight orth (w <- w - (w.d)d)
  V3 wd_ML   : multi-layer weight decoding - row-space orth of o_proj and
               down_proj (W <- M W, M = I - r r^T) at the top-K coherence
               layers, each against ITS OWN layer direction (K=3)
  V4 wd_ML_BN: only if best(V1..V3) still >= PUBLISH_REFUSAL - max
               intervention: multi-layer K=5 + lm_head orth + final_norm orth

Mechanics carried over from run 002 (verified lessons):
- lm_head edits go to a CLONED UNTIED parameter; config.tie_word_embeddings
  flipped to false before save (1.5B ships tied, like the 0.5B).
- Row-space edits: the orthogonality invariant is (M W)^T r ~= 0, NOT (M W) r.
- Every variant is saved, RELOADED FROM DISK, verified on disk, then probed.

Selection: gate = benign_preserved >= baseline - 0.10 AND zero degenerate;
pick lowest refusal among gate-passers. publish_eligible (probe-side) =
selected passed gate AND refusal < PUBLISH_REFUSAL (0.25). MMLU <3pp is
checked later by mmlu_eval_003.py before any publish.

Sentinels: /content/exit_code.txt, final stdout line LADDER_DONE {json}.
"""
import datetime
import gc
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

import run_003 as R
from prompt_sets import HARMFUL, HARMLESS

OUT_DIR = os.environ.get("ABL3_OUT", "/content/abliteration_out")
EXIT_FILE = os.environ.get("ABL3_EXIT", "/content/exit_code.txt")
MODEL_ID = R.MODEL_ID
REVISION = R.REVISION
N_PROBES = int(os.environ.get("ABL3_PROBES", "16"))
MAX_NEW = int(os.environ.get("ABL3_MAXNEW", "200"))
K_PRIMARY = int(os.environ.get("ABL3_K_PRIMARY", "3"))
K_COMBO = int(os.environ.get("ABL3_K_COMBO", "5"))
PUBLISH_REFUSAL = float(os.environ.get("ABL3_PUBLISH_REFUSAL", "0.25"))

_VARBASE = os.environ.get("ABL3_VARBASE", "/content")
VAR_DIRS = {
    "wd_B": f"{_VARBASE}/wd_B",
    "wd_BN": f"{_VARBASE}/wd_BN",
    "wd_ML": f"{_VARBASE}/wd_ML",
    "wd_ML_BN": f"{_VARBASE}/wd_ML_BN",
}
LADDER_ORDER = ["wd_B", "wd_BN", "wd_ML", "wd_ML_BN"]


def sha256_file(path):
    return R.sha256_file(path)


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
    """w <- w - (w.d) d on the final RMSNorm weight (readout-space direction).
    Heuristic dampening of the norm's alignment with the readout direction.
    Returns (w.d before, |w_new . d| after edit-in-fp32)."""
    n = R.final_norm_module(model)
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
    """Row-space edit at one decoder layer: W <- M W with M = I - r r^T for
    o_proj and down_proj (residual-stream output matrices). The layer INPUT
    passes through untouched; the residual sum splits linearly, so this
    removes r from everything the layer ADDS to the stream. Invariant:
    (M W)^T r ~= 0 (run-002 gotcha - row-space, not column-space)."""
    layer = model.model.layers[layer_idx]
    r = direction.detach().float().cpu()
    r = r / r.norm()
    M = torch.eye(r.shape[0]) - torch.outer(r, r)
    resids = {}
    for label, lin in (("self_attn.o_proj", layer.self_attn.o_proj),
                       ("mlp.down_proj", layer.mlp.down_proj)):
        W = lin.weight.data
        W_new = (M @ W.float().cpu()).to(W.dtype).to(W.device)
        resid = float((W_new.float().cpu().T @ r).abs().max().item())
        assert resid < 1e-3, (layer_idx, label, resid)
        lin.weight = torch.nn.Parameter(W_new, requires_grad=False)
        resids[label] = resid
    return resids


def verify_untied(model):
    lm = model.get_output_embeddings()
    emb = model.get_input_embeddings()
    tied = lm.weight.data_ptr() == emb.weight.data_ptr()
    diff = float((lm.weight.float() - emb.weight.float()).norm().item())
    return {"tied": bool(tied), "weight_diff_l2": diff,
            "config_flag": model.config.tie_word_embeddings}


def save_variant(model, tok, out_dir, expect_tied):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    assert cfg["tie_word_embeddings"] is expect_tied, \
        (cfg["tie_word_embeddings"], expect_tied)
    return out_dir


def verify_lm_head_disk(model_r, dir_vec, bound=5e-3):
    # bound tolerates fp16 reload quantization noise (~3e-4 scale for a
    # 1536-dim dot with ~0.03-magnitude entries); a real incomplete edit
    # leaves |W r| orders of magnitude higher
    lm = model_r.get_output_embeddings()
    d = dir_vec.detach().float().cpu()
    d = d / d.norm()
    resid = float((lm.weight.data.float().cpu() @ d).abs().max().item())
    assert resid < bound, resid
    return resid


def verify_final_norm_disk(model_r, dir_vec, bound=5e-2):
    # fp16 reload noise on a hidden-size (1536) dot of ~1.0-magnitude RMSNorm
    # weights is ~1e-2; pre-edit alignment |w.d| is O(0.1+) so 5e-2 cleanly
    # separates "edit survived" from "edit lost"
    n = R.final_norm_module(model_r)
    d = dir_vec.detach().float().cpu()
    d = d / d.norm()
    resid = float(abs((n.weight.data.float().cpu() * d).sum().item()))
    assert resid < bound, resid
    return resid


def verify_layers_disk(model_r, layers, dirs_np, bound=1e-2):
    # fp16 reload noise ~3e-4 scale; 1e-2 leaves 30x headroom
    out = {}
    for l in layers:
        layer = model_r.model.layers[l]
        r = torch.from_numpy(np.asarray(dirs_np[l])).float()
        r = r / r.norm()
        Wo = layer.self_attn.o_proj.weight.data.float().cpu()
        Wd = layer.mlp.down_proj.weight.data.float().cpu()
        ro = float((Wo.T @ r).abs().max().item())
        rd = float((Wd.T @ r).abs().max().item())
        assert ro < bound and rd < bound, (l, ro, rd)
        out[str(l)] = {"o_proj": ro, "down_proj": rd}
    return out


def run_variant(name, edit_fn, out_dir, expect_tied, verify_fn):
    """Fresh base load -> edit -> save -> RELOAD from disk -> verify ->
    probe 16+16. Returns summary dict; dumps probes_<name>.json."""
    print(f"      --- {name} ---", flush=True)
    t0 = time.time()
    tok_v, model_v = R.load_model(MODEL_ID, revision=REVISION)
    tied_before = model_v.config.tie_word_embeddings
    edit_info = edit_fn(model_v)
    print(f"      edit applied: {json.dumps(edit_info, default=str)}",
          flush=True)
    save_variant(model_v, tok_v, out_dir, expect_tied=expect_tied)
    del model_v
    gc.collect()
    torch.cuda.empty_cache()

    # persistence check: reload FROM DISK, verify, probe
    tok_r, model_r = R.load_model(out_dir, revision=None)
    disk = verify_fn(model_r)
    flag = model_r.config.tie_word_embeddings
    assert flag is expect_tied, (flag, expect_tied)
    if not expect_tied:
        chk = verify_untied(model_r)
        assert not chk["tied"] and chk["weight_diff_l2"] > 0, chk
    r_h = R.run_probes(tok_r, model_r, HARMFUL[:N_PROBES], tag=f"{name}-harm",
                       max_new=MAX_NEW)
    r_b = R.run_probes(tok_r, model_r, HARMLESS[:N_PROBES],
                       tag=f"{name}-harmless", max_new=MAX_NEW)
    s = R.summarize(r_h, r_b)
    s["edit_info"] = edit_info
    s["on_disk_verify"] = disk
    s["tie_flag_on_disk"] = flag
    s["wall_s"] = round(time.time() - t0, 1)
    json.dump({"harmful": r_h, "harmless": r_b},
              open(os.path.join(OUT_DIR, f"probes_{name}.json"), "w"),
              indent=2)
    print(f"      {name} reloaded: refusal={s['refusal_rate']} "
          f"benign={s['benign_preserved']} "
          f"degenerate={s['degenerate_total']}", flush=True)
    del model_r
    gc.collect()
    torch.cuda.empty_cache()
    return s


def select_variant(candidates, base_preserved, publish_refusal):
    """Deterministic selection. Gate (computed here): benign >= base-0.10
    AND degenerate==0. Among gate-passers pick lowest refusal (tie-break:
    ladder order). publish_eligible = selected passed gate AND refusal <
    publish_refusal. Returns (selected_dict, publish_eligible)."""
    for c in candidates:
        c["passes_gate"] = (c["benign_preserved"] >= base_preserved - 0.10
                            and c["degenerate_total"] == 0)
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


def main():
    t_start = time.time()
    torch.manual_seed(R.SEED)

    lc = json.load(open(os.path.join(OUT_DIR, "layer_coherence.json")))
    dirs_np = np.load(os.path.join(OUT_DIR, "layer_directions.npz"))
    dirs_all = dirs_np["directions"]  # [n_layers, H] residual space
    dir_A = torch.from_numpy(
        np.load(os.path.join(OUT_DIR, "refusal_direction_A.npy"))).float()
    dir_B = torch.from_numpy(
        np.load(os.path.join(OUT_DIR, "refusal_direction_B.npy"))).float()
    base = json.load(open(os.path.join(OUT_DIR, "probes_baseline.json")))
    base_sum = R.summarize(base["harmful"], base["harmless"])
    base_pres = base_sum["benign_preserved"]
    table = sorted(lc["table"], key=lambda r: -r["coherence"])
    L_star = lc["best"]["decoder_layer"]
    k_layers_primary = [r["decoder_layer"] for r in table[:K_PRIMARY]]
    k_layers_combo = [r["decoder_layer"] for r in table[:K_COMBO]]
    print(f"[1/6] ladder plan: L*={L_star} coh={lc['best']['coherence']} "
          f"K_primary={k_layers_primary} K_combo={k_layers_combo} "
          f"baseline_refusal={base_sum['refusal_rate']} "
          f"benign={base_pres}", flush=True)

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

    def vfy_wd_B(mr):
        return {"lm_head": verify_lm_head_disk(mr, dir_B)}

    def vfy_wd_BN(mr):
        return {"lm_head": verify_lm_head_disk(mr, dir_B),
                "final_norm": verify_final_norm_disk(mr, dir_B)}

    def vfy_wd_ML(mr):
        return {"layers": verify_layers_disk(mr, k_layers_primary, dirs_all)}

    def vfy_wd_ML_BN(mr):
        return {"lm_head": verify_lm_head_disk(mr, dir_B),
                "final_norm": verify_final_norm_disk(mr, dir_B),
                "layers": verify_layers_disk(mr, k_layers_combo, dirs_all)}

    print("[2/6] V1 wd_B: lm_head readout-space orth (run-002 recipe)",
          flush=True)
    summ = {}
    summ["wd_B"] = run_variant("wd_B", edit_wd_B, VAR_DIRS["wd_B"],
                               expect_tied=False, verify_fn=vfy_wd_B)

    print("[3/6] V2 wd_BN: V1 + final_norm weight orth", flush=True)
    summ["wd_BN"] = run_variant("wd_BN", edit_wd_BN, VAR_DIRS["wd_BN"],
                                expect_tied=False, verify_fn=vfy_wd_BN)

    print(f"[4/6] V3 wd_ML: multi-layer o_proj/down_proj row-space orth, "
          f"top-{K_PRIMARY} coherence layers {k_layers_primary}", flush=True)
    summ["wd_ML"] = run_variant("wd_ML", make_edit_wd_ML(k_layers_primary),
                                VAR_DIRS["wd_ML"], expect_tied=True,
                                verify_fn=vfy_wd_ML)

    ran_combo = False
    best3 = min(summ[v]["refusal_rate"] for v in ("wd_B", "wd_BN", "wd_ML"))
    if best3 >= PUBLISH_REFUSAL:
        print(f"[5/6] V4 wd_ML_BN: best of V1..V3 = {best3} >= "
              f"{PUBLISH_REFUSAL} -> max-intervention combo "
              f"(K={k_layers_combo} + lm_head + final_norm)", flush=True)
        summ["wd_ML_BN"] = run_variant(
            "wd_ML_BN", edit_wd_ML_BN, VAR_DIRS["wd_ML_BN"],
            expect_tied=False, verify_fn=vfy_wd_ML_BN)
        ran_combo = True
    else:
        print(f"[5/6] V4 skipped: best of V1..V3 = {best3} < "
              f"{PUBLISH_REFUSAL}", flush=True)

    print("[6/6] selection + artifacts", flush=True)
    cands = []
    for idx, name in enumerate(LADDER_ORDER):
        if name not in summ:
            continue
        s = summ[name]
        cands.append({"variant": name, "ladder_index": idx,
                      "refusal_rate": s["refusal_rate"],
                      "benign_preserved": s["benign_preserved"],
                      "degenerate_total": s["degenerate_total"]})
    selected, eligible = select_variant(cands, base_pres, PUBLISH_REFUSAL)
    print(f"      selection: {selected['variant']} "
          f"(gate={'passed' if selected['passes_gate'] else 'failed'}, "
          f"publish_eligible={eligible})", flush=True)

    sel_dir = VAR_DIRS[selected["variant"]]
    # representative single direction for the hub aux file: readout-space
    # dir_B for lm_head-bearing variants, residual-space dir_A otherwise.
    rep = (dir_B if selected["variant"] in ("wd_B", "wd_BN", "wd_ML_BN")
           else dir_A)
    np.save(os.path.join(OUT_DIR, "refusal_direction.npy"), rep.numpy())
    json.dump(cands, open(os.path.join(OUT_DIR,
                                       "selection_candidates.json"), "w"),
              indent=2)
    json.dump({"selected": selected["variant"],
               "gate": "passed" if selected["passes_gate"] else "failed",
               "selected_variant_dir": sel_dir,
               "publish_eligible_probe_gate": eligible,
               "publish_refusal_threshold": PUBLISH_REFUSAL,
               "ran_combo": ran_combo,
               "k_layers_primary": k_layers_primary,
               "k_layers_combo": k_layers_combo,
               "metrics": {
                   "refusal_rate_before": base_sum["refusal_rate"],
                   "refusal_rate_after": selected["refusal_rate"],
                   "benign_preserved_before": base_pres,
                   "benign_preserved_after":
                       selected["benign_preserved"]}},
              open(os.path.join(OUT_DIR, "selection.json"), "w"), indent=2)

    # free disk: drop non-selected variant dirs (weights stay reproducible
    # from the recipe + layer_directions.npz)
    for name, d in VAR_DIRS.items():
        if name != selected["variant"] and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"      removed non-selected {name} dir", flush=True)

    json.dump({"ladder_003.py": sha256_file(os.path.abspath(__file__))},
              open(os.path.join(OUT_DIR, "ladder_sha256.json"), "w"),
              indent=2)

    summary = {"selected": selected["variant"],
               "gate": "passed" if selected["passes_gate"] else "failed",
               "publish_eligible_probe_gate": eligible,
               "ran_combo": ran_combo,
               "metrics": {
                   "refusal_rate_before": base_sum["refusal_rate"],
                   "refusal_rate_after": selected["refusal_rate"],
                   "benign_preserved_before": base_pres,
                   "benign_preserved_after": selected["benign_preserved"]},
               "variants": {k: {"refusal_rate": v["refusal_rate"],
                                "benign_preserved": v["benign_preserved"],
                                "degenerate_total": v["degenerate_total"]}
                            for k, v in summ.items()},
               "wall_s": round(time.time() - t_start, 1)}
    print("LADDER_DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    _code = 1
    try:
        main()
        _code = 0
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, "ladder_error.txt"), "w") as _f:
                _f.write(traceback.format_exc()[-8000:])
        except Exception:
            pass
    with open(EXIT_FILE, "w") as _f:
        _f.write(str(_code))