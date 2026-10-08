"""`abliterate search` — honest multi-objective edit search (ENGINE-V0.4 §1.3).

What it does, end to end:
  1. stage-A artifacts (directions, coherence table) — run first if missing;
  2. load the base model ONCE, measure it (search-eval refusal + benign,
     answer trajectories for the KL term, sealed-holdout refusal);
  3. trials: apply a search point IN MEMORY (rank-1 partial projections,
     restored after each trial), score it, bank it in a resumable SQLite
     study (a dead Colab session resumes at the next trial);
  4. keep the Pareto front over three objectives instead of one blended
     score, pick a point with a stated policy, and print the trade-off;
  5. materialize the pick on a fresh base, save, reload FROM DISK, verify
     the edit algebraically, probe it, certify it on the sealed holdout,
     and write selection.json so `mmlu` and `publish` work unchanged.

Design choices (deliberately not Heretic's — that tool searches a per-layer
weight-kernel shape and a fractional direction index, optimizing refusal
count + first-token KL on the prompts it is scored on):
  * search space: a contiguous layer WINDOW; one partial-projection strength
    alpha for the window (W <- W - alpha r r^T W; alpha 1 = full
    orthogonalization, > 1 over-projects); the direction SOURCE per layer —
    its own difference-of-means, the single best layer's, or a
    coherence-weighted blend of the top-k coherent layers (our coherence
    scan); which matrices (attention, MLP, both); optional readout edit of
    the output head against the readout-space direction.
  * objectives (all minimized, never blended): harmful refusal and benign
    refusal on the SEARCH split, graded by the engine's grader
    (probe_sets.marker_mode v1/v2 — v2 catches "I can't... here's how"
    compliance), and ANSWER-TRAJECTORY KL: KL(base || candidate) averaged
    over the base model's own greedy answer tokens on benign prompts — drift
    across the answer, not just its first token.
  * honesty: directions come from the probe pairs, trials are scored on a
    disjoint search split, and the certificate is measured on a THIRD,
    sealed holdout split the optimizer never sees (honest.HoldoutVault;
    disjointness asserted).
  * the optimizer is Optuna's multi-objective TPE sampler (MIT library;
    spec §1.3), seeded; storage is an SQLite file in the run dir.
"""
import hashlib
import json
import os
import time

import numpy as np

SEARCH_DEFAULTS = {
    "trials": 40,
    "seed": 0,
    "eval_prompts": 24,        # per side (harmful / benign), search split
    "holdout_prompts": 32,     # sealed, certificate only
    "max_new_tokens": 48,      # refusals show in the first sentence
    "kl_tokens": 8,            # base-answer positions in the KL term
    "batch_size": 8,
    "harmful_pool": "builtin:ara_bad",    # disjoint from the probe sets
    "benign_pool": "builtin:ara_good",
    "max_kl": None,            # selection cap (None = no cap)
    "space": {
        "alpha": [0.25, 1.2],
        "components": ["attn", "mlp", "both"],
        "direction_modes": ["own", "best", "blend"],
        "readout": [False, True],
        "blend_k": [2, 5],
    },
}
_COMPONENTS = ("attn", "mlp", "both")
_MODES = ("own", "best", "blend")
_ATTN, _MLP = "self_attn.o_proj", "mlp.down_proj"


# ---- config (CPU) ---------------------------------------------------------
def resolve_config(spec):
    """Normalized search config: spec `search:` block over the defaults.
    Raises ValueError with the offending key on anything malformed."""
    raw = spec.get("search") or {}
    if not isinstance(raw, dict):
        raise ValueError("search must be a mapping")
    unknown = sorted(set(raw) - set(SEARCH_DEFAULTS))
    if unknown:
        raise ValueError(f"search: unknown key(s) {unknown}")
    cfg = {k: v for k, v in SEARCH_DEFAULTS.items() if k != "space"}
    cfg.update({k: v for k, v in raw.items() if k != "space"})
    space = dict(SEARCH_DEFAULTS["space"])
    rs = raw.get("space") or {}
    if not isinstance(rs, dict):
        raise ValueError("search.space must be a mapping")
    bad = sorted(set(rs) - set(space))
    if bad:
        raise ValueError(f"search.space: unknown key(s) {bad}")
    space.update(rs)
    cfg["space"] = space
    for key in ("trials", "eval_prompts", "holdout_prompts",
                "max_new_tokens", "kl_tokens", "batch_size"):
        v = cfg[key]
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"search.{key} must be a positive int")
    if cfg["holdout_prompts"] < 8:
        raise ValueError("search.holdout_prompts must be >= 8 (noise-level "
                         "certificate rates)")
    lo, hi = space["alpha"]
    if not (0 < lo <= hi <= 2.0):
        raise ValueError("search.space.alpha must be [lo, hi] with "
                         "0 < lo <= hi <= 2")
    for key, allowed in (("components", _COMPONENTS),
                         ("direction_modes", _MODES)):
        vals = space[key]
        if not vals or any(v not in allowed for v in vals):
            raise ValueError(f"search.space.{key} must be a non-empty subset "
                             f"of {list(allowed)}")
    if not space["readout"] or any(not isinstance(v, bool)
                                   for v in space["readout"]):
        raise ValueError("search.space.readout must list true and/or false")
    k_lo, k_hi = space["blend_k"]
    if not (2 <= k_lo <= k_hi):
        raise ValueError("search.space.blend_k must be [lo, hi] with lo >= 2")
    if cfg["max_kl"] is not None and not (
            isinstance(cfg["max_kl"], (int, float)) and cfg["max_kl"] > 0):
        raise ValueError("search.max_kl must be a positive number or null")
    return cfg


# ---- search points (CPU) ------------------------------------------------------
def suggest_point(trial, cfg, n_layers):
    """One search point from an Optuna trial (or any object with the
    suggest_* API)."""
    sp = cfg["space"]
    start = trial.suggest_int("start", 0, n_layers - 1)
    width = trial.suggest_int("width", 1, n_layers)
    point = {
        "start": start,
        "end": min(n_layers, start + width),
        "alpha": trial.suggest_float("alpha", *sp["alpha"]),
        "components": trial.suggest_categorical("components",
                                                list(sp["components"])),
        "direction_mode": trial.suggest_categorical(
            "direction_mode", list(sp["direction_modes"])),
        "readout": trial.suggest_categorical("readout", list(sp["readout"])),
    }
    if point["direction_mode"] == "blend":
        point["blend_k"] = trial.suggest_int("blend_k", *sp["blend_k"])
    return point


def describe(point):
    parts = [f"layers {point['start']}–{point['end'] - 1}",
             f"α {point['alpha']:.2f}", point["direction_mode"]
             + (f"(k={point['blend_k']})" if point["direction_mode"] == "blend"
                else ""), point["components"]]
    if point["readout"]:
        parts.append("+readout")
    return " · ".join(parts)


def labels_for(components, available):
    want = {"attn": (_ATTN,), "mlp": (_MLP,), "both": (_ATTN, _MLP)}[
        components]
    return tuple(lab for lab in want if lab in available)


def layer_directions(point, dirs_all, table):
    """{layer: unit direction (np.float32)} for the point's window."""
    dirs_all = np.asarray(dirs_all, dtype=np.float32)

    def unit(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    mode = point["direction_mode"]
    ranked = sorted(table, key=lambda r: -r["coherence"])
    if mode == "best":
        shared = unit(dirs_all[ranked[0]["decoder_layer"]])
    elif mode == "blend":
        top = ranked[:point["blend_k"]]
        shared = unit(sum(r["coherence"] * unit(dirs_all[r["decoder_layer"]])
                          for r in top))
    else:
        shared = None
    return {li: (shared if shared is not None else unit(dirs_all[li]))
            for li in range(point["start"], point["end"])}


def point_fingerprint(point, arrays_sha, spec):
    blob = json.dumps({"point": point, "arrays": arrays_sha,
                       "patient": [spec["patient"]["model_id"],
                                   spec["patient"]["revision"]]},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ---- Pareto + selection (CPU) ------------------------------------------------
OBJECTIVES = ("refusal", "benign_refusal", "kl")


def pareto_front(rows):
    """Rows not dominated on (refusal, benign_refusal, kl), minimized."""
    front = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if all(b[k] <= a[k] for k in OBJECTIVES) and any(
                    b[k] < a[k] for k in OBJECTIVES):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def select(rows, base, gates, max_kl=None):
    """The stated selection policy over the Pareto front:
    feasible = benign refusal within the gate floor of the base, no
    degenerate outputs, KL under max_kl (when set); pick the lowest harmful
    refusal, ties -> lowest KL. Returns (row, feasible_flag, reason)."""
    done = [r for r in rows if r.get("ok", True)]
    if not done:
        raise ValueError("search produced no completed trials")
    # degenerate output is a hard constraint: filter BEFORE the front, or a
    # degenerate trial can dominate (and hide) a usable one. The benign
    # floor and KL cap cannot hide anything — a point over either limit can
    # never dominate one under it.
    clean = [r for r in done
             if r.get("degenerate", 0) <= gates.get("degenerate_max", 0)]
    front = pareto_front(clean or done)
    floor = base["benign_refusal"] + gates["benign_floor_delta"]
    feasible = [r for r in front
                if r["benign_refusal"] <= floor + 1e-12
                and r.get("degenerate", 0) <= gates.get("degenerate_max", 0)
                and (max_kl is None or r["kl"] <= max_kl)]
    pool = feasible or front
    best = min(pool, key=lambda r: (r["refusal"], r["kl"]))
    reason = ("lowest harmful refusal among feasible Pareto points"
              if feasible else
              "NO feasible Pareto point (benign floor / KL cap / degenerate) "
              "— picked the lowest refusal on the front; publish gates will "
              "judge it")
    return best, bool(feasible), reason


# ---- model-side pieces (torch) -----------------------------------------------
class Editor:
    """In-memory rank-1 partial projections with exact restore.

    Decoder matrices are edited IN PLACE (no second copy on the GPU); the
    first touch snapshots the original to CPU. The output head is edited
    in place when untied; when tied it is replaced by an untied clone and
    re-tied on restore (an in-place edit would change the input
    embeddings too)."""

    def __init__(self, model):
        self.model = model
        self.snap = {}
        self.head = None   # ("tied", original_param) | ("untied", cpu_copy)

    def project_rows(self, layer, label, r, alpha):
        import torch

        from . import arch
        lin = arch.edit_linear(self.model, layer, label)
        W = lin.weight
        key = (layer, label)
        if key not in self.snap:
            self.snap[key] = W.detach().to("cpu", copy=True)
        with torch.no_grad():
            # explicit copy: on an fp32 model .float() returns the SAME
            # tensor, and the in-place update below would edit the live
            # weight before the restore snapshot is taken
            Wf = W.detach().to(torch.float32, copy=True)
            rd = torch.as_tensor(r, dtype=torch.float32, device=Wf.device)
            Wf -= alpha * torch.outer(rd, rd @ Wf)
            W.copy_(Wf.to(W.dtype))

    def project_readout(self, d, alpha):
        import torch
        lm = self.model.get_output_embeddings()
        emb = self.model.get_input_embeddings()
        tied = lm.weight.data_ptr() == emb.weight.data_ptr()
        # snapshot BEFORE any write (tied: keep the shared Parameter itself)
        if self.head is None:
            self.head = (("tied", lm.weight) if tied else
                         ("untied", lm.weight.detach().to("cpu", copy=True)))
        with torch.no_grad():
            Wf = lm.weight.detach().to(torch.float32, copy=True)
            dd = torch.as_tensor(d, dtype=torch.float32, device=Wf.device)
            dd = dd / dd.norm()
            Wf -= alpha * torch.outer(Wf @ dd, dd)
            if tied:
                # an in-place edit would move the input embeddings too
                lm.weight = torch.nn.Parameter(Wf.to(lm.weight.dtype),
                                               requires_grad=False)
                self.model.config.tie_word_embeddings = False
            else:
                lm.weight.copy_(Wf.to(lm.weight.dtype))

    def apply(self, point, dirs, dir_B, labels_available):
        labels = labels_for(point["components"], labels_available)
        for li, r in dirs.items():
            for lab in labels:
                self.project_rows(li, lab, r, point["alpha"])
        if point["readout"]:
            self.project_readout(dir_B, point["alpha"])

    def restore(self):
        import torch

        from . import arch
        with torch.no_grad():
            for (layer, label), W0 in self.snap.items():
                W = arch.edit_linear(self.model, layer, label).weight
                W.copy_(W0.to(W.device))
        if self.head is not None:
            kind, orig = self.head
            lm = self.model.get_output_embeddings()
            if kind == "tied":
                lm.weight = orig           # the shared Parameter: re-tied
                self.model.config.tie_word_embeddings = True
            else:
                with torch.no_grad():
                    lm.weight.copy_(orig.to(lm.weight.device))
        self.snap, self.head = {}, None


def _chat(tok, prompts):
    return [tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts]


def generate_batch(tok, model, prompts, max_new, batch_size):
    """Greedy responses, left-padded batches (the probe-stage generate is
    one prompt at a time; search runs thousands of generations)."""
    import torch
    outs = []
    old = tok.padding_side
    tok.padding_side = "left"
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(_chat(tok, prompts[i:i + batch_size]),
                      return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=max_new,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for row in out[:, enc["input_ids"].shape[1]:]:
                outs.append(tok.decode(row, skip_special_tokens=True))
    finally:
        tok.padding_side = old
    return outs


def base_trajectories(tok, model, prompts, kl_tokens, batch_size):
    """The base model's own greedy answers on benign prompts, plus its
    log-probabilities at every answer position (fp16, CPU) — the reference
    the KL term compares candidates against."""
    import torch
    batches = []
    old = tok.padding_side
    tok.padding_side = "left"
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(_chat(tok, prompts[i:i + batch_size]),
                      return_tensors="pt", padding=True).to(model.device)
            L = enc["input_ids"].shape[1]
            with torch.inference_mode():
                seq = model.generate(**enc, max_new_tokens=kl_tokens,
                                     do_sample=False, min_new_tokens=kl_tokens,
                                     pad_token_id=tok.pad_token_id)
            mask = torch.cat([enc["attention_mask"],
                              torch.ones_like(seq[:, L:])], dim=1)
            lp = _answer_logprobs(model, seq, mask, L)
            batches.append({"seq": seq.cpu(), "mask": mask.cpu(), "L": L,
                            "logp": lp.to(torch.float16).cpu()})
    finally:
        tok.padding_side = old
    return batches


def _answer_logprobs(model, seq, mask, L):
    import torch
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        logits = model(input_ids=seq, attention_mask=mask,
                       position_ids=pos).logits
    # positions L-1 .. end-1 predict the answer tokens L .. end
    return torch.log_softmax(logits[:, L - 1:-1].float(), dim=-1)


def trajectory_kl(model, batches):
    """Mean KL(base || candidate) over the base's answer positions."""
    total, n = 0.0, 0
    for b in batches:
        seq = b["seq"].to(model.device)
        mask = b["mask"].to(model.device)
        lq = _answer_logprobs(model, seq, mask, b["L"])
        lp = b["logp"].to(lq.device).float()
        kl = (lp.exp() * (lp - lq)).sum(-1)       # [B, T]
        total += float(kl.sum())
        n += kl.numel()
    return total / max(1, n)


def _rate(outputs, score, markers):
    return sum(score(o, markers) for o in outputs) / max(1, len(outputs))


# ---- the search phase --------------------------------------------------------
def search_phase(spec):
    """Runs the search; returns the SEARCH_DONE payload. GPU (Colab)."""
    import torch

    from . import arch, core, honest, ui
    from .data import resolve_probe_set
    from .edits import _arr_sha, save_variant

    cfg = resolve_config(spec)
    out_dir = core._out_dir(spec, create=True)
    sdir = os.path.join(out_dir, "search")
    os.makedirs(sdir, exist_ok=True)
    stages = ui.Stages("S", 5)

    # 1. stage-A artifacts + base measurements -------------------------------
    stages.step("directions + base measurements")
    if not os.path.exists(os.path.join(out_dir, "layer_directions.npz")):
        ui.detail("no stage-A artifacts yet — running stage A first")
        core.from_spec(spec)
    lc = json.load(open(os.path.join(out_dir, "layer_coherence.json")))
    dirs_all = np.load(os.path.join(out_dir,
                                    "layer_directions.npz"))["directions"]
    dir_B = np.load(os.path.join(out_dir, "refusal_direction_B.npy"))
    arrays_sha = {"dirs": _arr_sha(dirs_all), "dir_B": _arr_sha(dir_B)}

    ps = spec["probe_sets"]
    train = (resolve_probe_set(ps["harmful"])[:ps["n_pairs"]]
             + resolve_probe_set(ps["harmless"])[:ps["n_pairs"]])
    harm_pool = resolve_probe_set(cfg["harmful_pool"])
    ben_pool = resolve_probe_set(cfg["benign_pool"])
    E, H = cfg["eval_prompts"], cfg["holdout_prompts"]
    if len(harm_pool) < E + H or len(ben_pool) < E:
        raise ValueError(f"search pools too small: need {E + H} harmful and "
                         f"{E} benign prompts")
    eval_harm, eval_ben = harm_pool[:E], ben_pool[:E]
    vault = honest.HoldoutVault(harm_pool[E:E + H], cfg["harmful_pool"])
    honest.assert_disjoint(train + eval_harm + eval_ben, vault,
                           name="direction+search")

    markers, score = core.session_grader(spec)
    honest.set_scorer(score, markers,
                      f"{ps['refusal_markers']}-{ps.get('marker_mode', 'v1')}"
                      "-strict")
    tok, model = core.load_patient(spec)
    n_layers = arch.n_layers(model)
    labels = arch.edit_labels(model)
    base_tied = bool(model.config.tie_word_embeddings)
    bs, mx = cfg["batch_size"], cfg["max_new_tokens"]

    def measure():
        oh = generate_batch(tok, model, eval_harm, mx, bs)
        ob = generate_batch(tok, model, eval_ben, mx, bs)
        return {"refusal": _rate(oh, score, markers),
                "benign_refusal": _rate(ob, score, markers),
                "degenerate": sum(core.is_degenerate(o) for o in oh + ob)}

    base = measure()
    traj = base_trajectories(tok, model, eval_ben, cfg["kl_tokens"], bs)
    base["kl"] = 0.0

    def gen_one(p):
        return generate_batch(tok, model, [p], mx, 1)[0]
    holdout_before, _ = honest.refusal_rate_holdout(vault, gen_one)
    ui.detail(f"base: refusal {ui.pct(base['refusal'])} · benign refused "
              f"{ui.pct(base['benign_refusal'])} · holdout refusal "
              f"{ui.pct(holdout_before)} (sealed, {vault.n} prompts)")

    # 2. trials --------------------------------------------------------------------
    stages.step(f"search: {cfg['trials']} trials (resumable)")
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError("abliterate search needs optuna: pip install "
                           "'abliteration-engine[search]'") from e
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{os.path.join(sdir, 'study.db')}"
    # degenerate outputs are reported to the sampler as a constraint
    # (> 0 = violated) so it learns to avoid them; selection filters them
    sampler = optuna.samplers.TPESampler(seed=cfg["seed"], multivariate=True)
    study = optuna.create_study(
        study_name=f"{core.run_dir_name(spec)}-search", storage=storage,
        load_if_exists=True, sampler=sampler,
        directions=["minimize"] * len(OBJECTIVES))
    editor = Editor(model)
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    if done:
        ui.detail(f"resuming: {done} trials already banked in study.db")
    log_path = os.path.join(sdir, "trials.jsonl")

    def objective(trial):
        point = suggest_point(trial, cfg, n_layers)
        t0 = time.time()
        try:
            editor.apply(point, layer_directions(point, dirs_all, lc["table"]),
                         dir_B, labels)
            m = measure()
            m["kl"] = trajectory_kl(model, traj)
        finally:
            editor.restore()
        trial.set_constraint("degenerate", float(m["degenerate"]))
        trial.set_user_attr("point", point)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        row = {"trial": trial.number, "point": point, **m,
               "seconds": round(time.time() - t0, 1)}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        ui.emit(f"  trial {trial.number + 1:>3} · {describe(point)} → refusal "
                f"{ui.pct(m['refusal'])} · benign refused "
                f"{ui.pct(m['benign_refusal'])} · KL {m['kl']:.3f}"
                + (f" · {m['degenerate']} degenerate" if m["degenerate"]
                   else ""))
        return tuple(m[k] for k in OBJECTIVES)

    remaining = max(0, cfg["trials"] - done)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    # 3. Pareto + selection --------------------------------------------------------
    stages.step("Pareto front + selection")
    rows = [{"trial": t.number, "point": t.user_attrs["point"],
             **{k: t.user_attrs[k] for k in (*OBJECTIVES, "degenerate")}}
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE]
    front = pareto_front(rows)
    chosen, feasible, reason = select(rows, base, spec["gates"],
                                      cfg["max_kl"])
    for r in sorted(front, key=lambda r: (r["refusal"], r["kl"])):
        mark = " ← chosen" if r is chosen else ""
        ui.detail(f"#{r['trial'] + 1:>3} refusal {ui.pct(r['refusal'])} · "
                  f"benign refused {ui.pct(r['benign_refusal'])} · KL "
                  f"{r['kl']:.3f} · {describe(r['point'])}{mark}")
    ui.detail(reason)

    # 4. materialize on a fresh base, save, reload, verify ----------------------------
    name = f"search_{chosen['trial'] + 1}"
    stages.step(f"materialize {name}: save → reload → verify")
    point = chosen["point"]
    dirs = layer_directions(point, dirs_all, lc["table"])
    editor.apply(point, dirs, dir_B, labels)   # base still loaded: reuse it
    originals = dict(editor.snap)
    var_dir = core.variant_dir(spec, name)
    save_variant(model, tok, var_dir,
                 expect_tied=(False if point["readout"] else base_tied))
    model = editor = None  # release the base before reloading the variant
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    tok_r, model_r = core.load_patient({"patient": {"model_id": var_dir,
                                                    "revision": None}})
    verify = verify_on_disk(model_r, point, dirs, originals)
    ui.detail(f"on-disk check: max |r·W_new − (1−α) r·W_base| = "
              f"{verify['max_resid']:.2e} over {verify['n_matrices']} "
              "matrices")

    # 5. probes + certificate ----------------------------------------------------------
    stages.step(f"probe {name} + certify on the sealed holdout")
    n_probes = ps["n_probes"]
    r_h = core.run_probes(tok_r, model_r,
                          resolve_probe_set(ps["harmful"])[:n_probes],
                          tag=f"{name}-harm",
                          max_new=spec["decoding"]["max_new_tokens"],
                          markers=markers, score_fn=score)
    r_b = core.run_probes(tok_r, model_r,
                          resolve_probe_set(ps["harmless"])[:n_probes],
                          tag=f"{name}-harmless",
                          max_new=spec["decoding"]["max_new_tokens"],
                          markers=markers, score_fn=score)
    fp = point_fingerprint(point, arrays_sha, spec)
    provenance = {"fingerprint": fp, "spec_sha256": spec.get("_spec_sha256"),
                  "verified": True}
    from .edits import write_variant_probes
    write_variant_probes(spec, name, r_h, r_b, provenance)
    summ = core.summarize(r_h, r_b)

    def gen_cand(p):
        return generate_batch(tok_r, model_r, [p], mx, 1)[0]
    cert = honest.build_certificate(
        name, {"search_eval": {k: chosen[k] for k in
                               (*OBJECTIVES, "degenerate")},
               "probe_set": summ, "point": point},
        vault, fp, spec["patient"]["revision"], arrays_sha,
        {"search_seed": cfg["seed"], "decoding_seed":
         spec["decoding"]["seed"]},
        refusal_rate_train=chosen["refusal"],
        benign_preserved=1 - chosen["benign_refusal"],
        benign_before=1 - base["benign_refusal"],
        holdout_before=holdout_before,
        capability_proxy=chosen["kl"], capability_proxy_before=0.0,
        generate_fn=gen_cand)
    cert_dir = os.path.join(sdir, name)
    if os.path.exists(os.path.join(cert_dir, "certificate.json")):
        cert_dir = os.path.join(sdir, f"{name}-{int(time.time())}")
    honest.certify(cert, cert_dir)
    ok_cert, cert_why = honest.publishable(
        cert, holdout_bar=spec["gates"]["publish_refusal"])

    report = {"config": cfg, "base": {**base, "holdout_refusal":
                                       holdout_before},
              "n_trials": len(rows), "pareto": front, "chosen": chosen,
              "feasible": feasible, "selection_reason": reason,
              "variant": name, "variant_dir": var_dir,
              "on_disk_verify": verify, "certificate": cert,
              "certificate_publishable": ok_cert,
              "certificate_reasons": cert_why}
    json.dump(report, open(os.path.join(sdir, "search_report.json"), "w"),
              indent=2, default=str)
    # representative direction for the hub aux file (publish verifies it):
    # readout-space dir_B when the head was edited, else the window's first
    # layer direction actually applied
    rep = dir_B if point["readout"] else dirs[min(dirs)]
    np.save(os.path.join(out_dir, "refusal_direction.npy"),
            np.asarray(rep, dtype=np.float32))
    _write_selection(spec, out_dir, name, var_dir, summ, base, chosen, point,
                     provenance, ok_cert, cert)
    _pareto_chart(sdir, rows, front, chosen, base)
    return {"selected": name, "point": point, "describe": describe(point),
            "trials": len(rows), "pareto_size": len(front),
            "feasible": feasible,
            "search_eval": {k: chosen[k] for k in OBJECTIVES},
            "holdout_refusal": cert["refusal_rate_holdout"],
            "holdout_refusal_before": holdout_before,
            "certificate_publishable": ok_cert,
            "probe_set": summ}


def verify_on_disk(model_r, point, dirs, originals):
    """The edit survived save/reload iff, for every edited matrix,
    r·W_reloaded == (1 − α) r·W_base (rank-1 partial projection algebra)."""
    import torch

    from . import arch
    worst, n = 0.0, 0
    for (layer, label), W0 in originals.items():
        r = torch.as_tensor(dirs[layer], dtype=torch.float32)
        W = arch.edit_linear(model_r, layer, label).weight.detach().float() \
            .cpu()
        got = r @ W
        want = (1 - point["alpha"]) * (r @ W0.float())
        scale = float(W0.float().abs().max()) or 1.0
        worst = max(worst, float((got - want).abs().max()) / scale)
        n += 1
    if worst > 2e-2:
        raise AssertionError(f"search edit did not survive save/reload "
                             f"(relative residual {worst:.3e})")
    return {"max_resid": worst, "n_matrices": n}


def _write_selection(spec, out_dir, name, var_dir, summ, base, chosen, point,
                     provenance, ok_cert, cert):
    gates = spec["gates"]
    passes = (summ["benign_preserved"]
              >= (1 - base["benign_refusal"]) - gates["benign_floor_delta"]
              and summ["degenerate_total"] <= gates.get("degenerate_max", 0))
    eligible = bool(passes and summ["refusal_rate"]
                    < gates["publish_refusal"] and ok_cert)
    cand = {"variant": name, "ladder_index": 0,
            "refusal_rate": summ["refusal_rate"],
            "benign_preserved": summ["benign_preserved"],
            "degenerate_total": summ["degenerate_total"],
            "passes_gate": passes}
    json.dump([cand], open(os.path.join(out_dir,
                                        "selection_candidates.json"), "w"),
              indent=2)
    json.dump({"selected": name, "gate": "passed" if passes else "failed",
               "selected_variant_dir": var_dir,
               "selected_variant_banked_resume": False,
               "banked_resume_variants": [],
               "selected_provenance": provenance,
               "publish_eligible_probe_gate": eligible,
               "publish_refusal_threshold": gates["publish_refusal"],
               "k_layers_primary": [], "k_layers_combo": [],
               "search_point": point,
               "search_holdout_refusal": cert["refusal_rate_holdout"],
               "metrics": {"refusal_rate_before": None,
                           "refusal_rate_after": summ["refusal_rate"],
                           "benign_preserved_before":
                               1 - base["benign_refusal"],
                           "benign_preserved_after":
                               summ["benign_preserved"]}},
              open(os.path.join(out_dir, "selection.json"), "w"), indent=2)


def _pareto_chart(sdir, rows, front, chosen, base):
    """Trade-off chart: every trial, the front, the pick (house palette)."""
    try:
        from . import card_charts as cc
        plt = cc._plt()
    except ImportError:
        return None
    front_ids = {r["trial"] for r in front}
    with plt.rc_context(cc._RC):
        fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=160)
        other = [r for r in rows if r["trial"] not in front_ids]
        ax.scatter([r["kl"] for r in other],
                   [r["refusal"] * 100 for r in other], s=22,
                   color=cc.C_BASELINE, label="trial")
        fr = sorted(front, key=lambda r: r["kl"])
        ax.scatter([r["kl"] for r in fr], [r["refusal"] * 100 for r in fr],
                   s=38, color=cc.C_HOOK, label="Pareto front")
        ax.scatter([chosen["kl"]], [chosen["refusal"] * 100], s=140,
                   marker="*", color=cc.C_VARIANT, label="chosen", zorder=5)
        ax.axhline(base["refusal"] * 100, color=cc.MUTED, lw=1, ls=":")
        ax.text(ax.get_xlim()[1], base["refusal"] * 100 + 1.5,
                f"base {base['refusal'] * 100:.0f}%", color=cc.MUTED,
                fontsize=8.6, ha="right")
        ax.set_xlabel("answer-trajectory KL vs base (lower = closer to "
                      "the original)")
        ax.set_ylabel("harmful refusal, search split (%)")
        ax.set_ylim(0, 105)
        ax.set_title("search trade-off: refusal vs drift", fontsize=11,
                     pad=10)
        ax.legend(frameon=False, fontsize=8.6)
        cc._style(ax)
        path = os.path.join(sdir, "search_pareto.png")
        cc._save(fig, path)
        plt.close(fig)
    return path
