"""YAML run-spec loading + validation.

The spec is the ONLY run-specific data the engine reads. Validation is
fail-fast and hard-asserts the items that protect prior runs' guarantees:
pinned revision, probe-count sanity, gate ordering, ladder subset.
"""
import hashlib
import json
import os

import yaml


class SpecError(ValueError):
    pass


REQUIRED_TOP = ["spec_version", "run_card", "patient", "probe_sets",
                "decoding", "ladder", "gates"]

_GATES_DEFAULT = {"benign_floor_delta": 0.10, "degenerate_max": 0,
                  "publish_refusal": 0.25, "mmlu_max_loss_pp": 3.0}
_VERIFY_DEFAULT = {"lm_head": 0.005, "final_norm": 0.05, "layer_row": 0.01}
_LADDER_ALL = ["wd_B", "wd_BN", "wd_ML", "wd_ML_BN"]
_DECODING_STRATEGIES = ("greedy",)


def _is_ara_variant(name):
    return (isinstance(name, str) and name.startswith("ara_")
            and name[4:].isdigit() and int(name[4:]) >= 1)


def load_spec(path):
    """Load + validate a YAML run spec. Returns the normalized dict.

    Raises SpecError with an actionable message on any violation.
    """
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise SpecError(f"spec file not found: {path}") from e
    except yaml.YAMLError as e:
        raise SpecError(f"{path}: not valid YAML ({e})") from e
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: YAML must be a mapping, got {type(raw)}")

    for key in REQUIRED_TOP:
        if key not in raw:
            raise SpecError(f"{path}: missing required key '{key}'")

    if int(raw["spec_version"]) != 1:
        raise SpecError(f"unsupported spec_version {raw['spec_version']} "
                        "(this engine reads version 1)")

    rc = raw["run_card"]
    for key in ("run_number", "patient", "purpose"):
        if key not in rc:
            raise SpecError(f"run_card: missing '{key}'")

    pat = raw["patient"]
    for key in ("model_id", "revision"):
        if not pat.get(key):
            raise SpecError(f"patient: missing '{key}' "
                            "(pinned revisions are mandatory)")
    rev = pat["revision"]
    if not isinstance(rev, str):
        raise SpecError(f"patient.revision must be a string, got "
                        f"{type(rev).__name__} ({rev!r})")
    rev_ok = (len(rev) == 40 and rev.isalnum()
              or (rev.startswith("v") and len(rev) > 1
                  and rev[1].isdigit()))
    if not rev_ok:
        # allow either a 40-char sha or a v-prefixed tag pin; refuse
        # branch names and floating refs
        raise SpecError(f"patient.revision='{rev}' is not a pin - "
                        "full sha or v-prefixed tag required")

    se = pat.get("structure_expect") or {}
    if not isinstance(se, dict):
        raise SpecError("patient.structure_expect must be a mapping")

    ps = raw["probe_sets"]
    # required, validated, never injected: every GPU stage indexes these
    # directly, so an absent key used to plan green and KeyError on GPU
    # (the day-2 k_primary lesson, same class).
    for key in ("harmful", "harmless", "refusal_markers", "n_pairs",
                "n_probes"):
        if key not in ps:
            raise SpecError(f"probe_sets: missing '{key}'")
    if int(ps.get("n_pairs", 64)) < 8:
        raise SpecError("probe_sets.n_pairs < 8 gives degenerate directions")
    if int(ps.get("n_probes", 16)) < 4:
        raise SpecError("probe_sets.n_probes < 4 gives noise-level rates")

    dec = raw["decoding"]
    mnt = dec.get("max_new_tokens")
    if not isinstance(mnt, int) or isinstance(mnt, bool) or mnt < 1:
        raise SpecError("decoding.max_new_tokens must be a positive int "
                        f"(got {mnt!r})")
    if dec.get("strategy", "greedy") not in _DECODING_STRATEGIES:
        raise SpecError(f"decoding.strategy '{dec.get('strategy')}' "
                        "unsupported (v3: greedy only)")

    lad = raw["ladder"]
    variants = lad.get("variants", _LADDER_ALL)
    # v1 amendment (FTT-28, 2026-10-02): `ara_<rank>` variants — the ARA
    # optimizer as a persistent-edit variant class (src/abliteration_engine/
    # ara.py). Same status as the built-in wd_* names.
    bad = [v for v in variants if v not in _LADDER_ALL
           and not _is_ara_variant(v)]
    if bad:
        raise SpecError(f"ladder.variants must be a subset of {_LADDER_ALL}"
                        " (+ ara_<rank>), got "
                        f"{bad}")
    # NOTE (v1 amendment 2026-09-30): empty variants = hook-only
    # characterization run (Run 000 semantics): stage 5 skipped, no
    # selection.json, publish stage gated off. Optional 'hooks' block
    # below carries the hook-scope choice for exactly this run shape.
    if "wd_ML_BN" in variants and "wd_ML" not in variants:
        raise SpecError("ladder: wd_ML_BN presumes wd_ML (top-K layers come "
                        "from the coherence scan)")
    # v1 amendment (FTT-28): validate ladder.ara against the requested
    # ara_<rank> variant(s). Fail here (CPU, no model) — a spec that plans
    # green must run green. Never default-injected: absent ladder.ara with
    # no ara_* variant leaves the normalized spec byte-stable (Run-001
    # parity contract).
    if any(_is_ara_variant(v) for v in variants) or lad.get("ara"):
        from .ara import resolve_ara_config

        try:
            name, cfg = resolve_ara_config(lad)
        except ValueError as e:
            raise SpecError(f"ladder.ara: {e}") from e
        if name is None:
            raise SpecError("ladder.ara set but ladder.variants has no "
                            "ara_<rank> variant")
        lad["ara"] = cfg  # normalized config (injected, mirrors k_primary)
        n_layers = se.get("num_hidden_layers")
        bad_l = [x for x in (cfg.get("layers") or [])
                 if isinstance(n_layers, int) and x >= n_layers]
        if bad_l:
            raise SpecError(f"ladder.ara.layers {bad_l} out of range for "
                            f"{n_layers} decoder layers "
                            "(patient.structure_expect.num_hidden_layers)")
        n_ara = [v for v in variants if _is_ara_variant(v)]
        if len(n_ara) > 1:
            raise SpecError("multiple ara_<rank> variants in one ladder is "
                            "unsupported (one ARA config per ladder)")
    if int(lad.get("k_primary", 3)) < 1 or int(lad.get("k_combo", 5)) < 1:
        raise SpecError("ladder k_primary/k_combo must be >= 1")
    # day-2 lesson (first full-GPU ladder run, Run 002): defaults were
    # validated here but never INJECTED, so edits.run_ladder's lad["k_primary"]
    # KeyError'd at ladder start on a spec that omitted them. Validate AND
    # inject — a spec that plans green must run green.
    if "k_primary" not in lad:
        lad["k_primary"] = 3  # v2 mission-004 default (ABL3_K_PRIMARY)
    if "k_combo" not in lad:
        lad["k_combo"] = 5    # v2 mission-004 default (ABL3_K_COMBO)

    g = dict(_GATES_DEFAULT)
    g.update(raw.get("gates") or {})
    if not (0 < g["publish_refusal"] < 1):
        raise SpecError("gates.publish_refusal must be in (0,1)")
    if g["mmlu_max_loss_pp"] <= 0:
        raise SpecError("gates.mmlu_max_loss_pp must be > 0")
    dm = g["degenerate_max"]
    if not isinstance(dm, int) or isinstance(dm, bool) or dm < 0:
        raise SpecError("gates.degenerate_max must be an int >= 0")

    # v1 amendment: optional hooks block, validated but never default-injected
    # (absent = engine treats as {"scope": "selected"} at use time; keeps
    # normalized spec hashes stable for existing specs)
    hooks = raw.get("hooks") or {}
    if not isinstance(hooks, dict):
        raise SpecError("hooks must be a mapping")
    scope = hooks.get("scope", "selected")
    if scope not in ("selected", "all"):
        raise SpecError(f"hooks.scope must be 'selected' or 'all', "
                        f"got {scope!r}")

    # marker_mode (grader-under-study): v1 = frozen substring
    # grader (Run-001 parity contract); v2 = refusal_score_v2 (word
    # boundary + offer-tail exception). Validated only, never injected —
    # absent means v1 and keeps normalized spec hashes stable for
    # existing specs (hooks-block precedent).
    mm = (ps.get("marker_mode") or "v1") if isinstance(ps, dict) else "v1"
    if mm not in ("v1", "v2"):
        raise SpecError(f"probe_sets.marker_mode must be 'v1' or 'v2', "
                        f"got {mm!r}")

    # directions.readout_norm: how readout-space direction B is computed.
    # 'double' = frozen Run-001 computation (final norm applied on top of
    # hidden_states[-1], which HF already returns post-norm); 'single' =
    # the actual lm_head input. Validated only, never injected (absent =
    # double) so every existing spec_hash and parity anchor stays stable.
    # search: optional block for `abliterate search` (validated against the
    # search module's schema; never injected — absent keeps every existing
    # spec hash stable)
    if raw.get("search") is not None:
        from .search import resolve_config as _search_cfg
        try:
            _search_cfg(raw)
        except ValueError as e:
            raise SpecError(str(e)) from e

    dirs = raw.get("directions") or {}
    if not isinstance(dirs, dict):
        raise SpecError("directions must be a mapping")
    unknown = sorted(set(dirs) - {"readout_norm"})
    if unknown:
        raise SpecError(f"directions: unknown key(s) {unknown}")
    rn = dirs.get("readout_norm", "double")
    if rn not in ("double", "single"):
        raise SpecError(f"directions.readout_norm must be 'double' or "
                        f"'single', got {rn!r}")

    out = dict(raw)
    if hooks:
        out["hooks"] = {"scope": scope}
    if dirs:
        out["directions"] = {"readout_norm": rn}
    out["gates"] = g
    out["ladder"] = {**lad, "variants": list(variants)}
    out["decoding"] = {**dec, "seed": int(dec.get("seed", 0))}
    out.setdefault("publish", {})
    pub = out["publish"] or {}
    if not isinstance(pub, dict):
        raise SpecError("publish must be a mapping")
    if "card_charts" in pub and not isinstance(pub["card_charts"], bool):
        raise SpecError("publish.card_charts must be true or false "
                        f"(got {pub['card_charts']!r})")
    for key in ("hf_user", "card_byline", "card_marker"):
        if pub.get(key) is not None and not isinstance(pub[key], str):
            raise SpecError(f"publish.{key} must be a string")
    out.setdefault("colab", {"gpu": "t4"})
    out.setdefault("hitl", {"after_selection": True, "before_publish": True})
    if "verify_disk_bounds" not in out["publish"]:
        out["publish"]["verify_disk_bounds"] = dict(_VERIFY_DEFAULT)
    out["_spec_path"] = os.path.abspath(path)
    out["_spec_sha256"] = sha256_file(path)
    return out


def spec_hash(spec):
    """Sha256 of the NORMALIZED spec (defaults filled) - the provenance
    hash that goes into run_config.json so the paper can cite the exact
    effective config even if the raw YAML later gains new defaults."""
    norm = {k: v for k, v in spec.items() if not k.startswith("_")}
    blob = json.dumps(norm, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()