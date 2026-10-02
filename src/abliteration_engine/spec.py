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


def load_spec(path):
    """Load + validate a YAML run spec. Returns the normalized dict.

    Raises SpecError with an actionable message on any violation.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
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
    if int(ps.get("n_pairs", 64)) < 8:
        raise SpecError("probe_sets.n_pairs < 8 gives degenerate directions")
    if int(ps.get("n_probes", 16)) < 4:
        raise SpecError("probe_sets.n_probes < 4 gives noise-level rates")

    dec = raw["decoding"]
    if dec.get("strategy", "greedy") not in _DECODING_STRATEGIES:
        raise SpecError(f"decoding.strategy '{dec.get('strategy')}' "
                        "unsupported (v3: greedy only)")

    lad = raw["ladder"]
    variants = lad.get("variants", _LADDER_ALL)
    if not set(variants) <= set(_LADDER_ALL):
        raise SpecError(f"ladder.variants must be a subset of {_LADDER_ALL}")
    # NOTE (v1 amendment 2026-09-30): empty variants = hook-only
    # characterization run (Run 000 semantics): stage 5 skipped, no
    # selection.json, publish stage gated off. Optional 'hooks' block
    # below carries the hook-scope choice for exactly this run shape.
    if "wd_ML_BN" in variants and "wd_ML" not in variants:
        raise SpecError("ladder: wd_ML_BN presumes wd_ML (top-K layers come "
                        "from the coherence scan)")
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

    out = dict(raw)
    if hooks:
        out["hooks"] = {"scope": scope}
    out["gates"] = g
    out["ladder"] = {**lad, "variants": list(variants)}
    out["decoding"] = {**dec, "seed": int(dec.get("seed", 0))}
    out.setdefault("publish", {})
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