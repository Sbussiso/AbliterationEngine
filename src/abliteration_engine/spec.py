"""YAML run-spec loading + validation (FTT-19).

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
    if not variants or not set(variants) <= set(_LADDER_ALL):
        raise SpecError(f"ladder.variants must be a non-empty subset of "
                        f"{_LADDER_ALL}")
    if "wd_ML_BN" in variants and "wd_ML" not in variants:
        raise SpecError("ladder: wd_ML_BN presumes wd_ML (top-K layers come "
                        "from the coherence scan)")
    if int(lad.get("k_primary", 3)) < 1 or int(lad.get("k_combo", 5)) < 1:
        raise SpecError("ladder k_primary/k_combo must be >= 1")

    g = dict(_GATES_DEFAULT)
    g.update(raw.get("gates") or {})
    if not (0 < g["publish_refusal"] < 1):
        raise SpecError("gates.publish_refusal must be in (0,1)")
    if g["mmlu_max_loss_pp"] <= 0:
        raise SpecError("gates.mmlu_max_loss_pp must be > 0")

    out = dict(raw)
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