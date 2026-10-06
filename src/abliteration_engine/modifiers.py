"""Modifier registry — pluggable edit strategies (ENGINE-V0.4 spec §1.1).

A Modifier maps (model, direction bank, search point) -> edited model.
The ladder's variant names dispatch through this registry; the search
harness (spec §1.3) drives the same registry with search-selected points.

Built-in modifiers:
  projection  — the ladder's orthogonalization family (wd_* sites)
  ara         — Arbitrary-Rank Ablation (Weidmann 2026, math-only impl;
                wraps src/abliteration_engine/ara.py 1:1 — byte-stable
                behavior, no reimplementation)
  composite   — RESERVED slot (ARA + direction-bank projection in one
                joint objective; lands with the search harness)

Registry discipline (dev contract):
  - register() is the ONLY mutation point; duplicate names raise.
  - every modifier declares: ladder names, per-name flavor (which tie
    state it produces, derived from the patient at plan time), device
    discipline (BUG-1: math staged on CPU, writes routed via arch._Lin),
    and a verify hook usable on reload-from-disk.
  - AGPL boundary: nothing here derives from p-e-w/heretic code; ARA is
    implemented from the published math description.
"""

from abliteration_engine import arch  # noqa: F401  (registry consumers)

# ladder flavor constants (BUG-2 contract): what tie state a variant
# PRODUCES on a given patient
FLAVOR_UNTIE = "untie_head"      # lm_head edit on a cloned untied head
FLAVOR_KEEP = "keep_tie"         # decoder-layer edits only; tie untouched
FLAVOR_FROM_BASE = "from_base"   # expectation = the base's own state

_VARIANT_FLAVOR = {
    "wd_B": FLAVOR_UNTIE,
    "wd_BN": FLAVOR_UNTIE,
    "wd_ML": FLAVOR_KEEP,
    "wd_ML_BN": FLAVOR_UNTIE,
}

_REGISTRY = {}


def register(name, modifier):
    """The only mutation point; duplicate registration is a bug."""
    if name in _REGISTRY:
        raise ValueError(f"modifier {name!r} already registered")
    _REGISTRY[name] = modifier
    return modifier


def get(name):
    """The modifier for a ladder variant name: explicit names first, then
    the handles list, then prefix families (ara_* -> the ara modifier)."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    for mod in _REGISTRY.values():
        if name in (getattr(mod, "handles", None) or ()):
            return mod
        prefix = getattr(mod, "handles_prefix", None)
        if prefix and name.startswith(prefix):
            return mod
    raise KeyError(f"no modifier registered for variant {name!r} "
                   f"(registered: {sorted(_REGISTRY)})")


def names():
    return sorted(_REGISTRY)


def flavor_for(name, model=None):
    """'what tie state does this variant produce on THIS patient' —
    derived, never name-carried (BUG-2): untie flavors are untied on
    every base; keep flavors pass the base's own state through.
    model=None only valid for untie flavors (their answer is
    base-invariant)."""
    flavor = _VARIANT_FLAVOR.get(name, FLAVOR_FROM_BASE
                                 if name.startswith("ara_") else None)
    if flavor is None:
        raise KeyError(f"unknown variant flavor for {name!r}")
    if flavor == FLAVOR_UNTIE:
        return False
    if model is None:
        raise ValueError(
            f"variant {name!r} keeps the base tie state; deriving the "
            "expectation needs the model (BUG-2: name-only assumptions "
            "broke untied patients)")
    return bool(model.config.tie_word_embeddings)


class Modifier:
    """Protocol every edit strategy implements (duck-typed; no ABC
    ceremony). All tensor math MUST stage on CPU and route writes through
    arch._Lin.set_weight / explicit .to(module-weight.device) — the BUG-1
    discipline."""

    name = None

    def plan_info(self, name, cfg, model=None):
        """One-line human + dict summary for plan output / run_config."""
        raise NotImplementedError

    def run_variant(self, spec, name, cfg, ctx, provenance=None):
        """Full variant lifecycle (edit -> save -> reload -> verify ->
        probe), mirroring edits.run_variant's summary contract."""
        raise NotImplementedError


# ---- built-in modifier implementations --------------------------------------
class ProjectionModifier(Modifier):
    """wd_B / wd_BN / wd_ML / wd_ML_BN — direction orthogonalization at
    ladder-carried sites. The frozen edit/verify closures stay in
    edits.py (run-002 semantics + parity anchors depend on their bytes);
    this wrapper exposes the family under the protocol WITHOUT moving
    that code."""
    name = "projection"
    handles = ("wd_B", "wd_BN", "wd_ML", "wd_ML_BN")

    def plan_info(self, name, cfg, model=None):
        flavor_name = _VARIANT_FLAVOR.get(name)
        return {"modifier": self.name, "variant": name,
                "flavor": flavor_name or FLAVOR_KEEP}

    def run_variant(self, spec, name, cfg, ctx, provenance=None):
        raise NotImplementedError(
            "projection variants run through edits.run_ladder's frozen "
            "closures (parity contract); modifier-object routing for "
            "search lands with the search harness (spec §1.3)")


class AraModifier(Modifier):
    """ara_<rank> — ARA (Weidmann 2026): rank-r LoRA per edit matrix,
    L-BFGS on the preserve-good + steer-bad objective (published math
    only). Wraps ara.py 1:1 — byte-stable behavior."""
    name = "ara"
    handles_prefix = "ara_"

    def resolve_cfg(self, ladder):
        from abliteration_engine.ara import resolve_ara_config
        return resolve_ara_config(ladder)

    def plan_info(self, name, cfg, model=None):
        rank = int(name[4:]) if name[4:].isdigit() else cfg.get("rank")
        return {"modifier": self.name, "variant": name, "rank": rank,
                "flavor": FLAVOR_FROM_BASE,
                "layers": cfg.get("layers") or "ALL"}

    def run_variant(self, spec, name, cfg, ctx, provenance=None):
        from abliteration_engine.ara import run_ara_variant
        return run_ara_variant(spec, name, cfg, provenance=provenance)


register("projection", ProjectionModifier())
register("ara", AraModifier())