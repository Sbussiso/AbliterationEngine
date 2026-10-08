"""search v1 — CPU-only logic (config, space, directions, Pareto selection).
The model-side path runs in tests/test_e2e_cpu.py (cpu-e2e CI job)."""
import os
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from abliteration_engine import search  # noqa: E402
from abliteration_engine.spec import SpecError, load_spec, spec_hash  # noqa: E402

RUN001 = os.path.join(REPO, "specs", "run001_parity.yaml")
GATES = {"benign_floor_delta": 0.10, "degenerate_max": 0}


def _spec_with(tmp_path, block):
    d = yaml.safe_load(open(RUN001))
    d["search"] = block
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p)


def test_defaults_and_overrides():
    cfg = search.resolve_config({"search": {"trials": 7,
                                            "space": {"alpha": [0.5, 1.0]}}})
    assert cfg["trials"] == 7 and cfg["space"]["alpha"] == [0.5, 1.0]
    assert cfg["space"]["components"] == ["attn", "mlp", "both"]
    assert cfg["harmful_pool"] == "builtin:ara_bad"


@pytest.mark.parametrize("block,match", [
    ({"trails": 3}, "unknown key"),
    ({"trials": 0}, "trials"),
    ({"holdout_prompts": 4}, "holdout_prompts"),
    ({"space": {"alpha": [1.2, 0.5]}}, "alpha"),
    ({"space": {"components": ["attn", "ffn"]}}, "components"),
    ({"space": {"direction_modes": []}}, "direction_modes"),
    ({"space": {"blend_k": [1, 3]}}, "blend_k"),
    ({"max_kl": -1}, "max_kl"),
])
def test_bad_config_rejected_at_spec_load(tmp_path, block, match):
    with pytest.raises(SpecError, match=match):
        load_spec(_spec_with(tmp_path, block))


def test_search_block_absent_keeps_spec_hash():
    s = load_spec(RUN001)
    assert "search" not in s and spec_hash(s).startswith("54590e5d")


class _Trial:
    """suggest_* stand-in returning fixed choices."""
    def __init__(self, **v):
        self.v = v

    def suggest_int(self, name, lo, hi):
        return max(lo, min(hi, self.v.get(name, lo)))

    def suggest_float(self, name, lo, hi):
        return self.v.get(name, lo)

    def suggest_categorical(self, name, choices):
        return self.v.get(name, choices[0])


def test_suggest_point_clamps_window():
    cfg = search.resolve_config({})
    p = search.suggest_point(_Trial(start=5, width=50, alpha=0.8,
                                    direction_mode="blend", blend_k=3), cfg,
                             n_layers=8)
    assert (p["start"], p["end"]) == (5, 8) and p["blend_k"] == 3
    assert "layers 5–7" in search.describe(p)


def test_layer_directions_modes():
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(6, 4)).astype(np.float32)
    table = [{"decoder_layer": i, "coherence": c}
             for i, c in enumerate([0.1, 0.5, 0.9, 0.3, 0.2, 0.4])]
    base = {"start": 1, "end": 4, "alpha": 1.0}
    own = search.layer_directions(dict(base, direction_mode="own"), dirs,
                                  table)
    assert sorted(own) == [1, 2, 3]
    assert np.allclose(own[1], dirs[1] / np.linalg.norm(dirs[1]))
    best = search.layer_directions(dict(base, direction_mode="best"), dirs,
                                   table)
    assert all(np.allclose(v, dirs[2] / np.linalg.norm(dirs[2]))
               for v in best.values())
    blend = search.layer_directions(
        dict(base, direction_mode="blend", blend_k=2), dirs, table)
    u = lambda v: v / np.linalg.norm(v)  # noqa: E731
    want = u(0.9 * u(dirs[2]) + 0.5 * u(dirs[1]))
    assert np.allclose(blend[3], want, atol=1e-6)
    assert all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in blend.values())


def _row(t, refusal, benign, kl, deg=0):
    return {"trial": t, "point": {}, "refusal": refusal,
            "benign_refusal": benign, "kl": kl, "degenerate": deg}


def test_pareto_front_drops_dominated():
    rows = [_row(0, 0.5, 0.0, 0.1), _row(1, 0.2, 0.0, 0.3),
            _row(2, 0.6, 0.1, 0.4),  # dominated by 0
            _row(3, 0.2, 0.0, 0.5)]  # dominated by 1
    assert {r["trial"] for r in search.pareto_front(rows)} == {0, 1}


def test_select_respects_benign_floor_kl_cap_and_degenerates():
    base = {"benign_refusal": 0.0}
    rows = [_row(0, 0.05, 0.30, 0.9),   # breaks the benign floor
            _row(1, 0.10, 0.05, 0.6),
            _row(2, 0.20, 0.00, 0.1),
            _row(3, 0.00, 0.00, 0.2, deg=2)]  # degenerate outputs
    best, feasible, _ = search.select(rows, base, GATES)
    assert best["trial"] == 1 and feasible
    best, _, _ = search.select(rows, base, GATES, max_kl=0.5)
    assert best["trial"] == 2
    # nothing feasible: still returns the lowest refusal, flagged
    best, feasible, why = search.select([_row(0, 0.1, 0.9, 0.1)], base,
                                        GATES)
    assert not feasible and "NO feasible" in why


def test_plan_shows_search(tmp_path, capsys, monkeypatch):
    from abliteration_engine import cli
    monkeypatch.setenv("NO_COLOR", "1")
    p = _spec_with(tmp_path, {"trials": 12})
    assert cli.main(["plan", "--spec", p]) == 0
    out = capsys.readouterr().out
    assert "12 trials" in out and "abliterate search --spec" in out
