"""ENGINE-V0.4 item-2 gate: modifier abstraction + ARA-from-math vs
ANALYTIC ground truth on a toy model (spec build-order #2).

Projection gate: orthogonalize_layer_output produces exactly W' = M W
(M = I - r r^T) on o_proj AND mlp sites, the row-space invariant
(M W)^T r ≈ 0 holds at fp32, and OTHER layers stay untouched.
ARA gate — HONEST propositions only: the k>1 KNN-pull is NOT zero at a
teacher (mean-of-k-smallest self-distances = (0 + d_second)/2), so
"loss -> 0 at candidate==teacher" is never asserted. Asserted instead:
the hold-MSE quadratic (whose minimizer IS the teacher) lands W_eff
nearer a rank-r teacher than the base was; the loss decreases; the pull
term at the optimum is <= the pull at the base.
Registry dispatch contract (names/prefix/duplicate-guards), the
BUG-2 flavor single-source-of-truth, and the BUG-1 device discipline.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "nn") or not hasattr(torch.nn, "Module"):
    pytest.skip("torch stub in sys.modules (banked-resume fixture)",
                allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from abliteration_engine import arch  # noqa: E402
from abliteration_engine import edits as edits_mod  # noqa: E402
from abliteration_engine import modifiers as mods  # noqa: E402


class _LinWrap(torch.nn.Module):
    """block.self_attn.o_proj-shaped: a Linear DIRECTLY on the module."""

    def __init__(self, n_out, n_in):
        super().__init__()
        self.o_proj = torch.nn.Linear(n_in, n_out, bias=False)


class _MLPWrap(torch.nn.Module):
    def __init__(self, n_out, n_in):
        super().__init__()
        self.down_proj = torch.nn.Linear(n_in, n_out, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.self_attn = _LinWrap(d, d)
        self.mlp = _MLPWrap(d, d_ff)


class _Model(torch.nn.Module):
    def __init__(self, n_layers=2, d=8, d_ff=12):
        super().__init__()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([_Block(d, d_ff)
                                            for _ in range(n_layers)])
        inner.norm = torch.nn.RMSNorm(d)
        self.model = inner
        self.config = type("C", (), {"model_type": "qwen2",
                                     "tie_word_embeddings": False})()
        self.lm_head = torch.nn.Linear(d, 16, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.lm_head


def _toy():
    torch.manual_seed(11)
    return _Model()


# ---- projection vs analytic ---------------------------------------------------
def test_projection_equals_analytic_MW():
    m = _toy()
    d = 8
    r = torch.randn(d)
    rhat = r / r.norm()
    M = torch.eye(d) - torch.outer(rhat, rhat)
    # untouched-layer snapshot BEFORE any edit (a self-comparison asserts
    # nothing — the audit caught exactly that tautology)
    W0_L0 = arch.module_for(m, 0, "self_attn.o_proj").weight.detach().clone()
    # analytic target, computed independently on the SAME weight
    W0 = arch.module_for(m, 1, "self_attn.o_proj").weight.detach().clone()
    target = M @ W0
    edits_mod.orthogonalize_layer_output(m, 1, r)
    got = arch.module_for(m, 1, "self_attn.o_proj").weight.detach()
    assert torch.allclose(got, target, atol=1e-6), \
        (got - target).abs().max()
    # row-space invariant at fp32 precision
    resid = float((got.float().T @ rhat).abs().max())
    assert resid < 1e-5, resid
    # BOTH edit matrices edited on that layer; other layer untouched
    W0d = arch.edit_linear(m, 1, "mlp.down_proj").weight.detach().clone()
    edits_mod.orthogonalize_layer_output(m, 1, r)
    got_d = arch.module_for(m, 1, "mlp.down_proj").weight.detach()
    assert torch.allclose(got_d, M @ W0d, atol=1e-6)
    assert torch.equal(arch.module_for(m, 0, "self_attn.o_proj")
                       .weight.detach(), W0_L0)


def test_projection_mlp_site_analytic():
    m = _toy()
    d = 8
    r = torch.randn(d)
    rhat = r / r.norm()
    M = torch.eye(d) - torch.outer(rhat, rhat)
    W0 = arch.module_for(m, 0, "mlp.down_proj").weight.detach().clone()
    edits_mod.orthogonalize_layer_output(m, 0, r)
    got = arch.module_for(m, 0, "mlp.down_proj").weight.detach()
    assert torch.allclose(got, M @ W0, atol=1e-6)


# ---- ARA vs rank-r teacher (the optimizer can express the target) -------------
def test_ara_recovers_rank_r_teacher_delta():
    """ARA-vs-analytic gate (honest propositions, spec gate #2).

    The k>1 KNN-pull term is minimized by CLUSTERING the bad outputs
    toward their neighbors, not by matching a teacher — so "loss -> 0 at
    candidate==teacher" is mathematically FALSE and never asserted. What
    IS analytic here:
      (a) hold-MSE pressure: W_eff must land NEARER the rank-r teacher
          than W0 was (the preserve term is an exact quadratic in W_eff;
          its minimizer on good rows IS the teacher);
      (b) the optimizer drives the loss DOWN from the start point;
      (c) the pull term at the optimum is <= its value at the base
          (steering pressure moves outputs together, never apart).
    Also documents WHY: with k>1, mean-of-k-smallest self-distances on
    the good pool is (0 + d_second)/2 at the teacher — never zero.
    """
    from abliteration_engine.ara import mean_distances_to_knn, optimize_ara_weights

    m = _toy()
    d = 8
    rank = 4
    torch.manual_seed(3)
    lin = arch.module_for(m, 0, "self_attn.o_proj")
    W0 = lin.weight.detach().clone().float()
    B_t = torch.randn(d, rank) * 0.05
    A_t = torch.randn(rank, d) * 0.05
    W_teacher = W0 + B_t @ A_t
    P = 24
    torch.manual_seed(7)
    G = torch.randn(P, d)          # same rows both pools = clean geometry
    g_out = G @ W_teacher.T        # hold-MSE minimizer on this data
    b_base = G @ W0.T              # bad anchor = base outputs
    gm = torch.randn(P, 12)
    good_io = {0: {"self_attn.o_proj": (G, g_out),
                   "mlp.down_proj": (gm, torch.zeros(P, 8))}}
    bad_io = {0: {"self_attn.o_proj": (G, b_base),
                  "mlp.down_proj": (gm, torch.zeros(P, 8))}}
    cfg = {"rank": rank, "preserve_good_weight": 1.0,
           "steer_bad_weight": 2.0, "overcorrect_weight": 0.0,
           "neighbor_count": 2, "steps": 25, "lr": 1.0, "max_iter": 60,
           "history_size": 10, "preserve_row_magnitudes": False,
           "layers": [0]}
    info = optimize_ara_weights(m, 0, cfg, good_io, bad_io)
    i = info["self_attn.o_proj"]
    # (b) the optimizer worked
    assert i["loss_last"] < i["loss_first"], i
    # (a) landed nearer the teacher than the base was (analytic quadratic)
    got = arch.module_for(m, 0, "self_attn.o_proj").weight.detach().float()
    from_teacher = float((got - W_teacher).abs().max())
    from_base = float((got - W0).abs().max())
    assert from_teacher < from_base, (from_teacher, from_base)
    # (c) steering pressure: outputs pulled together vs the base point set
    with torch.no_grad():
        pull_at = float(mean_distances_to_knn(G @ got.T, g_out, 2).mean())
        pull_base = float(mean_distances_to_knn(b_base, g_out, 2).mean())
    assert pull_at < pull_base, (pull_at, pull_base)


def test_ara_loss_matches_reference_formula():
    """ara_loss == w_pg*MSE + w_sb*(pull - w_oc*push), term by term."""
    from abliteration_engine.ara import ara_loss, mean_distances_to_knn
    torch.manual_seed(5)
    g, b = torch.randn(6, 4), torch.randn(5, 4)
    ng, nb = torch.randn(6, 4), torch.randn(5, 4)
    p = {"preserve_good_weight": 0.7, "steer_bad_weight": 1.3,
         "overcorrect_weight": 0.4, "neighbor_count": 2}
    expect = (0.7 * ((ng - g) ** 2).mean()
              + 1.3 * (mean_distances_to_knn(nb, g, 2).mean()
                       - 0.4 * mean_distances_to_knn(nb, b, 2).mean()))
    assert torch.allclose(ara_loss(g, b, ng, nb, p), expect, atol=1e-6)


# ---- registry contract ----------------------------------------------------------
def test_registry_dispatch_and_guards():
    assert "projection" in mods.names() and "ara" in mods.names()
    assert mods.get("wd_B").name == "projection"
    assert mods.get("wd_ML_BN").name == "projection"
    assert mods.get("ara_50").name == "ara"
    assert mods.get("ara_8").name == "ara"
    with pytest.raises(KeyError):
        mods.get("banana_xyz")
    with pytest.raises(ValueError, match="already registered"):
        mods.register("ara", mods.AraModifier())
    with pytest.raises(NotImplementedError, match="search harness"):
        mods.get("wd_B").run_variant(None, "wd_B", None, None)


def test_flavor_single_source_of_truth():
    """edits.expect_tied_for DELEGATES to modifiers.flavor_for — one table,
    no drift (checked by identity of behavior on both flavors + bases)."""
    tied = type("X", (), {"config": type("C", (), {
        "tie_word_embeddings": True})()})()
    untied = type("X", (), {"config": type("C", (), {
        "tie_word_embeddings": False})()})()
    for name, expect_pair in (("wd_B", (False, False)),
                              ("wd_BN", (False, False)),
                              ("wd_ML", (True, False)),
                              ("wd_ML_BN", (False, False)),
                              ("ara_50", (True, False))):
        assert edits_mod.expect_tied_for(name, tied) is expect_pair[0], name
        assert edits_mod.expect_tied_for(name, untied) is expect_pair[1], name
        # registry says the same thing (single source of truth)
        assert mods.flavor_for(name, tied) == \
            edits_mod.expect_tied_for(name, tied)


def test_device_discipline_on_edit_path():
    """BUG-1 discipline pinned at the toy level: the edit MATH must stage
    on CPU (source of r + the M product) while the module write routes
    back to the module's device/dtype. On CPU-only tests both are cpu;
    the DISCIPLINE is pinned by source inspection (GPU unavailable here)
    + the capture-cpu comment contract."""
    import inspect
    src = inspect.getsource(
        edits_mod.orthogonalize_layer_output)
    assert ".float().cpu()" in src, "edit-math staging must pin to CPU"
    assert "set_weight" in src, "write-back must route through arch._Lin"


if __name__ == "__main__":
    raise SystemExit("run via pytest")