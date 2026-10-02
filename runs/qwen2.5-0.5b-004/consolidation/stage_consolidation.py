#!/usr/bin/env python3
"""Stage the consolidated repo tree for sbussiso/Qwen2.5-0.5B-abliterated.

Builds consolidation/stage/ from the recorded run artifacts:
  stage/eval/    run-002 evidence (flat; MMLU raws flattened from nested dirs)
  stage/eval2/   run-003 evidence + new consolidation evals (tq240, multilingual)
  stage/charts/  make_all_charts.py + its 10 PNG outputs (run AFTER evals)
  stage root     weights (wd_ML_BN) + tokenizer + configs + direction npys
                 + gen_card.py

Run order: evals first (EVALS_DONE marker), then stage, then charts, then card.
Every step asserts what it expects — fail loudly, never guess.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

CONS = Path("/root/research/abliteration/qwen2.5-0.5b-004/consolidation")
A2 = Path("/root/research/abliteration/qwen2.5-0.5b-002/artifacts")   # run-002
A3 = Path("/root/research/abliteration/qwen2.5-0.5b-004/artifacts")   # run-003
VAR = Path("/root/research/abliteration/qwen2.5-0.5b-004/variant_wd_ML_BN")
STAGE = CONS / "stage"
PY = "/root/research/venvs/ablate/bin/python"

NEW_WSHA = "0b2133342dce215d6ae9645259b9f05e33914c0f51dcde60c550f3bf92367125"


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_flat(src_dir, dst_dir, names, required=True):
    for n in names:
        s = src_dir / n
        d = dst_dir / n
        if s.exists():
            shutil.copy2(s, d)
        elif required:
            raise FileNotFoundError(f"required artifact missing: {s}")
    return dst_dir


def find_lm_eval_results(outdir):
    """Newest results_*.json under an lm-eval --output_path dir."""
    hits = sorted(outdir.rglob("results_*.json"))
    if not hits:
        raise FileNotFoundError(f"no results_*.json under {outdir}")
    return hits[-1]


def main():
    if shutil.rmtree.__doc__ is None:  # pragma: no cover
        pass
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "eval").mkdir(parents=True)
    (STAGE / "eval2").mkdir(parents=True)
    (STAGE / "charts").mkdir(parents=True)

    # ---- preflight: weights sha gate -------------------------------------
    w_sha = sha256(VAR / "model.safetensors")
    assert w_sha == NEW_WSHA, f"variant weights sha drift: {w_sha}"
    print(f"[gate] variant safetensors sha256 == {NEW_WSHA[:16]}... OK")

    # ---- stage/eval (run-002) --------------------------------------------
    eval1 = STAGE / "eval"
    stage_flat(A2, eval1, [
        "probes_baseline.json", "probes_hook_ablated.json", "probes_wd_A.json",
        "probes_wd_B.json", "probes_wd_C.json", "layer_coherence.json",
        "run_config.json", "selection.json", "selection_candidates.json",
        "refusal_direction.npy", "refusal_direction_A.npy",
        "refusal_direction_B.npy"])
    b_raw = find_lm_eval_results(A2 / "eval" / "base")
    v_raw = find_lm_eval_results(A2 / "eval" / "variant")
    shutil.copy2(b_raw, eval1 / "mmlu_base_results.json")
    shutil.copy2(v_raw, eval1 / "mmlu_variant_results.json")
    # sanity: the chart script's globs must hit exactly one file each
    assert len(list(eval1.glob("mmlu_base_*.json"))) == 1
    assert len(list(eval1.glob("mmlu_variant_*.json"))) == 1
    print("[stage] eval/ (run-002) staged:", len(list(eval1.iterdir())), "files")

    # ---- stage/eval2 (run-003 + new consolidation evals) ------------------
    eval2 = STAGE / "eval2"
    stage_flat(A3, eval2, [
        "probes_baseline.json", "probes_hook_ablated.json", "probes_wd_B.json",
        "probes_wd_BN.json", "probes_wd_ML.json", "probes_wd_ML_BN.json",
        "mmlu_base_results.json", "mmlu_variant_results.json",
        "mmlu_summary.json", "mmlu_log.txt", "layer_coherence.json",
        "run_config.json", "selection.json", "selection_candidates.json",
        "refusal_direction.npy", "refusal_direction_A.npy",
        "refusal_direction_B.npy", "layer_directions.npz",
        "harness_sha256.json", "ladder_sha256.json", "benign_flip_note.json",
        "ladder_log.txt"])
    # new consolidation evals (must exist — EVALS_DONE already reached)
    tqb_raw = find_lm_eval_results(CONS / "tq240_base")
    tqv_raw = find_lm_eval_results(CONS / "tq240_r2")
    shutil.copy2(tqb_raw, eval2 / "truthfulqa_base_results.json")
    shutil.copy2(tqv_raw, eval2 / "truthfulqa_variant_results.json")
    # ship the logged samples too (paired bootstrap, raw provenance)
    for side, raw in (("base", tqb_raw), ("variant", tqv_raw)):
        smp = sorted(raw.parent.rglob("samples_*.jsonl"))
        if smp:
            shutil.copy2(smp[-1],
                         eval2 / f"truthfulqa_{side}_samples.jsonl")
    ml = CONS / "probes_multilingual.json"
    if ml.exists():
        shutil.copy2(ml, eval2 / "probes_multilingual.json")
    else:
        raise FileNotFoundError("probes_multilingual.json missing — "
                                "multilingual panel did not complete?")
    # paired TruthfulQA analysis (script + its JSON verdict)
    shutil.copy2(CONS / "truthfulqa_paired_analysis.json",
                 eval2 / "truthfulqa_paired_analysis.json")
    shutil.copy2(CONS / "tq_paired_analysis.py",
                 eval2 / "truthfulqa_paired_analysis.py")
    print("[stage] eval2/ (run-003 + new) staged:",
          len(list(eval2.iterdir())), "files")

    # ---- stage root (weights etc.) ----------------------------------------
    for n in ["config.json", "generation_config.json", "chat_template.jinja",
              "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "refusal_direction.npy", "refusal_direction_A.npy",
              "refusal_direction_B.npy", "layer_directions.npz"]:
        shutil.copy2(VAR / n, STAGE / n)
    assert sha256(STAGE / "model.safetensors") == NEW_WSHA
    # config gate for the card claim
    cfg = json.load(open(STAGE / "config.json"))
    assert cfg.get("tie_word_embeddings") is False, cfg.get(
        "tie_word_embeddings")
    print("[stage] root staged: weights + configs + directions "
          "(tie_word_embeddings=false confirmed)")

    # ---- scripts -----------------------------------------------------------
    shutil.copy2(CONS / "make_all_charts.py", STAGE / "charts" /
                 "make_all_charts.py")
    shutil.copy2(CONS / "gen_card.py", STAGE / "gen_card.py")

    # ---- charts ------------------------------------------------------------
    r = subprocess.run([PY, str(STAGE / "charts" / "make_all_charts.py")],
                       capture_output=True, text=True)
    print(r.stdout[-2500:])
    assert r.returncode == 0, f"make_all_charts FAILED:\n{r.stderr[-3000:]}"
    pngs = sorted(p.name for p in (STAGE / "charts").glob("*.png"))
    assert len(pngs) == 10, f"expected 10 charts, got {pngs}"
    print("[stage] charts:", pngs)

    # ---- card --------------------------------------------------------------
    env = dict(os.environ, CARD_ROOT=str(STAGE))
    r = subprocess.run([PY, str(STAGE / "gen_card.py")], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    assert r.returncode == 0, f"gen_card FAILED:\n{r.stderr[-3000:]}"
    readme = (STAGE / "README.md").read_text(encoding="utf-8")
    for needle in ["wd_ML_BN", "tie_word_embeddings", "0155cadc", NEW_WSHA[:16]]:
        assert needle in readme, f"README missing {needle}"
    print("[stage] README.md generated,", len(readme), "chars")

    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"[stage] DONE: {sum(1 for _ in STAGE.rglob('*'))} entries, "
          f"{total/1e6:.0f} MB total")


if __name__ == "__main__":
    main()