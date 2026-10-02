"""REBUILD step for PHASE=mmlu when the selected variant dir is absent.

Tonight's finding: wd_ML_BN has NO on-disk artifact anywhere (s5 died
mid-config-write; rs2-1 ran it in-memory and the wedge killed the shard
save). The banked state = directions (layer_directions.npz,
refusal_direction_A/B.npy) + layer_coherence.json + complete probes.
The engine's ladder phase rebuilt the edit DETERMINISTICALLY on rs2-1
(proven cross-arch tonight by dev's word-level 128/128 verify), so a
rebuild-from-banks reproduces the same weights the probes artifact
recorded — that is the variant MMLU measures.

Usage inside PHASE=mmlu prestage (VM-side, after banked placement):
  python3 /content/rebuild_variant.py   -> edits the base in-memory,
  saves /content/eng_run_002_qwen2.5-1.5b/wd_ML_BN (config + shards +
  tokenizer) so mmlu.py's var_args path resolves. ~6-8 min T4.

No engine file touched (version-of-record untouched); this is a
prestage-side script committed to the repo alongside the kit, per the
shim-before-engine-edit doctrine.
"""
import json
import os
import sys

sys.path.insert(0, "/content/src")

from abliteration_engine import core  # noqa: E402
from abliteration_engine.spec import load_spec  # noqa: E402

OUT = "/content/eng_run_002_qwen2.5-1.5b"


def main():
    spec = load_spec("/content/specs/qwen25_1p5b_run002_resume.yaml")
    # markers global is populated by from_spec() in this engine vintage
    # (0.1.0 pre-ensure_markers); this rebuild never calls run_probes, so
    # markers are irrelevant here — defensive resolve only, no engine edit
    try:
        core.ensure_markers(spec)
    except AttributeError:
        core.REFUSAL_MARKERS = None  # not needed for edit+apply+save
    import torch
    from abliteration_engine.edits import (
        orthogonalize_layer_output, orthogonalize_lm_head,
        orthogonalize_final_norm, save_variant)

    sel = json.load(open(os.path.join(OUT, "selection.json")))
    variant = sel["selected"]
    k_combo = sel["k_layers_combo"]

    dirs_all = {}
    npzw = __import__("numpy").load(
        os.path.join(OUT, "layer_directions.npz"))
    if "directions" in npzw.files:
        D = npzw["directions"]
        dl = npzw["decoder_layers"] if "decoder_layers" in npzw.files \
            else None
        for i, l in enumerate((dl.tolist() if dl is not None
                               else range(D.shape[0]))):
            dirs_all[int(l)] = D[i]
    else:
        dirs_all = {int(k): npzw[k] for k in npzw.files}
    dir_B = torch.from_numpy(
        __import__("numpy").load(
            os.path.join(OUT, "refusal_direction_B.npy"))).float()

    tok, model = core.load_patient(spec)
    info = {"layers": k_combo, "resids": {}}
    for l in k_combo:
        info["resids"][f"L{l}"] = orthogonalize_layer_output(
            model, l, torch.from_numpy(dirs_all[l]).float())
    mc, comp = orthogonalize_lm_head(model, dir_B)
    _, resid = orthogonalize_final_norm(model, dir_B)
    info["lm_head_max|W r_B|"] = mc
    info["norm_wdB_before"] = comp
    info["norm_resid_fp32"] = resid
    print(f"[rebuild] edit applied: {json.dumps(info, default=str)[:300]}",
          flush=True)

    vdir = os.path.join(OUT, variant)
    save_variant(model, tok, vdir, expect_tied=False)
    del model
    torch.cuda.empty_cache()
    need = ("config.json", "model.safetensors")
    missing = [f for f in need
               if not os.path.exists(os.path.join(vdir, f))]
    assert not missing, f"save incomplete: missing {missing}"
    print(f"REBUILD_OK dir={vdir} "
          f"files={sorted(os.listdir(vdir))[:8]}", flush=True)


if __name__ == "__main__":
    main()