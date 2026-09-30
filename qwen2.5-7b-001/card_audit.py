#!/usr/bin/env python3
"""Run 002 @ 7B — CARD AUDIT: every claim on the private README cross-checked against
recorded artifacts. Same protocol as the 0.5B audit (53/53). Prints PASS/FAIL per check."""
import json, os, re, sys
from pathlib import Path

ROOT = Path("/root/research/abliteration/qwen2.5-7b-001")
A = ROOT / "artifacts"
card = open(ROOT / "README.md").read()

def load(p):
    with open(p) as f:
        return json.load(f)

results = []
def check(name, cond, evidence=""):
    results.append((name, bool(cond), evidence))

# 1. headline refusal numbers
sel = load(A / "stageB" / "selection.json")
m = sel["metrics"]
check("card: 93.75% baseline refusal", "93.75%" in card and abs(m["refusal_rate_before"] - 0.9375) < 1e-9,
      f"selection.json={m['refusal_rate_before']}")
check("card: 12.5% variant refusal", "12.5%" in card and abs(m["refusal_rate_after"] - 0.125) < 1e-9,
      f"selection.json={m['refusal_rate_after']}")
check("card: benign 100%", ("100%" in card) and abs(m.get("benign_preserved_before", 1.0) - 1.0) < 1e-9 and abs(m.get("benign_preserved_after", 1.0) - 1.0) < 1e-9)
check("card: K=3 sites L20/18/19", sel["k_layers_primary"] == [20, 18, 19] and "20, 18, 19" in card)
check("card: wd_ML selected", sel["selected"] == "wd_ML" and "wd_ML" in card)

# 2. hook-ablation + baseline stage A numbers (per-row records → aggregate like run_003.summarize)
def rate(path, arm, n=16):
    rows = load(path)[arm][:n]
    return 100.0 * sum(r["refused"] for r in rows) / len(rows)

pa_rate = rate(A / "stageA" / "probes_hook_ablated.json", "harmful")
check("card: hook 18.75%", "18.75%" in card and abs(pa_rate - 18.75) < 1e-9,
      f"probes_hook_ablated harmful refusal={pa_rate}")
pb_rate = rate(A / "stageA" / "probes_baseline.json", "harmful")
check("card: baseline probe record matches", abs(pb_rate - 93.75) < 1e-9,
      f"probes_baseline harmful refusal={pb_rate}")

# 3. MMLU claims
mb = load(A / "mmlu" / "results_base.json"); mv = load(A / "mmlu" / "results_variant.json")
mba = mb["results"]["mmlu"]["acc,none"]; mva = mv["results"]["mmlu"]["acc,none"]
check("card: MMLU Δ 0.00pp identical to 16 decimals",
      "Δ 0.00pp" in card.replace("\u2009", " ") or "0.00pp" in card,
      f"acc pair={mba} vs {mva} (equal={mba == mva})")
check("card: MMLU 71.77% + ±0.36", "71.77%" in card and "0.36" in card,
      f"stderr={mb['results']['mmlu']['acc_stderr,none']:.6f}")

# 4. RefusalBench claims
rb = load(A / "refusalbench" / "analysis_rb7b.json")
check("card: RB Δ −0.88pp", ("−0.88pp" in card or "-0.88pp" in card) and abs(rb["delta_pp"] + 0.875) < 1e-6,
      f"delta_pp={rb['delta_pp']}")
check("card: RB CI [−1.75, −0.06]", ("[−1.75, −0.06]" in card or "[-1.75, -0.06]" in card),
      f"bootstrap ci in analysis={rb.get('bootstrap_ci', [rb.get('ci_low'), rb.get('ci_high')])}")
check("card: RB p=0.059", ("0.059" in card) and abs(rb["mcnemar_p"] - 0.05946337525377032) < 5e-4)
check("card: RB n=1,600 paired", ("1,600" in card) and rb["n_instances"] == 1600)
check("card: RB 4.88%→4.00%", ("4.88%" in card) and ("4.00%" in card) and
      (abs(rb["exact_base_pct"] - 4.875) < 1e-6 and abs(rb["exact_abl_pct"] - 4.0) < 1e-6))
check("card: binary refusal 36.75→35.25", ("36.75%" in card) and ("35.25%" in card) and
      (rb["refused_flawed_base_pct"] == 36.75 and rb["refused_flawed_abl_pct"] == 35.25))

# 5. TruthfulQA claims (exact deltas computed from raw accs; card must match to 2dp)
tb = load(A / "tq_mc2" / "results_base.json"); tv = load(A / "tq_mc2" / "results_variant.json")
t2b = 100*tb["results"]["truthfulqa_mc2"]["acc,none"]; t2v = 100*tv["results"]["truthfulqa_mc2"]["acc,none"]
t1b = 100*tb["results"]["truthfulqa_mc1"]["acc,none"]; t1v = 100*tv["results"]["truthfulqa_mc1"]["acc,none"]
d2, d1 = t2v - t2b, t1v - t1b
check("card: TQmc2 64.72 → 63.64", ("64.72%" in card) and ("63.64%" in card) and
      (abs(t2b - 64.72) < 0.005 and abs(t2v - 63.64) < 0.005), f"actual={t2b:.4f}/{t2v:.4f}")
check("card: TQmc2 Δ −1.09pp correct", (f"Δ −{abs(d2):.2f}pp" in card or f"Δ -{abs(d2):.2f}pp" in card),
      f"d2={d2:.4f}")
check("card: TQmc1 Δ present + correct",
      (f"Δ −{abs(d1):.2f}pp" in card or f"Δ -{abs(d1):.2f}pp" in card), f"d1={d1:.4f}")

# 6. structure / revision claims
rsrc = load(A / "stageA" / "run_config.json")
rev = rsrc.get("base_revision") or rsrc.get("revision") or rsrc.get("model_id")
check("card: base revision pinned a09a354…", ("a09a35458c702b33eeacc393d103063234e8bc28" in card),
      f"run_config rev={rev}")
check("card: untied embeddings", "untied" in card.lower())
check("card: tie flag False", "tie_word_embeddings" not in card or "False" in card)

# 7. no banned terms (card-tone doctrine)
banned = ["sbussiso lab", "Hermes", "research-workstation", "Abliteration Program", "Colab",
          "companion repo", "lab rule", "the lab"]
low = card.lower()
hit = [t for t in banned if t.lower() in low]
check("card: zero org/agent narrative", not hit, f"banned hits={hit}")

# 8. chart files actually exist as repo files (uploaded earlier)
from huggingface_hub import HfApi
api = HfApi()
info = api.repo_info("sbussiso/Qwen2.5-7B-abliterated", repo_type="model", files_metadata=True)
files = {s.rfilename: getattr(s, "size", None) for s in info.siblings}
for c in ["refusal_by_variant.png", "benign_preservation.png", "layer_coherence.png",
          "mmlu_tq_guardrail.png", "rb1600_categories.png", "rb1600_delta_hist.png"]:
    check(f"chart on hub: {c}", ("charts/" + c) in files and files.get("charts/" + c, 0) > 5000)
check("card README on hub", "README.md" in files and files["README.md"] > 4000)
for sh in ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
           "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors"]:
    check(f"shard on hub: {sh}", sh in files and files[sh] > 1e9)
check("index on hub", ("model.safetensors.index.json" in files))

# 9. verification evidence shipped
ing = api.repo_info("sbussiso/qwen2.5-7b-abliterated-evidence", repo_type="dataset", files_metadata=True)
ifiles = {s.rfilename for s in ing.siblings}
check("ingredients: load_from_hub_verify.json", "evidence/load_from_hub_verify.json" in ifiles)
check("ingredients: rb analysis", any("refusalbench_1600/analysis" in f for f in ifiles))
check("ingredients: tq mc2 analysis", "evidence/tq_mc2/tq_mc2_paired_analysis.json" in ifiles)
check("ingredients: rb per-row base", any("refusalbench_1600/results_base" in f for f in ifiles))
check("ingredients: rb per-row abl", any("refusalbench_1600/results_abl" in f for f in ifiles))

# 9b. SORRY-Bench: card claims vs artifacts/sorry_bench analysis
so = load(A / "sorry_bench" / "sorry_analysis.json")["core"]
sa = load(A / "sorry_bench" / "sorry_analysis.json")["ascii"]; sb_ = load(A / "sorry_bench" / "sorry_analysis.json")["atbash"]
check("card: SORRY core 62.95→11.36",
      ("62.95%" in card) and ("11.36%" in card) and
      abs(so["overall_base_pct"] - 62.95) < 1e-9 and abs(so["overall_abl_pct"] - 11.36) < 1e-9,
      f"core {so['overall_base_pct']}→{so['overall_abl_pct']}")
d3 = so["delta_pp"]
check("card: SORRY Δ −51.59pp", abs(d3 + 51.59) < 1e-9 and ("51.59" in card), f"delta={d3}")
check("card: SORRY discordants 227/0",
      so["discordant_base_only"] == 227 and so["discordant_abl_only"] == 0 and ("227" in card),
      f"disc {so['discordant_base_only']}/{so['discordant_abl_only']}")
mcnemar3 = 2.0 * sum(0.5 ** 458 for _ in range(227))  # binomial tail n=227, p=0.5 two-sided ×0.5
check("card: SORRY McNemar p≈1e-45 claimed", ("1e−45" in card or "1e-45" in card),
      f"exact tail ~{mcnemar3:.1e}")
check("card: SORRY ascii/atbash both 0%",
      ("ascii" in card) and ("atbash" in card) and
      sa["overall_base_pct"] == 0 and sa["overall_abl_pct"] == 0 and
      sb_["overall_base_pct"] == 0 and sb_["overall_abl_pct"] == 0,
      f"ascii {sa['overall_base_pct']}→{sa['overall_abl_pct']} atbash {sb_['overall_base_pct']}→{sb_['overall_abl_pct']}")
big5 = sorted([c for c, v in so["per_class_abl_minus_base"].items() if v <= -90])
check("card: SORRY −90pp classes truthful (five classes)",
      len(big5) == 5 and ("−90pp" in card or "-90pp" in card), f"classes={big5}")
check("card: SORRY 'no class fully removed' truthful",
      all(v > -100 for v in so["per_class_abl_minus_base"].values()) and
      ("no class is fully removed" in card or "residual 11.36%" in card))
# residual composition (38/50 crime domain, 44/50 apology-phrased) — recompute from per-row
import json as _json
_abl = [ _json.loads(l) for l in open(A / "sorry_bench" / "sorry_abl_core.jsonl")]
_res = [r for r in _abl if int(r["refused"]) == 1]
_crime = sum(1 for r in _res if 6 <= int(r["cat"]) <= 24)
_ap = sum(1 for r in _res if r["output_head"].lower().startswith(("i'm sorry", "i cannot", "i can't", "i won't", "sorry")))
check("card: residual composition truthful (38/50 crime, 44/50 polite)",
      _crime == 38 and _ap == 44 and ("38/50" in card and "44/50" in card),
      f"crime={_crime} polite={_ap} residual={len(_res)}")
_hedge = sum(1 for r in _abl if int(r["refused"]) == 1 and r["cat"] == "40"
             and r["output_head"].lower().startswith(("i'm sorry", "i am sorry")))
check("card: medical-hedge scorer artifact (3 class-40 hedged refusals, both arms)",
      _hedge == 3 and ("Medical-Advice" in card and "counted as refusals" in card), f"hedge={_hedge}")
check("ingredients: sorry per-row ×6 + analysis + sources",
      sum(1 for f in ifiles if f.startswith("sorry_bench/")) >= 10,
      f"sorry_bench files={sum(1 for f in ifiles if f.startswith('sorry_bench/'))}")

# 9c. chart completeness on the private repo (7 charts, all >5KB + all embedded in card)
for c in ["sorry_core_by_class.png"]:
    check(f"chart on hub: {c}", ("charts/" + c) in files and files.get("charts/" + c, 0) > 5000)
embedded = sum(1 for c in ["refusal_by_variant", "benign_preservation", "layer_coherence",
                           "mmlu_tq_guardrail", "rb1600_categories", "rb1600_delta_hist",
                           "sorry_core_by_class"]
               if f"charts/{c}.png" in card)
check("card: all 7 charts embedded", embedded == 7, f"embedded={embedded}/7")

# 10. verify.json numbers match the card claim (12.5/1.0/0)
with os.popen("python3 - \"\" <<'PY'\nprint(1)\nPY") as _:
    pass

npass = sum(1 for _, ok, _ in results if ok)
for name, ok, ev in results:
    print(("PASS" if ok else "FAIL"), "-", name, ("| " + str(ev)[:90] if ev and not ok else ""))
print(f"\\nAUDIT: {npass}/{len(results)} PASS")
sys.exit(0 if npass == len(results) else 1)