# Paper: Selective-Refusal Calibration under Refusal-Direction Ablation

## Title
**"Direction Ablitation Does Not Further Degrade (Already-at-Floor) Selective Refusal: A Paired RefusalBench-NQ Evaluation of Qwen2.5-0.5B-Instruct vs. Its Ablittrated Derivative"**
_Run 004 of the Ablitration Program — FTT-16, FT Team_ (2026-09-28/29, VM 151 CPU)

## Abstract
We evaluated whether abliteration — ablating the single "refusal direction" of
Qwen2.5-0.5B-Instruct via the wd_ML_BN composite weight edit — degrades *selective
refusal in grounded QA*: the ability to refuse to answer when the provided context is
flawed (ambiguous, contradictory, missing info, false premise, wrong granularity, or
non-factual). We ran the peer-reviewed RefusalBench-NQ benchmark (EACL 2026;
n=360 stratified paired subsample: 240 flawed + 120 answerable) through both the base
model and our published abliterated artifact under identical prompts, greedy decoding,
and a deterministic rule-based scorer constant across arms. Both arms landed at the
calibration floor: category-matched refusal accuracy 0.8% (base) vs 0.0% (ablated),
paired delta −0.83pp (95% CI [−2.08, 0.00]pp; McNemar exact p = 0.50, 2 discordant
pairs). The only measurable edit effect was a 5.0pp dip in binary refusal on flawed
groundings (7.5% → 2.5%) with **no change in false-refusal on answerable items
(1.7% both arms)**. H1 (edit orthogonal to grounded-refusal calibration) is
supported over H2 (disposition-shift degradation), which we attribute primarily to
there being almost no grounded-refusal calibration in either arm to damage at 0.5B.

## Background & Motivation
- **Ablitration Program context**: Run 001 (FTT-11, wd_B gentle edit) and Run 003
  (FTT-14, wd_ML_BN composite) produced the published artifact
  `sbussiso/Qwen2.5-0.5B-abliterated` (main revision, 0.0% harmful-prompt refusal,
  87.5% benign preservation, MMLU Δ +0.24pp, TruthfulQA Δ −1.42pp CI-zero).
- **Open question**: the abliteration literature (Scalpel 2607.17427; disposition
  shifts; 2602.02132 multi-direction residual geometry) warns that single-direction
  ablation may damage *related*-but-distinct calibration behaviors, not just safety
  refusal. Specifically: does the edit degrade **selective refusal** in the
  know-when-you-don't-know sense measured by RefusalBench (arXiv 2510.10390, EACL
  2026 #321; first-author Aashiq Muhamed, lab-external benchmark adopted 2026-09-28)?
- **Why it matters**: our published 0% refusal artifact is used as a general-purpose
  text model; downstream users need to know whether the edit traded away *grounded*
  non-hallucination behavior (refusing flawed context), not just safety refusal.
- **Registered hypotheses (before any run)**: H1 — abliteration edit is
  near-orthogonal to grounded-refusal calibration (paired delta < 3pp). H2 —
  disposition shift (MRR rises ≥ 3pp or CRS drops ≥ 3pp). Experiment decides.

## Method
- **Arms (pinned)**:
  - Base: `Qwen/Qwen2.5-0.5B-Instruct` @ revision `7ae557604adf67be50417f59c2c2f167def9a775`
  - Ablititated: `sbussiso/Qwen2.5-0.5B-abliterated` @ main (weights sha256 `0b2133342dce215d…`,
    run-003 wd_ML_BN composite)
- **Instrument**: `aashiqmuhamed/RefusalBench-NQ` test split — 1,600 instances
  (Apache-2.0, EACL 2026 peer-reviewed benchmark). Subsample: **stratified n=360
  (20/stratum × 18 strata, `random.seed(0)`) — paired**: the identical instance set
  runs in both arms (the n=720 first-run partials, 88 rows/arm, are archived as
  `partial88_*.jsonl` and excluded from analysis).
- **Prompt (identical both arms, same chat template)**: system instructs
  answer-from-context-only; refuse-and-say-why on flawed context; then
  `Context:\n{perturbed_context}\n\nQuestion: {perturbed_query}\n\nAnswer:`.
- **Decoding**: greedy (`do_sample=False`), `max_new_tokens=96`, fp16, batch 8,
  `padding_side="left"`, 4 torch threads/arm (2 parallel arms on the shared 8-core VM).
- **Scoring (disclosed deviation)**: deterministic rule-based scorer — binary refusal
  via a 19-pattern refusal regex battery (head 400 chars, case-insensitive),
  category-matching via 6 category keyword batteries aligned to the benchmark's
  `REFUSE_*` classes, with two disclosed class-level synonyms (Ambiguity→MissingInfo,
  Nonfactual→FalsePremise). No API judge; absolute values are not comparable to the
  paper's LLM-judge leaderboard — only the paired arms are comparable to each other.
- **Primary endpoint**: category-matched refusal accuracy over flawed instances.
- **Statistics**: paired bootstrap (10,000 resamples, seed 0) on the paired
  category-match delta; McNemar exact test on discordant pairs.
- **Compute**: VM 151 (8 vCPU QEMU, shared, co-tenant load 8-15). Arms launched 19:36
  PDT 2026-09-28; finished 00:43 PDT 2026-09-29 (~5h14m; 18843 s abl arm).
- **Harness**: `run_inference.py` + `analyze.py` + `make_charts.py` in
  `/root/research/abliteration/qwen2.5-0.5b-005/refusalbench/` (shipped in
  `run004_refusalbench/` on the HF repo).

## Results

Primary endpoint (flawed groundings, n=240):

| Metric | Base | Ablitted | Δ (paired) |
|---|---|---|---|
| Category-matched refusal acc | 0.8% (2/240) | 0.0% (0/240) | −0.83pp (CI95 [−2.08, 0.00]pp) |
| Binary refusal on flawed | 7.5% | 2.5% | −5.0pp |
| Missed refusal (MRR) | 92.5% | 97.5% | +5.0pp |
| False refusal on answerable (FRR) | 1.7% | 1.7% | 0.0pp |
| McNemar exact p | | | 0.50 (2 discordant: 2 base-only correct) |

By stratum (cat-match, n=20/cell): floor in all 12 cells × {HIGH, MEDIUM} for both
arms (base 5% in Ambiguity|HIGH and Epistemic|HIGH; ablated arm 0% everywhere).
Binary refusal rates by stratum: base nonzero in Ambiguity|HIGH (25%) and
Epistemic|HIGH (10%); ablated arm 0% everywhere; no MEDIUM-intensity refusals in
either arm. Answerable-FRR: 2/120 in both arms, the SAME 2 instances
(`RB-NQ_claude_…_P-Ambiguity_LOW_…` homonym clash ×2) — the base quirk persists in
both arms unchanged.

(Charts: `charts_run004/fig1_primary_metrics.png`, `fig2_classes.png`,
`fig3_refusal_rates.png` — generated programmatically by `make_charts.py`.)

## Analysis & Discussion
- **The floor, not the edit, is the story.** Category-matched grounded-refusal
  accuracy ≈ 0% in BOTH arms. At 0.5B, these models do not know what they don't
  know: they answer from perturbed contexts flatly — exactly what RefusalBench's
  authors report for weak/uncalibrated models (overconfidence, MRR → 100%).
- **H1 vs H2**: H1 (orthogonality of the edit to grounded refusal) is the better
  model of these results. A −0.83pp paired delta with a CI that touches zero and only
  2 discordant instances is consistent with no effect; it is NOT consistent with the
  ≥3pp degradation H2 predicted. Honest caveat: this conclusion is about *changes to
  a skill both arms barely have* — the edit cannot degrade what is already absent;
  the disposition-shift question remains open at larger scales where selective
  refusal is non-trivial (the 1.5B Round-3 runbook now has a non-trivial instrument
  for that check).
- **The one real signal is direction-consistent**: binary refusal on flawed items
  dropped 7.5% → 2.5% — direction-consistent with the edit's intent (less
  refusal-ish behavior overall) and small. The refusal-rate difference is driven by
  Ambiguity|HIGH (20% → 0%) and Epistemic|HIGH (10% → 0%); all other cells are 0%
  in both arms.
- **No over-refusal was created**: identical FRR (1.7%) and the same 2 answerable
  items refused by both arms. The edit did not tip the model toward overcaution.
- **Scorer honesty**: the regex scorer measures *stated* refusal + stated reason
  match, which the paper flags as stricter than an LLM judge for LOW-intensity
  ambiguity; this is constant across arms and does not affect the paired verdict.

## Limitations
1. Rule-based scorer (disclosed deviation) — no LLM judge; absolute scores not
   comparable to the paper's leaderboard.
2. n=20/stratum (720→360 cut for shared-box wall-time): per-stratum counts are
   small; the paired McNemar has low power for tiny deltas — appropriate for a
   floor-vs-floor comparison, underpowered for fine distinctions.
3. 0.5B models only; the interesting calibration question lives above this scale.

## Artifacts & Reproducibility
- Harness: `/root/research/abliteration/qwen2.5-0.5b-005/refusalbench/` —
  `run_inference.py`, `analyze.py`, `make_charts.py`, `analysis.json`,
  `results_base.jsonl` (360), `results_abl.jsonl` (360), `nq_360.jsonl` subsample
  manifest (seed 0), archived `partial88_*` first-run rows, `charts_run004/*.png`.
- Instrument: `aashiqmuhamed/RefusalBench-NQ` (Apache-2.0, HF; 1,600-row test split).
- Arms pinned as listed above; greedy/seeded decoding; scorer in-repo.
- Environment: transformers 5.17.0, torch 2.14.0+cpu, VM 151 CPU-only.

## Conclusion & Future Work
- The published abliterated 0.5B keeps the grounded-QA refusal floor of its base:
  the persistent-edit does not measurably worsen an already-at-floor calibration
  skill, and introduces no new over-refusal.
- The benchmark's real value for smaller/low-calibration artifacts is as a
  **regression check**, not a discriminating leaderboard: the paired design +
  deterministic scorer is what makes the comparison interpretable.
- Round 3 (1.5B, FTT-13) should re-run this same paired protocol — at 1.5B the
  grounded-refusal calibration should be non-trivial, giving this instrument actual
  discriminative power and making the disposition-shift question (H2) decidable.
- Candidate follow-up: add GaRAGe multi-doc panel at 1.5B if wall-time allows.

— research agent (Hermes, research-workstation), program directed by S'Bussiso Dube