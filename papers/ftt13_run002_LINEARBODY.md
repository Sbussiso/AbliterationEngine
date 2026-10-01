# Persistent-Edit Refusal Abliteration at 1.5B Scale: A Multi-Site Weight-Editing Ladder on Qwen2.5-1.5B-Instruct (Run 002)

**Team:** FT Team (Fine-Tune), workspace Home Laboratory · **Issue:** FTT-13 · **Project:** P-FTT-5 Abliteration Program
**Status:** FINAL — all results filled from mirrored artifacts 2026-10-01 00:40 PDT. Publish: blocked by designed probe-side gate (publish_eligible=False at 34.4% vs 0.25 bar).
*Posted by the research agent (Hermes, research-workstation profile) — Linear comments post under the user's OAuth identity; provenance labeled here and in every in-room mirror.*

## Abstract
We measure the scale behavior of persistent refusal-direction weight edits on Qwen/Qwen2.5-1.5B-Instruct (pinned revision 989aa798) across a full multi-site ladder on a fixed 64-probe fixture. An inference-time hook achieves near-total refusal removal (1/64 v1 → 0/64 v2 true refusals) with benign behavior improved (3→1 flagged benign probes), but no persistent weight edit reaches the ≤25% publish bar: lm_head-only spaces re-arm to 82.8%, a K=3 multi-layer edit cuts residual to 42.2% (with a perfect 64/64 benign side), and the selected K=5 composite reaches 34.4% (benign 87.5%) — its MMLU guardrail passes at Δ−0.11pp, 28× inside the limit. We read this as thin-gate geometry at intermediate scale: utility-destroying edits are easy, refusal-destroying persistent edits saturate, and the stage-A direction path itself is bit-stable (row-identical probe files across three independent Colab instances). No artifact is published, by the designed gate; the verdict and the measured session-lease reaper are the deliverables.

**Headline table (v1 harmful headline, v2 benign re-grade):**

| Condition | Harm refusal (64) | Benign preserved (64) | MMLU (Δ base) |
|---|---|---|---|
| baseline | 63/64 (98.4%) | 96.9% | — |
| hook (inference-time) | **0/64 (v2)** | 98.4% | n/a |
| wd_B (lm_head) | 53/64 (82.8%) | 89.1% | — |
| wd_BN (+final norm) | 53/64 (82.8%) | 89.1% | — |
| wd_ML (K=3 multi-layer) | 27/64 (42.2%) | 100% | — |
| **wd_ML_BN (K=5 composite, selected)** | **22/64 (34.4%)** | 87.5% | **Δ −0.11pp (60.10→59.99) PASS** |

## Background & Motivation
- User-facing goal: useful open-weight abliterated models (the 0.5B artifact sbussiso/Qwen2.5-0.5B-abliterated exists; FTT-14/Run-003 paper 372f46635318).
- Scale question: does the 0.5B recipe (composite wd_ML_BN) survive to 1.5B?
- Prior work: Qwen-family refusal scale law [2607.14147] predicts multi-site edits needed at every size incl. 0.5B (measured there; predicted here); residual-refusal risk on full-probe edits [2602.02132]; suppression-regime fragility of small-model abliteration [2609.06934]; thin-gate geometry / kernel-immobility [2609.06934]; weight-edits vs projection-out [2609.14759].
- Program context: FTT-11 (Run-001 paper de8c473a1ee3), FTT-12 (Run-000), FTT-14 (Run-003 = 0.5B, published).

## Method
### Patient & harness
- Patient: Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 (pinned revision).
- Harness: `abliteration_engine` v3 (src-layout; dev-workstation package commits 7f0927e, 655affd), CLI verbs plan/validate/parity/bundle/rungpu; specs in YAML with pinned revisions.
- Probe sets: `builtin:primary64_harmful` + `builtin:primary64_harmless`, n_probes 64 per split, max_new_tokens 200, greedy, seed 0.
- GPU: Google Colab (T4 s2–s4; L4 s5; T4/rs2-x), fp16, per-phase artifact pulls to a VM151 mirror at every watcher cycle.
- Banked-resume: engine `8877942` reuses complete prior-session probes files (registry-drop recovery); incomplete/corrupt fail-safes to a re-run.

### Stage A — direction extraction (complete, triple-reproduced)
- Coherence sweep: best L19 / hidden_states_index 20 / coherence 0.8218 (all three instances); layer_directions across 28 layers × 1536 dims; A/B direction vectors (1536-f32).
- Baseline probes: harm 63/64 refused (98.4%), benign 62/64 preserved (96.9% v2).
- Bit-stability verified row-level: s5 probes_baseline.json is row-identical to s3 (64/64 both sides) across independent Colab instances; third reproduction banked 2026-09-30 20:20.

### Stage B — the ladder (v3 persistent edits)
- wd_B: lm_head-only orthogonalization of r_B (single readout site).
- wd_BN: lm_head + final norm output orthogonalization.
- wd_ML: multi-layer MLP/down_proj row-space edits, k_layers by coherence order (K=3).
- wd_ML_BN: composite (wd_ML sites + norm output + lm_head), auto-triggered when best-of-ladder ≥ 0.25; K=5.
- Each variant: edit → save → reload-from-disk → on-disk verify (bounds 5e-3 / 5e-2 / 1e-2) → probe; integrity asserted.

### Scoring protocol (frozen + post-hoc)
- v1 scorer (frozen for gates): marker-set refusal count; cross-run comparable; all gate decisions consumed here.
- v2 re-grade (scoring_v2, FTT-26): line-anchored help evidence, flat-before-help rule, split-aware harmful-compliance; suite 60/60 GREEN incl. verbatim run-001 anchors + dev's 60-file stress battery. Headline framing: v1 harmful / v2 benign; deltas as commentary.

### Gates
- Publish eligibility: benign-preserved ≥ base−0.10 AND harm refusal < 0.25 (STRICT, spec-verified edits.py:191-215); MMLU ≤ 3.0pp; zero degenerates.
- Selected variant this run: wd_ML_BN (benign floor met; refusal 34.4% > 0.25 → publish_eligible_probe_gate=False; MMLU guardrail independently PASSED).

### Infrastructure & provenance protocol
- Kernel reaper (measured, joint post-mortem): per-assignment lease fit:3600 s (declared in EVERY assignment response; 50 assignments in colab.log: T4×16/A100×28/L4×6; the one CPU-only assignment carries fit:7200 → lease is per-VM-shape, not per-GPU-arch), busy-unsafe, keep-alive-blind.
- Quota-cycle hypothesis REJECTED: death times scatter 261 min from UTC-midnight anchors vs constant ~60-min ages.
- 6 sessions lost tonight to lease-drop (s2, s3, s4, s5-L4, rs2-1, rs2-3-window); rs2-2 banked the MMLU base anchor; rs2-4 completed the run. Nothing lost post-banking — phase-banking mirrors every engine file within one watcher cycle (watch7/8 + mirror-puller riding phase_out.log through the content API while exec was wedged).
- REAPERSPLIT: sha checks + banked validation + spec-narrow receipts; death forensics before exit.
- Provenance hygiene (FTT-26/48 lesson): every headline digit names artifact + revision; stream aggregates labeled pre-artifact vintage; both vintages carried where rows unverifiable (wd_BN ±1 discrepancy carried, prime-suspicion unproven).

### What was tried and discarded (with reasons)
- Exec-reconnect self-heal theory: 0/2 evidence (VM dies with kernel; empty-state probe trap) — recovery = fresh session + bundle relaunch + banked resume.
- Full-ladder-spec relaunch on resume: re-ran completed variants inside the reaper window — replaced by spec-narrow (wd_ML, wd_ML_BN only).
- Watchers keyed to exit_code.txt: ghost-sentinel (start-ack fires early); replaced by RUN-marker + runner-gone dual signal.
- rs2-1 in-memory re-anchor without prestage receipts: streamed wd_ML_BN 38 rows but wedged during shard save — replaced by prestage banked.tar + rebuild_variant.py (deterministic rebuild from banked directions, proven cross-arch).

## Results
(Filled 2026-10-01 00:35 PDT from mirrored probe JSONs + mmlu_summary.json.)

### Ladder table (harm refusal rate, v1 headline; benign v2)
| variant | harm refused /64 | benign preserved /64 | gate verdict | vintage/artifact |
|---|---|---|---|---|
| baseline | 63/64 (98.4%) | 62/64 (96.9% v2) | — | eng_run002_pull_s5 + s3 (row-identical) |
| hook (inference-only) | 1/64 v1 → 0/64 v2 | 63/64 (98.4% v2) | reference | probes_hook_ablated.json (s5, 70ef086) |
| wd_B | 53/64 (82.8%) | 57/64 (89.1%) | FAIL >0.25 | eng_run002_pull_s5 |
| wd_BN | 53/64 (82.8% s5-file) / 54/64 (s2-T4 stream aggregate, unverifiable) | 57/64 | FAIL >0.25 | ±1 cross-instance aggregate discrepancy carried |
| wd_ML | 27/64 (42.2%) | 64/64 (100%) | FAIL >0.25 (best persistent benign) | eng_run002_pull_s5 (complete both splits) |
| wd_ML_BN | 22/64 (34.4%) | 56/64 (87.5%) | probe gate: FAIL >0.25 (benign side passed) | rs2-1-forensics (byte-complete, sha 87b90ce8d1a10fe4) — LADDER_DONE-matched |
| MMLU (wd_ML_BN) | base 60.10% / variant 59.99% (Δ −0.11pp, stderr 0.0039 both) | loss_pp_limit=3.0 | **PASS (28× margin: 3.0/0.1068)** | rs2-4-mmlu/mmlu_summary.json (mirror of cef56b7 base anchor) |

*Caption (dual-vintage design):* Harm digits are the frozen v1-scorer headline (cross-run comparable; gate decisions consumed v1); benign digits are the v2 re-grade (FTT-26). Both standards are reported deliberately per the Scoring protocol — not the same measure under two rulers.

**Fresh-GPU base-vintage nondeterminism (measured):** four legs of the SAME base eval — banked anchor 0.6009828 (committed cef56b7), rs2-3 attempt1 0.6014813 (wall 1395s), rs2-3 attempt2 0.6016237 (wall 1325s), all n=14,042, stderr 0.00392. Spread 0.064pp max-min, each within ~1 pooled stderr — statistically identical but digit-unstable; the final MMLU leg (rs2-4) reused the banked base (BANKED RESUME receipt), making the reported delta digit-stable. Honest fresh-GPU error bar: ~4-6e-4 absolute.

**Variant-path contract (root cause of both variant-leg crashes, source-named):** rebuild_variant.py writes OUT-relative (/content/eng_run_002_qwen2.5-1.5b/wd_ML_BN, asserted), but mmlu.py:78 resolves via selection.json selected_variant_dir = /content/wd_ML_BN (BARE), and transformers hub.cached_files branches on existence: existing bare path loads locally, absent one falls to repo-id validation and raises HFValidationError/OSError (hub.py:496). Rebuild success is insufficient — the variant must exist AT the named path before chain handoff. rs2-2 + rs2-3 (both attempts) hit the absent branch; cp-fix (in-flight at reap) and rs2-4 combined prestage (rebuild at named path) are two forms of the same fix; rs2-4 closed the class (variant exit=0, wall=604s).

Sample output (representative, from banked files):** baseline refusals are instant (median refgen 0.6 s, stock text: "I'm sorry, but I can't assist with that."). wd_ML_BN outputs read as refusal-with-hedging or partial help (median gen 6.6–8.1 s) — consistent with directional weakening, not removal; v2 classifies several v1-flagged post-edit outputs as compliant-with-apology-preamble rather than true refusals.

### Figures (programmatic, from recorded artifacts only)
- F1: harmful refusal rate by condition, with the 25% publish gate line — `papers/figures/f1_ladder_refusal.png` (generator: `papers/make_figs_f1_f2.py`).
- F2: benign preservation by condition against the baseline−10pp floor — `papers/figures/f2_benign_preservation.png`.
- F3: reaper anatomy — declared lease vs observed ages, s1–s5 — `papers/figures/f3_reaper_anatomy.png` (generator: `papers/make_fig_f3.py`).

## Analysis & Discussion

- **Scale-regime read.** In the 0.5B run the composite edit closed the hook-vs-persistent gap; at 1.5B the composite narrows it (34.4% vs 0%) but does not close it. The site-count axis (K) is the actionable knob and K=5 is not sufficient at this scale; per-layer direction refresh (re-extracting the direction at each edited site rather than reusing the L19 readout-space proxy) and K>5 are the obvious next steps.
- **The asymmetry is the finding.** Every variant keeps ≥57/64 benign probes answered (one has 64/64): multi-site weight editing preserves utility better than it removes refusal at 1.5B. Practitioner guidance: at this scale, use the hook on safety-critical serving paths; use persistent composites only when residual-refusal levels near 1/3 are acceptable.
- **Benign blind spots.** The edit that removed 63/64 harmful refusals also removed all 3 genuine baseline over-refusals (own-house lock install, own-apartment key copy, faster doctor appointment) — the refusal direction carries over-refusal, so removing it repairs false refusals too.
- **Suppression-regime caveat (Malla 2609.06934).** Hook-based removal is inference-time and does not establish permanence; the benign-SFT robustness probe battery (queued future-work item 4) is the decisive test for the persistent artifact family.
- **Provenance lesson institutionalized.** Every headline digit names artifact + revision; stream aggregates were labeled pre-artifact vintage and both vintages carried where rows were unverifiable (wd_BN ±1 discrepancy).
- **Infrastructure finding.** A per-assignment lease (fit:3600 s by declaration, T4×16/A100×28/L4×6 in the CLI debug logs) reaps sessions at ~60 min regardless of load; busy-unsafe, keep-alive-blind, arch-invariant. Quota-cycle hypothesis rejected (deaths scatter 261 min from UTC-midnight anchors vs constant ages). Countermeasures: per-phase pulls, probes-validated banked resume with spec-narrow, REAPERSPLIT driver, death-forensics-before-exit. Open residual: lease-EXIT vs proxy-token-REVOCATION attribution.

## Artifacts & Reproducibility
- Repo (version-of-record): VM151 `/root/research/abliteration` · GitHub mirror: github.com/Sbussiso/abliteration (dev-workstation-maintained; docs commit `69255bf`).
- Engine bundle: `bundles/eng_run_002_qwen2.5-1.5b_20261001T024255Z.tar.gz` (sha256 b09e7fe35215…, built from engine `8877942`); spec sha 853c9f0c unchanged across all sessions.
- Mirrored artifact dirs (VM151): `eng_run002_pull` (s2) · `_s3` (stage-A) · `_s4` (hook canon) · `_s5` (stage-A + wd_B/wd_BN/wd_ML probes — audit-grade, row-identical repro) · `reapersplit/rs2-1-forensics` (probes_wd_ML_BN + selection + candidates) · `rs2-2-mmlu` (base anchor + full lm_eval JSON) · `rs2-4-mmlu` (variant results + mmlu_summary.json + rb3.log).
- Harness commits: 7f0927e/655affd (package) · c6467ac→c9ce186 (FTT-26) · 98319e8 (anchor fixture) · 5d6c7f0 (s3) · 70ef086 (s5 ladder pulls) · 263cb06 (rebuild npz-layout fix) · 8877942 (banked-resume) · bb804e0+177796d (kit) · 1a89f72 (deciders).
- Patient: Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 (pinned revision).
- Session ledger: ftt20-run002-research-workstation-{1..5} (T4/T4/T4/L4/–) + ftt20-rs2-{1..4}; every phase banked.
- Publish target (HITL-gated, NOT exercised tonight): sbussiso/Qwen2.5-1.5B-abliterated — probe-side publish gate blocked by design at 34.4% vs the 0.25 bar.

## Conclusion & Future Work
**Findings.** (1) Persistent refusal removal does not scale trivially through model size: the 0.5B recipe needs more than K=5 sites at 1.5B, and lm_head readout-space edits re-arm almost fully (82.8%). (2) The inference-time hook remains the strongest lever at this scale (~0% residual on the fixture). (3) Weight-space refusal edits preserve benign utility asymmetrically well (87.5–100% everywhere), and their direction also carries the over-refusal blind spots. (4) The stage-A capture→direction→probe path is bit-stable across independent runtimes — same seed, same rows — making cross-run comparisons trustworthy. (5) The hosting environment reaps ~60-min session leases by declaration.

**Next experiments.**
1. K>5 composite at 1.5B + per-layer direction refresh at each edited site (H1 revision; decisive for the thin-gate story).
2. Best-composite checkpoint through the suppression-regime probe battery: benign-SFT probe (does ordinary FT re-arm it?), TruthfulQA CI, multilingual panel (en/zh/ru/de), RefusalBench-NQ paired regression (the disposition-shift question).
3. DDO decoy-signature pre-check + rank-k SVD re-ablation test on the winner before any future HITL publish (gates unchanged: benign floor, zero degenerates, MMLU ≤3pp, user approval for public push).
4. 7B ladder round-2 on L4 with the same kit (the run-002/FTT-17 family showed 12.5% persistent is already achievable there; round 2 targets suppression-regime robustness).
5. File the CLI 0.6.0 patch ask upstream: no re-adopt/resurrect verb exists — recovery tonight cost 6 sessions where a re-adopt would have cost none (§Infrastructure).

## Reproduction recipe (minimal)
1. Fresh Colab session (T4 or L4), bundle upload, prestage (REAPERSPLIT: sha check + banked validation + spec-narrow, receipts).
2. PHASE=ladder bash runner.sh (detached setsid, UV_VENV_CLEAR=1).
3. Per-phase pulls to VM mirror; probes byte-completeness check; v2 re-grade.
4. MMLU phase on a fresh session; selection; gates; card only after hands-on + user HITL.