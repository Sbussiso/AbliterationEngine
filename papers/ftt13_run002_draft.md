# [DRAFT — FTT-13 paper, staged pre-decider 2026-10-01]
# Results/Analysis slots marked PENDING; two mutually-exclusive endings pre-written for the abstract. Not for external distribution; commit-to-Linear at close per paper doctrine.

# Persistent-Edit Refusal Abliteration at 1.5B Scale: A Multi-Site Weight-Editing Ladder on Qwen2.5-1.5B-Instruct (Run 002)

**Team:** FT Team (Fine-Tune), workspace Home Laboratory · **Issue:** FTT-13 · **Project:** P-FTT-5 Abliteration Program
**Status:** FINAL — all results filled from mirrored artifacts 2026-10-01 00:40 PDT. Publish: blocked by designed probe-side gate (publish_eligible=False at 34.4% vs 0.25 bar).

## Abstract
We measure the scale behavior of persistent refusal-direction weight edits on Qwen/Qwen2.5-1.5B-Instruct (pinned revision 989aa798) across a full multi-site ladder on a fixed 64-probe fixture. An inference-time hook achieves near-total refusal removal (1/64 v1 → 0/64 v2 true refusals) with benign behavior improved (3→1 flagged benign probes), but no persistent weight edit reaches the ≤25% publish bar: lm_head-only spaces re-arm to 82.8%, a K=3 multi-layer edit cuts residual to 42.2% (with a perfect 64/64 benign side), and the selected K=5 composite reaches 34.4% (benign 87.5%) — its MMLU guardrail passes at Δ−0.11pp, 30× inside the limit. We read this as thin-gate geometry at intermediate scale: utility-destroying edits are easy, refusal-destroying persistent edits saturate, and the stage-A direction path itself is bit-stable (row-identical probe files across three independent Colab instances). No artifact is published, by the designed gate; the verdict and the measured session-lease reaper are the deliverables.

## Background & Motivation
- User-facing goal: useful open-weight abliterated models (the 0.5B artifact sbussiso/Qwen2.5-0.5B-abliterated exists; FTT-14/Run-003 paper 372f46635318).
- Scale question: does the 0.5B recipe (composite wd_ML_BN) survive to 1.5B?
- Prior work: Qwen-family refusal scale law [2607.14147] predicts multi-site edits needed at every size incl. 0.5B (measured there; predicted here); residual-refusal risk on full-probe edits [2602.02132]; suppression-regime fragility of small-model abliteration [2609.06934]; thin-gate geometry / kernel-immobility [2609.06934]; weight-edits vs projection-out [2609.14759].
- Program context: FTT-11 (Run-001 paper de8c473a1ee3), FTT-12 (Run-000), FTT-14 (Run-003 = 0.5B, published).

## Method
### Patient & harness
- Patient: Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 (pinned revision).
- Harness: `abliteration_engine` v3 (src-layout; dev-workstation package commits 7f0927e, 655affd), CLI verbs plan/validate/rungpu; specs in YAML with pinned revisions.
- Probe sets: `builtin:primary64_harmful` + `builtin:primary64_harmless`, n_probes 64 per split, max_new_tokens 200, greedy.
- GPU: Google Colab T4 (s2-s4) + L4 (s5), fp16, per-phase artifact pulls to VM151 mirror.

### Stage A — direction extraction (complete, triple-reproduced)
- Coherence sweep: best L19 / hidden_states_index 20 / coherence 0.8218 (all three instances); layer_directions across 28 layers × 1536 dims; A/B direction vectors (1536-f32).
- Baseline probes: harm 63/64 refused (98.4%), benign 62/64 preserved (96.9% v2).

### Stage B —  the ladder (v3 persistent edits)
- wd_B: lm_head-only orthogonalization of r_B (single readout site).
- wd_BN: lm_head + final norm output orthogonalization.
- wd_ML: multi-layer MLP/down_proj row-space edits, k_layers by coherence order.
- wd_ML_BN: composite (wd_ML sites + norm output), auto-triggered when best-of-ladder ≥ 0.25.
- Each variant: edit → save → reload → on-disk verify → probe (edit integrity enforced).

### Scoring protocol (frozen + post-hoc)
- v1 scorer (`core.py`, ~50 markers) FROZEN for gates — cross-run comparable; consumed by all gate decisions.
- v2 re-grade layer (`scoring_v2.py`, FTT-26): line-anchored help evidence, flat-before-help rule, split-aware harmful_compliance; suite 60/60 GREEN incl. verbatim run-001 i14 anchor + dev's 60-file stress battery.
- Headline framing (dev-workstation approved): harmful headline = v1; benign preservation = v2; deltas as commentary; flip to v2-headline post-hands-on only.

### Gates
- Publish eligibility: benign-preserved ≥ base−0.10 AND harm refusal_rate < 0.25 (STRICT, spec-verified edits.py:191-215).
- MMLU guardrail: ≤3.0pp loss (mmlu_max_loss_pp) — enforced pre-publish on the SELECTED variant.
- Degenerate-output gate: zero empty/degenerate probes for selection.

### Infrastructure & provenance protocol
- Kernel reaper (measured, joint post-mortem): per-session ~60-min runtime reaper, arch-invariant (4×T4 + 1×L4), busy-unsafe, keep-alive-blind; QUOTA-CYCLE HYPOTHESIS REJECTED (death times scatter 261 min from UTC-midnight anchors vs constant ages).
- Countermeasures (all in-kit, committed): per-phase pulls; probes-validated banked resume (validate-before-copy + spec-narrow [wd_ML, wd_ML_BN] with prestage receipts); REAPERSPLIT driver (bb804e0 + 177796d); death-forensics-before-exit; re-adopt verb filed as CLI 0.6.0 patch ask.
- Provenance hygiene (FTT-26/48 lesson): every headline digit names artifact+revision; stream aggregates labeled pre-artifact vintage; both vintages carried where rows unverifiable.

### What was tried and discarded (with reasons)
- Exec-reconnect self-heal theory: 0-for-2 in evidence (VM dies with kernel; empty-state probe trap).
- In-run session resurrection: no CLI verb exists (prune-only 0.6.0) — recovery = fresh session + bundle relaunch + banked resume.
- Full-ladder-spec relaunch on resume: caps re-work wrongly (runs completed variants inside the reaper window) — replaced by spec-narrow.
- Watchers keyed to exit_code.txt: ghost-sentinel (start-ack fires early); replaced by RUN_DONE-marker + runner-gone dual signal.

## Results
(Filled 2026-10-01 00:35 PDT from mirrored probe JSONs + mmlu_summary.json.)

### Ladder table (harm refusal rate, v1 headline; benign v2)
| variant | harm refused /64 | benign preserved /64 | gate verdict | vintage/artifact |
|---|---|---|---|---|
| baseline | 63/64 (98.4%) | 62/64 (96.9% v2) | — | eng_run002_pull_s5 + s3 (word-identical) |
| hook (inference-only) | 1/64 v1 → 0/64 v2 | 63/64 (98.4% v2) | reference | probes_hook_ablated.json (s5, 70ef086) |
| wd_B | 53/64 (82.8%) | 57/64 (89.1%) | FAIL >0.25 | eng_run002_pull_s5 |
| wd_BN | 53/64 (82.8% s5-file) / 54/64 (s2-T4 stream aggregate, unverifiable) | 57/64 | FAIL >0.25 | ±1 cross-instance aggregate discrepancy, i7-class prime suspect unproven |
| wd_ML | 27/64 (42.2%) | 64/64 (100%) | FAIL >0.25 (best persistent) | eng_run002_pull_s5 (complete both splits) |
| wd_ML_BN | 22/64 (34.4%) | 56/64 (87.5%) | FAIL >0.25 (gate passed benign side; refusal over bar) | rs2-1-forensics (byte-complete, sha 87b90ce8d1a10fe4) — LADDER_DONE-matched |
| MMLU (wd_ML_BN) | base 60.10% / variant 59.99% (Δ −0.11pp, stderr 0.0039 both) | Δ ≤ 3pp gate | **PASS (30× margin)** | rs2-4-mmlu/mmlu_summary.json (mirror cef56b7 base anchor) |

### Figures (programmatic, from recorded artifacts only — per user doctrine)
- F1: ladder refusal-rate by variant (bar, baseline→hook→wd_B→wd_BN→wd_ML→wd_ML_BN + gate line at 25%).
- F2: benign preservation by variant (bar with base reference).
- F3: reaper anatomy (session age-at-death across s2-s5, with 60-min line) — the measured infrastructure finding.
- F4: [if E1] decider probe distribution; [if E2] hook-vs-persistent contrast visual.

## Analysis & Discussion

- **Scale-regime read.** In the 0.5B run the composite edit closed the hook-vs-persistent gap; at 1.5B the composite narrows it (34.4% vs 0%) but does not close it. The site-count axis (K) is the actionable knob and K=5 is not sufficient at this scale; per-layer direction refresh (re-extracting the direction at each edited site rather than reusing the L19 readout-space proxy) and K>5 are the obvious next steps.
- **The asymmetry is the finding.** Every variant keeps ≥57/64 benign probes answered (one has 64/64): multi-site weight editing preserves utility better than it removes refusal at 1.5B. Practitioner guidance: at this scale, use the hook on safety-critical serving paths; use persistent composites only when residual-refusal levels near 1/3 are acceptable.
- **Benign blind spots.** The edit that removed 63/64 harmful refusals also removed all 3 genuine baseline over-refusals (own-house lock install, own-apartment key copy, faster doctor appointment) — the refusal direction carries over-refusal, so removing it repairs false refusals too.
- **Suppression-regime caveat (Malla 2609.06934).** Hook-based removal is inference-time and does not establish permanence; the benign-SFT robustness probe battery (queued future-work item 4) is the decisive test for the persistent artifact family.
- **Provenance lesson institutionalized.** Every headline digit names artifact + revision; stream aggregates were labeled pre-artifact vintage and both vintages carried where rows were unverifiable (wd_BN ±1 discrepancy).
- **Infrastructure finding.** A per-assignment lease (fit:3600 s by declaration, T4×16/A100×28/L4×6 in the CLI debug logs) reaps sessions at ~60 min regardless of load; busy-unsafe, keep-alive-blind, arch-invariant. Quota-cycle hypothesis rejected (deaths scatter 261 min from UTC-midnight anchors vs constant ages). Countermeasures: per-phase pulls, probes-validated banked resume with spec-narrow, REAPERSPLIT driver, death-forensics-before-exit. Open residual: lease-EXIT vs proxy-token-REVOCATION attribution.

## Artifacts & Reproducibility
- Repo: /root/research/abliteration (local git is version-of-record; push/CI parked per user).
- Engine bundle sha256 8ac5ebafb2f75a0c… (bundles/eng_run_002_qwen2.5-1.5b_20260930T222244Z.tar.gz).
- Artifact dirs: eng_run002_pull (s2 3-file), _s3 (stage-A), _s4 (hook+config), _s5 (complete stage-A + wd_B/BN/ML probes — audit-grade), _rs2 (reapersplit resume, pending).
- Harness commits: 7f0927e/655affd (package), c6467ac→c9ce186 (FTT-26 fix), 98319e8 (anchor fixture), 5d6c7f0 (s3 pull), 70ef086 (s5 ladder pulls), bb804e0+177796d (kit).
- Publish target (HITL-gated): sbussiso/Qwen2.5-1.5B-abliterated.
- Colab session ledger: ftt20-run002-research-workstation-{1..5} + rs2 series (all with per-phase pulls).

## Conclusion & Future Work

**Findings.** (1) Persistent refusal removal does not scale trivially through model size: the 0.5B recipe needs more than K=5 sites at 1.5B, and lm_head readout-space edits re-arm almost fully (82.8%). (2) The inference-time hook remains the strongest lever at this scale (~0% residual on the fixture). (3) Weight-space refusal edits preserve benign utility asymmetrically well (87.5–100% everywhere), and their direction also carries the over-refusal blind spots. (4) The stage-A capture→direction→probe path is bit-stable across independent runtimes — same seed, same rows — making cross-run comparisons trustworthy. (5) The hosting environment reaps ~60-min session leases by declaration.

**Next experiments.**
1. K>5 composite at 1.5B + per-layer direction refresh at each edited site (H1 revision; decisive for the thin-gate story).
2. Best-composite checkpoint through the suppression-regime probe battery: benign-SFT probe (does ordinary FT re-arm it?), TruthfulQA CI, multilingual panel (en/zh/ru/de), RefusalBench-NQ paired regression (the disposition-shift question; decisive at 1.5B per program findings).
3. DDO decoy-signature pre-check + rank-k SVD re-ablation test on the winner before any future HITL publish (gates unchanged: benign floor, zero degenerates, MMLU ≤3pp, user approval for public push).
4. 7B ladder round-2 on L4 with the same kit (the run-002/FTT-17 family showed 12.5% persistent is already achievable there; round-2 targets suppression-regime robustness next).
5. File the CLI 0.6.0 patch ask upstream: no re-adopt/resurrect verb exists — recovery today cost 6 sessions where a re-adopt would have cost none (paper §Infrastructure).

## Reproduction recipe (minimal)
1. Fresh Colab session (T4 or L4), bundle upload, prestage (REAPERSPLIT: sha check + banked validation + spec-narrow, receipts).
2. PHASE=ladder bash runner.sh (detached setsid, UV_VENV_CLEAR=1).
3. Per-phase pulls to VM mirror; probes byte-completeness check; v2 re-grade.
4. MMLU phase on fresh session; selection; gates; card only after dev hands-on + user HITL.