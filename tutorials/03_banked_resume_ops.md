# Tutorial 3 — Surviving session kills: banked resume + phase split

Time: ~20 min · GPU: yes (for the resumed phases)

Cloud GPU sessions die. Colab in particular reaps busy-but-idle-unsafe
kernels on a fixed clock (~60 min per assignment on T4/A100/L4; a
keep-alive on the OAuth plane stays 200 while the VM-path token dies —
so don't trust keep-alive as proof of life).

This engine is engineered so a kill costs an **~8-minute re-warmup**, not
re-computation. Three registry-drops in one real day produced
byte-identical probe reproductions and zero banked-work loss. Here's the
playbook.

## What banked resume does

`abliterate ladder` checks the engine out-dir **before** building a
variant. If `probes_<name>.json` already exists with:

1. exactly `n_probes` rows on **both** sides (harmful + harmless),
2. every row carrying grader fields (`refused` not null, `output` text
   present),

…that variant is **reused**, not re-run, and the artifacts record it:

```json
"banked_resume": true,
"edit_info":  {"banked_resume": "probes restored from prior session;
               edit+save+verify skipped"},
"on_disk_verify": {"banked_resume": "variant dir from disk | PROBES_ONLY"}
```

Anything corrupt, partial, or mismatched fails-safe: it re-runs normally.
Selection records which variants were banked (provenance the papers cite).

The MMLU phase has the same pattern over `results_*.json`.

## The phase-split loop (what a kill-safe session looks like)

```bash
# Session N (fresh kernel):
bash runner.sh                                   # stage A (or skip if banked)
PHASE=ladder bash runner.sh                      # as many variants as fit

# ~10 min before the session's end (see lease note):
#   pull artifacts off the VM — banking only helps AFTER a pull
```

Then, on a fresh session:

```bash
# re-upload the pulled artifacts into the engine's expected out-dir
PHASE=ladder bash runner.sh                      # banked variants skip;
                                                 # the rest resume
PHASE=mmlu   bash runner.sh
```

The rule of thumb: **bank early, bank often — a session you can't
re-enter is a session you lost, so pull the out-dir each cycle.**

## Sizing windows to the lease

Real observed deaths on Colab: 60.1–61.7 min after assignment (lease
`fit:3600` + sweep cadence), arch-invariant across T4/A100/L4. Size GPU
phases to finish *within* a window, or split:

| Hardware | Practical compute window | Plan |
|---|---|---|
| Colab T4 | ~45 min | 1 ladder phase / session, probes 64 max |
| Colab L4 | ~55 min | ladder + start of mmlu |
| Colab A100 | ~50 min | bigger patients; same wall-clock lease |

One A100 "single push" does not beat the lease — splitting phases is the
fix, not bigger hardware. (Root cause + probe receipts live in the FTT-13
session logs; the residual open question is lease-exit vs token-revocation
attribution.)

## Recovering a mid-sweep death

A variant killed mid-generation leaves an incomplete `probes_*.json` —
the banked-resume completeness check rejects it and the re-run
regenerates that variant fully. If the runner itself died without
markers, pull whatever partial state exists (`phase_out_partial.log`,
partial probe JSONs) before exiting; a completed-per-row prefix can be
validated and reused by the same completeness rule.

## Anti-gotchas

- **Banked ≠ verified-on-disk.** Probes can be banked while the variant's
  weight dir is absent (publish/MMLU refuse loudly if the *selected*
  variant is missing on disk).
- **Don't hand-edit banked probe files.** The completeness check + parity
  contract make hand-tuning self-defeating; commit + push instead.
- **Two engines of truth:** `spec_sha256` binds artifacts to configs;
  `harness_sha256.json` binds to the exact engine build. Pull both.