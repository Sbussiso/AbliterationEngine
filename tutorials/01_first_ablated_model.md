# Tutorial 1 — Your first ablated model (full mission loop)

Time: ~40 min on a Colab T4 · GPU: required for `run`/`ladder`/`mmlu`

You'll take **Qwen2.5-0.5B-Instruct**, find its refusal direction, probe it,
and leave with a benched variant — on a free T4 — without writing any
pipeline code. Every step below is the real flow this repo's runs used.

> Before GPU steps: every one of them requires the
> `--i-know-this-spends-quota` flag (or its runner-side equivalent). That's
> a feature: nothing here spends quota by accident.

## 0. CPU prep (on your machine)

```bash
git clone https://github.com/Sbussiso/abliteration.git
cd abliteration
uv sync --extra dev                       # CPU runtime + dev tools
uv run --no-sync abliterate plan --spec specs/run001_parity.yaml
```

Read the stage plan it prints: capture → directions → probes → ladder →
mmlu → publish. That's your mission. `plan` never loads a model, so this
is safe on a laptop.

Validate the spec too (fail-fast, no side effects):

```bash
uv run --no-sync abliterate --spec specs/run001_parity.yaml validate
```

## 1. Bundle it up (CPU)

`bundle` packages the frozen engine + your spec + a generated runner into
one tarball with a per-file sha256 manifest:

```bash
uv run --no-sync abliterate bundle --spec specs/run001_parity.yaml \
    --out-dir bundles
# -> bundles/eng_run_<NNN>_<patient>_<TS>.tar.gz (+ .sha256 sidecar)
```

Why: the tarball installs with `uv sync --frozen` **inside Colab's
existing CUDA environment** (`--system-site-packages` venv), so the exact
CI-validated dependency set meets Colab's preinstalled GPU torch — no env
drift, no 2GB re-downloads.

## 2. GPU session (Colab or any CUDA host)

Upload the tarball, then on the host:

```bash
bash runner.sh                     # PHASE defaults to run (stage A)
```

The runner is generated, not handwritten — never edit it; regenerate from
the spec instead. While it runs, poll the engine-owned sentinels:

| Sentinel | Meaning |
|---|---|
| `/content/exit_code.txt` | 0 = phase finished clean |
| stdout `ENG<STAGE>_DONE {json}` | structured stage-complete receipt |
| `<stage>_error.txt` | written on failure, with the reason |

Stage A leaves: `layer_directions.npz` (all-layer candidates),
`layer_coherence.json` (the coherence table — pick the winner),
`probes_baseline.json`, `probes_hook_ablated.json`,
`refusal_direction_A/B.npy`, `run_config.json` (spec-sha-bound).

## 3. Ladder + guardrail (same or next session)

```bash
PHASE=ladder bash runner.sh        # persistent edits: edit→save→reload→verify→probe
PHASE=mmlu   bash runner.sh        # MMLU guardrail on the selected variant
```

Each variant lands as `probes_wd_<NAME>.json` + a verified on-disk model
dir. The **selection gate** (in the engine, not vibes) then picks the
lowest-refusal variant that preserves benign completions within the
benign-floor delta and zero degenerate outputs — recorded in
`selection.json`. If your best variant ≥ 25% refusal
(`gates.publish_refusal`), the engine marks it publish-ineligible and
tells you.

Interrupted? Skip to Tutorial 3 — completed variants are reused from
disk instead of recomputed.

## 4. Verify, don't trust (CPU — pull back first)

Pull the artifacts (including `run_config.json`, which carries the
`spec_sha256` that binds every artifact to the exact effective config),
then diff against the known-good baseline:

```bash
uv run --no-sync abliterate parity --spec specs/run001_parity.yaml \
    --baseline qwen2.5-0.5b-002/artifacts --run-dir <your-artifacts-dir>
```

Deterministic JSON metrics must match exactly; fp16-nondeterministic
tensors get cosine+L1 tolerances. A parity failure is exit 1 with
`parity_ok: false` — this repo does not fake passes.

## 5. What just happened, mechanically

- The engine diffed **mean final-position activations** over harmful
  prompts against harmless ones, all 24 layers of the 0.5B patient.
- The direction whose per-layer readout coheres most with a
  refusal-like subspace wins (`layer_coherence.json`; on the Run 001
  patient that was decoder layer 17, coherence 0.664).
- Ablating that direction (projection in the residual stream for the
  hook; orthogonalization of `o_proj`/`down_proj` rows for persisted
  variants) measurably dropped harmful-probe refusal 87.5% → 0% on 0.5B
  while benign behavior stayed within the gate floor.

Now read [Tutorial 2](02_writing_a_spec.md) to bring your own model, or
[Tutorial 4](04_probe_files_and_scoring.md) to understand what's inside
those `probes_*.json` files.