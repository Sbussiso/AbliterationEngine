# Tutorial 5 — Publish your model (all gates, then upload)

**Level: advanced beginner.** You made a variant that passed its gates
(Tutorial 1 Steps 5–5b). Now: verify everything one last time on your
own machine, generate the model card *mechanically*, and upload to
Hugging Face.

Time: ~10 minutes on CPU. GPU: not needed — publishing runs on a
laptop, or in the same Colab session as the rest of the run.

---

## The big idea: every lock, then the door opens

Publishing is deliberately boring: the tool re-checks **every** claim
about your model, refuses on any failure, generates the model card
from your result files (no hand-typed numbers!), double-checks your
identity, uploads, and then reads the hub back to confirm nothing got
scrambled.

| Lock | What it checks | Error if it fails |
|---|---|---|
| 1. Selection gate | the ladder's winner genuinely passed | "selection gate not passed" |
| 2. Probe gate | model still refuses under 25% of harmful probes | "refusal >= threshold — DO NOT PUBLISH" |
| 3. MMLU guardrail | knowledge exam: less than 3 points lost vs original | "MMLU guardrail failed" |
| 4. Files on disk | the winner's actual weights exist locally | "missing variant dir" |
| 5. Right weights | `--variant-dir` is the winner's folder, and the exam summary graded the winner | "not the selected variant's dir" / "MMLU summary evaluated …" |
| 6. Identity | your Hugging Face login owns the `repo_id` namespace (your username or one of your orgs; `publish.hf_user` pins an exact account) | "identity check failed" |

Six locks, one key: you must pass `--i-know-this-publishes` yourself.
Nothing ships by accident. That's the whole philosophy of this tool:
**risky actions require explicit, human-initiated intent.**

---

## Step 1 — The command (after pulling your artifacts somewhere safe)

Publishing works anywhere Python runs — your laptop, or the same Colab
session that did the run (the results are already on its disk). On a
laptop with `uv`:

```bash
uv run --no-sync abliterate publish \
    --spec specs/my_first_run.yaml \
    --variant-dir <the-winner's-model-folder> \
    --mmlu <results-folder>/mmlu_summary.json \
    --i-know-this-publishes
```

Or, in the Colab session you already set up (drop `uv run --no-sync`,
the tool is already installed there):

```python
!abliterate publish \
    --spec specs/my_first_run.yaml \
    --variant-dir <the-winner's-model-folder> \
    --mmlu <results-folder>/mmlu_summary.json \
    --i-know-this-publishes
```

You'll need your Hugging Face token in the session either way —
`hf auth login` from a terminal, or in Colab:

```python
!pip install -q -U huggingface_hub
!hf auth login
```

(Older `huggingface_hub` versions call this `huggingface-cli login`.)

(The first command installs the tools; the second opens the login
prompt — paste your token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens).)

- `--variant-dir`: the folder the ladder saved the winning variant to.
  It's `selected_variant_dir` in `selection.json`, e.g.
  `/content/eng_run_001_qwen2.5-0.5b_variants/wd_B`, and the files must be
  on the disk of the machine you're publishing from. If you moved them,
  point `--variant-dir` at the new spot *and* update
  `selected_variant_dir` to match, or lock 5 refuses.
- `--mmlu`: the exam summary from Tutorial 1 Step 5b.

If any lock fails: **nothing uploads.** The error names the exact
gate. Fix the run (or the variant), don't the gate.

> The `hitl` block (Tutorial 2) is why that flag exists: with
> `before_publish: true` (the default), `publish` refuses without it.
> Nothing pauses mid-run. `ladder`, `mmlu` and `publish` are separate
> commands precisely so you get this moment of control between them.

---

## Step 2 — The model card writes itself

Every number on the Hugging Face card is generated from your result
files: probe percentages (baseline vs edited), benign-preservation,
degenerate counts, MMLU before/after, which layers were edited and how,
plus the model lineage (which base model, which pinned version) and a
plain-language safety note + the spec's `card_marker` line.

Real example — the whole card is one file:
[`tutorials/assets/example_card_7b.md`](assets/example_card_7b.md),
and the live version it became:
[Qwen2.5-7B-abliterated on HF](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated).

Why mechanical cards matter: hand-typed numbers drift (transposed
digits, stale runs, hopeful rounding). Generated cards can't lie
unless the result files lie — and those are hash-checked.

---

## Step 3 — After upload: the tool double-checks the hub

The publish stage reads your new Hugging Face repo *back* and asserts:

- `README.md` and `config.json` are there,
- the weights are there (`model.safetensors`, or the sharded index for
  big models),
- `refusal_direction.npy` is there,
- the hub README carries the `abliteration` tag and the pinned base
  revision.

A push that "worked" but uploaded the wrong metadata still fails. The
task isn't done when the bytes leave — it's done when the hub is
verified to say what you meant.

---

## Step 4 — The human checklist (5 minutes, worth it)

- [ ] Gates passed on the **verified** artifacts (Tutorial 1 Step 6's
      parity check) — not on hopes.
- [ ] Winner's weights on disk wherever you're publishing from,
      hash-checked.
- [ ] You understand what you're about to host: **an abliterated model
      answers harmful requests**. That's the research point; the card
      says it; you should too — in how you share it, whom you share it
      with, and what you attach to it.
- [ ] License: your card carries the base model's license (Qwen models
      are apache-2.0; this repo's code is MIT).
- [ ] If you also tag a GitHub repo: remember a pushed git tag does
      **not** create a GitHub *Release* object — run
      `gh release create vX.Y.Z` if you want the "Latest" badge to update.

---

## What you signed up for

This is interpretability research on open-weight models: understanding
where safety behavior lives inside a model and what removing it does to
everything else. Published models carry research-use caveats. Treat a
model that answers anything as the powerful, dangerous artifact it is.

---

**The full loop:** [Tutorial 1](01_first_ablated_model.md) →
[Tutorial 2](02_writing_a_spec.md) → [Tutorial 3](03_banked_resume_ops.md)
→ [Tutorial 4](04_probe_files_and_scoring.md) → **Tutorial 5 (you are
here)**. Congratulations — you've now run the whole machine.