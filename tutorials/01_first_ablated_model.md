# Tutorial 1 — Make your first uncensored model (complete walkthrough)

**Level: total beginner.** You don't need machine-learning experience —
if you can copy-paste, you can do this, and everything happens in your
browser on Google's free GPUs.

Time: about 40 minutes (most of it waiting). Cost: a free Google Colab
GPU session. No installation, no terminal, no local Python.

1. **Open Colab** (Python is already installed there) — 2 min
2. **Download the repo** — 1 min
3. **Install the tool** — 2 min
4. **Ask it what it plans to do** (it prints a plan) — 2 min
5. **Press one button** — 10 min
   → the AI that once said "I'm sorry, I can't" now answers the question
6. **Check the result against the record** — 5 min
7. **(Later, if you want it) publish to Hugging Face** — 5 min

---

## What are we even doing? (3 paragraphs, no jargon)

Big chat models — like Qwen, Llama, Mistral — are trained to **refuse**
certain requests: "I'm sorry, but I can't help with that." That
refusal habit lives somewhere specific *inside* the model: researchers
in 2024 discovered it behaves like a single direction in the stream of
numbers flowing through the model. Find that direction, delete it, and
the model still knows everything it knew — it just stops moralizing
and answers.

**Abliteration** is the name for "find the refusal direction and surgically
remove it." This repo contains a tool that does the whole thing for you,
properly: it measures instead of guessing, keeps a mathematical record of
everything it changes, and refuses to produce a broken or dumber model —
there's a built-in knowledge check (a standardized exam called MMLU) that
must still pass afterward.

**Why you can trust this guide:** every number in it comes from runs
stored in this exact repo, and every command is a real command the tool
accepts. Nothing here is invented for illustration.

---

## Step 1 — Open Colab (2 minutes)

Go to [colab.research.google.com](https://colab.research.google.com) (any
Google account works). Click **New notebook**. Colab's sessions already
come with Python, the ML stack, and a free GPU — nothing to install on
your machine.

Turn the GPU on before you do anything else:
**Runtime → Change runtime type → T4 GPU → Save.**

## Step 2 — Download the repo (1 minute)

In a fresh code cell:

```python
!git clone https://github.com/Sbussiso/abliteration.git
%cd abliteration
```

## Step 3 — Install the tool (2 minutes)

```python
!pip install -q . && pip install -q lm-eval && abliterate --help
```

`pip install .` reads the repo, builds the `abliterate` command, and
installs its few dependencies. The second install brings `lm-eval`, the
standardized-exam engine the tool uses for its knowledge check in
Step 5b — grabbing it now saves you a surprise later.

`abliterate --help` greets you with the
tool's available "verbs": plan, validate, run, ladder, mmlu, publish,
parity, bundle. Verbs are just the things the tool can do. Think of
this like a workshop: `plan` consults the blueprint, `run` does the
measuring on the GPU, `ladder` does the surgery, `mmlu` gives the
patient an exam, `publish` ships the result, `parity` double-checks
the work.

## Step 4 — Ask the tool what it plans to do (2 minutes)

A "spec" is a small settings file that describes one mission: which
model, which test questions, how strict the safety checks are. The repo
ships with real ones. Look at what a mission looks like:

```python
!abliterate plan --spec specs/run001_parity.yaml
```

You'll see something like:

```
=== abliterate plan — run 1 (qwen2.5-0.5b) ...
patient: Qwen/Qwen2.5-0.5B-Instruct @ 7ae55760
probe sets: harmful=... (64 prompts), harmless=... (64 prompts)
stage plan:
  1  load_patient ...
  2  capture     64+64 prompts, all-layer final-position residuals
  3  directions  coherence scan -> L* ...
  4  probe       baseline + hook-ablated ...
  5  ladder      variants ['wd_B', 'wd_BN', ...]
  5.5 mmlu       base vs variant, delta <= 3.0pp gate
  6  publish     sbussiso/... (HITL before_publish=True)
```

Plain-English translation of that plan:

- **capture** — read the model's "internal chatter" (activations) while
  it reads harmful and harmless questions.
- **directions** — find the chatter direction that shows up when the
  model is about to refuse.
- **probe** — actually ask the model questions, before and after a
  "test-run surgery", and grade the answers.
- **ladder** — try several versions of the permanent surgery, and grade
  each one.
- **mmlu** — give the model a knowledge exam. If it got dumber by more
  than 3 percentage points, the tool refuses to ship it.
- **publish** — upload to Hugging Face (only with your explicit OK).

> Nothing here is loaded or run yet. `plan` is completely safe.

## Step 5 — The GPU session (10 minutes of watching)

One cell:

```python
!abliterate run --spec specs/run001_parity.yaml \
  --i-know-this-spends-quota
```

(The flag is the tool's built-in "yes, spend my GPU minutes" — it makes
sure nothing expensive ever runs by accident.)

You'll watch it: download the model → capture chatter (a few minutes)
→ find the direction → ask test questions. When you see `RUN_DONE`,
stage A is complete.

**What happened in there?** The tool read the model's internal
activations on each of 24 layers for 128 questions, found the one
direction that screams "I'm about to refuse!", and measured how the
model behaves with that direction switched off in the moment (called a
"hook"). Record: before = refused 87.5% of harmful questions; with the
direction off = 0%.

> ⚠️ Free Colab sessions get reclaimed after about an hour. Don't
> panic: Tutorial 3 shows how to lose nothing and resume. The tool was
> literally built for this — four real session kills in one day cost
> nothing here. In practice: pull your results folder down
> (next step) after each stage, and start a fresh session when Google
> ends yours.

## Step 5b — Make your model's surgery permanent (10 minutes)

The hook is temporary. To make a model file you can keep:

```python
!abliterate ladder --spec specs/run001_parity.yaml \
  --i-know-this-spends-quota
```

`ladder` means: the tool tries a few different permanent surgeries,
makes a fresh copy of the model for each, re-loads them to be sure the
copies are correct, then asks the same test questions. Each attempt
becomes a file like `probes_wd_B.json`, and a judge within the tool
(the "selection gate") picks the winner: the one that refuses the least
*harmful* content while still serving *harmless* questions normally.
(There's also a self-tuning surgery, `ara_<rank>` — Tutorial 2's ladder
section introduces it; the default specs run the classic ladder.)

Then, the exam:

```python
!abliterate mmlu --spec specs/run001_parity.yaml \
  --i-know-this-spends-quota
```

This runs the MMLU knowledge test (57 subjects: history, law, medicine,
…) on both the original and your surgically-modified copy. If your
model got 3 or more percentage points dumber, the exam stage fails
(exit code 6, `GUARDRAIL FAILED` in the log) and `publish` will refuse
to upload it. Our published models lost essentially nothing (the 7B one:
71.77% → 71.77%, i.e. unchanged).

## Step 6 — Check your results against the record (5 minutes)

The repo carries the verified result files of earlier runs. Same cell —
this compares your fresh run against that record:

```python
!abliterate parity --spec specs/run001_parity.yaml \
  --baseline tests/fixtures
```

"Parity within tolerance" = your session did the same science the
published record did. A mismatch *is not a failure of yours* — sessions
and hardware drift, and the tool reports exactly which file moved.

Then grab your results: in Colab's file browser (folder icon, left),
find `/content/eng_run_001_qwen2.5-0.5b`, right-click the folder →
**Download**.

The model *weights* from the ladder live in a separate folder next to
it, `/content/eng_run_001_qwen2.5-0.5b_variants/`. Only the winning
surgery is kept there (the others are deleted to save disk), and its
exact path is written in `selection.json` as `selected_variant_dir`.
It's large (about 1 GB for 0.5B), so download it only if you plan to
publish from another machine.

## You did it — what do you have?

A folder that contains:

- the direction the model uses to refuse (`refusal_direction_*.npy`),
- before/after report cards on how the model behaved
  (`probes_baseline.json`, `probes_hook_ablated.json`),
- (if you ran the ladder) the permanent surgery recipes and their grades
  — and, in the sibling `_variants` folder, the winning model's weights,
- the knowledge exam results (`mmlu_summary.json`),
- a receipt that ties it all to your exact settings
  (`run_config.json`).

## Where to next

- **Curious how "refused" is judged?** →
  [Tutorial 4 — Probe files + scoring](04_probe_files_and_scoring.md)
- **Modify it for your model of choice** →
  [Tutorial 2 — Writing your own run spec](02_writing_a_spec.md)
- **Fear losing a Colab session?** →
  [Tutorial 3 — Banked resume](03_banked_resume_ops.md)
- **Publish to Hugging Face** →
  [Tutorial 5 — Publishing](05_publishing.md)

Two real, downloadable examples of what this makes:
[0.5B model](https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated) ·
[7B model](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated)

---

*A note on intent: this is interpretability research on open-weight
models — understanding how safety training surfaces inside a model. An
abliterated model will answer harmful requests; published cards say so
plainly. Use the same judgment you'd apply to any powerful tool.*