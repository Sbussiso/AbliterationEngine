# Tutorial 1 — Make your first uncensored model (complete walkthrough)

**Level: total beginner.** You don't need machine-learning experience —
if you can open a terminal and copy-paste, you can do this.

Time: about 40 minutes (most of it waiting). Cost: a free Google Colab
GPU session.

1. **Download** a tool named `abliteration_engine` — 5 min
2. **Ask it what it plans to do** (it prints a plan) — 2 min
3. **Pack everything into one upload file** (a bundle) — 2 min
4. **Open free Google Colab, upload, press one button** — 10 min
   → the AI that once said "I'm sorry, I can't" now answers the question
5. **Check the result against the record** — 5 min
6. **(Later, if you want it) publish to Hugging Face** — 5 min

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

## Before you start — what to install

You need two free things:

**1. Python 3.11 or newer** (`python3 --version` to check).
**2. `uv` — a Python tool manager.** One command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`. On Mac the `curl` line above works in Terminal. On Linux too.)

That's it. The tool itself installs its own dependencies next.

---

## Step 1 — Get the tool (5 minutes)

```bash
git clone https://github.com/Sbussiso/abliteration.git
cd abliteration
uv sync --extra dev
```

That last line downloads everything the tool needs. `--extra dev` means
"the CPU-only starter kit" — safe on any laptop, no GPU needed yet.

When it finishes, say hello to the tool:

```bash
uv run --no-sync abliterate --help
```

You should see the tool's available "verbs" (plan, validate, run, ladder,
mmlu, publish, parity, bundle). Verbs are just the things the tool can do.
Think of this like a workshop: `plan` consults the blueprint, `run` does
the measuring on the GPU, `ladder` does the surgery, `mmlu` gives the
patient an exam, `publish` ships the result, `bundle` packs up anything
to move, `parity` double-checks the work.

---

## Step 2 — Ask the tool what it plans to do (2 minutes)

A "spec" is a small settings file that describes one mission: which
model, which test questions, how strict the safety checks are. The repo
ships with real ones. Look at what a mission looks like:

```bash
uv run --no-sync abliterate plan --spec specs/run001_parity.yaml
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
  6  publish      sbussiso/... (HITL before_publish=True)
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

---

## Step 3 — Pack the mission into one file (2 minutes)

```bash
uv run --no-sync abliterate bundle --spec specs/run001_parity.yaml --out-dir bundles
```

Done? You have one file like
`bundles/eng_run_001_...tar.gz` — it contains the whole tool, locked to
the exact tested versions, plus the mission file. This single file is
what you'll upload to Google Colab, so no setup is needed there.

Why not just run the tool on Colab directly? Because Colab's Python is
slightly different from yours; the bundle freezes the exact tested
setup so the numbers come out identical.

---

## Step 4 — The GPU session (10 minutes of clicking)

1. Go to [colab.research.google.com](https://colab.research.google.com)
   (free Google account; pick a **T4 GPU** runtime: Runtime → Change
   runtime type → T4 GPU).
2. Upload the bundle: the little folder icon on the left → upload arrow →
   choose the `.tar.gz`.
3. New code cell, type exactly:

```python
!tar xzf eng_run_001*.tar.gz && cd eng_run_001* && nohup bash runner.sh > phase_out.log 2>&1 &
```

4. Every minute or so, check how it's doing:

```python
!cd eng_run_001* && tail -5 phase_out.log
```

You'll watch it: download the model → capture chatter (a few minutes)
→ find the direction → ask test questions. When you see
`ENGRUN_DONE` and `exit_code.txt` says `0`, the stage is complete.

**What happened in there?** The tool read the model's internal
activations on each of 24 layers for 128 questions, found the one
direction that screams "I'm about to refuse!", and measured how the
model behaves with that direction switched off in the moment (called a
"hook"). Record: before = refused 87.5% of harmful questions; with the
direction off = 0%.

> ⚠️ Free Colab sessions get reclaimed after about an hour. Don't
> panic: Tutorial 3 shows how to lose nothing and resume. The tool was
> literally built for this — three real session kills in one day cost
> nothing here.

---

## Step 5 — Make your model's surgery permanent (10 minutes)

The hook is temporary. To make a model file you can keep:

```bash
!cd eng_run_001* && nohup bash runner.sh > ladder_out.log 2>&1 &   # with PHASE=ladder
```

(the runner line for step 4 works, just add `PHASE=ladder` after
`runner.sh`).

`ladder` means: the tool tries a few different permanent surgeries,
makes a fresh copy of the model for each, re-loads them to be sure the
copies are correct, then asks the same test questions. Each attempt
becomes a file like `probes_wd_B.json`, and a judge within the tool
(the "selection gate") picks the winner: the one that refuses the least
*harmful* content while still serving *harmless* questions normally.

Then, the exam:

```bash
!cd eng_run_001* && PHASE=mmlu bash runner.sh
```

This runs the MMLU knowledge test (57 subjects: history, law, medicine,
…) on both the original and your surgically-modified copy. If your
model got more than 3 percentage points dumber — the run halts and
tells you. Our published models lost essentially nothing (the 7B one:
71.77% → 71.77%, i.e. unchanged).

---

## Step 6 — Download your results and check them (5 minutes)

Grab every artifact file the session produced (the `eng_run_*` folder),
then back home:

```bash
uv run --no-sync abliterate parity --spec specs/run001_parity.yaml \
  --baseline qwen2.5-0.5b-002/artifacts --run-dir <your downloaded folder>
```

This compares your fresh run byte-by-byte against the record from this
repo. Identical = your machine did the exact same science. Different =
something drifted and the tool tells you where.

---

## You did it — what do you have?

A folder that contains:

- the direction the model uses to refuse (`refusal_direction_*.npy`),
- before/after report cards on how the model behaved
  (`probes_baseline.json`, `probes_hook_ablated.json`),
- (if you ran the ladder) the permanent surgery recipes and their grades,
- the knowledge exam results (`mmlu_summary.json`),
- a receipt that ties it all to your exact settings
  (`run_config.json`).

## Where to next

- **Modify it for your model of choice** →
  [Tutorial 2 — Writing your own run spec](02_writing_a_spec.md)
- **Fear losing a Colab session?** →
  [Tutorial 3 — Banked resume](03_banked_resume_ops.md)
- **Curious how "refused" is judged?** →
  [Tutorial 4 — Probe files + scoring](04_probe_files_and_scoring.md)
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