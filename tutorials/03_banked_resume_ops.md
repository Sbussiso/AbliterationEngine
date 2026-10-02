# Tutorial 3 — When Google kills your session (and other survival skills)

**Level: total beginner.** Free cloud GPUs take your session back after
about an hour — that's normal, not something you did wrong. This
tutorial shows how the tool is built so you lose **almost nothing**,
and how to resume like a pro.

Time: 20 minutes reading + whatever compute you're resuming.

---

## The problem, in plain English

Google Colab assigns you a real GPU in a datacenter, but it's a loan:
after roughly **60 minutes the session dies** — even if you're actively
using it, even if a keep-alive heartbeat is running. (We measured it
across every GPU type they offer: the timer is about the same
everywhere. It's in their system plumbing, not on your usage page.)

So a 2-hour job **will** be interrupted. Most hobbyist workflows lose
everything and restart from zero. Ours doesn't, because of two ideas:

1. **Banking** — every finished piece of work is immediately written to
   disk as a file. Killed tomorrow? The work exists, on your machine,
   after a download.
2. **Banked resume** — restart the job, and it *checks the disk first*:
   "have I already done this variant?" If yes, it reuses the answer
   instead of redoing the math.

Real receipts, since you shouldn't take an agent's word for it: during
one day of Run 002, sessions were killed **four times**. Cost each
time: about 8 minutes of re-warmup. Result at the end: byte-for-byte
identical answers on the re-checked questions, zero lost work. That's
the design goal, achieved.

---

## Habit 1 — Install once per session, download every time a stage finishes

Fresh session = fresh machine, so you re-install the tool first (from
Tutorial 1's Step 3: `!pip install -q .`). Then bank your work: the tool
writes result files as soon as each step completes, but disk on Colab
dies with the session — so after each stage (or every ~10 minutes during
long stages):

```python
# in a Colab cell: zip the results folder and pull it down to your machine
!cd /content/eng_run_001* && zip -qr /tmp/artifacts.zip . && \
  python -c "from google.colab import files; files.download('/tmp/artifacts.zip')"
```

(You can also right-click the results folder in Colab's file browser and
pick Download — same thing, no code.)

**Rule of thumb: a result file that only lives on Colab doesn't exist.**

---

## Habit 2 — Resume instead of restart

Start a fresh session, re-install (Step 3 again), put your banked files
back into the results folder, and re-run with the same verb. Example:
your `ladder` got through 2 of 4 surgery attempts...

First cell, re-install (`%cd` makes the folder change stick for the
cells after it):

```python
!git clone https://github.com/Sbussiso/abliteration.git
%cd /content/abliteration
!pip install -q . && pip install -q lm-eval
```

Second, upload the `artifacts.zip` you downloaded earlier: drag it into
the Colab file browser's `/content` folder, or run this cell and pick
the file:

```python
from google.colab import files
files.upload()          # saves it as /content/artifacts.zip
```

Third, restore it and resume:

```python
!mkdir -p /content/eng_run_001_qwen2.5-0.5b && \
  unzip -o /content/artifacts.zip -d /content/eng_run_001_qwen2.5-0.5b
!abliterate ladder --spec specs/run001_parity.yaml --i-know-this-spends-quota
```

Before each surgery attempt, the engine runs a three-point disk check:

| Check | If it fails |
|---|---|
| right number of answer rows (both question sets) | re-run that variant |
| every row has a verdict + full text | re-run |
| the file's provenance fingerprint matches this run (same refusal directions, question/grader settings, and surgery parameters) | re-run, because the file is stale |
| (all checks pass) | **reuse it** — logs say `banked_resume` |

Files banked by older versions of the tool have no fingerprint. They're
still reused, but flagged `banked_provenance: unverified`.

Half-written or corrupt file? Also re-run, silently and safely. The
system is built to be *paranoid by default*: it would rather redo an
hour than trust one shaky file.

One catch: banking saves a variant's *answers*, not its *weights*. The
weights live in `/content/eng_run_001_qwen2.5-0.5b_variants/` and die
with the session unless you downloaded them. If the winning variant was
reused from the bank and its weights folder is gone, the `mmlu` exam
can't load it and fails. Fix: delete that variant's `probes_<name>.json`
from the results folder and run `ladder` again. Only that variant is
rebuilt; the rest stay banked.

The `mmlu` exam banks each side too, and only reuses an earlier score
when it provably came from the same model (same pinned base, same
selected variant and fingerprint). Anything else is moved aside to a
`.stale-<time>` folder and re-run.

The name-agnostic part matters when you mix surgeries: the resume
check reads the probes file name (any variant — `wd_*` *or*
`ara_<rank>`), so a session that died after `wd_ML` but before
`ara_50` (or the other way around) still banks and resumes both.

---

## Habit 3 — Plan your session around the ~60-minute timer

| If a stage takes... | Do this |
|---|---|
| under 45 min | one stage per session, done |
| 45–55 min (L4) | ladder + start of the exam; bank between |
| over an hour | **split it**: one verb per session (`run`, then `ladder`, then `mmlu`) |
| "the whole mission at once" | resist. you'll die mid-flight and pay twice |

The GPU verbs are the practical splitter, one at a time:
`run` (measure), `ladder` (surgeries), `mmlu` (exam). Each is designed
to be a complete, resume-able unit. (`publish` is the local-CPU
exception — it runs on whatever session you're sitting at, no GPU.)

Bigger GPUs don't buy you a longer timer — it's roughly an hour on T4,
L4, and A100 alike. The fix is splitting, not sizing up.

---

## Habit 4 — When it died *mid-write* (partial files)

Bad luck can kill the machine between two sentences of an answer file.
When you resume, the three-point check above catches every such file
and redoes it. To peek at exactly how far a phase got before dying:

```python
!cat /content/exit_code.txt          # "running" = killed mid-stage, "0" = finished
!ls -la /content/eng_run_001*/       # which result files exist + sizes
!cat /content/eng_run_001*/*_error.txt   # traceback, if a stage crashed (not killed)
```

Want the full printed log to survive too? Run the verb with
`2>&1 | tee /content/phase_out.log` on the end, and include that file
in your download.

A useful tip: answers that were flagged "refused" and took under a
second to generate are real refusals (a model saying "no" instantly is
believable). Anything odd else — cross-check with Tutorial 4's scoring.

---

## The one-line summary

> **Bank often, resume instead of restart, split anything over an
> hour — and let the tool's paranoia work for you.**

Next: [Tutorial 4 — how answers get graded](04_probe_files_and_scoring.md) ·
[Tutorial 5 — publishing](05_publishing.md)