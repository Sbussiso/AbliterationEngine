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

## Habit 1 — Download every time a stage finishes

The tool writes result files as soon as each step completes. But disk on
Colab dies with the session — so after each stage (or every ~10
minutes during long stages):

```python
# in a Colab cell: pull the whole results folder down to your machine
from google.colab import files
!cd eng_run_001* && zip -qr /tmp/artifacts.zip . && \
  python -c "from google.colab import files; files.download('/tmp/artifacts.zip')"
```

(Above is the clicky way; the command-line alternative is `colab
download`, which is what this repo's own operators use.)

**Rule of thumb: a result file that only lives on Colab doesn't exist.**

---

## Habit 2 — Resume instead of restart

Start a fresh session, upload the bundle again (Tutorial 1 step 4),
restore your banked files into the results folder, and re-run with the
same `PHASE=`. Example: your `ladder` got through 2 of 4 surgery
attempts...

```python
# fresh session: restore the banked work, then resume
!cd eng_run_001* && unzip -o artifacts.zip -d /content/restored && \
  cp /content/restored/* /content/eng_run_001*/ && \
  nohup bash runner.sh > phase2_out.log 2>&1 &
```

The engine, before each surgery attempt, does the three-point disk
check:

| Check | If it fails |
|---|---|
| right number of answer rows (both question sets) | re-run that variant |
| every row has a verdict + full text | re-run |
| (both checks pass) | **reuse it** — logs say `banked_resume` |

Half-written or corrupt file? Also re-run, silently and safely. The
system is built to be *paranoid by default*: it would rather redo an
hour than trust one shaky file.

A variant whose answers are banked but whose modified model file
didn't make it off the VM: the publish/exam stages will **refuse
loudly** rather than pretend the model exists — you'll re-run that
variant's save step, not its full measurement.

---

## Habit 3 — Plan your session around the ~60-minute timer

| If a stage takes... | Do this |
|---|---|
| under 45 min | one stage per session, done |
| 45–55 min (L4) | ladder + start of the exam; bank between |
| over an hour | **split it**: two sessions, `PHASE=` verbs exist for each |
| "the whole mission at once" | resist. you'll die mid-flight and pay twice |

The `PHASE=` verb on the generated runner is the practical splitter:
`PHASE=run` (measure), `PHASE=ladder` (surgeries), `PHASE=mmlu` (exam),
`PHASE=publish` (upload prep). Each is designed to be a complete,
resume-able unit.

Bigger GPUs don't buy you a longer timer — it's roughly an hour on T4,
L4, and A100 alike. The fix is splitting, not sizing up.

---

## Habit 4 — When it died *mid-write* (partial files)

Bad luck can kill the machine between two sentences of an answer file.
When you resume, the three-point check above catches every such file
and redoes it. To peek at exactly how far a phase got before dying:

```python
!tail -30 phase_out.log          # last lines before the kill
!ls eng_run_001*                 # which result files exist + sizes
```

A useful tip: answers that were flagged "refused" and took under a
second to generate are real refusals (a model saying "no" instantly is
believable). Anything odd else — cross-check with Tutorial 4's scoring.

---

## The one-line summary

> **Bank often, resume instead of restart, split anything over an
> hour — and let the tool's paranoia work for you.**

Next: [Tutorial 4 — how answers get graded](04_probe_files_and_scoring.md) ·
[Tutorial 5 — publishing](05_publishing.md)