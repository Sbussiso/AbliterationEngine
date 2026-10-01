# Tutorials — learn to uncensor an open-weight model, step by step

**Written for people without ML background.** If you can open a terminal
and copy-paste commands, these guides will take you from zero to a
working ablated model. Every command has been run literally as shown,
and every number comes from result files stored in this repo.

| # | Tutorial | What you'll do | Time | GPU? |
|---|---|---|---|---|
| 1 | [Make your first uncensored model](01_first_ablated_model.md) | the complete walkthrough, start to finish | ~40 min | free Colab T4 |
| 2 | [Write your own mission file](02_writing_a_spec.md) | point the tool at any model you like | 15 min | no |
| 3 | [When Google kills your session](03_banked_resume_ops.md) | lose nothing, resume instead of restart | 20 min | when resuming |
| 4 | [Reading the report cards](04_probe_files_and_scoring.md) | open result files, grade answers honestly | 15 min | no |
| 5 | [Publish your model](05_publishing.md) | final checks + upload to Hugging Face | 10 min | no |

**Never used a terminal?** Tutorial 1 starts with the two free installs
you need and defines every abbreviation it uses. Start there and just
go in order.

---

## The 60-second mental model

```text
   a chat model refuses sometimes          "I'm sorry, I can't..."

   that "no" habit lives at ONE address — a direction in the model's
   internal number-stream

   delete that direction → the model keeps ALL its knowledge & skills
   (this is the surgery, "abliteration")

   ...but now answers. Guardrails verify it didn't get dumber or broken
   (behavior checks + a knowledge exam must still pass)
```

The tool in this repo does all of the above **measurably**: it grades
the model's answers before and after, picks the best surgery
automatically, checks the model still takes decent care with normal
questions (benign behavior) and hasn't lost knowledge (MMLU exam), and
refuses to publish anything that fails a check.

Two ideas this repo adds over the usual one-off scripts:

- **One mission file.** Your whole run — model, questions, checks — is
  one small YAML file, stamped (hash-verified) into every result.
- **Resume-safe.** Cloud GPUs get taken away after ~an hour. Finished
  work is banked to disk instantly and re-used on restart, so a killed
  session costs minutes, not the whole job.

---

## Where to start

| You are... | Start with |
|---|---|
| brand new to all of it | [Tutorial 1](01_first_ablated_model.md) |
| bringing your own model | [Tutorial 2](02_writing_a_spec.md) → 1 |
| bitten by a dead session before | [Tutorial 3](03_banked_resume_ops.md) |
| confused by the result JSONs | [Tutorial 4](04_probe_files_and_scoring.md) |
| ready to ship | [Tutorial 5](05_publishing.md) |

Real examples of what comes out the other end:
[0.5B on Hugging Face](https://huggingface.co/sbussiso/Qwen2.5-0.5B-abliterated) ·
[7B on Hugging Face](https://huggingface.co/sbussiso/Qwen2.5-7B-abliterated)

---

*A note on intent: this project exists for interpretability research on
open-weight models — understanding how trained-in safety behavior is
implemented inside a model and what surgically removing it does to
everything else. An abliterated model will answer harmful requests;
published model cards state this plainly. Use the same judgment you
would with any powerful tool.*