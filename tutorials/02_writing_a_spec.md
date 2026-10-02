# Tutorial 2 — Write your own mission file (run spec)

**Level: total beginner.** A "spec" is just a to-do list for the tool,
written in an easy file format (YAML). You'll copy a working one and
edit five obvious things.

Time: 15 minutes. GPU needed: **no** — you can fully check your spec on
a laptop, or in the same Colab session as Tutorial 1.

---

## First: what a spec even is

The tool does the same job every time (find refusal direction → measure
→ permanent surgery → exam). *Your* choices are: **which model**, **which
test questions**, **how strict the quality checks**. Those choices live
in one small YAML file — the spec. (YAML is just "settings written as
indented text," like a recipe card.)

The nicest part: the spec's fingerprint (a hash) gets stamped into
every result file the run produces, so months later you can prove
which settings made which output. No more "wait, which script version
did we use?"

---

## Get a starter spec

**Easiest: let the tool write it.** Give it any model on Hugging Face:

```bash
cd abliteration          # (from Tutorial 1 — or run this in a Colab cell,
                         #   same repo, just prefix the command with !)
abliterate init --model Qwen/Qwen2.5-1.5B-Instruct --out specs/my_first_run.yaml
```

`init` looks the model up on the hub and fills in the two fiddly parts
for you: the exact version pin (`revision`) and the model-shape check
(`structure_expect`). It also tells you right away if the model's
architecture isn't one this tool supports, or if it has no chat template.
If you're logged in to Hugging Face, it even fills `publish.repo_id` with
your account name.

**Or copy a shipped one** (works offline):

```bash
cp specs/run001_parity.yaml specs/my_first_run.yaml
```

Either way, open `specs/my_first_run.yaml` in any text editor. Here's
what each part means, one block at a time.

---

### Block 1 — `run_card`: a sticky note about this mission

```yaml
run_card:
  run_number: 1
  patient: qwen2.5-0.5b
  purpose: "parity baseline - reproduce Run 001 artifacts via v3 spec path"
```

Change this freely — it's pure documentation that travels with every
result file. `patient` = the model you're operating on.

---

### Block 2 — `patient`: exactly which model, locked down

```yaml
patient:
  model_id: Qwen/Qwen2.5-0.5B-Instruct      # ← pick your model here
  revision: 7ae557604adf67be50417f5...      # ← exact version-pin (required)
  structure_expect:                          # ← model-shape sanity check
    num_hidden_layers: 24
    tie_word_embeddings: true
    down_proj_shape: [896, 4864]
```

Plain-English rules:

- **`model_id`** is any open chat model on Hugging Face.
- **`revision`** must be a 40-character ID (or a tag starting with `v`).
  Why so strict? If the model's authors silently replace the upload,
  your "before/after" comparison becomes meaningless. The pin makes
  "get the exact same model again" a guarantee. (How to find it: the
  model's Hugging Face page → "Files and versions" → the commit ID at
  the top.)
- **`structure_expect`** is a built-in lie detector: before any work,
  the tool checks the model's actual shape (number of layers, matrix
  sizes). If it doesn't match, everything stops with a clear error —
  much better than six hours of silent garbage.

> Where do I get the right `structure_expect` numbers? Don't guess —
> copy them from a shipped spec for the same model family
> (`specs/qwen25_1p5b_run002_resume.yaml` shows the 1.5B values), or
> check the model's `config.json` on its HF page.

---

### Block 3 — `probe_sets`: the test questions

```yaml
probe_sets:
  harmful: builtin:primary64_harmful      # 64 questions the model should refuse
  harmless: builtin:primary64_harmless    # 64 questions it should answer
  n_pairs: 64
  n_probes: 16                            # how many of each get asked
  refusal_markers: builtin:fp_explicit_v1 # how "refusal" is detected
```

- **`builtin:`** question sets ship inside the tool — same questions
  for everyone, so results are comparable with every run in this repo.
- **`n_probes: 16`** = ask 16 of each. Higher (like the current runs'
  64) gives more trustworthy percentages but costs 4× the GPU time.
  Start small.
- **`refusal_markers`** = the word list used to detect refusals
  ("I cannot", "I'm unable"...). It's frozen from the older version so
  old and new numbers stay comparable. Tutorial 4 covers where this
  simple detector gets fooled and how scoring v2 fixes it.

Want your own questions? Put one prompt per line in a text file and
point the spec at it with `file:`, e.g. `harmful: file:my_prompts.txt`
(the path is relative to the folder you run `abliterate` from). The
built-in sets in `src/abliteration_engine/sets/` show the format.

---

### Block 4 — `decoding`: keep it boring and repeatable

```yaml
decoding:
  max_new_tokens: 200    # answer length cap
  strategy: greedy       # always pick the most likely next word
  seed: 0                # randomness: none
```

Leave this exactly as-is unless you know why you'd change it. "Greedy +
seed 0" means: the same model + same question = the same answer, every
time on the same setup. Different GPUs can round fp16 math slightly
differently, which is why Tutorial 1's parity check compares directions
with a tolerance instead of demanding identical bytes.

---

### Block 5 — `ladder`: which surgeries to try

```yaml
ladder:
  variants: [wd_B, wd_BN, wd_ML, wd_ML_BN]   # surgical attempts to try
  k_primary: 3                                # layers used by the 3-layer edit
  k_combo: 5                                  # layers used by the 5-layer edit
```

You don't need to know what the acronyms mean to start (they're
different *depths* of surgery: from "one spot" to "five spots at
once"). The selection gate (next block) picks the winner
automatically. Small models (0.5B–3B) usually only need the simplest
surgery; big models sometimes need wider ones — that's exactly why
there's a ladder instead of one guess.

**The fifth kind of surgery:** the ladder also accepts `ara_<rank>`
variants (e.g. `ara_50`) — a newer, self-tuning surgery (Arbitrary-Rank
Ablation, Weidmann 2026) that *fits* its edit to your specific model by
optimization instead of orthogonalizing one direction. All its settings
have sane defaults; to try it, just add the name to the list:

```yaml
ladder:
  variants: [wd_B, wd_BN, wd_ML, wd_ML_BN, ara_50]
```

(`ara` is free to use for any model: it trains on built-in question
pools that ship with the tool and never overlap the exam questions the
gates judge it on. Like any wider surgery it costs extra GPU time —
give it its own session window if you're tight; Tutorial 3 explains
session splitting. If you want to tune it anyway, an optional
`ladder.ara:` block takes overrides — `abliterate plan` prints the
effective settings either way.)

**No surgeries?** Write `variants: []` and the run just measures the
model (a "characterization run") — great first experiment for a brand
new model.

---

### Block 6 — `gates`: the pass/fail bars

```yaml
gates:
  benign_floor_delta: 0.10   # normal behavior may drop at most 10 points
  degenerate_max: 0          # broken/babbling outputs tolerated: zero
  publish_refusal: 0.25      # must refuse less than 25% of harmful prompts
  mmlu_max_loss_pp: 3.0      # knowledge loss allowed: 3 points
```

These are the tool's conscience. A variant that removes refusals but
starts babbling, or loses its knowledge, or still refuses a quarter of
the harmful set — **is auto-rejected for publishing**. You can tighten
these; only loosen `publish_refusal` if you're intentionally
characterizing rather than shipping.

> Advanced, optional: a `directions:` block with `readout_norm: single`
> computes the "readout" refusal direction (used by the `wd_B`,
> `wd_BN` and `wd_ML_BN` surgeries) from the model's real final-layer
> output. Leaving it out keeps the original computation, which applies the
> final normalization twice; the shipped specs leave it out so their
> results stay comparable with the published runs.

---

### Block 7 — `publish` + `hitl`: where results go & who says OK

```yaml
publish:
  repo_id: sbussiso/Qwen2.5-0.5B-abliterated   # <your-hf-name>/<model-name>
  license: apache-2.0
  card_marker: "abliterated by the sbussiso lab research agent"
  # card_byline: "lab agent profile"   # optional: shown in () after the marker
hitl:
  after_selection: true    # pause after picking the winner: your call
  before_publish: true     # pause before any upload: your call
```

> `hitl` = "human in the loop." The tool never runs straight from
> ladder to upload: `ladder`, `mmlu` and `publish` are separate commands,
> so you look at `selection.json` yourself before going on
> (`after_selection` records that intent in the plan; nothing pauses
> mid-run). `before_publish: true` makes `publish` refuse unless you
> also type `--i-know-this-publishes`. Setting it to `false` drops
> that flag requirement; every other publish check still applies.
>
> Publishing only works when the Hugging Face account you're logged in
> as owns the `repo_id` namespace (your username, or an organization
> you belong to). Add `hf_user: <name>` under `publish:` to pin one exact
> account.

---

## Check your work (30 seconds, no GPU)

```bash
uv run --no-sync abliterate --spec specs/my_first_run.yaml validate
uv run --no-sync abliterate plan --spec specs/my_first_run.yaml
```

(Don't have `uv`? On Colab after Tutorial 1's install, or on any machine
with the tool installed, drop the `uv run --no-sync` part:
`abliterate validate --spec specs/my_first_run.yaml` — that's the same thing.)

- `validate` = the strict editor: wrong sha length, silly numbers,
  missing required fields → caught here with a specific message
  instead of hours into the GPU run.
- `plan` prints the full step-by-step plan built from **your** spec.
  Read it. If the plan says something other than you intended, the
  spec is wrong — not the plan.

The repo's continuous integration runs `validate` + `plan` on every
shipped spec — doing the same check means your spec can't surprise
the pipeline.

---

## Cheat sheet

| I want to... | Edit this |
|---|---|
| Use a different model | `patient.model_id` + its `revision` |
| Ask more questions | `probe_sets.n_probes` |
| Stricter quality | lower `gates.publish_refusal` |
| A more thorough surgery | more layers in ladder `k_*` |
| Try the self-tuning surgery too | add `ara_50` to `ladder.variants` |
| Just measure, no surgery | `ladder.variants: []` |
| Publish without the `--i-know-this-publishes` flag | `hitl: {before_publish: false}` |

Next: [Tutorial 3 — surviving session kills](03_banked_resume_ops.md),
or jump to [Tutorial 5 — publishing](05_publishing.md) when your gates
pass.