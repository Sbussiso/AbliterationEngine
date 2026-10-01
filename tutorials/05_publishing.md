# Tutorial 5 — Publishing: gates, model card, HF push

Time: ~10 min (CPU) · GPU: not required (runs locally)

Publishing is a **gate-verified push**, in this order:

1. every artifact gate asserted locally
2. model card generated **from artifacts** (you type no numbers)
3. identity check on the HF token
4. push + post-upload verification (card marker, pinned revision, config)

## What happens before a single byte moves

Each assertion is a hard stop with a readable message:

```text
assert sel["gate"] == "passed"                       # selection gate
assert sel["publish_eligible_probe_gate"] is True    # residual refusal < 25%
assert mmlu["guardrail_..."] is True                 # MMLU loss ≤ 3pp
assert os.path.isdir(VDIR)                           # selected variant on disk
assert who["name"] == "sbussiso"                     # HF identity = expected
```

No pass = no push, and the error names the gate. There is also the flag
contract — the verb refuses to run without it:

```bash
abliterate publish --spec specs/<yours>.yaml \
    --variant-dir <selected-variant-dir> \
    --mmlu <dir>/mmlu_summary.json \
    --i-know-this-publishes
```

(`--variant-dir` is where the ladder saved the winning variant's weights;
`--mmlu` is the guardrail's summary file.)

## The card is generated, not written

`publish` builds the HF card mechanically from the artifacts:

- **base model + pinned revision** from the spec (and re-verified against
  the uploaded hub card: `pinned revision missing from hub card` is an
  assert, not a hope);
- **the edit recipe**: which layers got orthogonalized, in what order —
  e.g. the published 7B card lists layers 20, 18, 19 with the exact
  `(W r) ≈ 0` post-edit property;
- **probe numbers** from `selection.json` + probe files: baseline/hook/
  variant refusal rates, benign preservation, degenerate counts;
- **MMLU numbers** from `mmlu_summary.json` — identical lm-eval config
  both sides, so the before/after is apples-to-apples;
- the spec's `card_marker` line and `license` (the published models carry
  apache-2.0 inherited from their Qwen bases; this repo's code is MIT).

Example of what lands on HF: [`qwen2.5-7b-001/artifacts/publish/CARD.md`](../qwen2.5-7b-001/artifacts/publish/CARD.md)
— the actual card of a live model.

## After the push: verification, again

The publish stage then reads the hub **back** and asserts: expected file
list, correct config flags, marker present in the README, pinned revision
present. A push that "succeeded" but uploaded wrong metadata still fails.

## Publish-day checklist (the human part)

- HITL: with `before_publish: true` in the spec, the run pauses for your
  explicit go/no-go — honor it; the contract is that no unapproved push
  exists.
- Make sure the **selected variant's weights are on disk locally** and
  were hash-verified (Tutorial 1 §4) — publish consumes the disk dir, not
  a Colab memory state.
- GitHub Release ≠ HF publish, and on GitHub a pushed tag does not
  create the Release object — if your repo's "Latest" badge matters,
  create the Release (`gh release create v<X.Y.Z>`).

## What you are signing up for

An abliterated model will answer harmful-policy probes it previously
refused; that is the point of the research. Cards carry this caveat
explicitly. Publish for **research and interpretability purposes**, with
the same caution you'd apply to any powerful uncensored artifact.