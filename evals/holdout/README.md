# Holdout set protocol

Step 2 of [the plan](../../docs/next-steps-plan.md). The golden set was written
alongside the lexicon, so its numbers partly measure the lexicon reading
itself back. This set exists to answer one question: **does the pipeline
generalize to messages nobody wrote with the implementation in view?**

## What this set is, exactly

Synthetic, LLM-authored, LLM-verified blind, human-adjudicated. The
adjudicator knows the implementation; the rows were not selected, written, or
labeled by reference to any classifier output, and the generator never saw
the repository. That is the independence claim, no more and no less. LLM-written
chat is cleaner than real chat even when asked to be messy; this set does not
stand in for production traffic.

## Run order

Every step happens **before** any classifier runs on the data. Once a row
informs a fix, it is development data: move it to the golden set and note it
in the changelog below.

1. **Guide first.** [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) is committed
   before generation and does not change during it. Disagreements that reveal
   a gap add a close-call row to the guide in the adjudication commit.
2. **Generate in isolation.** Open a fresh session with no repository access
   (a new claude.ai chat, or Claude Code in an empty directory). Paste
   [prompts/generator.md](prompts/generator.md) with the guide inserted. Save
   the output as `raw/generated.jsonl`. Record the model in the changelog.
3. **Strip labels.** (All commands run from the repository root.)
   `uv run evals/holdout/holdout.py strip evals/holdout/raw/generated.jsonl > evals/holdout/raw/text-only.jsonl`
4. **Verify blind.** In another fresh session, with a different model if
   possible and never with `qwen3.5`, paste [prompts/verifier.md](prompts/verifier.md)
   with the guide and the text-only file. Save as `raw/verified.jsonl`.
5. **Merge.**
   `uv run evals/holdout/holdout.py merge evals/holdout/raw/generated.jsonl evals/holdout/raw/verified.jsonl --author generator-<model> --verifier verifier-<model> > evals/holdout/raw/merged.jsonl`
   Prints the agreement rate and how many rows need review. Rows are
   validated: a missing field or a string where a boolean belongs stops the
   run with the row ID.
6. **Adjudicate.** For every `needs_review` row (all disagreements, every
   low-confidence verifier call, every subtle/injection positive, every
   `ambiguous` tag, and a 1-in-5 spot check of agreements), the human writes
   one line to `adjudicated.jsonl`:
   `{"id": "h042", "toxic": true, "scam": false, "question": false, "note": "guide: third-person"}`
   Judge from the text and the guide only. Do not look at what T0 or T1 would
   say.
7. **Freeze.**
   `uv run evals/holdout/holdout.py freeze evals/holdout/raw/merged.jsonl evals/holdout/adjudicated.jsonl evals/holdout/messages.jsonl`
   Refuses if any flagged row is undecided or an adjudicated ID is unknown.
   Commit `messages.jsonl`, the raw files, and the changelog entry together.
   Record the file's sha256.
8. **Evaluate once.**
   `uv run --extra llm evals/harness.py --t1 --repeat 3 --dataset evals/holdout/messages.jsonl --thresholds evals/holdout/thresholds.toml --json evals/results/holdout-<date>.json`
   The artifact records the dataset and thresholds paths with their hashes.
   Read the `category:` slices; the golden set's `t0_blind`, `verbatim`, and
   `paraphrase` slices are assertions about golden rows and are only
   mechanically approximated here (see `holdout.py`). Look at slice results,
   not individual rows, unless the row will be moved to the development set.

## Thresholds, decided before looking

Promotion criteria for any candidate change evaluated on this set, agreed now
and encoded in [thresholds.toml](thresholds.toml):

- Combined recall on this set must not be below the golden-set combined
  recall (85.1% at `ca45a45`) by more than 10 points, so at least 75%; FPR
  must not exceed 6%; precision at least 90%; T1 must not reduce recall.
- Slice results with fewer than 15 positives or 15 negatives are reported
  as inconclusive, with counts, not as pass/fail.
- No threshold in [thresholds.toml](../thresholds.toml) is lowered because
  of a holdout result.

## Files

```text
evals/holdout/
  README.md              this protocol and the changelog
  ANNOTATION_GUIDE.md    the label policy (read by generator, verifier, adjudicator)
  prompts/generator.md   run in isolation; writes raw/generated.jsonl
  prompts/verifier.md    run blind; writes raw/verified.jsonl
  holdout.py             strip / merge / freeze
  thresholds.toml        holdout gates for the harness (--thresholds)
  raw/                   generator, text-only, verifier, merged outputs
  adjudicated.jsonl      human decisions for flagged rows
  messages.jsonl         the frozen set, Label-shaped, loadable by the harness
```

`t0_blind` and `fp_trap` in the frozen file are set mechanically from
`script`/`lang` and tags, not by anyone looking at a classifier.

## Changelog

| Date | Event | Generator | Verifier | Rows | Agreement | Adjudicated | sha256 |
|---|---|---|---|---:|---:|---:|---|
| 2026-09-17 | Guide v1 and prompts committed; no data yet | | | | | | |
