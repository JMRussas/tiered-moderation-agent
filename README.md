# Tiered moderation agent

A moderation classifier and evaluation prototype for live chat. A cheap
deterministic tier handles known policy matches and a small set of complete
trivial messages; unresolved content goes to a local model.

The repository implements T0, T1, and their evaluation harness. It does not
include a live chat service, voice integration, or a tool-using T2 agent.

```text
message
  |
  T0: keyword policy + explicit trivial-message allowlist
  |
  +-- known policy match or trivial message --> T0 verdict
  |
  +-- unresolved / multilingual ------------> T1: Ollama classifier
                                                |
                                                +-- validated result
                                                +-- failure/overload: original T0 verdict
```

## The routing contract

`toxic=False, scam=False, needs_llm=True` means **unknown**, not clean.
Consumers must abstain or wait for classification rather than interpreting
uncertainty as permission to speak or act. Classification is advisory; a
validated model result can still be wrong.

T0 only clears complete trivial messages such as `lol`, `GG`, `thanks!`, and
`<3` without inference. Positive sentiment and short length do not establish
safety: `nice, you should disappear forever` and `die` both escalate. The
allowlist is an explicit policy choice, not a guarantee about contextual intent.
Non-Latin letters and detected Arabizi also trigger escalation.

The earlier router used positive keywords to suppress escalation. That produced
silent misses outside the golden set despite its 100% safe-miss score. See
[the review fixes](docs/failure-modes.md#fm-5--review-found-gaps-outside-the-golden-set).

## Quickstart

```bash
uv sync --locked
uv run pytest -q
uv run evals/harness.py
uv run tools/check_wheel.py
```

These checks require no model. The wheel check builds and installs the package
outside the checkout to verify that its default blocklist is actually shipped.

For T1, with Ollama running:

```bash
uv sync --locked --extra llm
ollama pull qwen3.5
uv run --extra llm evals/harness.py --t1 --repeat 3 --json evals/results/<date>.json
```

Every repeat must meet combined precision/recall/FPR, recall-lift, and successful
inference thresholds in [thresholds.toml](evals/thresholds.toml). An outage still
returns safe fallback objects, but fails the model quality gate. `--no-gate`
allows diagnostic runs to exit zero on metric failures; invalid configuration
and invalid datasets remain errors.

CI runs unit tests, the wheel check, and the T0 gate. It **does not run Ollama**.
Run the command above after changes to prompts, model versions, or inference
configuration and retain its artifact. `--json` writes a versioned report with
collected provenance (revision, input hashes, model digest, inference settings,
hardware), every repeat's metrics and gate decision, and every message's route,
status, and latency; nothing in it is typed by hand. Compare two runs with
`uv run evals/compare.py a.json b.json`. See
[evals/results/README.md](evals/results/README.md) for the schema and
[validation](docs/validation.md) for the most recent checks.

## Measurements and limits

The unchanged golden set contains 78 synthetic, hand-labeled messages. It is
adversarial-weighted, with 47 moderation positives and 31 negatives. It is not a
traffic sample or a held-out capability benchmark.

Current results (T1 column is the final run of three passing model evaluations):

| Metric | T0 | T0 + T1 |
|---|---:|---:|
| Recall | 53.2% | 85.1% |
| Precision | 96.2% | 97.6% |
| False positive rate | 3.2% | 3.2% |

T0's safe miss rate on these fixtures is 100% (22/22 misses escalated).
Escalation is 65.4% (51/78). T1 recall lift was +31.9 points on each of three
runs. Seven positive fixtures still received confident clean verdicts after
T1 in the final run. [Recorded results](evals/results/review-2026-09-17.json)
include those IDs and model provenance.

The routing fix raises escalation from 60.3% to 65.4% without changing T0's
classification scores. The existing 70% escalation ceiling remains unchanged.
No representative traffic replay has established production throughput or the
fraction of real messages that avoid inference.

The lexicon and original fixtures were written together. Splitting their
results exposes the generalization gap:

| Slice | Messages | T0 recall |
|---|---:|---:|
| Canonical wording (`verbatim`) | 45 | 95.8% |
| Natural rephrasings (`paraphrase`) | 8 | 14.3% |
| Structural blind spots (`t0_blind`) | 16 | 6.2% |

The original model run reported 87.2% combined recall. The current result is
85.1%; the older number describes the earlier routing/schema. The T1 thresholds
are acceptance targets and were not lowered to accommodate this change.

## Library use

```python
from tiermod import t0, t1
from tiermod.schema import Message

message = Message(id="incoming-1", text="can you play the other map next")
verdict = t0.score(message.text)
if verdict.needs_llm:
    verdict = t1.score_one(message, verdict)

# A consumer must check uncertainty as well as moderation flags.
may_respond = not (verdict.needs_llm or verdict.toxic or verdict.scam)
```

Ground-truth `Label` objects belong to evaluation, not runtime callers. T1
can only add flags: a deterministic T0 policy hit that was also escalated
(non-Latin script, Arabizi, `translate_mode`) keeps its `toxic`/`scam`
flags and reasons whatever the model answers. T1
requires moderator-readable reasons for flagged results. It makes one attempt;
invalid output or an inference exception returns the original T0 object and
logs the failure type without logging chat text. There are no automatic retries.
`t1.classify` returns the same verdict with an `Outcome` status (`ok`,
`capacity`, `transport`, `validation`, `model`), latency, and the exception
class name, so a consumer can tell admission control from a broken model.

`T1_CONCURRENCY` bounds active calls across batches and direct calls in one
process. Saturation immediately returns the original verdict. Batches are
materialized; integrations must bound batch size, queueing, and cross-process
load themselves. `T1_TIMEOUT_S` is an HTTP inactivity timeout, not an end-to-end
deadline. The client requests an 8192-token context and at most 512 generated
tokens; integrations must also limit input size.

## Blocklist and configuration

The packaged [blocklist](src/tiermod/blocklist.json) contains synthetic tokens
only. Set `BLOCKLIST_PATH` to a private JSON list of nonempty strings before
the first score. Loading is cached for the process lifetime; restart after
policy edits. Missing/unreadable files, malformed JSON, and invalid entries
raise `BlocklistError` rather than silently disabling policy. An explicit `[]`
disables blocklist matching. Load the policy during application startup:

```python
from tiermod.t0 import load_blocklist
load_blocklist()
```

[.env.example](.env.example) lists environment settings. The library does not
automatically load a `.env` file; export settings through your shell or launcher.
`translate_mode="non-english"` explicitly requests escalation; it is not a
language detector.

## Provenance

Extracted from a production TikTok LIVE moderation system and rebuilt
clean-room. No viewer data, credentials, or captured chat is included. All
fixtures are synthetic; the upstream deployment's behavior is described as
historical context, not functionality shipped in this repository.

The [label policy](evals/golden/README.md), [failure history](docs/failure-modes.md),
and [review plan](docs/review-plan.md) explain the evaluation and design choices.
