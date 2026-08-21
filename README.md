# Tiered moderation agent

**An agent that mostly doesn't call a model.**

Live chat is high-volume and low-entropy. Most of it is `lol`. A moderation
pipeline that sends every message to an LLM is paying frontier prices to
classify `GG` — so here the model is the *last resort*, and each tier has to
earn its invocation.

The interesting engineering is not the model call. It is the routing: deciding
what needs a model, and — the part that actually matters — **being honest about
what the cheap tier could not read.**

```
every chat message
      │
      ▼
 [T0]  deterministic keyword scoring          <1ms · no network · no cost
       toxic · scam · question · sentiment · needs_llm
      │
      │  needs_llm: non-Latin script, Arabizi, or T0-uncertain
      ▼
 [T1]  local LLM re-score + translate         ~2.3s · Ollama · bounded
       structured output, retries, degrades to T0 on any failure
      │
      ▼
 [T2]  agent + tools, for the ambiguous few   (not built — see Status)
```

## Measured, on 70 hand-labeled adversarial messages

Reproduce with `uv run evals/harness.py --t1`.

| | T0 alone | T0 + T1 |
|---|---:|---:|
| recall | 60.0% | **90.0%** |
| precision | 96.0% | **97.3%** |
| false positive rate | 3.3% | 3.3% |
| wall clock | 2.1ms / 70 msgs | 40 model calls |

**T1 lift: +30.0 recall points**, with precision going up rather than being
traded away. That number is the entire justification for the tier existing. If
it ever approaches zero, T1 should be deleted, not tuned.

Two routing numbers matter more than either column:

| | value | meaning |
|---|---:|---|
| **safe miss rate** | **100%** | every positive T0 missed, it flagged `needs_llm`. Zero confident-and-wrong verdicts. |
| escalation rate | 57.1% | share of traffic reaching a model. High here **because this set is adversarial-weighted** — not a production throughput claim. |

`safe_miss_rate` is gated at 1.0 in CI with no margin. A silent miss is a
message the system is confidently wrong about, and downstream that reads as
permission to act. See [docs/failure-modes.md](docs/failure-modes.md).

## Why `needs_llm` is the important field

T0 is a keyword matcher. It is *structurally* blind to Arabic script, to
Arabizi, to sarcasm, and to any insult nobody put in a list. That is fine — it
is a cheap tier and it is allowed to miss.

What is not fine is missing **silently**, because `toxic=False` is what every
consumer downstream reads as "safe". This repo exists because of a specific
case where that went wrong: an AI stream host that speaks aloud, gating its
voice on a tier that could not read the message it was about to warmly greet.

That case is fixture [`g053`](evals/golden/messages.jsonl), and the mechanism is
written up in [docs/failure-modes.md](docs/failure-modes.md). The harness found
a second instance of the same shape on its first run, in code written the same
afternoon.

## Quickstart

```bash
uv sync
uv run pytest                        # unit tests, no network
uv run evals/harness.py              # T0 only — no model needed
```

For the model tier, with [Ollama](https://ollama.com) running:

```bash
uv sync --extra llm
ollama pull qwen3.5
uv run evals/harness.py --t1
```

The harness exits non-zero when a threshold in
[`evals/thresholds.toml`](evals/thresholds.toml) is breached, so a prompt tweak
that quietly costs recall fails the build.

## Layout

| Path | What |
|---|---|
| [`src/tiermod/t0.py`](src/tiermod/t0.py) | Deterministic scorer. Pure, no I/O, no model. |
| [`src/tiermod/t1.py`](src/tiermod/t1.py) | LangChain + Ollama re-score. Structured output; degrades to T0 on any failure. |
| [`evals/golden/`](evals/golden/) | 70 labeled messages + [label policy](evals/golden/README.md). |
| [`evals/metrics.py`](evals/metrics.py) | Per-slice precision/recall/FPR, plus the two routing metrics. |
| [`evals/harness.py`](evals/harness.py) | Runner, report, CI gate. |
| [`docs/failure-modes.md`](docs/failure-modes.md) | The two bugs, with before/after numbers. |

## Notes on the parts that are easy to get wrong

**Structured output, not prompt-and-parse.** T1 uses
`with_structured_output` rather than asking for JSON in prose and parsing the
reply. Small local models emit *almost*-JSON constantly — a prose preamble, a
markdown fence, a trailing comma — and hand-parsing turns each of those into a
silently dropped verdict. Moving the contract into the tool-call layer makes a
mismatch a retry instead.

**Explicit `num_ctx`.** Ollama applies its own context window at serve time and
truncates an over-long prompt *silently*. The client value wins, so T1 always
sends one rather than inheriting whatever the server is configured with today.

**Every failure path returns T0's verdict.** A dead Ollama, a timeout, a schema
the model will not satisfy — all mean "no lift", never "clean". Each tier
degrades to the one below it, which is what makes the model tier safe to
depend on and safe to switch off.

**The golden set is synthetic and contains no real slurs.** Both are deliberate
engineering decisions with reasoning in
[evals/golden/README.md](evals/golden/README.md) — live chat is other people's
speech, and the blocklist is a substring test whose correctness does not depend
on what the strings mean.

## Status

T0, T1, the golden set, the metrics, and the CI gate are built and measured.

**T2 is not built.** It is the ~1% where a decision genuinely depends on a
viewer's history plus current room state, and it is the only tier that should
be an agent in the `create_agent` sense — tools, a bounded loop, an advisory
result. Building it before the eval harness existed would have meant no way to
tell whether it helped.

## Provenance

Extracted from a production TikTok LIVE moderation system, rebuilt clean-room.
No viewer data, credentials, or captured chat from that system is present here,
and none of the fixtures were written by a real person.
