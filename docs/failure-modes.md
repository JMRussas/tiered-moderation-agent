# Failure modes

Four ways a tiered classifier goes wrong. Three are about **routing** rather
than accuracy — which is why the harness measures routing as a first-class
thing and why `safe_miss_rate` is gated at 1.0 with no margin. The fourth is
about the measurement itself.

The shape shared by FM-1 through FM-3: *a cheap tier returns a confident
negative on a message it never had the means to read, and something downstream
treats that as a clearance.*

| | what | found by |
|---|---|---|
| FM-1 | A voice gate reading a tier that cannot read | production incident (motivating case) |
| FM-2 | Hostility seen, target unconfirmed | the harness, first run |
| FM-3 | The degradation contract was documented, not implemented | code review |
| FM-4 | The measurement was partly measuring itself | code review |

Worth noting that only FM-1 came from the original system. FM-2 through FM-4
were introduced *while building the thing that exists to prevent FM-1* — which
is the honest argument for why the harness and the review both had to happen.

---

## FM-1 — A voice gate reading a tier that cannot read

**Status:** motivating case. Fixture `g053`, tagged `voice-gate-repro`.

### The setting

The upstream system this repo was extracted from runs an AI host that speaks
aloud on stream. It greets people, and it acknowledges viewers surfaced by a
"worth-a-response" scorer. Before it speaks about a viewer's message, it
moderation-gates that message.

The gate called the T0 scorer. Synchronously. Because it had to — the voice
decision happens now, and T1 is a background task that lands ~2.3s later.

### The mechanism

```
chat message
  │
  ├─ T0 scores it synchronously ......... toxic=False, needs_llm=True
  │     │                                          ▲
  │     │                                          └── "I could not read this"
  │     ├─ worth-a-response board ....... question=True (Arabic ؟ matches)
  │     │                                 + first message ever
  │     │                                 = 75 points -> tier "hot"
  │     │
  │     └─ voice gate ................... re-checks with T0. Same blind answer.
  │                                       Sees toxic=False. Speaks.
  │
  └─ T1 re-scores in the background ..... toxic=True   <-- 2.3s too late,
                                                            and wired to the
                                                            mod feed, not the
                                                            voice
```

Nothing here is a crash, an exception, or a failed call. Every component does
exactly what it was written to do. The bug lives in the *seam*: an asynchronous
tier's verdict and a synchronous gate that cannot wait for it.

### Why the fixture scores the way it does

`g053` — `ليه انتي وحشة كده؟` ("why are you so ugly?")

| Signal | Value | Why |
|---|---|---|
| `question` | `True` | U+061F Arabic question mark is in the QUESTION pattern |
| `toxic` | `False` | the insult is not in the small Arabic lexicon |
| `needs_llm` | `True` | non-Latin script — T0 knows it is blind |

The `question=True` is what makes this dangerous rather than merely wrong. In
the upstream scorer, `question` is worth 40 points and `first message ever` is
worth 35. That is 75, over the 70 threshold for "hot", which is the only tier
the auto-responder speaks about. **The message's hostility is invisible, but its
question mark is not** — so being unreadable actively increases the odds of
being spoken about.

### The contract

> Any consumer that gates on a T0 verdict MUST treat `needs_llm=True` as
> **unknown**, not as clean. `toxic=False, needs_llm=True` means "I could not
> read this", and no consumer may convert that into permission.

For a voice, "abstain" is the correct resolution: the mod surface still shows
the viewer, and only the spoken response is withheld. This costs almost
nothing — the response board decays over 120s and T1 lands in ~2.3s, so waiting
is also viable. What is *not* viable is treating silence from a keyword matcher
as evidence of innocence.

---

## FM-2 — Hostility seen, target unconfirmed

**Status:** found by this harness on its first run. Fixed in `t0.py` rule (e).
Fixture `g043` is the regression test.

### What happened

T0 avoids flagging "this game is trash" by requiring a **second-person target**
alongside a hostile token. Good rule; it is why `fp_traps` held at 0% FPR.

But the escalation rule that catches unknown-unknowns was written as *"no
lexicon matched at all"* — literally `friendliness is None`. A message that
trips `NEGATIVE` has a sentiment, so it fails that condition and never
escalates.

Put those together and there is a gap exactly where the two rules overlap:

```
"eres una idiota"
  NEGATIVE matches "idiota"     -> friendliness = -0.6
  SECOND_PERSON fails           -> the Spanish copula "eres" is not in the list
  => toxic = False              (correct: T0 truly cannot confirm the target)
  => needs_llm = False          (WRONG: it is not confident, it is stuck)
```

The result is FM-1's shape again: a confident negative on a message the tier
could not resolve. The first harness run named it in one line:

```
  SILENT MISSES (1) — wrong AND confident.
    g043  [        es] 'eres una idiota'
```

### The fix

Rule (e): *seeing hostility without being able to confirm its target is
uncertainty, not innocence.*

```python
or (neg and not toxic)
```

### What it cost

Measured on the golden set. Rule (e) only touches `needs_llm`, so precision and
recall of `toxic`/`scam` are unchanged by it:

| | before | after |
|---|---|---|
| safe miss rate | 93.8% | **100%** |
| silent misses | `g043` | none |
| escalation rate | 51.4% | 57.1% |

That is a real cost, not a free win: rule (e) also escalates the "this game is
trash" population, which in production traffic is large. The trade is a higher
model bill for the elimination of confident-and-wrong verdicts, and it is the
right trade only because of what those verdicts feed. Track it — if escalation
creeps toward the `max_escalation_rate` gate, the fix is a better target check,
not a lower floor.

> A separate, unrelated change in the same commit — correcting fixture `g017`
> so its documented false positive actually fires — moved `all` precision from
> 100% to 96.0% and FPR from 0% to 3.3%. That is the fixture getting more
> honest, not the classifier getting worse.

---

## FM-3 — The degradation contract was documented, not implemented

**Status:** found in code review, after the first commit. Fixed. Regression
tests in `tests/test_t1_degradation.py`.

The same shape as FM-1 and FM-2, this time in the code written to prevent them.

`t1.score_one(label, fallback=None)` took its degradation target as an
**optional** parameter. `score_batch` then called it through
`ThreadPoolExecutor.map` with a single iterable, so `fallback` was always
`None` and every failure fell through to a blank `Verdict()` — not to T0's.

With Ollama unreachable, on `g053`:

```
T0      : toxic=False  needs_llm=True   question=True
T1(down): toxic=False  needs_llm=False  question=False
```

`needs_llm` went **True → False**. A dead model server silently converted "I
could not read this" into "this is clean", which is precisely what FM-1 says
must never happen. `question` was destroyed too — upstream that is a 40-point
signal on the response board, so the failure also changes who the host talks to.

Nothing caught it. `Routing` measures T0 alone and T0 was honest, so
`safe_miss_rate` correctly reported 100%; there was simply no metric covering
the combined tier. The README asserted the contract as a headline property.

### The fix

Make the mistake unwritable rather than merely fixed:

```python
def score_one(label: Label, base: Verdict) -> Verdict:          # no default
def score_batch(pairs: Sequence[tuple[Label, Verdict]]) -> ...  # pairs, not labels
```

The degradation target now travels with each message and cannot be dropped at
the call site. `test_fallback_has_no_default` asserts the signature, so a later
refactor cannot reintroduce the optional parameter quietly.

### What it cost

Nothing measurable — T0 metrics are byte-identical before and after, which is
the expected result for a pure correctness change and was used to confirm it.

A second, weaker check was added alongside: `combined_confident_misses` reports
positives still missed after T1 that no longer carry `needs_llm`. It is
**reported, not gated**, because it mixes an unavoidable condition (T1 ran and
was wrong) with a defect (T1 failed and cleared the flag). Gating it at zero
would demand perfect T1 recall, so the gate would sit red and get switched off.
The defect half is caught precisely, and without a model, by the unit tests.

---

## Historical model measurement (before the review fixes)

The point of tolerating T0's blindness is that T1 recovers it. On the golden
set, with `ollama:qwen3.5` re-scoring only the 47 messages T0 escalated:

| | T0 alone | T0 + T1 |
|---|---|---|
| recall | 53.2% | **87.2%** |
| precision | 96.2% | **97.6%** |
| FPR | 3.2% | 3.2% |

**+34.0 recall points** across 3 runs with spread 0.0, precision going *up*
rather than being traded away. That is the number that justifies the tier
existing. If a change ever drives it toward zero, T1 should be deleted rather
than tuned — and the harness is how you would find out.

### FM-4 — the measurement was partly measuring itself

Worth recording as a failure mode in its own right, because it is the one that
would have survived longest unnoticed.

The golden set and T0's lexicon were authored together, so `t0_readable` recall
of 95.8% was substantially the regexes being read back to themselves — four
fixtures were near-verbatim copies of the patterns matching them. Splitting
canonical wording from natural rephrasings:

| slice | recall |
|---|---:|
| `verbatim` | 95.8% |
| `paraphrase` | **14.3%** |

T0 generalizes to roughly 1 in 7 ordinary rephrasings of categories it
nominally covers. Nothing was broken; the number was just answering an easier
question than the one it appeared to answer.

The fix was to add the harder rows rather than delete the easy ones, so the two
questions are reported separately and `verbatim` survives as a regression guard.
Thresholds were re-baselined in the same commit — `all.recall` 0.55 → 0.48,
`t0_readable.recall` 0.90 → 0.70 — with the new numbers in the commit message,
which is the only honest way to lower a floor.

---

## Adding a failure mode

1. Write the fixture row **first**, with a `note` explaining the mechanism.
2. Run the harness. Confirm it fails, and that the output names the case.
3. Fix it.
4. Record the before/after numbers here, including what the fix cost.

A regression test that only ever existed after its bug was fixed never proved
that it catches anything.


## FM-5 -- Review found gaps outside the golden set

**Status:** fixed; regression cases in `tests/test_review_regressions.py`.
The older failure modes above record historical implementations and results.

A positive prefix changed `you should disappear forever` from uncertain to
confidently clean. Short hostility and scripts outside the enumerated Unicode
ranges could also bypass escalation. A 100% safe-miss ratio on the golden set
was never a guarantee about other input.

The router now clears only a small allowlist of complete trivial messages.
Unmatched content escalates regardless of sentiment or length. This replaces
FM-2's narrower rule (e), rather than adding more keyword exceptions. On the
same 78 fixtures, escalation rises from 47/78 (60.3%) to 51/78 (65.4%); T0
precision, recall, and safe-miss ratio are unchanged. The 70% cost gate stays.

The review also found that combined T1 scores were not gated, empty evaluations
passed, and installed wheels omitted the blocklist. The harness now gates every
T1 repeat and rejects missing evaluation/configuration inputs. The packaged
policy is tested from an installed wheel and invalid policy fails visibly.
T1 accepts runtime messages instead of requiring ground truth, validates
explanations, and preserves the original verdict on malformed results. Shared
process-level admission limits cover direct calls and concurrent batches.

Tests cover a completely broken model, a failed first repeat followed by a
successful second repeat, missing policy, malformed output, and overload. These
checks establish implementation contracts, not perfect model accuracy.
