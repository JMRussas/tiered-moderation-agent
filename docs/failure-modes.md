# Failure modes

Two ways a tiered classifier gets people hurt. Both are about **routing**, not
accuracy — which is why the harness measures routing as a first-class thing and
why `safe_miss_rate` is gated at 1.0 with no margin.

The shared shape: *a cheap tier returns a confident negative on a message it
never had the means to read, and something downstream treats that as a
clearance.*

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

## What the tiering buys, measured

The point of tolerating T0's blindness is that T1 recovers it. On the golden
set, with `ollama:qwen3.5` re-scoring only the 40 messages T0 escalated:

| | T0 alone | T0 + T1 |
|---|---|---|
| recall | 60.0% | **90.0%** |
| precision | 96.0% | **97.3%** |
| FPR | 3.3% | 3.3% |

**+30.0 recall points**, with precision going *up* rather than being traded
away. That is the number that justifies the tier existing. If a change ever
drives it toward zero, T1 should be deleted rather than tuned — and the harness
is how you would find out.

---

## Adding a failure mode

1. Write the fixture row **first**, with a `note` explaining the mechanism.
2. Run the harness. Confirm it fails, and that the output names the case.
3. Fix it.
4. Record the before/after numbers here, including what the fix cost.

A regression test that only ever existed after its bug was fixed never proved
that it catches anything.
