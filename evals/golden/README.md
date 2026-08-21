# Golden set — construction and label policy

78 hand-labeled messages. Small on purpose: every row is here to answer a
specific question, and a set you can read end to end in five minutes is a set
people actually audit.

## What this set is for

It is **not** a representative sample of live chat. Real chat is ~95% benign
trivia, and a set shaped like that would spend 74 of 78 rows confirming that
"lol" is fine. This set is deliberately **adversarial-weighted**: roughly 60%
benign (including 10 false-positive traps) and 40% positive, so that per-slice
precision and recall are measurable at this size.

Tier throughput on realistic traffic is a separate measurement, taken from a
replay stream — not from this file. Do not quote the escalation rate here as
if it were a production number.

## Provenance — everything here is synthetic

No message in this file was written by a real viewer. Every row was authored
for this repo against a category checklist. That is a hard requirement, not a
convenience: live chat is other people's speech, most of it from people who
never consented to appear in a public benchmark, and some of it identifying.

The upstream system this set was extracted from logs real chat to Postgres and
gitignores it. That data does not leave the machine it was captured on.

## No real slurs

The blocklist path is exercised with **synthetic placeholder tokens**
(`zzsynthslur`, `zzsynthslur2`) checked into `data/blocklist.json`.

This is a deliberate engineering decision, not squeamishness:

1. A public repo whose test fixtures are a slur list is a repo nobody can open
   at work, and the first thing any reviewer flags.
2. The blocklist is a **substring membership test**. Its correctness does not
   depend on what the strings mean. A synthetic token exercises exactly the
   same code path with exactly the same coverage.
3. Real deployments point `BLOCKLIST_PATH` at a private, per-creator list that
   is maintained by moderators and never committed. Testing against a
   hardcoded public list would test the wrong thing anyway.

What this set *does* cover, in place of slurs, is the harder and more
interesting category: hostility that carries no flaggable token at all —
sarcasm (`g040`), coded implication (`g039`), third-person disparagement
(`g068`), and character-level evasion (`g036`, `g037`).

## Label policy

Two independent booleans, because downstream consumers branch on them
separately:

| Field | Means | Decided by |
|---|---|---|
| `toxic` | Directed hostility, threat, or sexual content aimed at a person | Would a moderator act on this? |
| `scam` | Solicitation, off-platform lure, impersonation, fraud | Is the intent to extract money or contact? |
| `question` | Message is asking something | Structural — drives the response board, not moderation |

Calls that were genuinely close, recorded so they can be argued with:

- **`g018`** *"you are so dumb at this game hahaha"* → `toxic: true`. Directed
  second-person insult. In a friend group this is banter; from a stranger in a
  live chat it is what moderators actually remove. Labeled on the moderator's
  standard, not the sender's intent.
- **`g064`** *"why do you always ignore the questions?"* → `toxic: false`.
  Frustrated, entitled, unpleasant to receive — and not actionable. This is the
  line the system most needs to hold; over-flagging here is how a mod tool
  loses its users' trust.
- **`g017`** *"check my bio for my art commissions"* → `scam: false`, and
  it **fires the link-redirect rule anyway**. Left in as a known false
  positive. The precision number should carry this cost visibly rather than
  have the fixture quietly bless a bad rule.
- **`g068`** *"she is not even that pretty tbh"* → `toxic: true`. Third-person
  disparagement of the streamer in her own chat. Moderators treat this as
  harassment; a second-person target check cannot see it.

## Metadata fields

- **`t0_blind`** — an assertion about the *fixture*, not about any
  implementation: the meaning of this message is unreachable by Latin keyword
  matching. Used to report blind-spot exposure separately from ordinary misses.
  16 of 78 rows.
- **`paraphrase`** — a natural-chat rephrasing of a category the lexicon covers
  in its canonical form. See below. 8 of 78 rows.
- **`fp_trap`** — benign text containing hostile-looking tokens. A system that
  scores well overall while failing these is not usable in production.
- **`tags`** — free-form. `voice-gate-repro` marks `g053`, the case that
  motivated this repo.

## The paraphrase rows, and why they exist

Rows `g001`-`g070` were written in the same sitting as T0's lexicon. That is a
real methodological problem: four of them (`g022`, `g023`, `g026`, `g030`) are
near-verbatim reproductions of the regexes that match them, so "T0 recall"
was substantially the lexicon being read back to itself.

Rather than delete the coverage, `g071`-`g078` were added: ordinary ways a
viewer might phrase the *same categories*, written against the category
definitions rather than the patterns. The two slices then separate cleanly:

| slice | recall | reading |
|---|---:|---|
| `verbatim` | 95.8% | the lexicon matching what it was written against |
| `paraphrase` | 14.3% | the lexicon generalizing — 1 positive in 7 |

The gap is the finding. `verbatim` is a regression guard, not a capability
claim, and the README says so where it reports the number.

`g077` is the deliberate control: rephrased, but still containing a canonical
token ("cash app"). It is caught. Without a row like it, a drop in the
paraphrase slice could not be distinguished from the fixtures simply being
harder than intended.

`g078` is the paraphrase set's negative — rude but not actionable. A slice of
positives only measures recall and would happily reward a classifier that
flagged everything.

## Adding rows

Add the case *before* the fix. A row that only ever existed after its bug was
resolved never proved anything.
