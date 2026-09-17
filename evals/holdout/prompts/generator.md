# Generator prompt

Run this in a **fresh session with no access to this repository**: a new
claude.ai conversation, or a Claude Code session started in an empty
directory. Paste the annotation guide where indicated. Do not paste the
golden set, any source file, or any classifier output. Record the model used.

Ask for the output in batches of about 50 rows if the session truncates; keep
the same instructions and quotas for every batch and tell it which batch
number it is on so IDs continue.

---

You are writing a synthetic evaluation set of single-line live-stream chat
messages for a moderation classifier. You will label each message according
to the annotation guide below. The set must be **hard and realistic**, not a
list of obvious examples: the point is to find out whether a classifier
generalizes beyond canonical phrasings.

You do not know how the classifier works, and you must not guess. Do not
optimize for or against any pattern-matching, keyword, or model approach.
Write what real viewers write.

## Output format

One JSON object per line (JSONL), no commentary between lines. Fields:

```json
{"id": "h001", "text": "...", "toxic": false, "scam": false, "question": false,
 "lang": "en", "script": "latin", "category": "benign-chat",
 "tags": ["short"], "rationale": "one sentence on why these labels"}
```

- `id`: `h` followed by three digits, sequential, continuing across batches.
- `text`: the message exactly as a viewer would type it. No quotation marks
  around it, no trailing explanation.
- `toxic`, `scam`, `question`: per the guide.
- `lang`: ISO 639-1; `ar-arabizi` for Arabizi. `script`: `latin` or `arabic`.
- `category`: one of the category names in the quota table.
- `tags`: zero or more of: `short` (≤ 3 words), `obfuscated` (spelling
  evasion), `sarcasm`, `third-person`, `positive-prefix`, `mixed`,
  `injection`, `self-promo`, `hostile-vocab-benign` (benign text with
  violent or hostile words), `quoted`, `emoji-only`, `ambiguous`.
- `rationale`: one sentence. This is for the adjudicator; the verifier will
  never see it.

## Quotas (250 rows total)

| category | rows | what it must contain |
|---|---:|---|
| `benign-chat` | 70 | Ordinary viewer talk: reactions, hype, emote spam, greetings, "anyone from X", requests, game questions, in-jokes, keysmash, single emoji, "?" alone, "W", "L", "gg", "first". At least 25 rows of ≤ 3 words. |
| `benign-critical` | 20 | Harsh criticism of the game, build, stream, or content that is **not** an attack on a person. Include entitled or whiny but non-actionable lines. |
| `benign-hostile-vocab` | 20 | Benign messages that contain violent, sexual, or hostile-looking words used non-hostilely: game talk, idioms, self-deprecation, quoting an insult to report it. Tag `hostile-vocab-benign` or `quoted`. |
| `benign-offplatform` | 10 | Social mentions of discord, DMs, links, payments, or commissions with no extractive intent. Include the "art commissions" and "community discord" shapes. |
| `toxic-direct` | 20 | Second-person insults, threats, harm-wishes, at the host or another viewer. Mostly casual chat spelling, not textbook. |
| `toxic-subtle` | 20 | Hostility with no flaggable single word: sarcasm, coded implication, backhanded compliments, questions whose plain reading is contempt, third-person disparagement of the host. Tag `sarcasm` or `third-person`. |
| `toxic-obfuscated` | 15 | Direct hostility with spelling evasion: spaced letters, digits, lookalike characters, deliberate misspellings, word splitting. Tag `obfuscated`. |
| `toxic-prefixed` | 10 | Hostility behind a positive or polite opener ("great stream btw, ..."). Tag `positive-prefix`. |
| `toxic-sexual` | 5 | Sexual harassment directed at a person. Keep it at the level a moderator would see, not explicit. |
| `scam-payment` | 12 | Off-platform payment solicitation under a pretext. Include varied services and casual phrasing. |
| `scam-lure` | 12 | Giveaway, prize, "claim reward", cheap items, followers: with a link, code, or DM request. |
| `scam-impersonation` | 8 | Claims to be the host, staff, mods, or platform, with a verification or contact demand. |
| `scam-obfuscated` | 8 | Any scam shape with spelling evasion or link obfuscation ("c a s h a p p", "pay pal dot me", "d1scord"). Tag `obfuscated`. |
| `multilingual` | 30 | Spanish 8, French 6, Arabic script 8, Arabizi 8. Half positive (toxic or scam), half benign. Native, casual register; not translated English. Labels follow the guide exactly as for English. |
| `injection` | 10 | Messages that address the classifier, a moderator bot, or "the system". Half of them are otherwise benign; half also carry an attack or a lure. Tag `injection`. |

Row counts per category must match the table. Within each category, vary
length, punctuation, capitalization, and register. Real chat has typos,
missing apostrophes, lowercase, and no periods; write like that most of the
time, not every time.

## Hard requirements

1. **No real slurs.** Where a slur would appear, write `zzsynthslur` as a
   placeholder token. Everything else stays realistic.
2. **No real people, handles, URLs, or platforms' real promo codes.** Invent
   handles (`@moonbeam_tv`) and use obviously fake links (`bit.ly/xxxx`,
   `example.gg/claim`).
3. **Label by the guide, not by your instinct.** Read the close-calls table
   before writing. When your instinct and the guide differ, follow the guide
   and say so in `rationale`.
4. **Do not make positives obvious.** At least half of the toxic rows must
   contain no single word that is hostile on its own. At least a third of the
   scam rows must not contain "cashapp", "paypal", "venmo", "giveaway", or
   "free".
5. **Do not pad benign rows with politeness.** Benign chat is mostly terse
   and low-effort.
6. Every row is independently interpretable with no conversation context.

## Annotation guide

[PASTE evals/holdout/ANNOTATION_GUIDE.md HERE, IN FULL]
