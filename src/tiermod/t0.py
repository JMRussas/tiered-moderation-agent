"""T0 -- deterministic scoring. No model, no network, no I/O.

Runs on every message. Target is single-digit microseconds, because this tier
is the reason the architecture is affordable: if ~90% of live chat resolves
here, the model budget only has to cover the remainder.

The design property that matters more than accuracy
---------------------------------------------------
T0 is keyword-based, so it is structurally blind to anything outside its
lexicon -- Arabic script, Arabizi, and any hostility phrased in words nobody
put in a list. A tier that is confidently wrong on those is dangerous, because
downstream consumers (a mod feed, a voice) read ``toxic=False`` as "safe".

So T0's real job is not "catch everything". It is: **catch what it can, and be
honest about the rest.** That honesty is ``needs_llm``. A caller that gates on
a T0 verdict without consulting ``needs_llm`` has misread this module -- see
``docs/failure-modes.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import Verdict

# --- Lexicons -------------------------------------------------------------
# Deliberately multilingual (EN/ES/FR/AR) because live chat is. Deliberately
# incomplete, because pretending otherwise is what produces the false
# confidence this tier exists to avoid.

POSITIVE = re.compile(
    r"\b(love|great|amazing|beautiful|thank|thanks|congrats|awesome|nice|cute|"
    r"gracias|hermosa|bonita|genial|merci|belle|super)\b|احبك|جميل|شكرا",
    re.I,
)

NEGATIVE = re.compile(
    r"\b(hate|stupid|idiot|dumb|trash|ugly|shut up|loser|boring|worst|cringe|"
    r"estupido|idiota|tonto|feo|callate|basura|deteste|imbecile|nul)\b"
    r"|غبي|قذر|اكرهك|اخرس|حقير",
    re.I,
)

# Second person, used to distinguish "this game is trash" (an opinion about a
# thing) from "you are trash" (an attack on a person). This one check removes
# most of the false positives a naive sentiment list produces on gaming chat.
SECOND_PERSON = re.compile(
    r"\b(you|your|u|ur|tu|te|toi|vous|usted)\b|انت|انتي", re.I
)

# Sexual / solicitation aimed at a streamer. Kept specific: broad patterns here
# are the single largest source of false positives on ordinary flirty chat.
SEXUAL = re.compile(
    r"\b(send\s+nudes?"
    r"|show\s+(?:me\s+)?(?:your\s+)?(?:body|feet|chest)"
    r"|take\s+(?:it|them)\s+off"
    r"|strip\s+for\s+"
    r"|sit\s+on\s+my"
    r"|what\s+are\s+you\s+wearing\s+under)",
    re.I,
)

THREAT = re.compile(
    r"\b(i(?:’|')?ll\s+find\s+you"
    r"|i\s+know\s+where\s+you\s+live"
    r"|watch\s+your\s+back"
    r"|you(?:’|')?re\s+dead"
    r"|kill\s+your\s*self"
    r"|kys"
    r"|go\s+die"
    r"|hope\s+you\s+die)\b",
    re.I,
)

QUESTION = re.compile(
    r"[?؟¿]"
    r"|\b(who|what|when|where|why|how|which|can|do|does|did|is|are|will|would"
    r"|que|qué|quien|quién|como|cómo|donde|dónde|cuando|cuándo"
    r"|qui|quoi|quand|ou|où|comment|pourquoi)\b",
    re.I,
)

SCAM = [
    (re.compile(r"\b(?:free\s+)?(?:v-?bucks|robux|gift\s*cards?)\b", re.I), "free-currency lure"),
    (re.compile(r"\bdm\s+me\b.*\b(?:crypto|invest|profit|forex|trading)\b", re.I), "DM investment lure"),
    (
        re.compile(r"\b(?:i'?m|this\s+is)\s+the\s+(?:real\s+)?(?:host|creator|streamer|owner)\b", re.I),
        "impersonation",
    ),
    (re.compile(r"\bcash\s*app\b|\bvenmo\b|\bzelle\b|\bpaypal\.me\b", re.I), "off-platform payment"),
    (re.compile(r"\b(?:click|check)\s+(?:my|the)\s+(?:bio|link|profile)\b", re.I), "link redirect"),
    (
        re.compile(r"\b(?:whats\s*app|telegram|snap(?:chat)?)\b.{0,20}\b(?:me|my|number|add)\b", re.I),
        "off-platform contact",
    ),
]

LINK = re.compile(
    r"(https?://|www\.)\S+|[a-z0-9-]+\.(?:com|net|io|xyz|link|live|gg|shop|store)\b", re.I
)

# Arabic, Cyrillic, CJK blocks. Presence of any means the Latin lexicons above
# had no chance of applying.
NON_LATIN = re.compile(r"[؀-ۿݐ-ݿ一-鿿Ѐ-ӿ]")

# --- Arabizi --------------------------------------------------------------
# Egyptian Arabic written in Latin letters with digits standing in for Arabic
# sounds (3=ayn, 7=haa, 2=hamza, 5=khaa, 9=qaf). Extremely common in TikTok
# chat and invisible to both Latin keyword lists and naive language detection
# -- it looks like English to every cheap heuristic, which is exactly why it
# has to be escalated rather than scored.

_DIGIT_IN_WORD = re.compile(r"[a-z][23579][a-z]|[a-z][23579]\b|\b[23579][a-z]", re.I)
_ARABIZI_STRONG = {"habibi", "7abibi", "ya3ni", "inshallah", "yalla", "khalas", "5alas", "wallahi"}
_ARABIZI_WEAK = {
    "enti", "enta", "ana", "eh", "leh", "keda", "mesh",
    "bas", "kol", "meen", "fen", "ezay", "3ayez", "3ayza",
}


def looks_arabizi(text: str) -> bool:
    """Two independent signals; either one strong hit or two weak ones."""
    low = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", low))
    if tokens & _ARABIZI_STRONG:
        return True
    # A digit *inside* a word is the giveaway. English essentially never does
    # this; "3ayez", "7elwa", "2alby" all do.
    if _DIGIT_IN_WORD.search(low) and len(tokens) >= 2:
        return True
    return len(tokens & _ARABIZI_WEAK) >= 2


# --- Blocklist ------------------------------------------------------------
# Loaded from disk, never inlined. The shipped file contains SYNTHETIC
# placeholder tokens only: this repo tests the blocklist *mechanism* without
# publishing a slur list. Point it at a real list in deployment. See
# evals/golden/README.md for why the golden set contains no real slurs.

_BLOCKLIST: set[str] | None = None
_DEFAULT_BLOCKLIST = Path(__file__).resolve().parents[2] / "data" / "blocklist.json"


def load_blocklist(path: str | Path | None = None) -> set[str]:
    global _BLOCKLIST
    if path is None and _BLOCKLIST is not None:
        return _BLOCKLIST
    p = Path(path) if path else _DEFAULT_BLOCKLIST
    try:
        terms = {str(t).lower() for t in json.loads(p.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        terms = set()
    if path is None:
        _BLOCKLIST = terms
    return terms


def _blocklist_hit(low: str, terms: set[str]) -> bool:
    return any(t in low for t in terms)


def score(
    text: str,
    *,
    translate_mode: str | None = None,
    blocklist: set[str] | None = None,
) -> Verdict:
    """Score one message.

    Pure apart from the lazily-cached blocklist read; same input, same output.
    """
    t = (text or "").strip()
    if not t:
        return Verdict(tier="T0")

    low = t.lower()
    terms = blocklist if blocklist is not None else load_blocklist()

    # --- Toxicity ---------------------------------------------------------
    # NEGATIVE alone is not enough: it needs a second-person target, or every
    # "this game is trash" gets flagged.
    toxic = bool(
        THREAT.search(t)
        or SEXUAL.search(t)
        or _blocklist_hit(low, terms)
        or (NEGATIVE.search(t) and SECOND_PERSON.search(t))
    )

    # --- Scam -------------------------------------------------------------
    scam = any(rx.search(t) for rx, _ in SCAM)

    # --- Sentiment --------------------------------------------------------
    pos = bool(POSITIVE.search(t))
    neg = bool(NEGATIVE.search(t)) or toxic
    friendliness: float | None
    if pos and not neg:
        friendliness = 0.6
    elif neg and not pos:
        friendliness = -0.6
    elif pos and neg:
        friendliness = 0.0
    else:
        friendliness = None

    # --- Escalation -------------------------------------------------------
    # The honesty flag. Escalate whenever this tier has structural reason to
    # doubt its own verdict:
    arabizi = looks_arabizi(t)
    words = re.sub(r"[^\w]+", " ", t, flags=re.UNICODE).split()
    forced = translate_mode == "non-english" and not _is_plain_english(t)
    needs_llm = bool(
        NON_LATIN.search(t)  # (a) a script the lexicons cannot address
        or arabizi  # (b) Latin letters, non-Latin language
        or forced  # (c) caller says this room is multilingual
        # (d) real words, and every lexicon returned nothing -- no sentiment,
        #     no toxicity, no scam. Total silence on a substantive message is
        #     itself evidence the lexicons do not cover it.
        or (
            not toxic
            and not scam
            and friendliness is None
            and len(words) >= 2
            and len(t) >= 8
        )
        # (e) hostility seen, target unconfirmed. Rule (d) alone was not
        #     enough: it requires `friendliness is None`, so a message that
        #     trips a NEGATIVE keyword but fails the second-person check exits
        #     with a CONFIDENT `toxic=False` and no escalation. That is the
        #     worst possible combination -- wrong and silent -- and it is how
        #     "eres una idiota" scored clean (the second-person list has no
        #     Spanish copula). Seeing hostility without being able to confirm
        #     its target is uncertainty, not innocence.
        #
        #     Cost: this escalates the "this game is trash" population too.
        #     Measured on the golden set it moved escalation 51.4% -> 57.1%,
        #     and safe-miss 93.8% -> 100%. See docs/failure-modes.md.
        or (neg and not toxic)
    )

    return Verdict(
        toxic=toxic,
        scam=scam,
        question=bool(QUESTION.search(t)),
        friendliness=friendliness,
        needs_llm=needs_llm,
        arabizi=arabizi,
        tier="T0",
    )


def _is_plain_english(t: str) -> bool:
    return not NON_LATIN.search(t) and not looks_arabizi(t)
