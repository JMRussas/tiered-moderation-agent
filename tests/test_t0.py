"""T0 unit tests.

These encode the decisions that are easy to "clean up" and hard to recover:
the second-person target check, the Arabizi signals, and above all the
escalation contract. Several of these tests exist because the behaviour they
pin down looks like a bug until you know why it is there.
"""

from __future__ import annotations

import pytest

from tiermod import t0


# --- The escalation contract ---------------------------------------------
# The single most important property in this module: T0 may be wrong, but it
# must not be wrong *and* confident.

def test_negative_without_target_escalates_rather_than_clearing():
    """Rule (e). This is the regression test for the silent miss the eval found.

    "eres una idiota" trips NEGATIVE but fails the second-person check, because
    the Spanish copula is not in the list. Before rule (e) it exited as a
    confident `toxic=False, needs_llm=False` -- wrong and silent.
    """
    v = t0.score("eres una idiota")
    assert v.toxic is False, "T0 genuinely cannot resolve this; it must not guess"
    assert v.needs_llm is True, "but it MUST admit that, or a voice gate reads it as clean"


def test_arabic_script_always_escalates():
    v = t0.score("انتي مقرفة جدا")
    assert v.needs_llm is True


def test_arabizi_escalates():
    v = t0.score("enti mesh 7elwa khalas")
    assert v.arabizi is True
    assert v.needs_llm is True


def test_confident_verdicts_do_not_escalate():
    """The cost side. A message T0 genuinely resolved must not reach a model."""
    assert t0.score("kys").needs_llm is False
    assert t0.score("free vbucks click my bio").needs_llm is False


def test_trivial_messages_never_escalate():
    """Chat is mostly this. If 'lol' reaches a GPU the architecture is dead."""
    for text in ("lol", "GG", "<3", "🔥🔥"):
        assert t0.score(text).needs_llm is False, text


# --- Target check ---------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "this game is trash lol",
        "that boss is stupid hard",
        "hate this lag so much",
        "the worst part is I already bought it",
    ],
)
def test_hostile_token_without_target_is_not_toxic(text):
    """Criticism of a *thing*. Flagging these is how a mod tool loses trust."""
    assert t0.score(text).toxic is False


@pytest.mark.parametrize(
    "text",
    ["you are so dumb at this game hahaha", "shut up you idiot", "ur ugly and this stream is boring"],
)
def test_hostile_token_with_target_is_toxic(text):
    assert t0.score(text).toxic is True


# --- Arabizi detection ----------------------------------------------------

def test_arabizi_strong_token():
    assert t0.looks_arabizi("ya habibi ezayak") is True


def test_arabizi_digit_inside_word():
    assert t0.looks_arabizi("3ayza a2olek 7aga") is True


def test_arabizi_two_weak_tokens():
    assert t0.looks_arabizi("meen enti bardo") is True


@pytest.mark.parametrize(
    "text",
    ["I have 3 kids", "level 7 boss", "top 5 plays", "im 21 btw"],
)
def test_english_with_digits_is_not_arabizi(text):
    """The digit signal requires the digit *inside* a word. Ordinary English
    numerals must not trip it or every stream stat escalates."""
    assert t0.looks_arabizi(text) is False


# --- Questions ------------------------------------------------------------

def test_arabic_question_mark_counts():
    """U+061F. Missed here, an Arabic question never reaches the response
    board -- and in the upstream system this flag is also what lifted a
    message to 'hot'. See docs/failure-modes.md."""
    assert t0.score("انتي منين؟").question is True


def test_interrogative_without_punctuation():
    assert t0.score("can you play the other map next").question is True


# --- Blocklist ------------------------------------------------------------

def test_blocklist_is_injectable():
    """Never read the real list in tests, and never inline terms in source."""
    assert t0.score("you are a zzsynthslur", blocklist={"zzsynthslur"}).toxic is True
    assert t0.score("you are a zzsynthslur", blocklist=set()).toxic is False


# --- Purity ---------------------------------------------------------------

def test_score_is_deterministic():
    text = "meen enti 3ashan tetkalemi keda"
    assert t0.score(text) == t0.score(text)


def test_empty_input_is_safe():
    for text in ("", "   ", None):
        v = t0.score(text)
        assert v.toxic is False and v.scam is False and v.needs_llm is False


# --- Configuration --------------------------------------------------------

def test_blocklist_path_env_is_honoured(tmp_path, monkeypatch):
    """BLOCKLIST_PATH was documented in .env.example and read nowhere. A
    deployment pointing at a private list would have silently kept using the
    committed placeholders."""
    real = tmp_path / "blocklist.json"
    real.write_text('["zzcustomterm"]', encoding="utf-8")
    monkeypatch.setenv("BLOCKLIST_PATH", str(real))
    monkeypatch.setattr(t0, "_BLOCKLIST", None)  # bypass the process-wide cache

    assert t0.load_blocklist() == {"zzcustomterm"}


def test_missing_blocklist_file_is_a_configuration_error(tmp_path, monkeypatch):
    """A bad path must never silently disable moderation policy."""
    monkeypatch.setenv("BLOCKLIST_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setattr(t0, "_BLOCKLIST", None)

    with pytest.raises(t0.BlocklistError, match="Cannot load blocklist"):
        t0.load_blocklist()


# --- Explainability -------------------------------------------------------

def test_verdict_carries_a_reason():
    """Advisories a moderator cannot audit are advisories they learn to
    ignore. The reason is known at match time and used to be discarded."""
    assert "off-platform payment" in t0.score("cashapp me 50").reasons
    assert "threat" in t0.score("kys").reasons
    assert t0.score("yooo this stream is amazing").reasons == []
