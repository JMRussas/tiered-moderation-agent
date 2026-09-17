"""T1's degradation contract, verified without a model or a network.

This file exists because the contract was stated in a docstring and not
implemented. `score_one` took its fallback as an OPTIONAL parameter, and
`score_batch` called it through `pool.map` with a single iterable -- so every
failure degraded to a blank `Verdict()` rather than to T0's.

The observable effect on `g053`, the fixture this whole project is built
around, with Ollama unreachable:

    T0      : toxic=False  needs_llm=True   question=True
    T1(down): toxic=False  needs_llm=False  question=False

`needs_llm` went True -> False. "I could not read this" silently became "this
is clean" -- the precise failure mode documented in docs/failure-modes.md, in
the code written to prevent it.

The fix made the parameter mandatory so the mistake is unwritable. These tests
make sure it stays that way.
"""

from __future__ import annotations

import inspect

import pytest

from tiermod import t0, t1
from tiermod.schema import Label, Verdict


@pytest.fixture
def unreadable() -> tuple[Label, Verdict]:
    """An Arabic-script message T0 correctly refuses to judge."""
    label = Label(
        id="t1-degrade",
        text="ليه انتي وحشة كده؟",
        toxic=True,
        scam=False,
        question=True,
        lang="ar",
        script="arabic",
        category="toxic-foreign",
        t0_blind=True,
    )
    verdict = t0.score(label.text)
    assert verdict.needs_llm is True, "fixture precondition"
    assert verdict.toxic is False, "fixture precondition"
    return label, verdict


class _Down:
    """A constructed client whose every call fails, as with an unreachable
    server. Construction itself is lazy and never connects."""

    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("ollama is down")


def _explode():
    return _Down()


def test_failed_call_returns_the_t0_verdict_unchanged(unreadable, monkeypatch):
    """The whole contract, in one assertion."""
    label, base = unreadable
    monkeypatch.setattr(t1, "_get_llm", _explode)

    got = t1.score_one(label, base)

    assert got == base, "a failed T1 must hand back T0's verdict untouched"


def test_failed_call_preserves_needs_llm(unreadable, monkeypatch):
    """Stated separately because this is the field that gets someone hurt.

    A consumer reads `toxic=False, needs_llm=False` as permission to act.
    """
    label, base = unreadable
    monkeypatch.setattr(t1, "_get_llm", _explode)

    assert t1.score_one(label, base).needs_llm is True


def test_failed_call_preserves_question(unreadable, monkeypatch):
    """`question` is structural and T0 already decided it. Upstream it is worth
    40 points on the response board, so dropping it silently changes who the
    host talks to."""
    label, base = unreadable
    monkeypatch.setattr(t1, "_get_llm", _explode)

    assert t1.score_one(label, base).question is True


def test_batch_degrades_every_message(unreadable, monkeypatch):
    """`score_batch` is the real call site -- the original bug was here, not in
    `score_one`."""
    label, base = unreadable
    monkeypatch.setattr(t1, "_get_llm", _explode)

    pairs = t1.score_batch([(label, base), (label, base)])

    assert len(pairs) == 2
    assert all(v == base for _, v in pairs)


def test_fallback_has_no_default():
    """Structural guard. The bug was possible because this argument was
    optional; if a later refactor makes it optional again, the same class of
    failure comes straight back."""
    sig = inspect.signature(t1.score_one)
    base = sig.parameters["base"]
    assert base.default is inspect.Parameter.empty, (
        "score_one's fallback must stay mandatory -- see this module's docstring"
    )


def test_batch_requires_pairs_not_labels(unreadable, monkeypatch):
    """Passing bare labels -- the original mistake -- must fail loudly rather
    than silently score without a degradation target."""
    label, _ = unreadable
    monkeypatch.setattr(t1, "_get_llm", _explode)

    with pytest.raises((TypeError, ValueError)):
        t1.score_batch([label, label])


def test_empty_batch_is_safe():
    assert t1.score_batch([]) == []
