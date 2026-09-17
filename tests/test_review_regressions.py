"""Cases from review that the original golden set did not exercise."""

import json

import pytest

import harness
from metrics import evaluate
from tiermod import t0, t1
from tiermod.schema import Verdict


@pytest.mark.parametrize("text", [
    "you should disappear forever",
    "nice, you should disappear forever",
    "thanks, you should disappear forever",
    "die", "죽어", "しね", "לך", "7mar", "gg you should disappear forever",
])
def test_unresolved_text_is_not_cleared(text):
    verdict = t0.score(text)
    assert verdict.toxic or verdict.scam or verdict.needs_llm


@pytest.mark.parametrize("text", ["lol", "GG", "<3", "🔥🔥", "  thanks!  "])
def test_complete_trivial_messages_remain_cheap(text):
    assert not t0.score(text).needs_llm


def test_empty_evaluation_fails_gate():
    assert harness.check_thresholds(evaluate([]))


def test_missing_thresholds_fails_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "THRESHOLDS", tmp_path / "missing.toml")
    with pytest.raises((OSError, ValueError)):
        harness.check_thresholds(evaluate([]))


def test_all_clean_t1_fails_gate():
    pairs = [(label, t0.score(label.text)) for label in harness.load_golden()]
    cleared = [(label, Verdict(tier="T1")) for label, base in pairs if base.needs_llm]
    assert harness.check_thresholds(evaluate(pairs, t1_pairs=cleared))


@pytest.mark.parametrize("payload", [None, {}, "word", [""], ["  "], [1]])
def test_invalid_blocklist_is_rejected(tmp_path, payload):
    path = tmp_path / "blocklist.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        t0.load_blocklist(path)


def test_missing_model_result_preserves_fallback(monkeypatch):
    class Model:
        def invoke(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(t1, "_get_llm", lambda: Model())
    label = harness.load_golden()[0]
    base = Verdict(needs_llm=True, question=True)
    assert t1.score_one(label, base) is base
