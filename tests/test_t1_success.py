"""Runtime inputs, output validation, and shared inference admission."""

from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event

import pytest

from tiermod import t1
from tiermod.schema import Message, Verdict


def result():
    return dict(toxic=True, scam=False, sentiment="negative", lang="ar",
                translation="You are ugly", reasons=["Directed insult"])


def test_runtime_message_and_explanation(monkeypatch):
    message = Message(id="incoming", text="some private text")
    base = Verdict(needs_llm=True, question=True, arabizi=True)

    class Model:
        def invoke(self, prompt):
            assert prompt[-1] == ("human", "<message>some private text</message>")
            return result()

    monkeypatch.setattr(t1, "_get_llm", lambda: Model())
    [(returned, verdict)] = t1.score_batch([(message, base)])
    assert returned is message
    assert verdict.tier == "T1" and verdict.toxic
    assert verdict.reasons == ["Directed insult"]
    assert verdict.translation == "You are ugly"
    assert verdict.question and verdict.arabizi and not verdict.needs_llm


def test_model_cannot_clear_a_deterministic_t0_hit(monkeypatch):
    """A T0 policy match that was also escalated (non-Latin, Arabizi,
    translate_mode) keeps its flags and reasons whatever the model says."""
    class Model:
        def invoke(self, _prompt):
            return dict(toxic=False, scam=False, sentiment="neutral", lang="en",
                        translation=None, reasons=[])

    monkeypatch.setattr(t1, "_get_llm", lambda: Model())
    base = Verdict(toxic=True, scam=True, needs_llm=True, arabizi=True,
                   reasons=["threat", "off-platform payment"])
    verdict = t1.score_one(Message(id="m", text="kys"), base)
    assert verdict.tier == "T1" and not verdict.needs_llm
    assert verdict.toxic and verdict.scam
    assert verdict.reasons == ["threat", "off-platform payment"]


def test_model_reasons_merge_with_t0_reasons(monkeypatch):
    class Model:
        def invoke(self, _prompt):
            return dict(toxic=True, scam=False, sentiment="negative", lang="en",
                        translation=None, reasons=["threat", "Directed insult"])

    monkeypatch.setattr(t1, "_get_llm", lambda: Model())
    base = Verdict(toxic=True, needs_llm=True, reasons=["threat"])
    verdict = t1.score_one(Message(id="m", text="x"), base)
    assert verdict.reasons == ["threat", "Directed insult"]


@pytest.mark.parametrize("changes", [
    {"reasons": []}, {"reasons": [" "]}, {"sentiment": "bad"},
])
def test_invalid_results_degrade_and_release_capacity(monkeypatch, changes):
    class Model:
        def invoke(self, _prompt):
            return result() | changes

    slots = BoundedSemaphore(1)
    monkeypatch.setattr(t1, "_slots", slots)
    monkeypatch.setattr(t1, "_get_llm", lambda: Model())
    base = Verdict(needs_llm=True, reasons=["Existing reason"])
    assert t1.score_one(Message(id="m", text="text"), base) is base
    assert slots.acquire(blocking=False)
    slots.release()


def test_exception_logging_does_not_expose_message(monkeypatch, caplog):
    def fail():
        raise RuntimeError("private viewer text")

    monkeypatch.setattr(t1, "_get_llm", fail)
    base = Verdict(needs_llm=True)
    assert t1.score_one(Message(id="m", text="private viewer text"), base) is base
    assert "RuntimeError" in caplog.text
    assert "private viewer text" not in caplog.text


def test_limit_is_shared_by_batches_and_direct_calls(monkeypatch):
    entered, release = Event(), Event()

    class Model:
        def invoke(self, _prompt):
            entered.set()
            assert release.wait(5), "test did not release model"
            return result()

    monkeypatch.setattr(t1, "_slots", BoundedSemaphore(1))
    monkeypatch.setattr(t1, "_get_llm", lambda: Model())
    message, base = Message(id="m", text="text"), Verdict(needs_llm=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(t1.score_batch, [(message, base)])
        try:
            assert entered.wait(5)
            assert t1.score_one(message, base) is base
            assert t1.score_batch([(message, base)])[0][1] is base
        finally:
            release.set()
        assert first.result()[0][1].tier == "T1"
    assert t1.score_one(message, base).tier == "T1"


def test_lazy_model_is_initialized_once(monkeypatch):
    entered, release = Event(), Event()
    calls = []
    model = object()

    def create():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return model

    monkeypatch.setattr(t1, "_llm", None)
    monkeypatch.setattr(t1, "_create_llm", create)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(t1._get_llm) for _ in range(4)]
        try:
            assert entered.wait(5)
        finally:
            release.set()
        assert all(f.result() is model for f in futures)
    assert len(calls) == 1


# --- Outcome statuses --------------------------------------------------------
# A consumer or an eval must be able to tell admission control apart from a
# broken model without reading logs. Every non-ok status returns the base.

class _Http(Exception):
    """Stands in for httpx.HTTPError: classified by class name, not import."""


class HTTPError(_Http):
    pass


class ReadTimeout(HTTPError):
    pass


@pytest.mark.parametrize("exc, status", [
    (ReadTimeout("slow"), "transport"),
    (ConnectionRefusedError(), "transport"),
    (TimeoutError(), "transport"),
    (ValueError("bad json"), "validation"),
    (RuntimeError("server 500"), "model"),
])
def test_exception_classification(monkeypatch, exc, status):
    def raise_it(_prompt):
        raise exc

    monkeypatch.setattr(t1, "_get_llm", lambda: type("M", (), {"invoke": staticmethod(raise_it)})())
    base = Verdict(needs_llm=True)
    outcome = t1.classify(Message(id="m", text="x"), base)
    assert outcome.status == status and not outcome.succeeded
    assert outcome.verdict is base
    assert outcome.error == type(exc).__name__
    assert outcome.latency_ms >= 0


def test_invalid_output_is_a_validation_outcome(monkeypatch):
    monkeypatch.setattr(t1, "_get_llm", lambda: type("M", (), {"invoke": lambda self, _p: {"toxic": True, "reasons": []}})())
    outcome = t1.classify(Message(id="m", text="x"), Verdict(needs_llm=True))
    assert outcome.status == "validation" and outcome.error == "ValidationError"


def test_capacity_outcome_when_saturated(monkeypatch):
    slots = BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(t1, "_slots", slots)
    monkeypatch.setattr(t1, "_get_llm", lambda: pytest.fail("must not call the model"))
    base = Verdict(needs_llm=True)
    outcome = t1.classify(Message(id="m", text="x"), base)
    assert outcome.status == "capacity" and outcome.verdict is base and outcome.error is None
    slots.release()


def test_ok_outcome_and_wrappers_agree(monkeypatch):
    monkeypatch.setattr(t1, "_get_llm", lambda: type("M", (), {"invoke": lambda self, _p: result()})())
    message, base = Message(id="m", text="x"), Verdict(needs_llm=True)
    outcome = t1.classify(message, base)
    assert outcome.succeeded and outcome.status == "ok" and outcome.verdict.tier == "T1"
    assert t1.score_one(message, base) == outcome.verdict
    [(_, verdict)] = t1.score_batch([(message, base)])
    assert verdict == outcome.verdict
    [(_, batched)] = t1.classify_batch([(message, base)])
    assert batched.status == "ok"


def test_inference_settings_match_client_construction():
    settings = t1.inference_settings()
    assert settings["model"] == t1.MODEL and settings["num_ctx"] == t1.NUM_CTX
    assert settings["concurrency"] == t1.CONCURRENCY and settings["timeout_s"] == t1.TIMEOUT_S
    assert settings["timeout_kind"] == "http-inactivity"
