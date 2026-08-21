"""Metrics for a tiered classifier.

Aggregate accuracy is close to useless here. A system that is 94% accurate on
live chat has probably learned to say "benign" — which is what chat mostly is —
and the 6% it gets wrong is the entire product. So everything below is reported
per-slice, and two of the numbers are about *routing* rather than accuracy.

The two that carry the architecture
-----------------------------------
**Escalation recall** (`safe_miss_rate`). Of the positives T0 got wrong, what
fraction did it at least mark ``needs_llm``? This is the number that decides
whether the tiering is safe. T0 is *allowed* to miss — it is a keyword matcher.
It is not allowed to miss *silently*, because a confident ``toxic=False`` is
what a downstream voice gate reads as permission to speak. A missed positive
that was escalated is recoverable by T1; one that was not is a silent error.

**Escalation rate**. What fraction of all traffic gets routed to a model. This
is the cost side of the same trade, and the claim the whole architecture rests
on. If it drifts up, the thesis quietly stops being true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from tiermod.schema import Label, Verdict


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def fpr(self) -> float | None:
        """False positive rate. The trust metric: a mod tool that cries wolf
        gets ignored, and an ignored tool has zero recall in practice."""
        d = self.fp + self.tn
        return self.fp / d if d else None

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
        elif not predicted and actual:
            self.fn += 1
        else:
            self.tn += 1


@dataclass
class Slice:
    name: str
    n: int
    toxic: Counts
    scam: Counts
    # Either label being true; what a moderator surface actually branches on.
    flagged: Counts


@dataclass
class Routing:
    """Cost and safety of T0's escalation decision."""

    escalated: int = 0
    total: int = 0
    # Positives T0 missed, split by whether it admitted uncertainty.
    missed_escalated: int = 0
    missed_silent: int = 0
    silent_ids: list[str] = field(default_factory=list)

    @property
    def escalation_rate(self) -> float | None:
        return self.escalated / self.total if self.total else None

    @property
    def safe_miss_rate(self) -> float | None:
        """Of T0's misses, the fraction it flagged as unreliable.

        1.0 means every miss is recoverable downstream. Anything below means
        there are messages the system is confidently, silently wrong about.
        """
        d = self.missed_escalated + self.missed_silent
        return self.missed_escalated / d if d else None


@dataclass
class Report:
    slices: list[Slice]
    routing: Routing
    # recall(T0 then T1) minus recall(T0), on `flagged`. None when T1 was off.
    t1_lift: float | None = None
    t1_evaluated: int = 0
    # Positives still missed after T1 that no longer carry needs_llm --
    # "confident and wrong" at the end of the pipeline.
    #
    # REPORTED, NOT GATED, and the distinction matters. This bucket mixes two
    # different things: T1 ran and got it wrong (unavoidable -- no classifier
    # has perfect recall), and T1 failed but cleared needs_llm anyway (a defect,
    # and the one this project is about). Gating it at zero would demand 100%
    # T1 recall, so the gate would be permanently red and get switched off.
    #
    # The defect half is caught precisely, and without a model, by
    # tests/test_t1_degradation.py: a failed T1 call must return its T0 base
    # object unchanged. That is the hard gate. This number is the residual
    # exposure to look at afterwards.
    combined_confident_misses: list[str] = field(default_factory=list)

    def slice_by(self, name: str) -> Slice | None:
        return next((s for s in self.slices if s.name == name), None)


def _count(rows: Sequence[tuple[Label, Verdict]]) -> Slice:
    toxic, scam, flagged = Counts(), Counts(), Counts()
    for label, verdict in rows:
        toxic.add(verdict.toxic, label.toxic)
        scam.add(verdict.scam, label.scam)
        flagged.add(verdict.toxic or verdict.scam, label.toxic or label.scam)
    return Slice(name="", n=len(rows), toxic=toxic, scam=scam, flagged=flagged)


SLICES: dict[str, Callable[[Label], bool]] = {
    "all": lambda l: True,
    "t0_blind": lambda l: l.t0_blind,
    "t0_readable": lambda l: not l.t0_blind,
    "fp_traps": lambda l: l.fp_trap,
    "arabic_script": lambda l: l.script == "arabic",
    "arabizi": lambda l: l.lang == "ar-arabizi",
    "english": lambda l: l.lang == "en",
}


def evaluate(
    pairs: Iterable[tuple[Label, Verdict]],
    *,
    t1_pairs: Sequence[tuple[Label, Verdict]] | None = None,
) -> Report:
    """Score T0 verdicts, optionally crediting T1 with what it recovered.

    `t1_pairs` holds only the messages T0 escalated, with T1's verdict. The
    combined tier takes T1's answer where it exists and T0's everywhere else —
    which is what the production path does.
    """
    pairs = list(pairs)

    slices = []
    for name, pred in SLICES.items():
        rows = [(l, v) for l, v in pairs if pred(l)]
        if not rows:
            continue
        s = _count(rows)
        s.name = name
        slices.append(s)

    routing = Routing(total=len(pairs))
    for label, verdict in pairs:
        if verdict.needs_llm:
            routing.escalated += 1
        actual = label.toxic or label.scam
        predicted = verdict.toxic or verdict.scam
        if actual and not predicted:
            if verdict.needs_llm:
                routing.missed_escalated += 1
            else:
                routing.missed_silent += 1
                routing.silent_ids.append(label.id)

    report = Report(slices=slices, routing=routing)

    if t1_pairs:
        recovered = {l.id: v for l, v in t1_pairs}
        combined = [
            (l, recovered.get(l.id, v)) for l, v in pairs
        ]
        base = report.slice_by("all")
        after = _count(combined)
        if base and base.flagged.recall is not None and after.flagged.recall is not None:
            report.t1_lift = after.flagged.recall - base.flagged.recall
        report.t1_evaluated = len(t1_pairs)
        after.name = "all+t1"
        report.slices.append(after)

        for label, verdict in combined:
            actual = label.toxic or label.scam
            predicted = verdict.toxic or verdict.scam
            if actual and not predicted and not verdict.needs_llm:
                report.combined_confident_misses.append(label.id)

    return report
