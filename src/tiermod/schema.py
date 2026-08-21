"""Typed contracts shared by every tier and by the eval harness.

The whole architecture rests on one claim: most messages resolve without a
model. That claim is only meaningful if every tier answers the *same question*
in the *same shape*, so the harness can score them against each other. These
types are that shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["T0", "T1", "T2"]


class Verdict(BaseModel):
    """What any tier returns about a single message.

    `toxic` and `scam` are the two decisions the downstream consumers actually
    branch on: the mod feed surfaces them, and the voice layer refuses to speak
    about anything carrying either.
    """

    toxic: bool = False
    scam: bool = False
    question: bool = False
    # [-1, 1]; None when the tier had no opinion (no sentiment keywords hit).
    friendliness: float | None = None

    # --- Routing -----------------------------------------------------------
    # The most important field in this file. T0 sets it to say "I could not
    # read this -- do not trust my verdict." Everything downstream that gates
    # on a T0 verdict MUST consult it; a `toxic=False` carrying
    # `needs_llm=True` means "unknown", not "clean".
    needs_llm: bool = False
    arabizi: bool = False

    tier: Tier = "T0"
    lang: str | None = None
    translation: str | None = None

    # Why this verdict fired, in moderator-readable terms ("off-platform
    # payment", "threat"). The whole product is advisory -- a human decides and
    # can mark the call wrong -- and an advisory a moderator cannot audit is
    # one they learn to ignore. T0 knows the reason at match time; dropping it
    # there means no later tier can reconstruct it.
    reasons: list[str] = Field(default_factory=list)


class Label(BaseModel):
    """Ground truth for one golden-set message.

    `t0_blind` is an assertion about the *fixture*, not about any
    implementation: it marks messages whose meaning is unreachable by
    Latin-keyword matching, so the harness can report blind-spot exposure
    separately from ordinary misses.
    """

    id: str
    text: str
    toxic: bool
    scam: bool
    question: bool

    lang: str
    script: Literal["latin", "arabic"] = "latin"
    category: str
    t0_blind: bool = False
    note: str = ""
    # Marks deliberate false-positive traps: benign text that contains
    # hostile-looking tokens ("this game is trash", "I'd kill for that skin").
    fp_trap: bool = False
    tags: list[str] = Field(default_factory=list)
