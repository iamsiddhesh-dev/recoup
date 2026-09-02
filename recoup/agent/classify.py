"""Root-cause classification: symptoms in, recovery-relevant cause out.

Razorpay tells you where a payment broke. This decides what to do about it, which
is a different question — `insufficient_funds` and `card_expired` are both
"declined", and one is worth retrying after payday while the other can never
succeed again.

The rules live in config/classifier.yaml and are written from Razorpay's public
documentation. They cannot see config/world.yaml; see
tests/test_no_ground_truth_leak.py. Because the documentation enumerates `source`
and `step` exhaustively but publishes `reason` only by example, the ruleset is
structurally complete about *where* and necessarily incomplete about *why*. What
falls through is not a bug in the rules, it is the shape of the available
information — and it is the caseload the LLM fallback exists to handle.

Nothing here calls a model. The fallback is a seam, wired up later, and
deliberately narrow: it sees only what the deterministic pass could not resolve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field

from recoup.domain import FailureCause, PaymentEntity

MATCHABLE = ("method", "source", "step", "reason")


class Resolution(StrEnum):
    """How a classification was reached. Reported, not inferred."""

    DETERMINISTIC = "deterministic"
    FALLBACK = "fallback"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Classification:
    cause: FailureCause | None
    confidence: float
    resolution: Resolution
    rule_id: str | None = None
    note: str | None = None

    @property
    def actionable(self) -> bool:
        return self.cause is not None


class Rule(BaseModel):
    id: str
    match: dict[str, str]
    cause: FailureCause
    confidence: float
    note: str | None = None

    @property
    def specificity(self) -> int:
        """How many fields the rule constrains.

        A rule naming a reason beats one naming only a source, because it is
        making a narrower claim and is more likely to be right about it.
        """
        return len(self.match)

    def matches(self, fields: dict[str, str | None]) -> bool:
        return all(fields.get(key) == value for key, value in self.match.items())


class ClassifierRules(BaseModel):
    version: int
    min_confidence: float = 0.5
    rules: list[Rule] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path = "config/classifier.yaml") -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# A fallback takes the unresolved fields and returns a cause, or None if it cannot
# do better than the rules did. Wired to an LLM later; absent, classification
# simply reports what it could not resolve.
Fallback = Callable[[dict[str, str | None]], Classification | None]


class Classifier:
    def __init__(
        self, rules: ClassifierRules | None = None, fallback: Fallback | None = None
    ) -> None:
        self._rules = rules or ClassifierRules.load()
        self._fallback = fallback

        # Most specific first, declaration order breaking ties. Sorted once here
        # rather than per call — this runs against every failure in a batch.
        self._ordered = sorted(
            self._rules.rules, key=lambda rule: -rule.specificity
        )

    @staticmethod
    def fields_of(payment: PaymentEntity) -> dict[str, str | None]:
        return {
            "method": str(payment.method),
            "source": payment.error_source,
            "step": payment.error_step,
            "reason": payment.error_reason,
        }

    def classify(self, payment: PaymentEntity) -> Classification:
        fields = self.fields_of(payment)

        for rule in self._ordered:
            if not rule.matches(fields):
                continue
            if rule.confidence < self._rules.min_confidence:
                break
            return Classification(
                cause=rule.cause,
                confidence=rule.confidence,
                resolution=Resolution.DETERMINISTIC,
                rule_id=rule.id,
                note=rule.note,
            )

        if self._fallback is not None:
            resolved = self._fallback(fields)
            if resolved is not None:
                return resolved

        return Classification(
            cause=None,
            confidence=0.0,
            resolution=Resolution.UNRESOLVED,
            note=f"no rule for {fields['method']}/{fields['source']}/{fields['reason']}",
        )

    def unresolved_fields(self, payments: list[PaymentEntity]) -> list[dict[str, str | None]]:
        """The distinct symptom combinations the rules could not handle.

        Distinct, not per-payment: this is what gets batched into a single model
        call rather than one call per failure. On a 5,000-payment run that is the
        difference between roughly 200 requests and one, which is the difference
        between fitting in the free tier and not.
        """
        seen: dict[tuple, dict[str, str | None]] = {}
        for payment in payments:
            if self.classify(payment).actionable:
                continue
            fields = self.fields_of(payment)
            seen.setdefault(tuple(fields.get(k) for k in MATCHABLE), fields)
        return list(seen.values())
