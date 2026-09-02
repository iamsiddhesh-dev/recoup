"""What the agent knows when it decides.

Two things live here: the per-payment facts a decision needs, and the recovery
model the agent learns during a run.

## The learned model

The policy needs P(recover | cause, issuer, hour). Estimating that directly is
hopeless — eight causes by seven issuers by twenty-four hours is 1,344 cells, and
a run produces on the order of a thousand attempts. Most cells would be empty and
the rest would hold two observations, one of which succeeded, giving a confident
50%.

So estimates are built hierarchically and shrunk toward their parent:

    prior  →  (cause)  →  (cause, hour)  →  (cause, issuer, hour)

Each level blends its own observations with the level above, weighted by sample
size: `(n·observed + k·parent) / (n + k)`. With no data a cell reports its
parent's estimate; with plenty it reports its own; in between it moves gradually.
`k` is `learning.min_observations` in policy.yaml.

This is deliberately statistics rather than a model. Retry timing is exactly the
kind of problem where an LLM would be reached for and would be worse: the answer
is a number derived from counts, it needs to be defensible to a merchant, and it
has to be recomputed thousands of times per run.

The agent starts from a prior that is deliberately *not* the world's truth — see
`test_agent_prior_does_not_secretly_match_the_world`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from recoup.adapters.base import supports_silent_retry
from recoup.agent.classify import Classification
from recoup.agent.config import PolicyConfig
from recoup.domain import DowntimeEntity, FailureCause, PaymentEntity


@dataclass(frozen=True)
class Estimate:
    """A probability plus the evidence behind it.

    Carried together because the control room shows both: a 40% built from four
    hundred observations and a 40% built from two are different claims, and a
    merchant asking why their customer was charged at 10am deserves to know
    which one they are looking at.
    """

    probability: float
    observations: int
    level: str

    def __float__(self) -> float:
        return self.probability


class RecoveryModel:
    """Hierarchical success rates, learned from observed outcomes only."""

    def __init__(self, policy: PolicyConfig) -> None:
        self._policy = policy
        self._k = policy.learning.min_observations
        self._counts: dict[tuple, list[int]] = {}

    def record(
        self, cause: FailureCause, issuer: str | None, hour: int, succeeded: bool
    ) -> None:
        for key in self._keys(cause, issuer, hour):
            bucket = self._counts.setdefault(key, [0, 0])
            bucket[0] += 1
            bucket[1] += int(succeeded)

    @staticmethod
    def _keys(cause: FailureCause, issuer: str | None, hour: int) -> list[tuple]:
        keys: list[tuple] = [(str(cause),), (str(cause), hour)]
        if issuer:
            keys.append((str(cause), issuer, hour))
        return keys

    def _shrink(self, key: tuple, parent: float) -> tuple[float, int]:
        attempts, successes = self._counts.get(key, [0, 0])
        if attempts == 0:
            return parent, 0
        blended = (attempts * (successes / attempts) + self._k * parent) / (attempts + self._k)
        return blended, attempts

    def estimate(
        self, cause: FailureCause | None, issuer: str | None, hour: int
    ) -> Estimate:
        if cause is None:
            # An unclassified failure gets the most pessimistic thing that is
            # still honest: the lowest prior on the board. Acting confidently on
            # a cause you could not identify is how an agent spends money on
            # nothing.
            return Estimate(
                probability=min(self._policy.prior_recovery_probability.values()),
                observations=0,
                level="unclassified",
            )

        probability = self._policy.prior_recovery_probability[cause]
        level = "prior"

        for key, name in (
            ((str(cause),), "cause"),
            ((str(cause), hour), "cause+hour"),
            ((str(cause), issuer, hour), "cause+issuer+hour"),
        ):
            if name == "cause+issuer+hour" and not issuer:
                continue
            probability, observations = self._shrink(key, probability)
            if observations:
                level = name

        return Estimate(
            probability=max(0.0, min(1.0, probability)),
            observations=self._counts.get((str(cause), issuer, hour), [0])[0],
            level=level,
        )

    def best_hour(
        self, cause: FailureCause | None, issuer: str | None, candidates: list[int]
    ) -> tuple[int, Estimate]:
        """The hour with the highest estimated success rate among candidates."""
        scored = [(h, self.estimate(cause, issuer, h)) for h in candidates]
        return max(scored, key=lambda pair: pair[1].probability)


@dataclass
class DecisionContext:
    """Everything the policy is allowed to look at for one failure."""

    payment: PaymentEntity
    classification: Classification
    now: datetime

    attempts: int = 0
    contacts_in_window: int = 0
    last_contact_at: datetime | None = None
    consecutive_failures: int = 0

    issuer_code: str | None = None
    downtime: DowntimeEntity | None = None
    can_retry_silently: bool = False

    estimate: Estimate | None = None
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def cause(self) -> FailureCause | None:
        return self.classification.cause

    @property
    def amount(self) -> int:
        return self.payment.amount

    @property
    def customer_ref(self) -> str | None:
        return self.payment.customer_ref

    @property
    def degraded(self) -> bool:
        return self.downtime is not None

    def hours_since_last_contact(self) -> float | None:
        if self.last_contact_at is None:
            return None
        return (self.now - self.last_contact_at).total_seconds() / 3600


class ContextBuilder:
    """Assembles decision context from what the agent has observed.

    State is held here rather than recomputed from the ledger on every decision:
    a run makes tens of thousands of decisions, and re-folding the event stream
    for each one would dominate the runtime. The ledger remains the record; this
    is the working set.
    """

    def __init__(self, policy: PolicyConfig, model: RecoveryModel | None = None) -> None:
        self._policy = policy
        self.model = model or RecoveryModel(policy)

        self._attempts: dict[str, int] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._contacts: dict[str, list[datetime]] = {}
        self._downtimes: dict[str, DowntimeEntity] = {}

    # -- observations --------------------------------------------------------

    def note_attempt(self, payment_id: str, succeeded: bool) -> None:
        self._attempts[payment_id] = self._attempts.get(payment_id, 0) + 1
        if succeeded:
            self._consecutive_failures[payment_id] = 0
        else:
            self._consecutive_failures[payment_id] = (
                self._consecutive_failures.get(payment_id, 0) + 1
            )

    def note_contact(self, customer_ref: str, at: datetime) -> None:
        self._contacts.setdefault(customer_ref, []).append(at)

    def note_downtime(self, event: str, downtime: DowntimeEntity) -> None:
        """Track the degradation feed.

        Resolution matters as much as onset: an agent that defers on downtime and
        never learns it ended sits on recoverable money until the horizon runs
        out.
        """
        key = _downtime_key(downtime)
        if event.endswith("started"):
            self._downtimes[key] = downtime
        else:
            self._downtimes.pop(key, None)

    def active_downtime(
        self, method: str, issuer_code: str | None
    ) -> DowntimeEntity | None:
        for key in (f"{method}:{issuer_code}", f"{method}:"):
            if key in self._downtimes:
                return self._downtimes[key]
        return None

    def contacts_in_window(self, customer_ref: str | None, now: datetime) -> int:
        if not customer_ref:
            return 0
        cutoff = now - timedelta(days=self._policy.annoyance.window_days)
        return sum(1 for at in self._contacts.get(customer_ref, []) if at >= cutoff)

    def last_contact(self, customer_ref: str | None) -> datetime | None:
        history = self._contacts.get(customer_ref or "", [])
        return max(history) if history else None

    # -- assembly ------------------------------------------------------------

    def build(
        self, payment: PaymentEntity, classification: Classification, now: datetime
    ) -> DecisionContext:
        issuer_code = payment.bank
        instrument = {
            k: v
            for k, v in (("token_id", payment.token_id), ("card_id", payment.card_id))
            if v
        }

        return DecisionContext(
            payment=payment,
            classification=classification,
            now=now,
            attempts=self._attempts.get(payment.id, 0),
            consecutive_failures=self._consecutive_failures.get(payment.id, 0),
            contacts_in_window=self.contacts_in_window(payment.customer_ref, now),
            last_contact_at=self.last_contact(payment.customer_ref),
            issuer_code=issuer_code,
            downtime=self.active_downtime(str(payment.method), issuer_code),
            can_retry_silently=supports_silent_retry(payment.method, instrument),
            estimate=self.model.estimate(classification.cause, issuer_code, now.hour),
        )


def _downtime_key(downtime: DowntimeEntity) -> str:
    return f"{downtime.method}:{downtime.instrument.get('bank', '')}"
