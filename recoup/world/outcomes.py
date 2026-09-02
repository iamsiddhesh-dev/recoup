"""Whether a recovery attempt actually works.

This is the hidden model. The agent never sees any of it; it only observes what
happened and has to infer the shape from outcomes.

## Common random numbers

The one design decision here that is worth explaining. Every draw is derived
deterministically from `(seed, payment_id, attempt_number)` rather than pulled
from a running stream. That means **the same payment retried for the same time
gets the same die roll in every arm of the experiment**.

Why it matters: the headline claim is incremental recovery over a naive baseline.
If each arm drew independent randomness, the difference between arms would mix the
effect of the policy with the luck of the draw, and at these sample sizes the luck
is large enough to swamp a real 2-3 point improvement. Fixing the draw per attempt
means arms differ *only* by their decisions — if the agent retries at 10am and the
baseline retries at 2am, the same uniform is compared against different success
probabilities, and the gap is attributable to the choice.

This is variance reduction, not cheating: it makes an honest comparison sharper.
It also makes the whole run replayable, which the audit ledger depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from recoup.domain import FailureCause, PaymentMethod
from recoup.rng import substream
from recoup.world.config import WorldConfig
from recoup.world.customers import Customer, Population
from recoup.world.generator import PaymentTruth
from recoup.world.issuers import IssuerBook


@dataclass(frozen=True)
class AttemptOutcome:
    succeeded: bool
    probability: float
    factors: dict[str, float]


class Outcomes:
    """The world's answer to "did that work?"."""

    def __init__(
        self, config: WorldConfig, issuers: IssuerBook, population: Population
    ) -> None:
        self._config = config
        self._issuers = issuers
        self._population = population
        self._seed = config.run.seed

    # -- deterministic draws -------------------------------------------------

    def _draw(self, kind: str, key: str, index: int) -> float:
        """A uniform fixed by identity, not by call order.

        Two arms asking about the same attempt get the same number.
        """
        return substream(self._seed, f"{kind}:{key}:{index}").random()

    # -- retries -------------------------------------------------------------

    def retry_probability(
        self, truth: PaymentTruth, attempt_number: int, when: datetime
    ) -> tuple[float, dict[str, float]]:
        """Probability this retry succeeds, and the factors that produced it.

        Returned as a breakdown rather than a scalar because the same decomposition
        drives the world's own diagnostics, and because a single opaque number is
        impossible to debug when a sweep moves it.
        """
        cfg = self._config.recovery
        cause = truth.cause

        if cause is None:
            return 0.0, {}

        base = cfg.base_probability[cause]
        decay = cfg.attempt_decay ** max(0, attempt_number - 1)
        hour = cfg.hour_multiplier[when.hour]
        reliability = self._issuers.reliability(truth.issuer_code)
        downtime = self._issuers.multiplier_at(when, truth.method, truth.issuer_code)

        # Salary credits cluster at month start, so a balance that was short last
        # week may not be short now. This is the single most exploitable pattern in
        # the world and the agent has to find it from outcomes alone.
        payday = 1.0
        if cause is FailureCause.INSUFFICIENT_FUNDS and when.day in cfg.payday.days_of_month:
            payday = cfg.payday.multiplier

        factors = {
            "base": base,
            "attempt_decay": decay,
            "hour": hour,
            "issuer_reliability": reliability,
            "downtime": downtime,
            "payday": payday,
        }

        probability = base * decay * hour * reliability * downtime * payday
        return max(0.0, min(1.0, probability)), factors

    def retry_succeeds(
        self, truth: PaymentTruth, attempt_number: int, when: datetime
    ) -> AttemptOutcome:
        probability, factors = self.retry_probability(truth, attempt_number, when)
        draw = self._draw("retry", truth.payment_id, attempt_number)
        return AttemptOutcome(
            succeeded=draw < probability, probability=probability, factors=factors
        )

    # -- customer actions ----------------------------------------------------

    def nudge_lands(
        self,
        customer: Customer,
        payment_id: str,
        contact_number: int,
        channel: str = "sms",
        cause: FailureCause | None = None,
    ) -> bool:
        """Whether a contact produces action.

        Three things move it.

        **Channel**, because a WhatsApp message and a transactional email are not
        the same ask — and if they were, an agent optimising cost would rationally
        email forever.

        **Cause**, because what a message can achieve depends on what broke. There
        is nothing a customer can do about their bank being down, and everything
        they can do about an OTP they never entered. This is the reason knowing
        the cause is worth paying for.

        **Prior contacts**, because the fourth message about one failed payment
        persuades nobody, and for an annoyance-sensitive customer it is worse than
        nothing. The agent pays a modelled penalty for repeat contact; here that
        penalty turns out to have been real.
        """
        decay = 1.0 / (1.0 + customer.annoyance_sensitivity * max(0, contact_number - 1))
        channel_factor = self._config.customers.channel_response.get(channel, 1.0)
        cause_factor = (
            self._config.customers.cause_response.get(cause, 1.0) if cause else 1.0
        )
        draw = self._draw("nudge", payment_id, contact_number)
        return draw < customer.nudge_response * decay * channel_factor * cause_factor

    def link_completed(
        self,
        customer: Customer,
        payment_id: str,
        contact_number: int,
        channel: str = "sms",
        cause: FailureCause | None = None,
    ) -> bool:
        """Whether a customer who opened a recovery link finishes paying.

        Strictly harder than responding to a nudge: they have to re-enter an
        instrument. Modelled as responsiveness with a fixed completion haircut.
        """
        if not self.nudge_lands(customer, payment_id, contact_number, channel, cause):
            return False
        draw = self._draw("link", payment_id, contact_number)
        return draw < 0.68

    # -- instruments ---------------------------------------------------------

    def instrument_can_ever_succeed(self, truth: PaymentTruth) -> bool:
        """Some failures are terminal for the instrument, not the payment.

        An expired card cannot be argued with. The only path is a new instrument,
        which is why `INSTRUMENT_INVALID` is a hard stop on retry but not on
        contact.
        """
        return truth.cause not in (
            FailureCause.INSTRUMENT_INVALID,
            FailureCause.RISK_BLOCKED,
        )

    def method_of(self, truth: PaymentTruth) -> PaymentMethod:
        return truth.method
