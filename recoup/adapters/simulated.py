"""The simulated adapter — the world behind the seam.

Implements `PaymentsAdapter` and `Notifier` against the generated world. This is
the only adapter that imports `recoup.world`, and that is deliberate: it is the
translation layer, so the agent never has to be.

Everything it returns is shaped exactly like the live adapter's output, including
the error fields on a failed retry. If the agent could tell the difference, the
portability claim would be marketing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from recoup.adapters.base import (
    ChargeRequest,
    ChargeResult,
    LinkRequest,
    NudgeRequest,
    NudgeResult,
    RecoveryLink,
    supports_silent_retry,
)
from recoup.domain import Channel, DowntimeEntity, PaymentEntity, PaymentStatus
from recoup.world.config import WorldConfig
from recoup.world.customers import Population
from recoup.world.generator import Batch
from recoup.world.issuers import IssuerBook
from recoup.world.outcomes import Outcomes


class SimulatedAdapter:
    """Drives the generated world."""

    name = "simulated"
    live_capable = False

    def __init__(
        self,
        config: WorldConfig,
        batch: Batch,
        issuers: IssuerBook,
        population: Population,
    ) -> None:
        self._config = config
        self._batch = batch
        self._issuers = issuers
        self._population = population
        self._outcomes = Outcomes(config, issuers, population)

        self._payments: dict[str, PaymentEntity] = {p.id: p for p in batch.payments}
        self._attempts: dict[str, int] = {}
        self._links: dict[str, RecoveryLink] = {}

        # Idempotency is enforced, not assumed. Replaying the ledger must not
        # produce a second charge, and the only way to be sure is to make the
        # adapter itself refuse.
        self._seen_keys: dict[str, ChargeResult] = {}

    # -- reads ---------------------------------------------------------------

    def fetch_payment(self, payment_id: str) -> PaymentEntity | None:
        return self._payments.get(payment_id)

    def active_downtimes(self, at: datetime) -> list[DowntimeEntity]:
        windows = self._issuers.active_at(at)
        return [
            DowntimeEntity(
                id=window.id,
                method=window.method,
                begin=int(window.begin.timestamp()),
                end=None,
                status="started",
                severity=window.severity,
                instrument={"bank": window.issuer_code} if window.issuer_code else {},
            )
            for window in windows
        ]

    def attempts_for(self, payment_id: str) -> int:
        return self._attempts.get(payment_id, 0)

    # -- money ---------------------------------------------------------------

    def attempt_charge(self, request: ChargeRequest) -> ChargeResult:
        if request.idempotency_key in self._seen_keys:
            return self._seen_keys[request.idempotency_key]

        original = self._payments[request.payment_id]

        # The same refusal the live adapter gives. Without this the agent would
        # learn a retry policy that works in simulation and is impossible in
        # production — exactly the divergence the shared protocol exists to
        # prevent, and the one a simulated evaluation is most likely to hide.
        instrument = {
            k: v
            for k, v in (("token_id", original.token_id), ("card_id", original.card_id))
            if v
        }
        if not supports_silent_retry(request.method, instrument):
            result = ChargeResult(
                succeeded=False,
                payment=original,
                error_source="business",
                error_step="payment_initiation",
                error_reason="customer_action_required",
                error_description=(
                    f"{request.method} has no standing authorisation; this payment "
                    "cannot be charged without the customer. Use a recovery link."
                ),
            )
            self._seen_keys[request.idempotency_key] = result
            return result

        truth = self._batch.truth_for(request.payment_id)
        when = request.at or self._config.run.start_at

        attempt_number = self._attempts.get(request.payment_id, 1) + 1
        self._attempts[request.payment_id] = attempt_number

        outcome = self._outcomes.retry_succeeds(truth, attempt_number, when)

        if outcome.succeeded:
            recovered = original.model_copy(
                update={
                    "id": f"{request.payment_id}_r{attempt_number}",
                    "status": PaymentStatus.CAPTURED,
                    "captured": True,
                    "error_code": None,
                    "error_source": None,
                    "error_step": None,
                    "error_reason": None,
                    "error_description": None,
                    "created_at": int(when.timestamp()),
                }
            )
            result = ChargeResult(succeeded=True, payment=recovered)
        else:
            # A failed retry reports the same symptoms the original did. Real
            # issuers do not explain themselves differently on the second ask.
            result = ChargeResult(
                succeeded=False,
                payment=original,
                error_source=original.error_source,
                error_step=original.error_step,
                error_reason=original.error_reason,
                error_description=original.error_description,
                raw={"probability": outcome.probability, "factors": outcome.factors},
            )

        self._seen_keys[request.idempotency_key] = result
        return result

    def create_recovery_link(self, request: LinkRequest) -> RecoveryLink:
        if request.idempotency_key in self._links:
            return self._links[request.idempotency_key]

        when = request.at or self._config.run.start_at
        link = RecoveryLink(
            id=f"plink_{request.payment_id}",
            url=f"https://rzp.invalid/l/{request.payment_id}",
            expires_at=when + timedelta(days=3),
        )
        self._links[request.idempotency_key] = link
        return link


class SimulatedNotifier:
    """Customer messaging against the simulated population.

    Delivery and action are separate outcomes on purpose. A message that reaches a
    phone and changes nothing still cost money and still spent goodwill, and an
    agent that conflates the two will over-nudge.
    """

    name = "simulated"

    def __init__(
        self,
        config: WorldConfig,
        population: Population,
        issuers: IssuerBook,
        batch: Batch | None = None,
    ) -> None:
        self._population = population
        self._outcomes = Outcomes(config, issuers, population)
        # The world knows why each payment failed. The agent does not, and the
        # request carries no cause — it is looked up here so the true cause can
        # shape the outcome without ever crossing the wire.
        self._batch = batch
        self._contacts: dict[str, int] = {}
        self._seen_keys: dict[str, NudgeResult] = {}

    def contacts_for(self, customer_ref: str) -> int:
        return self._contacts.get(customer_ref, 0)

    def consented_channels(self, customer_ref: str) -> set[Channel]:
        customer = self._population.get(customer_ref)
        if customer is None:
            return set()
        return {channel for channel in Channel if customer.may_contact(channel)}

    def send(self, request: NudgeRequest) -> NudgeResult:
        if request.idempotency_key in self._seen_keys:
            return self._seen_keys[request.idempotency_key]

        customer = self._population.get(request.customer_ref)
        if customer is None:
            result = NudgeResult(delivered=False, detail="unknown customer")
            self._seen_keys[request.idempotency_key] = result
            return result

        if not customer.may_contact(request.channel):
            # The compliance gate should have caught this upstream. Failing here
            # too means a bug in the gate shows up as a refusal rather than as an
            # unconsented message.
            result = NudgeResult(
                delivered=False, detail=f"no consent for {request.channel}"
            )
            self._seen_keys[request.idempotency_key] = result
            return result

        contact_number = self._contacts.get(request.customer_ref, 0) + 1
        self._contacts[request.customer_ref] = contact_number

        # A message carrying a payment link asks strictly more of the customer
        # than one that only informs: they have to open it and supply an
        # instrument. `link_completed` is the harder bar, and using it here is
        # what stops the run counting an opened notification as recovered money.
        channel = str(request.channel)
        cause = None
        if self._batch is not None and request.payment_id in self._batch.truths:
            cause = self._batch.truth_for(request.payment_id).cause

        if request.link is not None:
            acted = self._outcomes.link_completed(
                customer, request.payment_id, contact_number, channel, cause
            )
        else:
            acted = self._outcomes.nudge_lands(
                customer, request.payment_id, contact_number, channel, cause
            )

        result = NudgeResult(
            delivered=True,
            acted_on=acted,
            detail=f"contact #{contact_number} via {request.channel}",
        )
        self._seen_keys[request.idempotency_key] = result
        return result
