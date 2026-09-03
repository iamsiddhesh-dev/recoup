"""Turning a decision into an action, and an action into a ledger entry.

The executor is the only place the agent touches money or customers. It is
deliberately thin: it does not decide anything, it does not second-guess the
gate, and it holds no policy of its own. Everything it does is either a call
through the adapter protocol or a line in the ledger.

Two responsibilities worth naming.

**Idempotency keys.** Every money action carries one, derived from
`(payment_id, attempt, action)`. A recovery agent that replays its queue after a
crash and charges twice has not had an outage, it has had an incident. The key is
built here rather than accepted from a caller so it cannot be forgotten.

**Recovery links.** Most failures cannot be retried silently — no standing
authorisation exists — so the customer has to act. A message about a failed
payment that gives them no way to pay is a notification, not a recovery, and the
executor attaches a link whenever the instrument cannot be charged again on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from recoup.adapters.base import (
    ChargeRequest,
    LinkRequest,
    Notifier,
    NudgeRequest,
    PaymentsAdapter,
    RecoveryLink,
)
from recoup.agent.actions import ActionKind, Decision
from recoup.agent.context import ContextBuilder, DecisionContext
from recoup.agent.llm.copywriter import Copywriter
from recoup.ledger.events import EventKind, LedgerEvent


@dataclass(frozen=True)
class Execution:
    """What happened when the decision was carried out."""

    action: ActionKind
    at: datetime
    succeeded: bool
    recovered_paise: int = 0
    cost_paise: int = 0
    contacted: bool = False
    detail: str = ""


class Executor:
    def __init__(
        self,
        arm: str,
        adapter: PaymentsAdapter,
        notifier: Notifier,
        context: ContextBuilder,
        cost_of,
        copywriter: Copywriter | None = None,
        merchant: str = "the merchant",
    ) -> None:
        self._arm = arm
        self._adapter = adapter
        self._notifier = notifier
        self._context = context
        self._cost_of = cost_of
        # Defaults to a copywriter with no model, which serves the hand-written
        # fallback templates. Messaging must never depend on a model being
        # reachable.
        self._copywriter = copywriter or Copywriter(use_llm=False)
        self._merchant = merchant

    @staticmethod
    def idempotency_key(payment_id: str, attempt: int, action: ActionKind) -> str:
        return f"{payment_id}:{attempt}:{action}"

    def consented_channels(self, customer_ref: str):
        """Passed through from the messaging layer, which owns consent."""
        return self._notifier.consented_channels(customer_ref)

    def _link_for(self, context: DecisionContext, attempt: int) -> RecoveryLink | None:
        """A way to pay, for failures the agent cannot charge on its own."""
        if context.can_retry_silently:
            return None

        return self._adapter.create_recovery_link(
            LinkRequest(
                payment_id=context.payment.id,
                order_id=context.payment.order_id,
                amount=context.amount,
                customer_ref=context.customer_ref or "",
                description="Complete your payment",
                idempotency_key=self.idempotency_key(
                    context.payment.id, attempt, ActionKind.STOP
                )
                + ":link",
                at=context.now,
            )
        )

    def execute(
        self, decision: Decision, context: DecisionContext
    ) -> tuple[Execution, list[LedgerEvent]]:
        action = decision.action
        at = decision.chosen.at if decision.chosen else context.now
        attempt = context.attempts + 1
        events: list[LedgerEvent] = []

        def record(kind: EventKind, **data) -> None:
            events.append(
                LedgerEvent(
                    at=at,
                    kind=kind,
                    arm=self._arm,
                    payment_id=context.payment.id,
                    customer_id=context.customer_ref,
                    amount=data.pop("amount", None),
                    data=data,
                )
            )

        if action is ActionKind.STOP:
            record(EventKind.STOPPED, reason=decision.reason)
            return Execution(action=action, at=at, succeeded=False), events

        if action is ActionKind.ESCALATE_HUMAN:
            # Humans are outside the system. The agent hands the case over and
            # stops touching it; whether it is resolved is not something this
            # run may claim credit for.
            cost = self._cost_of(str(action))
            record(
                EventKind.EXECUTED,
                action=str(action),
                cost=cost,
                detail="handed to human review",
            )
            record(EventKind.STOPPED, reason="escalated to human review")
            return (
                Execution(action=action, at=at, succeeded=False, cost_paise=cost),
                events,
            )

        if action.is_retry:
            return self._charge(decision, context, at, attempt, record, events)

        return self._contact(decision, context, at, attempt, record, events)

    # -- money ---------------------------------------------------------------

    def _charge(self, decision, context, at, attempt, record, events):
        instrument = {
            k: v
            for k, v in (
                ("token_id", context.payment.token_id),
                ("card_id", context.payment.card_id),
            )
            if v
        }

        result = self._adapter.attempt_charge(
            ChargeRequest(
                payment_id=context.payment.id,
                order_id=context.payment.order_id,
                amount=context.amount,
                method=context.payment.method,
                idempotency_key=self.idempotency_key(
                    context.payment.id, attempt, decision.action
                ),
                instrument=instrument,
                at=at,
            )
        )

        cost = self._cost_of(str(decision.action), context.attempts)
        self._context.note_attempt(context.payment.id, result.succeeded)

        if context.cause is not None:
            self._context.model.record(
                context.cause, context.issuer_code, at, result.succeeded
            )

        record(
            EventKind.EXECUTED,
            action=str(decision.action),
            cost=cost,
            succeeded=result.succeeded,
            error_reason=result.error_reason,
        )

        if result.succeeded:
            record(EventKind.RECOVERED, amount=context.amount, via=str(decision.action))

        return (
            Execution(
                action=decision.action,
                at=at,
                succeeded=result.succeeded,
                recovered_paise=context.amount if result.succeeded else 0,
                cost_paise=cost,
                detail=result.error_reason or "",
            ),
            events,
        )

    # -- customers -----------------------------------------------------------

    def _contact(self, decision, context, at, attempt, record, events):
        link = self._link_for(context, attempt)
        customer_ref = context.customer_ref or ""

        language = self._notifier.preferred_language(customer_ref)
        body, copy_source = self._copywriter.render(
            context.cause,
            language,
            decision.action.channel,
            amount_paise=context.amount,
            link=link.url if link else "",
            merchant=self._merchant,
        )

        result = self._notifier.send(
            NudgeRequest(
                customer_ref=customer_ref,
                channel=decision.action.channel,
                language=language,
                body=body,
                payment_id=context.payment.id,
                idempotency_key=self.idempotency_key(
                    context.payment.id, attempt, decision.action
                ),
                link=link,
                at=at,
            )
        )

        cost = self._cost_of(str(decision.action))

        if result.delivered and customer_ref:
            self._context.note_contact(customer_ref, at)

        record(
            EventKind.EXECUTED,
            action=str(decision.action),
            cost=cost,
            delivered=result.delivered,
            acted_on=result.acted_on,
            with_link=link is not None,
            detail=result.detail,
            # The message that was actually sent, so the case screen can show a
            # reader what the customer received rather than describing it.
            body=body,
            language=str(language),
            copy_source=copy_source,
        )

        if result.acted_on:
            record(EventKind.RECOVERED, amount=context.amount, via=str(decision.action))

        return (
            Execution(
                action=decision.action,
                at=at,
                succeeded=result.acted_on,
                recovered_paise=context.amount if result.acted_on else 0,
                cost_paise=cost if result.delivered else 0,
                contacted=result.delivered,
                detail=result.detail,
            ),
            events,
        )
