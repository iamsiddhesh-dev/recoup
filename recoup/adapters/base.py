"""The seam.

One protocol, two implementations. The agent holds a `PaymentsAdapter` and cannot
tell whether it is driving a simulated world or live Razorpay test mode — which is
the answer to the obvious objection that a simulated evaluation grades its own
homework. If the same agent binary runs unmodified against the real API, the
simulator is a load generator rather than a comfortable fiction.

Two things shape this interface:

**Razorpay has no "retry" call.** You cannot re-run a failed payment. You create a
*new* attempt against the same order, or a payment link the customer completes, or
for a mandate you trigger a fresh charge. Modelling that honestly as
`attempt_charge` and `create_recovery_link` costs nothing here and would cost a
rewrite later.

**Nudges are not Razorpay's problem.** Messaging lives behind a separate
`Notifier` protocol. Conflating "move money" with "talk to a customer" would put
the compliance gate's consent rules in the wrong place.

Everything is synchronous. The simulated path is pure computation and the live
path makes a handful of calls; async would buy nothing and complicate the
evaluation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from recoup.domain import Channel, DowntimeEntity, Language, PaymentEntity, PaymentMethod


class TestModeViolation(RuntimeError):
    """Raised when an adapter is pointed at anything other than test mode.

    Recoup executes money actions. The guard that makes a live action structurally
    impossible is worth more than any amount of care about not misconfiguring it.
    """


@dataclass(frozen=True)
class ChargeRequest:
    """A fresh attempt against an existing order.

    `idempotency_key` is mandatory rather than optional. A recovery agent that
    replays its queue after a crash and charges twice has not had an outage, it
    has had an incident.
    """

    payment_id: str
    order_id: str
    amount: int
    method: PaymentMethod
    idempotency_key: str
    instrument: dict[str, str] = field(default_factory=dict)
    at: datetime | None = None


@dataclass(frozen=True)
class ChargeResult:
    succeeded: bool
    payment: PaymentEntity | None = None
    error_source: str | None = None
    error_step: str | None = None
    error_reason: str | None = None
    error_description: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LinkRequest:
    """A link the customer completes themselves.

    The path for failures no retry can fix — an expired card, an invalid VPA, a
    wallet that is not linked. The instrument has to change, and only the customer
    can change it.
    """

    payment_id: str
    order_id: str
    amount: int
    customer_ref: str
    description: str
    idempotency_key: str
    at: datetime | None = None


@dataclass(frozen=True)
class RecoveryLink:
    id: str
    url: str
    expires_at: datetime | None = None


@dataclass(frozen=True)
class NudgeRequest:
    customer_ref: str
    channel: Channel
    language: Language
    body: str
    payment_id: str
    idempotency_key: str
    link: RecoveryLink | None = None
    at: datetime | None = None


@dataclass(frozen=True)
class NudgeResult:
    delivered: bool
    acted_on: bool = False
    detail: str = ""


def supports_silent_retry(method: PaymentMethod, instrument: dict[str, str]) -> bool:
    """Whether this instrument can be charged again without the customer present.

    Worth stating plainly because it constrains the whole policy space, and it is
    the detail a recovery agent built against a toy API would get wrong.

    Razorpay cannot re-run a failed payment. A charge with nobody watching is only
    possible where standing authorisation exists — an e-mandate, or a saved card
    or UPI Autopay token. Everything else (one-time cards, netbanking, wallets, a
    plain UPI collect) legally and technically requires the customer to act again,
    which makes it a link-and-nudge problem rather than a retry problem.

    An agent that assumes it can silently retry a netbanking failure is proposing
    an action that does not exist.
    """
    if method is PaymentMethod.EMANDATE:
        return True
    return bool(instrument.get("token_id"))


@runtime_checkable
class PaymentsAdapter(Protocol):
    """Everything the agent can do that touches money."""

    name: str
    live_capable: bool

    def fetch_payment(self, payment_id: str) -> PaymentEntity | None: ...

    def attempt_charge(self, request: ChargeRequest) -> ChargeResult: ...

    def create_recovery_link(self, request: LinkRequest) -> RecoveryLink: ...

    def active_downtimes(self, at: datetime) -> list[DowntimeEntity]: ...


@runtime_checkable
class Notifier(Protocol):
    """Everything the agent can do that touches a customer."""

    name: str

    def send(self, request: NudgeRequest) -> NudgeResult: ...
