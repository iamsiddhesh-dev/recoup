"""Shared vocabulary between the world and the agent.

Everything here is deliberately *shape*, not truth. The world fills these objects
in and the agent reads them, but nothing in this module tells the agent whether a
payment was ever going to succeed.

The one thing to notice: `PaymentEntity` carries Razorpay's error fields — source,
step, reason — and **no cause**. `FailureCause` exists in this module because it is
shared terminology, but it travels only inside the world's own bookkeeping and
inside the agent's inference. It is never serialised onto an event. That absence is
what makes root-cause classification a real problem rather than a lookup.

Event shapes mirror Razorpay's webhook payloads closely enough that the same agent
code parses both a simulated event and a live test-mode one:
    https://razorpay.com/docs/webhooks/payloads/payments/
    https://razorpay.com/docs/api/payments/downtime/
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class PaymentMethod(StrEnum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMANDATE = "emandate"


class FailureCause(StrEnum):
    """Recoup's own recovery-relevant taxonomy — not Razorpay's.

    Razorpay tells you *where* a payment broke (source, step, reason). This says
    what to do about it, which is a different question: two failures with the same
    reason string can want opposite treatment depending on the instrument.
    """

    SOFT_ISSUER_DECLINE = "SOFT_ISSUER_DECLINE"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    AUTH_ABANDONED = "AUTH_ABANDONED"
    TECHNICAL_GATEWAY = "TECHNICAL_GATEWAY"
    INSTRUMENT_INVALID = "INSTRUMENT_INVALID"
    RISK_BLOCKED = "RISK_BLOCKED"
    MANDATE_PROBLEM = "MANDATE_PROBLEM"
    CUSTOMER_INTENT = "CUSTOMER_INTENT"


class PaymentStatus(StrEnum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Language(StrEnum):
    HINGLISH = "hinglish"
    ENGLISH = "english"
    REGIONAL = "regional"


class Channel(StrEnum):
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    VOICE = "voice"


# ---------------------------------------------------------------------------
# Entities — Razorpay-shaped
# ---------------------------------------------------------------------------


class PaymentEntity(BaseModel):
    """A payment as Razorpay reports it.

    Deliberately carries the error fields flat (error_source, error_step,
    error_reason) exactly as the payment entity does, rather than nesting them,
    because that is what the agent will parse off a live webhook.
    """

    id: str
    entity: Literal["payment"] = "payment"
    amount: int  # paise
    currency: str = "INR"
    status: PaymentStatus
    order_id: str
    method: PaymentMethod
    captured: bool = False
    international: bool = False

    # Instrument, populated per method — mirrors Razorpay's sparse entity.
    bank: str | None = None
    vpa: str | None = None
    card_id: str | None = None
    wallet: str | None = None
    token_id: str | None = None

    contact: str | None = None
    email: str | None = None
    description: str | None = None

    error_code: str | None = None
    error_description: str | None = None
    error_source: str | None = None
    error_step: str | None = None
    error_reason: str | None = None

    acquirer_data: dict[str, Any] = Field(default_factory=dict)
    notes: dict[str, str] = Field(default_factory=dict)
    created_at: int = 0

    @property
    def customer_ref(self) -> str | None:
        """Merchants routinely stash their own customer id in notes; so do we."""
        return self.notes.get("customer_id")


class DowntimeEntity(BaseModel):
    """Razorpay's payment downtime entity.

    This is the agent's degradation signal and the reason it can refuse to retry
    into a known outage instead of burning attempts against the cap.
    """

    id: str
    entity: Literal["payment.downtime"] = "payment.downtime"
    method: PaymentMethod
    begin: int
    end: int | None = None
    status: Literal["started", "resolved"]
    scheduled: bool = False
    severity: Severity
    instrument: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Webhook envelope
# ---------------------------------------------------------------------------


class WebhookEvent(BaseModel):
    """The envelope Razorpay POSTs to a webhook URL.

    The agent's ingest path accepts exactly this, whether it came from the
    simulator or from a live test-mode webhook. That is the whole point of the
    adapter seam: swapping the source changes nothing downstream.
    """

    entity: Literal["event"] = "event"
    account_id: str
    event: str
    contains: list[str]
    payload: dict[str, dict[str, Any]]
    created_at: int

    @classmethod
    def for_payment(cls, account_id: str, event: str, payment: PaymentEntity) -> WebhookEvent:
        return cls(
            account_id=account_id,
            event=event,
            contains=["payment"],
            payload={"payment": {"entity": payment.model_dump(mode="json")}},
            created_at=payment.created_at,
        )

    @classmethod
    def for_downtime(
        cls, account_id: str, event: str, downtime: DowntimeEntity, at: int
    ) -> WebhookEvent:
        return cls(
            account_id=account_id,
            event=event,
            contains=["payment.downtime"],
            payload={"payment.downtime": {"entity": downtime.model_dump(mode="json")}},
            created_at=at,
        )

    def payment(self) -> PaymentEntity | None:
        block = self.payload.get("payment")
        return PaymentEntity.model_validate(block["entity"]) if block else None

    def downtime(self) -> DowntimeEntity | None:
        block = self.payload.get("payment.downtime")
        return DowntimeEntity.model_validate(block["entity"]) if block else None
