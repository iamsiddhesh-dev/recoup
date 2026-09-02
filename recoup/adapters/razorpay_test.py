"""The live adapter — Razorpay test mode.

Same protocol as the simulated adapter, so the agent runs against it unmodified.
That is the point of the whole seam and the reason a simulated evaluation is worth
anything: the code being measured is the code that would run in production.

Two things here are load-bearing.

**The test-mode guard.** `__init__` refuses any key that is not `rzp_test_`. This
project executes money actions; a misconfigured environment variable should not be
able to turn a demo into a real charge. It is four lines and it makes the failure
mode impossible rather than unlikely.

**Silent retry is not universally available.** Razorpay has no "retry this
payment" call. Where standing authorisation exists — an e-mandate, a saved token —
a fresh charge can be raised with the customer absent, via the recurring endpoint.
Where it does not, the only honest answer is that the customer has to act, and the
adapter says so instead of pretending. See `supports_silent_retry`.
"""

from __future__ import annotations

import os
from datetime import datetime

import httpx

from recoup.adapters.base import (
    ChargeRequest,
    ChargeResult,
    LinkRequest,
    RecoveryLink,
    TestModeViolation,
    supports_silent_retry,
)
from recoup.domain import DowntimeEntity, PaymentEntity, Severity

API_BASE = "https://api.razorpay.com/v1"
TEST_KEY_PREFIX = "rzp_test_"


class RazorpayTestAdapter:
    """Talks to real Razorpay, in test mode only."""

    name = "razorpay_test"
    live_capable = True

    def __init__(
        self,
        key_id: str | None = None,
        key_secret: str | None = None,
        *,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        key_id = key_id or os.environ.get("RAZORPAY_KEY_ID", "")
        key_secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET", "")

        if not key_id or not key_secret:
            raise TestModeViolation(
                "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set. Copy .env.example "
                "to .env and fill in your test-mode keys."
            )

        if not key_id.startswith(TEST_KEY_PREFIX):
            raise TestModeViolation(
                f"refusing to run against a non-test key (got {key_id[:8]}...). "
                f"Recoup only ever operates in test mode; keys must start with "
                f"{TEST_KEY_PREFIX!r}."
            )

        self._key_id = key_id
        self._client = client or httpx.Client(
            base_url=API_BASE,
            auth=(key_id, key_secret),
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RazorpayTestAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- reads ---------------------------------------------------------------

    def fetch_payment(self, payment_id: str) -> PaymentEntity | None:
        response = self._client.get(f"/payments/{payment_id}")
        if response.status_code == 400:
            return None
        response.raise_for_status()
        return _to_payment(response.json())

    def active_downtimes(self, at: datetime) -> list[DowntimeEntity]:
        """Razorpay's live degradation feed.

        The same signal the agent consumes in the simulator, which is why the
        policy's downtime gating needs no branch for which world it is in.
        https://razorpay.com/docs/api/payments/downtime/
        """
        response = self._client.get("/payments/downtimes")
        response.raise_for_status()

        downtimes: list[DowntimeEntity] = []
        for item in response.json().get("items", []):
            if item.get("status") != "started":
                continue
            downtimes.append(
                DowntimeEntity(
                    id=item["id"],
                    method=item["method"],
                    begin=item.get("begin", 0),
                    end=item.get("end"),
                    status="started",
                    scheduled=item.get("scheduled", False),
                    severity=Severity(item.get("severity", "low")),
                    instrument=item.get("instrument", {}),
                )
            )
        return downtimes

    # -- money ---------------------------------------------------------------

    def attempt_charge(self, request: ChargeRequest) -> ChargeResult:
        if not supports_silent_retry(request.method, request.instrument):
            return ChargeResult(
                succeeded=False,
                error_source="business",
                error_step="payment_initiation",
                error_reason="customer_action_required",
                error_description=(
                    f"{request.method} has no standing authorisation; this payment "
                    "cannot be charged without the customer. Use a recovery link."
                ),
            )

        payload = {
            "amount": request.amount,
            "currency": "INR",
            "order_id": request.order_id,
            "recurring": "1",
            "token": request.instrument.get("token_id"),
        }

        response = self._client.post(
            "/payments/create/recurring",
            json={k: v for k, v in payload.items() if v is not None},
            headers={"X-Razorpay-Idempotency-Key": request.idempotency_key},
        )

        if response.status_code >= 400:
            error = response.json().get("error", {})
            return ChargeResult(
                succeeded=False,
                error_source=error.get("source"),
                error_step=error.get("step"),
                error_reason=error.get("reason"),
                error_description=error.get("description"),
                raw=response.json(),
            )

        body = response.json()
        return ChargeResult(succeeded=True, payment=_to_payment(body), raw=body)

    def create_recovery_link(self, request: LinkRequest) -> RecoveryLink:
        response = self._client.post(
            "/payment_links",
            json={
                "amount": request.amount,
                "currency": "INR",
                "description": request.description,
                "reference_id": request.idempotency_key,
                "notes": {
                    "recoup_payment_id": request.payment_id,
                    "customer_id": request.customer_ref,
                },
            },
        )
        response.raise_for_status()
        body = response.json()
        return RecoveryLink(
            id=body["id"],
            url=body["short_url"],
            expires_at=(
                datetime.fromtimestamp(body["expire_by"]) if body.get("expire_by") else None
            ),
        )


def _to_payment(body: dict) -> PaymentEntity:
    """Razorpay's payment entity, kept to the fields Recoup uses.

    Tolerant by construction: the live API returns fields this project does not
    model, and new ones appear over time. Dropping them is correct; failing on
    them would make the adapter brittle for no benefit.
    """
    return PaymentEntity.model_validate(
        {
            "id": body["id"],
            "amount": body["amount"],
            "currency": body.get("currency", "INR"),
            "status": body["status"],
            "order_id": body.get("order_id") or "",
            "method": body["method"],
            "captured": body.get("captured", False),
            "international": body.get("international", False),
            "bank": body.get("bank"),
            "vpa": body.get("vpa"),
            "card_id": body.get("card_id"),
            "wallet": body.get("wallet"),
            "token_id": body.get("token_id"),
            "contact": body.get("contact"),
            "email": body.get("email"),
            "description": body.get("description"),
            "error_code": body.get("error_code"),
            "error_description": body.get("error_description"),
            "error_source": body.get("error_source"),
            "error_step": body.get("error_step"),
            "error_reason": body.get("error_reason"),
            "acquirer_data": body.get("acquirer_data") or {},
            "notes": body.get("notes") or {},
            "created_at": body.get("created_at", 0),
        }
    )
