"""Parsed against a real Razorpay response, not an imagined one.

`tests/fixtures/live_payment_failed.json` is a genuine `payment.failed` webhook,
captured from Razorpay test mode through a cloudflared tunnel. Contact details and
the account id are redacted; every other field is exactly as delivered.

It is committed for two reasons.

First, it is the only thing standing between "our model parses what we think
Razorpay sends" and "our model parses what Razorpay sends". Capturing it took a
tunnel, a webhook registration and a real card payment; re-running that on every
change is not viable, and a fixture costs nothing.

Second, it caught two wrong assumptions the moment it arrived — see FAILURES.md.
Both were invented values that looked entirely plausible sitting in a config file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recoup.domain import PaymentMethod, PaymentStatus, WebhookEvent

FIXTURE = Path(__file__).parent / "fixtures" / "live_payment_failed.json"


@pytest.fixture(scope="module")
def event() -> WebhookEvent:
    return WebhookEvent.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_a_real_webhook_envelope_parses(event):
    assert event.event == "payment.failed"
    assert event.contains == ["payment"]


def test_a_real_payment_entity_parses(event):
    payment = event.payment()

    assert payment is not None
    assert payment.status is PaymentStatus.FAILED
    assert payment.method is PaymentMethod.CARD
    assert payment.amount == 10000
    assert payment.captured is False


def test_unmodelled_fields_are_tolerated(event):
    """The live entity carries `card`, `fee`, `reward` and more that Recoup ignores.

    Dropping them is correct; failing on them would make the adapter brittle
    against an API that adds fields over time.
    """
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    entity = raw["payload"]["payment"]["entity"]

    assert "card" in entity and "reward" in entity
    assert event.payment() is not None


def test_the_error_fields_recoup_classifies_on_are_present(event):
    """The three fields the root-cause classifier keys on, as actually delivered."""
    payment = event.payment()

    assert payment.error_source == "gateway"
    assert payment.error_step == "payment_authorization"
    assert payment.error_reason == "payment_failed"
    assert payment.error_code == "BAD_REQUEST_ERROR"


def test_merchant_notes_survive_the_round_trip(event):
    """Recoup tags payments through `notes`; that has to come back intact."""
    assert event.payment().notes["recoup_payment_id"] == "probe"


def test_error_code_is_not_derivable_from_source(event):
    """The assumption this fixture disproved.

    The generator mapped `gateway` source onto `GATEWAY_ERROR`. Razorpay returned
    `BAD_REQUEST_ERROR` for a gateway-sourced failure, so the two are independent
    and `code` has to be carried explicitly rather than inferred.
    """
    payment = event.payment()
    assert payment.error_source == "gateway"
    assert payment.error_code != "GATEWAY_ERROR"
