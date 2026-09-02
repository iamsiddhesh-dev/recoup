"""The webhook endpoint.

A webhook route is an unauthenticated POST handler on the public internet. These
tests exist mostly to pin down its refusal behaviour: what it rejects, with which
status, and that a rejection never reaches storage.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from recoup.adapters.webhooks import expected_signature
from recoup.web.app import create_app
from recoup.web.sink import WebhookSink

SECRET = "whsec_test_value"

BODY = json.dumps(
    {
        "entity": "event",
        "account_id": "acc_test",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_probe_1",
                    "amount": 10000,
                    "status": "failed",
                    "order_id": "order_probe",
                    "method": "upi",
                    "vpa": "failure@razorpay",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_source": "customer",
                    "error_step": "payment_authentication",
                    "error_reason": "payment_timeout",
                    "error_description": "Payment was not completed in time",
                    "created_at": 1780000000,
                }
            }
        },
        "created_at": 1780000000,
    }
).encode()


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)


@pytest.fixture
def sink(tmp_path) -> WebhookSink:
    return WebhookSink(tmp_path / "webhooks.jsonl")


@pytest.fixture
def client(sink) -> TestClient:
    return TestClient(create_app(sink))


def _post(client: TestClient, body: bytes, signature: str | None, event_id: str = "evt_1"):
    headers = {"Content-Type": "application/json", "X-Razorpay-Event-Id": event_id}
    if signature is not None:
        headers["X-Razorpay-Signature"] = signature
    return client.post("/webhooks/razorpay", content=body, headers=headers)


def test_health_reports_how_many_webhooks_have_arrived(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "webhooks_received": 0}


def test_a_correctly_signed_webhook_is_stored(client, sink):
    response = _post(client, BODY, expected_signature(BODY, SECRET))

    assert response.status_code == 200
    assert response.json()["stored"] is True
    assert sink.count() == 1
    assert sink.recent()[0]["event"] == "payment.failed"


def test_a_bad_signature_is_a_400_and_is_not_stored(client, sink):
    response = _post(client, BODY, "not-the-right-signature")

    assert response.status_code == 400
    assert "mismatch" in response.json()["error"]
    assert sink.count() == 0


def test_a_missing_signature_is_refused(client, sink):
    response = _post(client, BODY, None)

    assert response.status_code == 400
    assert sink.count() == 0


def test_a_tampered_amount_is_refused(client, sink):
    """The signature covers the body, so changing the amount invalidates it."""
    signature = expected_signature(BODY, SECRET)
    response = _post(client, BODY.replace(b"10000", b"99999"), signature)

    assert response.status_code == 400
    assert sink.count() == 0


def test_a_duplicate_delivery_is_acknowledged_but_not_double_counted(client, sink):
    """Razorpay retries anything it does not get a 2xx for.

    A duplicate must still return 200 — otherwise it is retried forever — but must
    not be recorded twice, or the money at risk is counted twice.
    """
    signature = expected_signature(BODY, SECRET)

    first = _post(client, BODY, signature, event_id="evt_same")
    second = _post(client, BODY, signature, event_id="evt_same")

    assert first.json()["stored"] is True
    assert second.status_code == 200
    assert second.json()["stored"] is False
    assert sink.count() == 1


def test_recent_endpoint_surfaces_what_arrived(client):
    _post(client, BODY, expected_signature(BODY, SECRET))

    body = client.get("/webhooks/recent").json()

    assert body["count"] == 1
    assert body["events"][0]["payload"]["event"] == "payment.failed"


def test_the_sink_survives_a_restart(tmp_path):
    """Evidence has to outlive the process that received it."""
    path = tmp_path / "webhooks.jsonl"

    first = TestClient(create_app(WebhookSink(path)))
    _post(first, BODY, expected_signature(BODY, SECRET))

    reopened = WebhookSink(path)
    assert reopened.count() == 1
