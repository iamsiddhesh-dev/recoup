"""The seam has to hold, and the live-mode guard has to be unbypassable.

Live Razorpay is never contacted here. The test-mode adapter is exercised through
`httpx.MockTransport`, which means these run offline, in CI, with no credentials —
and still cover the request shaping and the refusal paths.
"""

from __future__ import annotations

import json

import httpx
import pytest

from recoup.adapters.base import (
    ChargeRequest,
    LinkRequest,
    Notifier,
    NudgeRequest,
    PaymentsAdapter,
    TestModeViolation,
    supports_silent_retry,
)
from recoup.adapters.razorpay_test import RazorpayTestAdapter
from recoup.adapters.simulated import SimulatedAdapter, SimulatedNotifier
from recoup.adapters.webhooks import SignatureError, expected_signature, parse, verify
from recoup.domain import Channel, Language, PaymentMethod, PaymentStatus
from recoup.world.config import WorldConfig
from recoup.world.generator import build_batch

# Built by concatenation so the literal never matches the credential pattern in
# tests/test_repo_hygiene.py. The hygiene test genuinely does fire on a key-shaped
# string in the tree, which is the point of it.
FAKE_LIVE_KEY = "rzp_live_" + "A1b2C3d4E5f6"
FAKE_TEST_KEY = "rzp_test_" + "A1b2C3d4E5f6"


@pytest.fixture(scope="module")
def config() -> WorldConfig:
    return WorldConfig.load()


@pytest.fixture(scope="module")
def run(config):
    return build_batch(config)


@pytest.fixture
def adapter(config, run):
    batch, population, issuers = run
    return SimulatedAdapter(config, batch, issuers, population)


@pytest.fixture
def notifier(config, run):
    _, population, issuers = run
    return SimulatedNotifier(config, population, issuers)


def _instrument_of(payment) -> dict[str, str]:
    return {
        k: v
        for k, v in (("token_id", payment.token_id), ("card_id", payment.card_id))
        if v
    }


@pytest.fixture(scope="module")
def retryable(run):
    """Failures the agent could actually charge again without the customer."""
    batch, _, _ = run
    return [
        p for p in batch.failures if supports_silent_retry(p.method, _instrument_of(p))
    ]


@pytest.fixture(scope="module")
def needs_customer(run):
    """Failures that can only be recovered by getting the customer to act."""
    batch, _, _ = run
    return [
        p
        for p in batch.failures
        if not supports_silent_retry(p.method, _instrument_of(p))
    ]


# ---------------------------------------------------------------------------
# The live-mode guard
# ---------------------------------------------------------------------------


def test_live_key_is_refused():
    """The guard that makes a real money action structurally impossible."""
    with pytest.raises(TestModeViolation, match="non-test key"):
        RazorpayTestAdapter(FAKE_LIVE_KEY, "secret")


def test_missing_credentials_are_refused(monkeypatch):
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    with pytest.raises(TestModeViolation, match="must be set"):
        RazorpayTestAdapter()


def test_test_key_is_accepted():
    adapter = RazorpayTestAdapter(FAKE_TEST_KEY, "secret")
    assert adapter.live_capable is True
    adapter.close()


# ---------------------------------------------------------------------------
# Both implementations satisfy the same protocol
# ---------------------------------------------------------------------------


def test_simulated_adapter_satisfies_the_protocol(adapter):
    assert isinstance(adapter, PaymentsAdapter)


def test_razorpay_adapter_satisfies_the_protocol():
    adapter = RazorpayTestAdapter(FAKE_TEST_KEY, "secret")
    assert isinstance(adapter, PaymentsAdapter)
    adapter.close()


def test_simulated_notifier_satisfies_the_protocol(notifier):
    assert isinstance(notifier, Notifier)


# ---------------------------------------------------------------------------
# Silent retry availability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "instrument", "expected"),
    [
        (PaymentMethod.EMANDATE, {}, True),
        (PaymentMethod.CARD, {"token_id": "token_x"}, True),
        (PaymentMethod.CARD, {"card_id": "card_x"}, False),
        (PaymentMethod.NETBANKING, {}, False),
        (PaymentMethod.UPI, {"vpa": "a@okhdfcbank"}, False),
        (PaymentMethod.WALLET, {}, False),
    ],
)
def test_silent_retry_requires_standing_authorisation(method, instrument, expected):
    """Razorpay cannot re-run a payment without standing authorisation.

    Getting this wrong means the policy proposes actions that do not exist.
    """
    assert supports_silent_retry(method, instrument) is expected


def test_both_adapters_refuse_the_same_charges(adapter, needs_customer):
    """The seam is only worth anything if both sides agree on what is possible.

    If the simulator allowed retries the live API refuses, the agent would learn a
    policy that cannot be executed — the exact failure a simulated evaluation is
    most likely to hide, and the reason this test exists.
    """
    assert needs_customer, "expected some failures to require customer action"

    for failure in needs_customer[:50]:
        result = adapter.attempt_charge(
            ChargeRequest(
                payment_id=failure.id,
                order_id=failure.order_id,
                amount=failure.amount,
                method=failure.method,
                idempotency_key=f"refuse-{failure.id}",
            )
        )
        assert result.succeeded is False
        assert result.error_reason == "customer_action_required"


def test_a_meaningful_share_of_failures_is_retryable(retryable, needs_customer):
    """Both recovery paths have to matter, or the policy space is degenerate.

    All-retryable makes this a pure scheduling problem; none-retryable makes it a
    pure messaging problem. The interesting product lives in between.
    """
    total = len(retryable) + len(needs_customer)
    share = len(retryable) / total
    assert 0.10 < share < 0.60, f"{share:.1%} of failures retryable — degenerate"


def test_live_adapter_refuses_a_charge_without_standing_authorisation():
    adapter = RazorpayTestAdapter(FAKE_TEST_KEY, "secret")
    result = adapter.attempt_charge(
        ChargeRequest(
            payment_id="pay_1",
            order_id="order_1",
            amount=10000,
            method=PaymentMethod.NETBANKING,
            idempotency_key="k1",
        )
    )
    assert result.succeeded is False
    assert result.error_reason == "customer_action_required"
    adapter.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_repeating_an_idempotency_key_does_not_charge_twice(adapter, run):
    batch, _, _ = run
    failure = batch.failures[0]
    request = ChargeRequest(
        payment_id=failure.id,
        order_id=failure.order_id,
        amount=failure.amount,
        method=failure.method,
        idempotency_key="stable-key",
    )

    first = adapter.attempt_charge(request)
    attempts_after_first = adapter.attempts_for(failure.id)
    second = adapter.attempt_charge(request)

    assert first == second
    assert adapter.attempts_for(failure.id) == attempts_after_first


def test_recovery_links_are_idempotent(adapter, run):
    batch, _, _ = run
    failure = batch.failures[0]
    request = LinkRequest(
        payment_id=failure.id,
        order_id=failure.order_id,
        amount=failure.amount,
        customer_ref=failure.customer_ref or "cust_00000",
        description="Complete your payment",
        idempotency_key="link-key",
    )
    assert adapter.create_recovery_link(request) == adapter.create_recovery_link(request)


# ---------------------------------------------------------------------------
# Common random numbers
# ---------------------------------------------------------------------------


def test_the_same_attempt_gets_the_same_luck_in_every_arm(config, run, retryable):
    """The property that makes the holdout comparison sharp.

    Two independent adapters — standing in for two experiment arms — must agree on
    the outcome of the same payment retried the same number of times. Otherwise
    the difference between arms mixes policy with luck.
    """
    batch, population, issuers = run
    failure = retryable[0]

    def charge(adapter):
        return adapter.attempt_charge(
            ChargeRequest(
                payment_id=failure.id,
                order_id=failure.order_id,
                amount=failure.amount,
                method=failure.method,
                idempotency_key=f"arm-{id(adapter)}",
                at=config.run.start_at,
            )
        ).succeeded

    arm_a = SimulatedAdapter(config, batch, issuers, population)
    arm_b = SimulatedAdapter(config, batch, issuers, population)

    assert charge(arm_a) == charge(arm_b)


# ---------------------------------------------------------------------------
# Charge results
# ---------------------------------------------------------------------------


def test_a_failed_retry_reports_the_same_symptoms(adapter, retryable):
    """Issuers do not explain themselves differently on the second ask."""
    for failure in retryable[:40]:
        result = adapter.attempt_charge(
            ChargeRequest(
                payment_id=failure.id,
                order_id=failure.order_id,
                amount=failure.amount,
                method=failure.method,
                idempotency_key=f"sym-{failure.id}",
            )
        )
        if not result.succeeded:
            assert result.error_reason == failure.error_reason
            assert result.error_source == failure.error_source


def test_a_successful_retry_clears_the_error_fields(adapter, retryable, config):
    for index, failure in enumerate(retryable[:300]):
        result = adapter.attempt_charge(
            ChargeRequest(
                payment_id=failure.id,
                order_id=failure.order_id,
                amount=failure.amount,
                method=failure.method,
                idempotency_key=f"ok-{index}",
                at=config.run.start_at,
            )
        )
        if result.succeeded:
            assert result.payment.status is PaymentStatus.CAPTURED
            assert result.payment.captured is True
            assert result.payment.error_reason is None
            return

    pytest.fail("no retry succeeded in 300 attempts, which is implausible")


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


def test_a_channel_without_consent_is_not_delivered(notifier, run):
    _, population, _ = run
    without_consent = next(
        c for c in population.customers if not c.may_contact(Channel.WHATSAPP)
    )

    result = notifier.send(
        NudgeRequest(
            customer_ref=without_consent.id,
            channel=Channel.WHATSAPP,
            language=Language.HINGLISH,
            body="Your payment did not go through",
            payment_id="pay_000001",
            idempotency_key="nudge-1",
        )
    )
    assert result.delivered is False
    assert "consent" in result.detail


def test_delivery_and_action_are_distinct_outcomes(notifier, run):
    """A message that arrives and changes nothing still cost money and goodwill."""
    _, population, _ = run
    reachable = next(c for c in population.customers if c.may_contact(Channel.SMS))

    results = [
        notifier.send(
            NudgeRequest(
                customer_ref=reachable.id,
                channel=Channel.SMS,
                language=Language.ENGLISH,
                body="Your payment did not go through",
                payment_id=f"pay_{index:06d}",
                idempotency_key=f"nudge-{index}",
            )
        )
        for index in range(20)
    ]

    assert all(r.delivered for r in results)
    assert any(not r.acted_on for r in results)


# ---------------------------------------------------------------------------
# Webhook verification
# ---------------------------------------------------------------------------

SECRET = "whsec_example_value"
BODY = json.dumps(
    {
        "entity": "event",
        "account_id": "acc_test",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_live_shape",
                    "amount": 45000,
                    "status": "failed",
                    "order_id": "order_1",
                    "method": "upi",
                    "error_reason": "payment_timeout",
                }
            }
        },
        "created_at": 1780000000,
    }
).encode()


def test_a_valid_signature_verifies():
    verify(BODY, expected_signature(BODY, SECRET), SECRET)


def test_a_tampered_body_is_rejected():
    signature = expected_signature(BODY, SECRET)
    with pytest.raises(SignatureError, match="mismatch"):
        verify(BODY.replace(b"45000", b"99999"), signature, SECRET)


def test_the_wrong_secret_is_rejected():
    with pytest.raises(SignatureError, match="mismatch"):
        verify(BODY, expected_signature(BODY, "other-secret"), SECRET)


def test_a_missing_signature_header_is_rejected():
    with pytest.raises(SignatureError, match="missing"):
        verify(BODY, None, SECRET)


def test_an_unset_secret_refuses_rather_than_trusts(monkeypatch):
    monkeypatch.delenv("RAZORPAY_WEBHOOK_SECRET", raising=False)
    with pytest.raises(SignatureError, match="not set"):
        verify(BODY, "anything", None)


def test_a_verified_event_parses_into_the_shared_shape():
    event = parse(BODY, expected_signature(BODY, SECRET), SECRET)
    payment = event.payment()

    assert event.event == "payment.failed"
    assert payment.id == "pay_live_shape"
    assert payment.method is PaymentMethod.UPI
    assert payment.error_reason == "payment_timeout"


# ---------------------------------------------------------------------------
# Live request shaping, without a network
# ---------------------------------------------------------------------------


def test_downtime_feed_is_parsed_into_the_shared_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/payments/downtimes"
        return httpx.Response(
            200,
            json={
                "entity": "collection",
                "count": 2,
                "items": [
                    {
                        "id": "down_live_1",
                        "method": "card",
                        "begin": 1780000000,
                        "end": None,
                        "status": "started",
                        "scheduled": False,
                        "severity": "high",
                        "instrument": {"bank": "HDFC"},
                    },
                    {
                        "id": "down_live_2",
                        "method": "upi",
                        "begin": 1779000000,
                        "end": 1779003600,
                        "status": "resolved",
                        "severity": "low",
                        "instrument": {},
                    },
                ],
            },
        )

    client = httpx.Client(
        base_url="https://api.razorpay.com/v1", transport=httpx.MockTransport(handler)
    )
    adapter = RazorpayTestAdapter(FAKE_TEST_KEY, "secret", client=client)

    downtimes = adapter.active_downtimes(None)

    assert [d.id for d in downtimes] == ["down_live_1"]
    assert downtimes[0].instrument == {"bank": "HDFC"}
    adapter.close()


def test_a_recurring_charge_sends_an_idempotency_header():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "pay_recurring_1",
                "amount": 59900,
                "status": "captured",
                "order_id": "order_9",
                "method": "emandate",
                "captured": True,
                "created_at": 1780000001,
            },
        )

    client = httpx.Client(
        base_url="https://api.razorpay.com/v1", transport=httpx.MockTransport(handler)
    )
    adapter = RazorpayTestAdapter(FAKE_TEST_KEY, "secret", client=client)

    result = adapter.attempt_charge(
        ChargeRequest(
            payment_id="pay_1",
            order_id="order_9",
            amount=59900,
            method=PaymentMethod.EMANDATE,
            idempotency_key="idem-42",
        )
    )

    assert result.succeeded is True
    assert seen["x-razorpay-idempotency-key"] == "idem-42"
    adapter.close()


def test_a_declined_live_charge_surfaces_razorpay_error_fields():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "BAD_REQUEST_ERROR",
                    "description": "Payment failed due to insufficient funds",
                    "source": "issuer_bank",
                    "step": "payment_authorization",
                    "reason": "insufficient_funds",
                }
            },
        )

    client = httpx.Client(
        base_url="https://api.razorpay.com/v1", transport=httpx.MockTransport(handler)
    )
    adapter = RazorpayTestAdapter(FAKE_TEST_KEY, "secret", client=client)

    result = adapter.attempt_charge(
        ChargeRequest(
            payment_id="pay_1",
            order_id="order_9",
            amount=59900,
            method=PaymentMethod.EMANDATE,
            idempotency_key="idem-43",
        )
    )

    assert result.succeeded is False
    assert result.error_reason == "insufficient_funds"
    assert result.error_source == "issuer_bank"
    adapter.close()
