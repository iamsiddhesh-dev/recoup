"""Root-cause classification, measured against the world's hidden truth.

The classifier is agent code and cannot see `config/world.yaml`. These tests are
not agent code, so they can — which is the only way to ask the question that
matters: how often is it right, and what does it miss?

The coverage number this produces is a headline result, not a health check. It is
the honest answer to "what does the LLM actually add", because the fallback's
entire caseload is what these rules leave behind.
"""

from __future__ import annotations

import pytest

from recoup.agent.classify import Classification, Classifier, ClassifierRules, Resolution
from recoup.domain import FailureCause, PaymentEntity, PaymentMethod, PaymentStatus
from recoup.world.config import WorldConfig
from recoup.world.generator import build_batch


@pytest.fixture(scope="module")
def classifier() -> Classifier:
    return Classifier()


@pytest.fixture(scope="module")
def run():
    return build_batch(WorldConfig.load())


def _payment(**fields) -> PaymentEntity:
    base = {
        "id": "pay_test",
        "amount": 50000,
        "status": PaymentStatus.FAILED,
        "order_id": "order_test",
        "method": PaymentMethod.CARD,
    }
    return PaymentEntity(**{**base, **fields})


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_rules_file_loads(classifier):
    rules = ClassifierRules.load()
    assert rules.version == 1
    assert rules.rules


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("insufficient_funds", FailureCause.INSUFFICIENT_FUNDS),
        ("insufficient_balance", FailureCause.INSUFFICIENT_FUNDS),
        ("invalid_otp", FailureCause.AUTH_ABANDONED),
        ("payment_timeout", FailureCause.AUTH_ABANDONED),
        ("payment_cancelled", FailureCause.CUSTOMER_INTENT),
        ("card_expired", FailureCause.INSTRUMENT_INVALID),
        ("invalid_vpa", FailureCause.INSTRUMENT_INVALID),
        ("gateway_technical_error", FailureCause.TECHNICAL_GATEWAY),
        ("payment_blocked_risk", FailureCause.RISK_BLOCKED),
        ("mandate_revoked", FailureCause.MANDATE_PROBLEM),
        ("payment_declined", FailureCause.SOFT_ISSUER_DECLINE),
    ],
)
def test_documented_reasons_classify_deterministically(classifier, reason, expected):
    result = classifier.classify(_payment(error_reason=reason))

    assert result.cause is expected
    assert result.resolution is Resolution.DETERMINISTIC


def test_a_more_specific_rule_wins(classifier):
    """A rule naming source, step and reason beats one naming source alone.

    `source: network` maps to TECHNICAL_GATEWAY, but a network-sourced failure
    whose reason is `insufficient_funds` is not an outage.
    """
    result = classifier.classify(
        _payment(
            method=PaymentMethod.UPI,
            error_source="network",
            error_step="payment_debit_request",
            error_reason="insufficient_funds",
        )
    )
    assert result.cause is FailureCause.INSUFFICIENT_FUNDS


def test_source_level_rules_catch_undocumented_reasons(classifier):
    """Structural inference where the reason string is vendor-specific."""
    result = classifier.classify(
        _payment(
            method=PaymentMethod.UPI,
            error_source="customer_psp",
            error_step="payment_request",
            error_reason="some_psp_specific_string_nobody_published",
        )
    )
    assert result.cause is FailureCause.TECHNICAL_GATEWAY
    assert result.confidence < 0.7


def test_the_live_captured_decline_classifies(classifier):
    """The archetype verified against real Razorpay test mode."""
    result = classifier.classify(
        _payment(
            error_source="gateway",
            error_step="payment_authorization",
            error_reason="payment_failed",
        )
    )
    assert result.cause is FailureCause.SOFT_ISSUER_DECLINE
    assert result.rule_id == "payment_failed_gateway"


def test_an_unknown_failure_is_reported_unresolved_not_guessed(classifier):
    result = classifier.classify(
        _payment(
            method=PaymentMethod.WALLET,
            error_source="issuer",
            error_step="payment_eligibility_check",
            error_reason="wallet_not_linked",
        )
    )
    assert result.cause is None
    assert result.resolution is Resolution.UNRESOLVED
    assert result.actionable is False


# ---------------------------------------------------------------------------
# Accuracy against the world
# ---------------------------------------------------------------------------


def test_deterministic_rules_are_right_when_they_are_confident(classifier, run):
    """Precision matters more than coverage here.

    A wrong cause sends a payment down the wrong recovery path and spends money on
    it. An unresolved one costs a fallback call. The rules are allowed to be
    incomplete; they are not allowed to be confidently wrong.
    """
    batch, _, _ = run

    decided = wrong = 0
    mistakes: dict[tuple, int] = {}

    for payment in batch.failures:
        result = classifier.classify(payment)
        if not result.actionable:
            continue
        decided += 1
        truth = batch.truth_for(payment.id).cause
        if result.cause is not truth:
            wrong += 1
            key = (result.rule_id, str(result.cause), str(truth))
            mistakes[key] = mistakes.get(key, 0) + 1

    precision = 1 - (wrong / decided)
    assert precision > 0.97, (
        f"classifier precision {precision:.1%} — most common mistakes: "
        f"{sorted(mistakes.items(), key=lambda kv: -kv[1])[:3]}"
    )


def test_coverage_leaves_a_real_but_bounded_gap(classifier, run):
    """The population the fallback exists for.

    If coverage were total the model would be decoration; if it were poor the
    deterministic pass would not be pulling its weight. Both ends are failures.
    """
    batch, _, _ = run

    resolved = sum(1 for p in batch.failures if classifier.classify(p).actionable)
    coverage = resolved / len(batch.failures)

    assert 0.80 < coverage < 0.97, f"deterministic coverage {coverage:.1%}"


def test_unresolved_failures_carry_more_than_their_share_of_the_money(classifier, run):
    """The real case for a fallback, and it is not the coverage percentage.

    Undocumented reason strings are not spread evenly. They cluster in netbanking
    — `bank_unavailable` is the largest single gap — and netbanking has the
    biggest tickets in the book. So the failures the rules cannot classify hold
    several times their headcount share of the money at risk.

    Judged on count, a 3% gap looks like rounding. Judged on rupees it is the
    difference between a fallback being worth a model call and not.
    """
    batch, _, _ = run

    unresolved = [p for p in batch.failures if not classifier.classify(p).actionable]

    count_share = len(unresolved) / len(batch.failures)
    money_share = sum(p.amount for p in unresolved) / batch.amount_at_risk

    assert money_share > count_share * 2, (
        f"unresolved: {count_share:.1%} of failures but {money_share:.1%} of money"
    )


def test_unresolved_symptoms_collapse_to_a_handful_of_cases(classifier, run):
    """Why the fallback fits in a free tier.

    Hundreds of unresolved payments reduce to a few distinct symptom combinations.
    Classifying those is one batched request, not one request per payment.
    """
    batch, _, _ = run

    unresolved_payments = [
        p for p in batch.failures if not classifier.classify(p).actionable
    ]
    distinct = classifier.unresolved_fields(batch.failures)

    assert unresolved_payments
    assert len(distinct) <= 15
    assert len(distinct) < len(unresolved_payments) / 10


# ---------------------------------------------------------------------------
# Fallback seam
# ---------------------------------------------------------------------------


def test_a_fallback_only_sees_what_the_rules_could_not_resolve(run):
    seen: list[dict] = []

    def fallback(fields):
        seen.append(fields)
        return Classification(
            cause=FailureCause.TECHNICAL_GATEWAY,
            confidence=0.6,
            resolution=Resolution.FALLBACK,
            rule_id="stub",
        )

    batch, _, _ = run
    classifier = Classifier(fallback=fallback)

    for payment in batch.failures[:400]:
        classifier.classify(payment)

    assert seen, "the fallback was never consulted"
    assert all(f["reason"] not in ("invalid_otp", "insufficient_funds") for f in seen)


def test_a_fallback_that_declines_leaves_the_failure_unresolved(run):
    batch, _, _ = run
    classifier = Classifier(fallback=lambda _: None)

    unresolved = [
        p for p in batch.failures if classifier.classify(p).resolution is Resolution.UNRESOLVED
    ]
    assert unresolved
