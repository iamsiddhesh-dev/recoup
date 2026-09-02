"""The learned recovery model and the facts a decision is allowed to see.

The model here is the thing most projects would hand to an LLM: "when should we
retry?". It is deliberately statistics — hierarchical shrinkage over observed
outcomes — because the answer is a number derived from counts, it has to be
defensible to a merchant, and it gets recomputed tens of thousands of times per
run. These tests pin the behaviour that makes it trustworthy: it starts at the
prior, it moves only with evidence, and one lucky success does not swing it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recoup.agent.classify import Classification, Resolution
from recoup.agent.config import ComplianceConfig, PolicyConfig
from recoup.agent.context import ContextBuilder, RecoveryModel
from recoup.domain import (
    DowntimeEntity,
    FailureCause,
    PaymentEntity,
    PaymentMethod,
    PaymentStatus,
    Severity,
)

NOW = datetime(2026, 6, 10, 14, 30)


@pytest.fixture(scope="module")
def policy() -> PolicyConfig:
    return PolicyConfig.load()


@pytest.fixture
def model(policy) -> RecoveryModel:
    return RecoveryModel(policy)


@pytest.fixture
def builder(policy) -> ContextBuilder:
    return ContextBuilder(policy)


def _payment(**fields) -> PaymentEntity:
    base = {
        "id": "pay_000001",
        "amount": 250000,
        "status": PaymentStatus.FAILED,
        "order_id": "order_000001",
        "method": PaymentMethod.CARD,
        "bank": "HDFC",
        "notes": {"customer_id": "cust_00001"},
    }
    return PaymentEntity(**{**base, **fields})


def _classified(cause: FailureCause | None = FailureCause.INSUFFICIENT_FUNDS) -> Classification:
    return Classification(
        cause=cause,
        confidence=0.9 if cause else 0.0,
        resolution=Resolution.DETERMINISTIC if cause else Resolution.UNRESOLVED,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_policy_and_compliance_configs_load():
    assert PolicyConfig.load().assumed_margin > 0
    assert ComplianceConfig.load().attempts.max_per_payment > 0


def test_retry_cost_escalates_with_prior_attempts(policy):
    """Retries look free per attempt and are not."""
    first = policy.cost_of("RETRY_NOW", prior_attempts=0)
    third = policy.cost_of("RETRY_NOW", prior_attempts=2)

    assert third > first


def test_non_retry_actions_do_not_escalate(policy):
    assert policy.cost_of("NUDGE_SMS", 0) == policy.cost_of("NUDGE_SMS", 3)


def test_quiet_hours_wrap_midnight():
    """21:00–09:00 is a union of two ranges, not a range."""
    quiet = ComplianceConfig.load().contact.quiet_hours

    assert quiet.covers(datetime(2026, 6, 1, 23, 0).time()) is True
    assert quiet.covers(datetime(2026, 6, 1, 3, 0).time()) is True
    assert quiet.covers(datetime(2026, 6, 1, 14, 0).time()) is False
    assert quiet.covers(datetime(2026, 6, 1, 9, 0).time()) is False


# ---------------------------------------------------------------------------
# The learned model
# ---------------------------------------------------------------------------


def test_with_no_evidence_the_estimate_is_the_prior(model, policy):
    estimate = model.estimate(FailureCause.SOFT_ISSUER_DECLINE, "HDFC", 10)

    assert estimate.probability == pytest.approx(
        policy.prior_recovery_probability[FailureCause.SOFT_ISSUER_DECLINE]
    )
    assert estimate.level == "prior"


def test_one_lucky_success_barely_moves_the_estimate(model, policy):
    """The failure mode shrinkage exists to prevent.

    Two observations, one success, is not a 50% recovery rate. An unshrunk
    estimator would say it is and the policy would spend money on it.
    """
    prior = policy.prior_recovery_probability[FailureCause.SOFT_ISSUER_DECLINE]

    model.record(FailureCause.SOFT_ISSUER_DECLINE, "HDFC", 10, succeeded=True)
    model.record(FailureCause.SOFT_ISSUER_DECLINE, "HDFC", 10, succeeded=False)

    estimate = model.estimate(FailureCause.SOFT_ISSUER_DECLINE, "HDFC", 10)

    assert abs(estimate.probability - prior) < 0.06


def test_sustained_evidence_overrides_the_prior(model, policy):
    prior = policy.prior_recovery_probability[FailureCause.SOFT_ISSUER_DECLINE]

    for _ in range(400):
        model.record(FailureCause.SOFT_ISSUER_DECLINE, "HDFC", 10, succeeded=True)

    estimate = model.estimate(FailureCause.SOFT_ISSUER_DECLINE, "HDFC", 10)

    assert estimate.probability > prior + 0.3
    assert estimate.level == "cause+issuer+hour"
    assert estimate.observations == 400


def test_evidence_at_a_coarser_level_informs_an_unseen_cell(model, policy):
    """Hierarchical backoff.

    A cause observed heavily at other hours should shift the estimate for an hour
    never seen, because most of what is being learned is about the cause.
    """
    prior = policy.prior_recovery_probability[FailureCause.AUTH_ABANDONED]

    for hour in range(0, 12):
        for _ in range(40):
            model.record(FailureCause.AUTH_ABANDONED, "ICIC", hour, succeeded=True)

    unseen_hour = model.estimate(FailureCause.AUTH_ABANDONED, "ICIC", 23)

    assert unseen_hour.probability > prior
    assert unseen_hour.level == "cause"


def test_an_unclassified_failure_gets_the_most_pessimistic_prior(model, policy):
    """Acting confidently on a cause you could not identify spends money on nothing."""
    estimate = model.estimate(None, "HDFC", 10)

    assert estimate.probability == min(policy.prior_recovery_probability.values())
    assert estimate.level == "unclassified"


def test_best_hour_picks_the_highest_estimate(model):
    for _ in range(300):
        model.record(FailureCause.INSUFFICIENT_FUNDS, "SBIN", 11, succeeded=True)
        model.record(FailureCause.INSUFFICIENT_FUNDS, "SBIN", 3, succeeded=False)

    hour, estimate = model.best_hour(
        FailureCause.INSUFFICIENT_FUNDS, "SBIN", [3, 11, 20]
    )

    assert hour == 11
    assert estimate.probability > 0.5


def test_estimates_stay_within_bounds(model):
    for _ in range(1000):
        model.record(FailureCause.TECHNICAL_GATEWAY, "PUNB", 5, succeeded=True)

    assert 0.0 <= model.estimate(FailureCause.TECHNICAL_GATEWAY, "PUNB", 5).probability <= 1.0


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def test_context_carries_the_facts_a_decision_needs(builder):
    context = builder.build(_payment(), _classified(), NOW)

    assert context.amount == 250000
    assert context.customer_ref == "cust_00001"
    assert context.issuer_code == "HDFC"
    assert context.cause is FailureCause.INSUFFICIENT_FUNDS
    assert context.estimate is not None


def test_attempts_accumulate(builder):
    for _ in range(3):
        builder.note_attempt("pay_000001", succeeded=False)

    context = builder.build(_payment(), _classified(), NOW)

    assert context.attempts == 3
    assert context.consecutive_failures == 3


def test_a_success_resets_the_consecutive_failure_run(builder):
    builder.note_attempt("pay_000001", succeeded=False)
    builder.note_attempt("pay_000001", succeeded=False)
    builder.note_attempt("pay_000001", succeeded=True)

    context = builder.build(_payment(), _classified(), NOW)

    assert context.attempts == 3
    assert context.consecutive_failures == 0


def test_contacts_outside_the_window_do_not_count(builder, policy):
    window = policy.annoyance.window_days

    builder.note_contact("cust_00001", NOW - timedelta(days=window + 2))
    builder.note_contact("cust_00001", NOW - timedelta(days=1))

    context = builder.build(_payment(), _classified(), NOW)

    assert context.contacts_in_window == 1
    assert context.hours_since_last_contact() == pytest.approx(24.0)


def test_only_tokenised_instruments_can_be_retried_silently(builder):
    without = builder.build(_payment(card_id="card_x"), _classified(), NOW)
    with_token = builder.build(_payment(token_id="token_x"), _classified(), NOW)

    assert without.can_retry_silently is False
    assert with_token.can_retry_silently is True


# ---------------------------------------------------------------------------
# Downtime tracking
# ---------------------------------------------------------------------------


def _downtime(method: PaymentMethod, bank: str | None) -> DowntimeEntity:
    return DowntimeEntity(
        id="down_0001",
        method=method,
        begin=int(NOW.timestamp()),
        status="started",
        severity=Severity.HIGH,
        instrument={"bank": bank} if bank else {},
    )


def test_a_started_downtime_is_visible_in_context(builder):
    builder.note_downtime("payment.downtime.started", _downtime(PaymentMethod.CARD, "HDFC"))

    context = builder.build(_payment(), _classified(), NOW)

    assert context.degraded is True
    assert context.downtime.severity is Severity.HIGH


def test_a_resolved_downtime_stops_being_visible(builder):
    """Resolution matters as much as onset.

    An agent that defers on downtime and never learns it ended sits on
    recoverable money until the horizon runs out.
    """
    downtime = _downtime(PaymentMethod.CARD, "HDFC")
    builder.note_downtime("payment.downtime.started", downtime)
    builder.note_downtime("payment.downtime.resolved", downtime)

    assert builder.build(_payment(), _classified(), NOW).degraded is False


def test_a_downtime_for_another_bank_does_not_apply(builder):
    builder.note_downtime("payment.downtime.started", _downtime(PaymentMethod.CARD, "SBIN"))

    assert builder.build(_payment(bank="HDFC"), _classified(), NOW).degraded is False


def test_a_method_wide_downtime_applies_to_every_bank(builder):
    """UPI outages often sit at the network layer and hit everyone at once."""
    builder.note_downtime("payment.downtime.started", _downtime(PaymentMethod.UPI, None))

    context = builder.build(
        _payment(method=PaymentMethod.UPI, bank="HDFC"), _classified(), NOW
    )
    assert context.degraded is True
