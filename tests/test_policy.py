"""The expected-value engine.

This is the code that decides where money goes, so the tests are about its
economics rather than its plumbing: does it optimise margin instead of gross, does
it trade delay against success rate, does repeated contact actually get expensive,
and does it stop when nothing is worth doing.

Each of those is a way a naive version of this engine goes wrong while still
looking like it works.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recoup.agent.actions import ActionKind
from recoup.agent.classify import Classification, Resolution
from recoup.agent.config import PolicyConfig
from recoup.agent.context import ContextBuilder, DecisionContext
from recoup.agent.policy import PolicyEngine
from recoup.domain import (
    DowntimeEntity,
    FailureCause,
    PaymentEntity,
    PaymentMethod,
    PaymentStatus,
    Severity,
)

NOW = datetime(2026, 6, 15, 14, 0)


@pytest.fixture(scope="module")
def policy() -> PolicyConfig:
    return PolicyConfig.load()


@pytest.fixture
def engine(policy) -> PolicyEngine:
    return PolicyEngine(policy)


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
        "token_id": "token_000001",
        "notes": {"customer_id": "cust_00001"},
    }
    return PaymentEntity(**{**base, **fields})


def _context(
    builder: ContextBuilder,
    cause: FailureCause | None = FailureCause.SOFT_ISSUER_DECLINE,
    now: datetime = NOW,
    **payment_fields,
) -> DecisionContext:
    classification = Classification(
        cause=cause,
        confidence=0.9 if cause else 0.0,
        resolution=Resolution.DETERMINISTIC if cause else Resolution.UNRESOLVED,
    )
    return builder.build(_payment(**payment_fields), classification, now)


# ---------------------------------------------------------------------------
# Shape of a decision
# ---------------------------------------------------------------------------


def test_a_decision_carries_its_arithmetic(engine, builder):
    """Explainability is a scoring criterion; it has to be in the data."""
    decision = engine.decide(_context(builder))

    assert decision.chosen is not None
    assert decision.chosen.breakdown
    assert "probability" in decision.chosen.breakdown
    assert decision.reason


def test_everything_considered_is_recorded_not_just_the_winner(engine, builder):
    decision = engine.decide(_context(builder))

    assert len(decision.considered) > 3
    assert decision.chosen.ev == max(c.ev for c in decision.considered)


def test_the_reason_is_a_sentence_a_merchant_could_check(engine, builder):
    decision = engine.decide(_context(builder))

    assert "₹" in decision.reason
    assert "%" in decision.reason


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_retries_are_not_proposed_without_standing_authorisation(engine, builder):
    """The policy must not propose actions the adapter cannot execute."""
    context = _context(builder, token_id=None, card_id="card_x")
    decision = engine.decide(context)

    assert not any(c.action.is_retry for c in decision.considered)


def test_retries_are_proposed_with_a_token(engine, builder):
    decision = engine.decide(_context(builder))

    assert any(c.action.is_retry for c in decision.considered)


@pytest.mark.parametrize(
    "cause",
    [FailureCause.INSTRUMENT_INVALID, FailureCause.RISK_BLOCKED, FailureCause.CUSTOMER_INTENT],
)
def test_causes_no_retry_can_fix_are_never_priced_as_retries(engine, builder, cause):
    """Distinct from a compliance stop.

    These are not forbidden, they are worthless — an expired card cannot be
    argued with. The arithmetic should exclude them on its own, without needing a
    rule.
    """
    decision = engine.decide(_context(builder, cause=cause))

    assert not any(c.action.is_retry for c in decision.considered)


def test_a_payment_without_a_customer_gets_no_contact_options(engine, builder):
    decision = engine.decide(_context(builder, notes={}))

    assert not any(c.action.is_contact for c in decision.considered)


# ---------------------------------------------------------------------------
# The economics
# ---------------------------------------------------------------------------


def test_expected_value_is_computed_on_margin_not_gross(engine, builder, policy):
    """Chasing gross revenue overstates every action's worth."""
    decision = engine.decide(_context(builder))

    priced = [c for c in decision.considered if "margin" in c.breakdown]
    assert priced

    for candidate in priced:
        assert candidate.breakdown["margin"] == policy.assumed_margin
        assert candidate.breakdown["gross"] < candidate.breakdown["amount"]


def test_delay_reduces_the_value_of_a_recovery(engine, builder):
    """Without decay the optimiser always waits for the best hour of the month."""
    context = _context(builder)
    retries = [c for c in engine.decide(context).considered if c.action.is_retry]

    soonest = min(retries, key=lambda c: c.delay_hours)
    latest = max(retries, key=lambda c: c.delay_hours)

    assert latest.breakdown["decay"] < soonest.breakdown["decay"]


def test_repeated_contact_gets_expensive(engine, builder):
    """The annoyance penalty is superlinear, or the product is a spam cannon."""
    fresh = _context(builder)
    fresh_nudge = next(c for c in engine.decide(fresh).considered if c.action.is_contact)

    for day in range(3):
        builder.note_contact("cust_00001", NOW - timedelta(days=day))

    chased = _context(builder)
    chased_nudge = next(c for c in engine.decide(chased).considered if c.action.is_contact)

    assert chased_nudge.breakdown["annoyance"] > 0
    assert chased_nudge.ev < fresh_nudge.ev
    assert chased_nudge.probability < fresh_nudge.probability


def test_retry_cost_escalates_with_attempts(engine, builder):
    first = next(c for c in engine.decide(_context(builder)).considered if c.action.is_retry)

    for _ in range(3):
        builder.note_attempt("pay_000001", succeeded=False)

    later = next(c for c in engine.decide(_context(builder)).considered if c.action.is_retry)

    assert later.breakdown["cost"] > first.breakdown["cost"]


def test_a_tiny_payment_is_not_worth_chasing(engine, builder):
    """The threshold that stops the agent spending ₹40 to recover ₹6."""
    decision = engine.decide(_context(builder, amount=1500))

    assert decision.action is ActionKind.STOP
    assert "threshold" in decision.reason


def test_a_large_payment_is_worth_acting_on(engine, builder):
    decision = engine.decide(_context(builder, amount=5000000))

    assert decision.acted


def test_escalation_only_competes_on_large_amounts(engine, builder):
    """A ₹120 human review cannot pay for itself on a small ticket."""
    small = engine.decide(_context(builder, amount=40000))
    large = engine.decide(_context(builder, amount=9000000))

    small_escalation = next(
        c for c in small.considered if c.action is ActionKind.ESCALATE_HUMAN
    )
    large_escalation = next(
        c for c in large.considered if c.action is ActionKind.ESCALATE_HUMAN
    )

    assert small_escalation.ev < 0
    assert large_escalation.ev > 0


# ---------------------------------------------------------------------------
# Learning changes behaviour
# ---------------------------------------------------------------------------


def test_learned_timing_moves_the_chosen_retry_hour(engine, builder):
    """The point of the learned model: schedule where the evidence is.

    Nothing tells the agent which hours are good. After observing that retries at
    18:00 succeed and retries at 15:00 do not, the schedule should move.
    """
    good = NOW.replace(hour=18)
    bad = NOW.replace(hour=15)

    for _ in range(300):
        builder.model.record(FailureCause.SOFT_ISSUER_DECLINE, "HDFC", good, succeeded=True)
        builder.model.record(FailureCause.SOFT_ISSUER_DECLINE, "HDFC", bad, succeeded=False)

    decision = engine.decide(_context(builder, now=bad))
    retries = [c for c in decision.considered if c.action.is_retry]
    best_retry = max(retries, key=lambda c: c.ev)

    assert best_retry.at.hour == 18


def test_an_unclassified_failure_prices_its_retry_pessimistically(engine, builder):
    """Like for like: the same action, priced under uncertainty."""
    classified = engine.decide(_context(builder, cause=FailureCause.SOFT_ISSUER_DECLINE))
    unknown = engine.decide(_context(builder, cause=None))

    known_p = max(c.probability for c in classified.considered if c.action.is_retry)
    unknown_p = max(c.probability for c in unknown.considered if c.action.is_retry)

    assert unknown_p < known_p


def test_an_unclassified_failure_tends_toward_human_review(engine, builder):
    """Emergent, not hard-coded, and the right instinct.

    Nothing says "escalate when confused". An unresolved cause is priced at the
    most pessimistic prior on the board, which drops the value of every automated
    option while human review — whose success rate does not depend on knowing the
    cause — is unaffected. So the arithmetic routes unexplained failures to a
    person, which is what an ops team would want.
    """
    decision = engine.decide(_context(builder, cause=None, amount=250000))

    assert decision.action is ActionKind.ESCALATE_HUMAN


# ---------------------------------------------------------------------------
# Downtime
# ---------------------------------------------------------------------------


def test_an_active_outage_adds_a_wait_and_recheck_option(engine, builder):
    """The arithmetic can only choose to wait if waiting is on the table."""
    builder.note_downtime(
        "payment.downtime.started",
        DowntimeEntity(
            id="down_1",
            method=PaymentMethod.CARD,
            begin=int(NOW.timestamp()),
            status="started",
            severity=Severity.HIGH,
            instrument={"bank": "HDFC"},
        ),
    )

    context = _context(builder)
    assert context.degraded is True

    decision = engine.decide(context)
    delays = {round(c.delay_hours, 2) for c in decision.considered if c.action.is_retry}

    assert 0.25 in delays
