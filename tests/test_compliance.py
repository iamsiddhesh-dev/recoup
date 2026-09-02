"""The compliance gate.

Every hard rule gets a test, because these are the rules most likely to be quietly
weakened under deadline pressure — they only ever *stop* the agent earning, and
their value shows up as an absence.

The two structural properties matter as much as the individual rules: a veto must
fall through to the next permitted option rather than collapsing to inaction, and
every refusal must be recorded. The refusal list is a deliverable, and it exists
only because vetoes are written down as positively as actions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recoup.agent.actions import ActionKind, Candidate
from recoup.agent.classify import Classification, Resolution
from recoup.agent.compliance import ComplianceGate, next_permitted_contact_time
from recoup.agent.config import ComplianceConfig, PolicyConfig
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

NOON = datetime(2026, 6, 15, 12, 0)
MIDNIGHT = datetime(2026, 6, 15, 2, 0)


@pytest.fixture(scope="module")
def policy() -> PolicyConfig:
    return PolicyConfig.load()


@pytest.fixture(scope="module")
def rules() -> ComplianceConfig:
    return ComplianceConfig.load()


@pytest.fixture
def gate(rules) -> ComplianceGate:
    return ComplianceGate(rules)


@pytest.fixture
def builder(policy) -> ContextBuilder:
    return ContextBuilder(policy)


@pytest.fixture
def engine(policy) -> PolicyEngine:
    return PolicyEngine(policy)


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
    now: datetime = NOON,
    consents: tuple[str, ...] = ("whatsapp", "voice"),
    **payment_fields,
) -> DecisionContext:
    classification = Classification(
        cause=cause,
        confidence=0.9 if cause else 0.0,
        resolution=Resolution.DETERMINISTIC if cause else Resolution.UNRESOLVED,
    )
    context = builder.build(_payment(**payment_fields), classification, now)
    for channel in consents:
        context.notes[f"consent:{channel}"] = "true"
    return context


def _candidate(action: ActionKind, at: datetime = NOON, ev: int = 10000) -> Candidate:
    return Candidate(action=action, at=at, ev=ev, probability=0.4)


# ---------------------------------------------------------------------------
# Hard stops
# ---------------------------------------------------------------------------


def test_a_risk_blocked_payment_is_never_retried(gate, builder):
    """Retrying is an attempt to route around a control."""
    context = _context(builder, cause=FailureCause.RISK_BLOCKED)

    chosen, vetoes = gate.screen([_candidate(ActionKind.RETRY_NOW)], context)

    assert chosen is None
    assert vetoes[0].rule == "hard_stop:RISK_BLOCKED"


def test_a_risk_blocked_payment_is_not_even_mentioned_to_the_customer(gate, builder):
    """Contacting them leaks the existence of the block."""
    context = _context(builder, cause=FailureCause.RISK_BLOCKED)

    chosen, vetoes = gate.screen([_candidate(ActionKind.NUDGE_SMS)], context)

    assert chosen is None
    assert vetoes


def test_a_revoked_mandate_is_never_debited_again(gate, builder):
    context = _context(builder, cause=FailureCause.MANDATE_PROBLEM)

    chosen, _ = gate.screen([_candidate(ActionKind.RETRY_NOW)], context)

    assert chosen is None


def test_an_expired_instrument_may_be_messaged_but_not_retried(gate, builder):
    """The only path is a new instrument, and only the customer can supply it."""
    context = _context(builder, cause=FailureCause.INSTRUMENT_INVALID)

    retry, _ = gate.screen([_candidate(ActionKind.RETRY_NOW)], context)
    nudge, _ = gate.screen([_candidate(ActionKind.NUDGE_SMS)], context)

    assert retry is None
    assert nudge is not None


def test_a_deliberate_cancellation_gets_one_follow_up_at_most(gate, builder):
    """Anything beyond one is chasing someone who already said no."""
    first, _ = gate.screen(
        [_candidate(ActionKind.NUDGE_SMS)],
        _context(builder, cause=FailureCause.CUSTOMER_INTENT),
    )
    assert first is not None

    builder.note_contact("cust_00001", NOON - timedelta(days=1))

    second, vetoes = gate.screen(
        [_candidate(ActionKind.NUDGE_SMS)],
        _context(builder, cause=FailureCause.CUSTOMER_INTENT),
    )
    assert second is None
    assert "max_contacts" in vetoes[0].rule


# ---------------------------------------------------------------------------
# Attempt caps
# ---------------------------------------------------------------------------


def test_retries_stop_at_the_attempt_cap(gate, builder, rules):
    for _ in range(rules.attempts.max_per_payment):
        builder.note_attempt("pay_000001", succeeded=False)

    chosen, vetoes = gate.screen([_candidate(ActionKind.RETRY_NOW)], _context(builder))

    assert chosen is None
    assert vetoes[0].rule == "attempts:max_per_payment"


def test_consecutive_failures_trigger_a_cooling_off(gate, builder, rules):
    for _ in range(rules.attempts.cooling_off_after_failures):
        builder.note_attempt("pay_000001", succeeded=False)

    _, vetoes = gate.screen([_candidate(ActionKind.RETRY_NOW)], _context(builder))

    assert any(v.rule == "attempts:cooling_off" for v in vetoes)


def test_the_attempt_cap_does_not_block_contact(gate, builder, rules):
    """Running out of retries is not running out of options."""
    for _ in range(rules.attempts.max_per_payment + 2):
        builder.note_attempt("pay_000001", succeeded=False)

    chosen, _ = gate.screen([_candidate(ActionKind.NUDGE_SMS)], _context(builder))

    assert chosen is not None


# ---------------------------------------------------------------------------
# Contact limits
# ---------------------------------------------------------------------------


def test_nothing_is_sent_during_quiet_hours(gate, builder):
    context = _context(builder, now=MIDNIGHT)

    chosen, vetoes = gate.screen(
        [_candidate(ActionKind.NUDGE_SMS, at=MIDNIGHT)], context
    )

    assert chosen is None
    assert vetoes[0].rule == "contact:quiet_hours"


def test_quiet_hours_apply_to_email_too(gate, builder):
    """A 3am payment-failure notification is alarming however cheap the channel."""
    _, vetoes = gate.screen(
        [_candidate(ActionKind.NUDGE_EMAIL, at=MIDNIGHT)],
        _context(builder, now=MIDNIGHT),
    )
    assert vetoes


def test_retries_are_allowed_during_quiet_hours(gate, builder):
    """Quiet hours protect the customer's attention, not the issuer's."""
    chosen, _ = gate.screen(
        [_candidate(ActionKind.RETRY_NOW, at=MIDNIGHT)],
        _context(builder, now=MIDNIGHT),
    )
    assert chosen is not None


def test_the_weekly_contact_cap_holds(gate, builder, rules):
    for day in range(rules.contact.max_per_customer_per_7d):
        builder.note_contact("cust_00001", NOON - timedelta(days=day + 1))

    chosen, vetoes = gate.screen([_candidate(ActionKind.NUDGE_SMS)], _context(builder))

    assert chosen is None
    assert vetoes[0].rule == "contact:max_per_customer"


def test_contacts_must_be_spaced_apart(gate, builder):
    builder.note_contact("cust_00001", NOON - timedelta(hours=2))

    _, vetoes = gate.screen([_candidate(ActionKind.NUDGE_SMS)], _context(builder))

    assert any(v.rule == "contact:min_interval" for v in vetoes)


def test_channels_needing_consent_are_refused_without_it(gate, builder):
    context = _context(builder, consents=())

    chosen, vetoes = gate.screen([_candidate(ActionKind.NUDGE_WHATSAPP)], context)

    assert chosen is None
    assert "consent" in vetoes[0].rule


def test_consent_is_not_required_for_sms(gate, builder):
    chosen, _ = gate.screen(
        [_candidate(ActionKind.NUDGE_SMS)], _context(builder, consents=())
    )
    assert chosen is not None


# ---------------------------------------------------------------------------
# Outages
# ---------------------------------------------------------------------------


def _downtime(severity: Severity, end: int | None = None) -> DowntimeEntity:
    return DowntimeEntity(
        id="down_1",
        method=PaymentMethod.CARD,
        begin=int(NOON.timestamp()),
        end=end,
        status="started",
        severity=severity,
        instrument={"bank": "HDFC"},
    )


def test_retrying_into_a_severe_outage_is_refused(gate, builder):
    builder.note_downtime("payment.downtime.started", _downtime(Severity.HIGH))

    chosen, vetoes = gate.screen([_candidate(ActionKind.RETRY_NOW)], _context(builder))

    assert chosen is None
    assert vetoes[0].rule == "downtime:active"
    assert "cannot be refilled" in vetoes[0].why


def test_a_low_severity_outage_does_not_block_a_retry(gate, builder):
    builder.note_downtime("payment.downtime.started", _downtime(Severity.LOW))

    chosen, _ = gate.screen([_candidate(ActionKind.RETRY_NOW)], _context(builder))

    assert chosen is not None


def test_a_retry_scheduled_after_the_outage_clears_survives(gate, builder):
    """Waiting it out is the correct response and must not be vetoed."""
    ends = int((NOON + timedelta(hours=2)).timestamp())
    builder.note_downtime("payment.downtime.started", _downtime(Severity.HIGH, end=ends))

    chosen, _ = gate.screen(
        [_candidate(ActionKind.RETRY_SCHEDULED, at=NOON + timedelta(hours=3))],
        _context(builder),
    )

    assert chosen is not None


def test_an_outage_does_not_block_contacting_the_customer(gate, builder):
    builder.note_downtime("payment.downtime.started", _downtime(Severity.HIGH))

    chosen, _ = gate.screen([_candidate(ActionKind.NUDGE_SMS)], _context(builder))

    assert chosen is not None


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_small_payments_are_not_escalated(gate, builder):
    chosen, vetoes = gate.screen(
        [_candidate(ActionKind.ESCALATE_HUMAN)], _context(builder, amount=250000)
    )

    assert chosen is None
    assert vetoes[0].rule == "escalation:below_threshold"


def test_large_payments_may_be_escalated(gate, builder):
    chosen, _ = gate.screen(
        [_candidate(ActionKind.ESCALATE_HUMAN)], _context(builder, amount=9000000)
    )
    assert chosen is not None


def test_escalation_is_capped_per_run(gate, builder, rules):
    """An agent that escalates everything is not a product."""
    context = _context(builder, amount=9000000)

    for _ in range(rules.escalation.max_escalations_per_run):
        chosen, _ = gate.screen([_candidate(ActionKind.ESCALATE_HUMAN)], context)
        gate.note_executed(chosen)

    chosen, vetoes = gate.screen([_candidate(ActionKind.ESCALATE_HUMAN)], context)

    assert chosen is None
    assert vetoes[0].rule == "escalation:run_cap"


def test_run_caps_reset(gate, builder, rules):
    context = _context(builder, amount=9000000)
    for _ in range(rules.escalation.max_escalations_per_run):
        gate.note_executed(_candidate(ActionKind.ESCALATE_HUMAN))

    gate.reset()

    chosen, _ = gate.screen([_candidate(ActionKind.ESCALATE_HUMAN)], context)
    assert chosen is not None


# ---------------------------------------------------------------------------
# Structural behaviour
# ---------------------------------------------------------------------------


def test_a_veto_falls_through_to_the_next_permitted_option(gate, builder):
    """The property that stops compliance turning into paralysis.

    A blocked best option should hand off to the second best, not stop the agent
    doing anything at all.
    """
    builder.note_downtime("payment.downtime.started", _downtime(Severity.HIGH))

    ranked = [
        _candidate(ActionKind.RETRY_NOW, ev=50000),
        _candidate(ActionKind.NUDGE_SMS, ev=20000),
    ]

    chosen, vetoes = gate.screen(ranked, _context(builder))

    assert chosen.action is ActionKind.NUDGE_SMS
    assert len(vetoes) == 1


def test_every_refusal_on_the_way_down_is_recorded(gate, builder):
    """The refusal list is made of these."""
    context = _context(builder, cause=FailureCause.RISK_BLOCKED)

    ranked = [
        _candidate(ActionKind.RETRY_NOW),
        _candidate(ActionKind.NUDGE_SMS),
        _candidate(ActionKind.NUDGE_EMAIL),
    ]

    chosen, vetoes = gate.screen(ranked, context)

    assert chosen is None
    assert len(vetoes) == 3
    assert all(v.why for v in vetoes)


def test_a_veto_explains_itself_in_prose(gate, builder):
    """Someone reads these. "policy_violation_3f" would be useless."""
    context = _context(builder, cause=FailureCause.RISK_BLOCKED)

    _, vetoes = gate.screen([_candidate(ActionKind.RETRY_NOW)], context)

    assert len(vetoes[0].why) > 40
    assert "risk" in vetoes[0].why.lower()


def test_stop_is_never_vetoed(gate, builder):
    """Doing nothing must always remain available."""
    _, vetoes = gate.screen([_candidate(ActionKind.STOP)], _context(builder))
    assert vetoes == []


def test_a_hard_stop_naming_something_that_is_not_a_cause_is_rejected(rules):
    """The bug this validator exists for.

    `MANDATE_REVOKED` is a reason string, not a FailureCause. Keyed that way the
    rule matched nothing and a revoked mandate could have been debited again —
    with no error, no warning, and a passing test suite.
    """
    broken = rules.model_dump()
    broken["hard_stops"]["MANDATE_REVOKED"] = broken["hard_stops"]["MANDATE_PROBLEM"]

    with pytest.raises(ValueError, match="would never fire"):
        ComplianceConfig.model_validate(broken)


def test_every_configured_hard_stop_actually_resolves(rules):
    """Belt and braces: each key round-trips to a live rule."""
    assert rules.hard_stops
    for key in rules.hard_stops:
        assert rules.hard_stop_for(FailureCause(key)) is not None


# ---------------------------------------------------------------------------
# Deferral
# ---------------------------------------------------------------------------


def test_quiet_hours_produce_a_send_time_not_an_abandonment(builder, rules):
    """The right answer to "it is 3am" is 9am, not "give up on the money"."""
    when = next_permitted_contact_time(_context(builder, now=MIDNIGHT), rules)

    assert when is not None
    assert when.hour == rules.contact.quiet_hours.end.hour


def test_a_permitted_moment_is_returned_unchanged(builder, rules):
    assert next_permitted_contact_time(_context(builder, now=NOON), rules) == NOON


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_policy_and_gate_compose(engine, gate, builder):
    """The full path: price everything, then take the best permitted option."""
    builder.note_downtime("payment.downtime.started", _downtime(Severity.HIGH))

    context = _context(builder)
    decision = engine.decide(context)
    chosen, vetoes = gate.screen(decision.considered, context)

    assert vetoes, "a live severe outage should have blocked the retries"
    assert chosen is None or not chosen.action.is_retry
