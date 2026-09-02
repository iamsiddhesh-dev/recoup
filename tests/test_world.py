"""The simulator has to be boring in the right ways.

Two properties matter more than anything else here:

* **Determinism.** Every number this project reports is regenerated from a seed.
  If the same seed produces a different batch, `reproduce` is a lie.
* **The wire carries no answers.** The agent must not be able to read the failure
  cause off an event. That is checked structurally, not by reading the code.

The rest are sanity bands on the generated distributions — loose enough not to be
brittle, tight enough to catch the class of bug where a parameter is off by a
factor of a hundred.
"""

from __future__ import annotations

import statistics
from datetime import timedelta

import pytest

from recoup.domain import FailureCause, PaymentMethod, PaymentStatus
from recoup.world.clock import Timeline
from recoup.world.config import WorldConfig
from recoup.world.generator import build_batch


@pytest.fixture(scope="module")
def config() -> WorldConfig:
    return WorldConfig.load()


@pytest.fixture(scope="module")
def run(config):
    return build_batch(config)


@pytest.fixture(scope="module")
def batch(run):
    return run[0]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_an_identical_batch(config):
    first, _, _ = build_batch(config)
    second, _, _ = build_batch(config)

    assert [p.model_dump() for p in first.payments] == [p.model_dump() for p in second.payments]


def test_a_different_seed_produces_a_different_batch(config):
    other = config.model_copy(deep=True)
    other.run.seed = config.run.seed + 1
    first, _, _ = build_batch(config)
    second, _, _ = build_batch(other)

    assert first.amount_at_risk != second.amount_at_risk


def test_downtime_calendar_is_deterministic(config):
    _, _, first = build_batch(config)
    _, _, second = build_batch(config)
    assert first.windows == second.windows


# ---------------------------------------------------------------------------
# The wire carries no answers
# ---------------------------------------------------------------------------


def test_no_failure_cause_is_serialised_onto_a_payment(batch):
    """The single most important test in this file.

    If `cause` ever appears on the entity the agent receives, root-cause
    classification becomes a dictionary lookup and every recovery number this
    project reports is meaningless.
    """
    causes = {c.value for c in FailureCause}

    for payment in batch.failures[:200]:
        dumped = payment.model_dump(mode="json")
        assert "cause" not in dumped
        assert not (causes & {str(v) for v in dumped.values()}), (
            f"a FailureCause leaked onto payment {payment.id}: {dumped}"
        )


def test_truth_is_held_separately_and_does_carry_the_cause(batch):
    failed = batch.failures[0]
    truth = batch.truth_for(failed.id)
    assert truth.failed is True
    assert truth.cause is not None


# ---------------------------------------------------------------------------
# Generated distributions
# ---------------------------------------------------------------------------


def test_batch_size_matches_config(batch, config):
    assert len(batch.payments) == config.run.batch_size


def test_method_mix_is_approximately_respected(batch, config):
    counts = {method: 0 for method in PaymentMethod}
    for payment in batch.payments:
        counts[payment.method] += 1

    for method, expected in config.method_mix.items():
        observed = counts[method] / len(batch.payments)
        assert abs(observed - expected) < 0.03, f"{method}: {observed:.3f} vs {expected:.3f}"


def test_overall_failure_rate_is_plausible(batch):
    rate = len(batch.failures) / len(batch.payments)
    assert 0.24 < rate < 0.38, f"failure rate {rate:.3f} outside plausible band"


@pytest.mark.parametrize(
    ("method", "low_rupees", "high_rupees"),
    [
        (PaymentMethod.UPI, 150, 1200),
        (PaymentMethod.CARD, 700, 5000),
        (PaymentMethod.NETBANKING, 1200, 10000),
        (PaymentMethod.WALLET, 100, 800),
        (PaymentMethod.EMANDATE, 250, 1800),
    ],
)
def test_median_ticket_size_is_sane(batch, method, low_rupees, high_rupees):
    """Guards the rupees-versus-paise bug.

    `mu` is ln(median in paise). Writing ln(median in rupees) instead is a silent
    factor-of-100 error that makes every recovery look worthless without failing
    anything. It happened once; this stops it happening again.
    """
    amounts = [p.amount for p in batch.payments if p.method is method]
    median_rupees = statistics.median(amounts) / 100

    assert low_rupees < median_rupees < high_rupees, (
        f"{method} median is ₹{median_rupees:,.0f}, expected between "
        f"₹{low_rupees:,} and ₹{high_rupees:,}"
    )


def test_amounts_respect_configured_bounds(batch, config):
    for payment in batch.payments:
        spec = config.amounts[payment.method]
        assert spec.min <= payment.amount <= spec.max


# ---------------------------------------------------------------------------
# Error fields
# ---------------------------------------------------------------------------


def test_failed_payments_carry_complete_razorpay_error_fields(batch):
    for payment in batch.failures:
        assert payment.error_code
        assert payment.error_source
        assert payment.error_step
        assert payment.error_reason
        assert payment.error_description


def test_captured_payments_carry_no_error_fields(batch):
    captured = [p for p in batch.payments if p.status is PaymentStatus.CAPTURED]
    assert captured
    for payment in captured:
        assert payment.error_reason is None
        assert payment.error_code is None


def test_error_source_and_step_come_from_the_documented_taxonomy(batch, config):
    """Only values Razorpay actually documents may appear on the wire."""
    allowed: set[tuple[str, str, str]] = {
        (str(method), entry.source, entry.step)
        for method, entries in config.error_taxonomy.items()
        for entry in entries
    }

    for payment in batch.failures:
        triple = (str(payment.method), payment.error_source, payment.error_step)
        assert triple in allowed, f"undocumented source/step on {payment.id}: {triple}"


def test_every_failure_links_back_to_a_customer(batch, run):
    _, population, _ = run
    for payment in batch.failures[:500]:
        assert payment.customer_ref is not None
        assert population.get(payment.customer_ref) is not None


# ---------------------------------------------------------------------------
# Downtime
# ---------------------------------------------------------------------------


def test_downtime_emits_matched_started_and_resolved_events(run):
    _, _, issuers = run
    events = issuers.events()

    started = [e for e in events if e[1] == "payment.downtime.started"]
    resolved = [e for e in events if e[1] == "payment.downtime.resolved"]

    assert len(started) == len(resolved) == len(issuers.windows)
    assert {e[2].id for e in started} == {e[2].id for e in resolved}


def test_downtime_events_are_in_chronological_order(run):
    _, _, issuers = run
    times = [when for when, _, _ in issuers.events()]
    assert times == sorted(times)


def test_downtime_suppresses_success(run, config):
    _, _, issuers = run
    window = issuers.windows[0]
    midpoint = window.begin + (window.end - window.begin) / 2

    during = issuers.multiplier_at(midpoint, window.method, window.issuer_code)
    after = issuers.multiplier_at(
        window.end + timedelta(hours=1), window.method, window.issuer_code
    )

    assert during < 1.0
    assert after == 1.0


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


def test_timeline_yields_in_chronological_order(config):
    timeline = Timeline(config.run.start_at)
    for offset in (5, 1, 9, 3):
        timeline.schedule(config.run.start_at + timedelta(hours=offset), offset)

    assert [payload for _, payload in timeline.run()] == [1, 3, 5, 9]


def test_timeline_breaks_ties_by_insertion_order(config):
    timeline = Timeline(config.run.start_at)
    at = config.run.start_at + timedelta(hours=1)
    for label in ("first", "second", "third"):
        timeline.schedule(at, label)

    assert [payload for _, payload in timeline.run()] == ["first", "second", "third"]


def test_timeline_refuses_to_schedule_into_the_past(config):
    timeline = Timeline(config.run.start_at)
    timeline.schedule(config.run.start_at + timedelta(hours=2), "advance")
    list(timeline.run())

    with pytest.raises(ValueError, match="past"):
        timeline.schedule(config.run.start_at, "too late")


def test_handlers_may_schedule_while_the_timeline_runs(config):
    timeline = Timeline(config.run.start_at)
    timeline.schedule(config.run.start_at + timedelta(hours=1), "first")

    seen = []
    for _, payload in timeline.run():
        seen.append(payload)
        if payload == "first":
            timeline.schedule_after(timedelta(hours=1), "second")

    assert seen == ["first", "second"]


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def test_population_size_and_segments(run, config):
    _, population, _ = run
    assert len(population) == config.customers.count

    segments = {c.segment for c in population.customers}
    assert segments == set(config.customers.segments)


def test_contact_details_are_non_routable(run):
    """Synthetic identifiers must be unusable, not merely fake."""
    _, population, _ = run
    for customer in population.customers[:200]:
        assert customer.email.endswith("@example.invalid")
        assert customer.contact.startswith("+91555")
