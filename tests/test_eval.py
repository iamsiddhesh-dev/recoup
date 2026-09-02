"""The evaluation harness.

The headline claim is incremental recovery over a naive baseline. These tests
guard the things that would make that claim dishonest rather than merely wrong:
an unfair comparison, a baseline held to a lower standard, arms that see different
worlds, or recoveries counted after the horizon closed.

A wrong number here is worse than a bug, because it looks like a result.
"""

from __future__ import annotations

import pytest

from recoup.agent.config import ComplianceConfig, PolicyConfig
from recoup.eval import BASELINE, run_all
from recoup.eval.arms import ContactOnlyArm, NaiveRetryArm, RecoupArm, build_arms
from recoup.eval.metrics import Comparison
from recoup.ledger.events import EventKind
from recoup.world.config import WorldConfig


@pytest.fixture(scope="module")
def world() -> WorldConfig:
    return WorldConfig.load()


@pytest.fixture(scope="module")
def run(world):
    results, ledger = run_all(world)
    yield results, ledger
    ledger.close()


@pytest.fixture(scope="module")
def results(run):
    return run[0]


@pytest.fixture(scope="module")
def ledger(run):
    return run[1]


def _arm(results, name):
    return next(m for m in results if m.arm == name)


# ---------------------------------------------------------------------------
# Fairness of the comparison
# ---------------------------------------------------------------------------


def test_every_arm_sees_the_same_failures(results):
    """Arms must differ by decisions, not by what they were shown."""
    observed = {m.observed for m in results}
    at_risk = {m.amount_at_risk for m in results}

    assert len(observed) == 1
    assert len(at_risk) == 1


def test_the_baseline_is_held_to_the_same_compliance_rules(results):
    """Naive, not reckless.

    If the baseline could retry revoked mandates and message people at 3am, part
    of the agent's measured advantage would be "we follow the rules and they do
    not" — a rigged comparison rather than a product difference.
    """
    baseline = _arm(results, BASELINE)

    assert baseline.vetoes > 0
    assert any("hard_stop" in rule for rule in baseline.veto_by_rule)


def test_every_arm_runs_through_a_compliance_gate(world):
    policy = PolicyConfig.load()
    compliance = ComplianceConfig.load()

    for arm in build_arms(policy, compliance):
        assert hasattr(arm, "gate")


def test_arms_do_not_share_attempt_state(ledger):
    """A fresh adapter per arm, or one arm's retries exhaust another's caps."""
    for arm in ("naive_baseline", "recoup_agent"):
        executed = list(ledger.events(arm=arm, kind=EventKind.EXECUTED))
        assert executed


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


def test_the_baseline_recovers_something(results):
    """A zero baseline makes any lift look infinite and means the arm is broken.

    It was: the runner re-derived the scheduled decision instead of carrying it,
    so the naive arm deferred its retry forever and never executed one.
    """
    baseline = _arm(results, BASELINE)

    assert baseline.recovered_count > 0
    assert baseline.recovered_paise > 0
    assert baseline.actions > 0


def test_the_agent_beats_the_naive_baseline(results):
    comparison = Comparison(arm=_arm(results, "recoup_agent"), baseline=_arm(results, BASELINE))

    assert comparison.incremental_paise > 0
    assert comparison.incremental_net > 0


def test_the_agent_beats_contact_only_on_fewer_contacts(results):
    """The part of the result that is judgment rather than coverage.

    Most of the lift over the naive baseline comes from addressing failures it
    structurally ignores. The interesting comparison is against an arm that also
    contacts customers: the agent should recover more while sending less.
    """
    agent = _arm(results, "recoup_agent")
    contact = _arm(results, "contact_only")

    assert agent.recovered_paise > contact.recovered_paise
    assert agent.contacts < contact.contacts
    assert agent.contacts_per_recovery < contact.contacts_per_recovery


def test_recovery_stays_within_the_money_at_risk(results):
    """An arm cannot recover more than was failing."""
    for metrics in results:
        assert metrics.recovered_paise <= metrics.amount_at_risk
        assert metrics.recovered_count <= metrics.observed


def test_net_is_margin_minus_cost_not_gross(results):
    for metrics in results:
        assert metrics.net_paise == metrics.margin_recovered - metrics.cost_paise
        assert metrics.margin_recovered < metrics.recovered_paise


# ---------------------------------------------------------------------------
# Behaviour worth asserting
# ---------------------------------------------------------------------------


def test_the_agent_uses_more_than_one_channel(results):
    """Guards the bug where every channel looked equally effective.

    Priced identically, the optimiser picks the cheapest one forever — the first
    run sent 1,582 emails and nothing else. Correct arithmetic over a wrong model.
    """
    agent = _arm(results, "recoup_agent")
    channels = {a for a in agent.actions_by_kind if a.startswith("NUDGE")}

    assert len(channels) >= 3


def test_the_agent_both_retries_and_contacts(results):
    agent = _arm(results, "recoup_agent")

    assert any(a.startswith("RETRY") for a in agent.actions_by_kind)
    assert any(a.startswith("NUDGE") for a in agent.actions_by_kind)


def test_the_naive_baseline_never_contacts_anyone(results):
    baseline = _arm(results, BASELINE)

    assert baseline.contacts == 0
    assert all(not a.startswith("NUDGE") for a in baseline.actions_by_kind)


def test_the_refusal_list_is_readable(results):
    """Vetoes are a deliverable, so they have to mean something.

    An earlier version screened every retry-time variant and logged each refusal,
    producing 80,945 vetoes — fifty near-identical rows per payment, differing
    only by scheduled hour. Deduplicated by rule, the list reads as a compliance
    report.
    """
    agent = _arm(results, "recoup_agent")

    assert agent.vetoes < agent.observed * 3
    assert len(agent.veto_by_rule) >= 4


def test_escalation_respects_its_run_cap(results):
    cap = ComplianceConfig.load().escalation.max_escalations_per_run

    for metrics in results:
        assert metrics.actions_by_kind.get("ESCALATE_HUMAN", 0) <= cap


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_the_same_seed_produces_the_same_result(world):
    first, first_ledger = run_all(world)
    second, second_ledger = run_all(world)

    for a, b in zip(first, second, strict=True):
        assert a.recovered_paise == b.recovered_paise
        assert a.cost_paise == b.cost_paise
        assert a.vetoes == b.vetoes

    assert first_ledger.digest() == second_ledger.digest()

    first_ledger.close()
    second_ledger.close()


def test_a_different_seed_moves_the_result(world):
    other = world.model_copy(deep=True)
    other.run.seed = world.run.seed + 7

    baseline_a, ledger_a = run_all(world)
    baseline_b, ledger_b = run_all(other)

    assert _arm(baseline_a, BASELINE).recovered_paise != _arm(baseline_b, BASELINE).recovered_paise

    ledger_a.close()
    ledger_b.close()


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------


def test_the_arms_are_what_the_writeup_claims(world):
    policy = PolicyConfig.load()
    compliance = ComplianceConfig.load()
    arms = build_arms(policy, compliance)

    assert [type(a) for a in arms] == [NaiveRetryArm, ContactOnlyArm, RecoupArm]
    assert arms[0].name == BASELINE
    assert all(a.description for a in arms)
