"""Audit and refusals.

"Here is what we deliberately did not touch, and why" is a deliverable rather
than an error log, which means two things have to hold. The list has to be
accurate, and the record it is drawn from has to be demonstrably unedited — a
refusal list from a mutable log is a story, not evidence.

The tests that matter most here are the ones about honesty: that a rule showing
zero is explained rather than looking broken, and that the large headline figure
is not quietly presented as the cost of compliance when nothing has established
that.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from recoup.agent.config import ComplianceConfig
from recoup.eval import run_all
from recoup.eval.store import ledger_path, load_summary, open_ledger, save_summary
from recoup.web.app import create_app
from recoup.web.views import build_audit
from recoup.world.config import WorldConfig

ARM = "recoup_agent"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("audit")
    world = WorldConfig.load()

    results, ledger = run_all(world, ledger_path=ledger_path(directory))
    digests = {m.arm: ledger.digest(m.arm) for m in results}
    ledger.close()

    save_summary(
        results,
        seed=world.run.seed,
        horizon_days=world.run.horizon_days,
        batch_size=world.run.batch_size,
        margin=world.merchant_margin,
        directory=directory,
        digests=digests,
    )
    return directory


@pytest.fixture(scope="module")
def audit(run_dir):
    ledger = open_ledger(run_dir)
    summary = load_summary(run_dir)
    view = build_audit(
        ledger,
        ARM,
        summary.digests[ARM],
        configured_hard_stops=list(ComplianceConfig.load().hard_stops),
    )
    ledger.close()
    return view


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_the_stream_hashes_to_what_it_hashed_to_when_written(audit):
    """The claim an audit trail makes, checked rather than asserted."""
    assert audit.replay_ok is True
    assert audit.digest


def test_a_wrong_recorded_digest_fails_verification(run_dir):
    """The check has to be capable of failing, or it proves nothing."""
    ledger = open_ledger(run_dir)
    view = build_audit(ledger, ARM, "0" * 64)
    ledger.close()

    assert view.replay_ok is False


def test_append_only_is_enforced_by_the_database(audit):
    assert set(audit.triggers) == {"ledger_no_update", "ledger_no_delete"}


def test_those_triggers_actually_refuse_writes(run_dir):
    """Reading trigger names off the schema is not the same as them working."""
    connection = sqlite3.connect(ledger_path(run_dir))

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("UPDATE ledger SET amount = 1")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("DELETE FROM ledger")

    connection.close()


def test_event_counts_cover_the_whole_stream(audit):
    assert audit.total_events == sum(audit.by_kind.values())
    assert audit.by_kind["observed"] > 0
    assert audit.by_kind["vetoed"] > 0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_rules_are_aggregated_with_their_exposure(audit):
    assert audit.rules
    for rule in audit.rules:
        assert rule.refusals >= rule.payments, "a payment cannot be refused less than once"
        assert rule.why, "a refusal that cannot explain itself is not auditable"
        assert rule.actions


def test_rules_are_ordered_by_money_not_count(audit):
    amounts = [r.amount for r in audit.rules]
    assert amounts == sorted(amounts, reverse=True)


def test_machine_rule_keys_get_human_names(audit):
    """Someone reads these. "contact:min_interval" is not a sentence."""
    for rule in audit.rules:
        assert rule.headline
        if rule.rule == "contact:min_interval":
            assert rule.headline != rule.rule


def test_refused_payments_split_into_abandoned_and_recovered_anyway(audit):
    assert audit.payments_touched == len(audit.refused)
    assert audit.abandoned_count + audit.recovered_anyway_count == audit.payments_touched

    for row in audit.refused:
        assert row.abandoned == (row.outcome != "recovered")


def test_compliance_blocking_one_route_does_not_end_the_payment(audit):
    """A veto should fall through to the next permitted option, not to nothing."""
    assert audit.recovered_anyway_count > 0
    assert audit.recovered_anyway_amount > 0


def test_the_list_is_ordered_by_money(audit):
    amounts = [r.amount for r in audit.refused]
    assert amounts == sorted(amounts, reverse=True)


# ---------------------------------------------------------------------------
# Honesty
# ---------------------------------------------------------------------------


def test_hard_stops_that_never_fired_are_reported_not_hidden(audit):
    """A rule showing zero looks inactive; the reason it is zero is the point.

    The policy declines to propose retries for causes no retry can fix, so
    compliance is never asked to refuse them. Both layers working looks, from the
    gate's side, like one layer doing nothing.
    """
    assert "hard_stop:RISK_BLOCKED" in audit.dormant
    assert "hard_stop:INSTRUMENT_INVALID" in audit.dormant


def test_a_rule_that_fired_is_not_listed_as_dormant(audit):
    fired = {r.rule for r in audit.rules}

    for dormant in audit.dormant:
        assert not any(rule.startswith(dormant) for rule in fired)


def test_dormant_detection_handles_suffixed_rules(audit):
    """`hard_stop:CUSTOMER_INTENT:max_contacts` firing means that stop is active."""
    fired = {r.rule for r in audit.rules}

    if any(r.startswith("hard_stop:CUSTOMER_INTENT") for r in fired):
        assert "hard_stop:CUSTOMER_INTENT" not in audit.dormant


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@pytest.fixture
def client(run_dir) -> TestClient:
    return TestClient(create_app(data_dir=run_dir))


def test_the_audit_page_renders(client):
    response = client.get(f"/audit?arm={ARM}")

    assert response.status_code == 200
    assert "What we deliberately did not do" in response.text
    assert "verified" in response.text


def test_the_page_refuses_to_call_the_figure_a_compliance_cost(client):
    """A number that flatters or alarms in the UI and qualifies itself in a doc
    is a number nobody reads the doc for."""
    html = client.get(f"/audit?arm={ARM}").text

    assert "not the cost of compliance" in html
    assert "counterfactual" in html


def test_dormant_rules_are_explained_on_the_page(client):
    html = client.get(f"/audit?arm={ARM}").text

    assert "Never had to fire" in html
    assert "hard_stop:RISK_BLOCKED" in html


def test_overlapping_amounts_are_disclosed(client):
    """A payment can be stopped by several rules, so the column does not sum."""
    assert "do not sum" in client.get(f"/audit?arm={ARM}").text


def test_the_audit_page_without_a_run_explains_itself(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path / "empty"))
    response = client.get("/audit")

    assert response.status_code == 200
    assert "No run to audit yet" in response.text
