"""Turning ledger events into what a screen shows.

The case view is where Recoup's central claim gets tested in the most direct way
available: can someone open one payment and follow the reasoning end to end,
including the options not taken and the rules that stopped them, without reading
code? These tests pin the parts of that which are easy to break quietly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recoup.eval import run_all
from recoup.eval.store import ledger_path, open_ledger, save_summary
from recoup.ledger.events import EventKind
from recoup.web.app import create_app
from recoup.web.views import build_case, build_queue, queue_facets
from recoup.world.config import WorldConfig

ARM = "recoup_agent"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("views")
    world = WorldConfig.load()

    results, ledger = run_all(world, ledger_path=ledger_path(directory))
    ledger.close()

    save_summary(
        results,
        seed=world.run.seed,
        horizon_days=world.run.horizon_days,
        batch_size=world.run.batch_size,
        margin=world.merchant_margin,
        directory=directory,
    )
    return directory


@pytest.fixture(scope="module")
def ledger(run_dir):
    instance = open_ledger(run_dir)
    yield instance
    instance.close()


@pytest.fixture(scope="module")
def rows(ledger):
    return build_queue(ledger, ARM, limit=500)


@pytest.fixture(scope="module")
def recovered_case(ledger, rows):
    """A payment that took several actions before it worked.

    Trivial cases prove nothing — the interesting story is the one where the
    agent tried, failed, and changed approach.
    """
    busy = next(r for r in rows if r.outcome == "recovered" and r.actions >= 2)
    return build_case(ledger, busy.payment_id, ARM)


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


def test_the_queue_has_one_row_per_failure(ledger, rows):
    observed = {e.payment_id for e in ledger.events(arm=ARM, kind=EventKind.OBSERVED)}

    assert rows
    assert len({r.payment_id for r in rows}) == len(rows)
    assert {r.payment_id for r in rows} <= observed


def test_the_queue_is_ordered_by_money(rows):
    """A queue ordered by payment id is a queue nobody reads twice."""
    amounts = [r.amount for r in rows]
    assert amounts == sorted(amounts, reverse=True)


def test_filtering_by_outcome(ledger):
    recovered = build_queue(ledger, ARM, outcome="recovered", limit=500)

    assert recovered
    assert all(r.outcome == "recovered" for r in recovered)
    assert all(r.recovered_paise > 0 for r in recovered)


def test_filtering_by_cause(ledger):
    filtered = build_queue(ledger, ARM, cause="INSUFFICIENT_FUNDS", limit=500)

    assert filtered
    assert all(r.cause == "INSUFFICIENT_FUNDS" for r in filtered)


def test_facets_only_offer_values_that_exist(ledger):
    """An empty filter option is a dead end the UI should never present."""
    facets = queue_facets(ledger, ARM)

    assert facets["causes"]
    for cause in facets["causes"]:
        assert build_queue(ledger, ARM, cause=cause, limit=5)


def test_rows_carry_the_cost_of_being_worked(rows):
    worked = [r for r in rows if r.actions > 0]
    assert worked
    assert any(r.cost_paise > 0 for r in worked)


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------


def test_an_unknown_payment_has_no_case(ledger):
    assert build_case(ledger, "pay_does_not_exist", ARM) is None


def test_a_case_opens_with_the_failure_and_its_classification(recovered_case):
    kinds = [e.kind for e in recovered_case.entries]

    assert kinds[0] is EventKind.OBSERVED
    assert kinds[1] is EventKind.CLASSIFIED
    assert recovered_case.amount > 0
    assert recovered_case.method


def test_a_case_is_in_chronological_order(recovered_case):
    times = [e.at for e in recovered_case.entries]
    assert times == sorted(times)


def test_a_recovered_case_ends_recovered(recovered_case):
    assert recovered_case.outcome == "recovered"
    assert recovered_case.recovered_paise == recovered_case.amount
    assert recovered_case.entries[-1].kind is EventKind.RECOVERED


def test_every_action_has_a_decision_that_explains_it(recovered_case):
    """The claim is that every money action is explainable.

    An earlier version logged a decision only on first sight of a payment, so a
    case history showed the opening move's arithmetic and then three actions that
    appeared from nowhere. A cheaper audit trail that explains only the first
    action is not a smaller version of the claim, it is a false one.
    """
    decisions = [e for e in recovered_case.entries if e.kind is EventKind.DECIDED]
    executions = [e for e in recovered_case.entries if e.kind is EventKind.EXECUTED]

    assert len(executions) >= 2
    assert len(decisions) >= len(executions)


def test_a_decision_shows_its_arithmetic(recovered_case):
    decided = next(e for e in recovered_case.entries if e.working and e.working.steps)
    working = decided.working

    assert working.ev != 0
    assert working.reason
    assert [s.key for s in working.steps]
    assert any(s.is_money for s in working.steps)


def test_the_arithmetic_actually_adds_up(recovered_case):
    """The sum on screen has to reconstruct the number beside it.

    Money is the subject; a breakdown that does not reproduce its own total is
    worse than no breakdown, because it invites a reader to check and then
    quietly loses their trust.
    """
    for entry in recovered_case.entries:
        working = entry.working
        if not working or not working.steps:
            continue

        by_key = {s.key: s.value for s in working.steps}

        if working.action.startswith("RETRY"):
            total = by_key["gross"] * by_key["decay"] - by_key["cost"]
        elif working.action.startswith("NUDGE"):
            total = by_key["gross"] - by_key["cost"] - by_key.get("annoyance", 0)
        else:
            continue

        assert abs(total - working.ev) <= 2, f"{working.action} sum does not reconstruct its total"


def test_alternatives_are_deduplicated_by_action(recovered_case):
    """The policy scores one action at many candidate times.

    Listed raw that reads as "RETRY_SCHEDULED ₹486, RETRY_SCHEDULED ₹486" and
    looks like a glitch rather than a schedule.
    """
    for entry in recovered_case.entries:
        if not entry.working:
            continue
        actions = [a["action"] for a in entry.working.alternatives]
        assert len(actions) == len(set(actions))
        assert entry.working.action not in actions


def test_a_case_reports_what_it_cost(recovered_case):
    assert recovered_case.cost_paise > 0
    assert recovered_case.net_paise == (
        recovered_case.recovered_paise - recovered_case.cost_paise
    )


def test_a_stopped_case_says_why(ledger, rows):
    stopped = next(r for r in rows if r.outcome == "stopped")
    case = build_case(ledger, stopped.payment_id, ARM)

    assert case.outcome == "stopped"
    assert case.recovered_paise == 0
    assert any(e.kind is EventKind.STOPPED and e.detail for e in case.entries)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client(run_dir) -> TestClient:
    return TestClient(create_app(data_dir=run_dir))


def test_the_queue_page_renders(client):
    response = client.get(f"/queue?arm={ARM}")

    assert response.status_code == 200
    assert "Failures" in response.text
    assert "/case/pay_" in response.text


def test_the_queue_page_survives_filters_matching_nothing(client):
    response = client.get(f"/queue?arm={ARM}&outcome=recovered&cause=RISK_BLOCKED")

    assert response.status_code == 200
    assert "Nothing matches those filters" in response.text


def test_a_case_page_renders_its_working(client, recovered_case):
    response = client.get(f"/case/{recovered_case.payment_id}?arm={ARM}")

    assert response.status_code == 200
    assert "expected value" in response.text
    assert "contribution margin" in response.text
    assert "also considered" in response.text


def test_an_unknown_case_is_a_404_not_a_crash(client):
    assert client.get(f"/case/pay_nope?arm={ARM}").status_code == 404


def test_a_case_page_without_a_run_is_a_404(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path / "empty"))
    assert client.get("/case/pay_000001").status_code == 404


def test_small_amounts_keep_their_paise(client, recovered_case):
    """A WhatsApp message costs ₹0.35. Rounded to ₹0 the sum stops adding up."""
    import re

    response = client.get(f"/case/{recovered_case.payment_id}?arm={ARM}")

    assert re.search(r"₹\d+\.\d{2}", response.text), (
        "expected at least one sub-hundred-rupee figure to keep its paise"
    )
