"""The audit trail.

Two claims are tested here rather than trusted: that the ledger cannot be edited,
and that two identical runs produce byte-identical streams. Both underpin
statements this project makes to a reader — "here is why we charged this
customer" and "clone it and get the same number" — and both are the kind of claim
that quietly stops being true midway through a build.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from recoup.ledger.events import EventKind, Ledger, LedgerEvent
from recoup.ledger.replay import final_state, first_divergence, fold, state_at

START = datetime(2026, 6, 1, 9, 0)


def _event(offset_hours: int, kind: EventKind, **fields) -> LedgerEvent:
    base = {
        "at": START + timedelta(hours=offset_hours),
        "kind": kind,
        "arm": "agent",
        "payment_id": "pay_000001",
    }
    return LedgerEvent(**{**base, **fields})


@pytest.fixture
def ledger() -> Ledger:
    with Ledger() as instance:
        yield instance


def _seed(ledger: Ledger, arm: str = "agent") -> None:
    ledger.extend(
        [
            _event(0, EventKind.OBSERVED, arm=arm, amount=45000, customer_id="cust_1"),
            _event(0, EventKind.CLASSIFIED, arm=arm, data={"cause": "INSUFFICIENT_FUNDS"}),
            _event(1, EventKind.DECIDED, arm=arm, data={"action": "RETRY_SCHEDULED", "ev": 8200}),
            _event(2, EventKind.EXECUTED, arm=arm, data={"succeeded": False}),
            _event(26, EventKind.DECIDED, arm=arm, data={"action": "NUDGE_SMS", "ev": 3100}),
            _event(26, EventKind.VETOED, arm=arm, data={"rule": "quiet_hours"}),
            _event(35, EventKind.EXECUTED, arm=arm, data={"succeeded": True}),
            _event(35, EventKind.RECOVERED, arm=arm, amount=45000),
        ]
    )


# ---------------------------------------------------------------------------
# Append-only, enforced by the database
# ---------------------------------------------------------------------------


def test_rows_cannot_be_updated(ledger):
    """A ledger that is append-only by convention stops being append-only.

    The trigger is what makes the audit trail an audit trail rather than a log
    someone can tidy up while debugging.
    """
    ledger.append(_event(0, EventKind.OBSERVED, amount=1000))

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger._conn.execute("UPDATE ledger SET amount = 999 WHERE seq = 1")


def test_rows_cannot_be_deleted(ledger):
    ledger.append(_event(0, EventKind.OBSERVED, amount=1000))

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger._conn.execute("DELETE FROM ledger WHERE seq = 1")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_events_come_back_in_insertion_order(ledger):
    _seed(ledger)
    kinds = [e.kind for e in ledger.events()]

    assert kinds[0] is EventKind.OBSERVED
    assert kinds[-1] is EventKind.RECOVERED
    assert [e.seq for e in ledger.events()] == sorted(e.seq for e in ledger.events())


def test_the_story_of_one_payment_is_retrievable(ledger):
    """What the case-detail screen renders."""
    _seed(ledger)
    story = ledger.story_of("pay_000001", arm="agent")

    assert len(story) == 8
    assert story[0].kind is EventKind.OBSERVED
    assert story[-1].kind is EventKind.RECOVERED


def test_arms_are_isolated(ledger):
    _seed(ledger, arm="agent")
    _seed(ledger, arm="baseline")

    assert ledger.arms() == ["agent", "baseline"]
    assert ledger.count(arm="agent") == 8
    assert all(e.arm == "baseline" for e in ledger.events(arm="baseline"))


def test_filtering_by_kind(ledger):
    _seed(ledger)
    assert ledger.count(kind=EventKind.DECIDED) == 2
    assert ledger.count(kind=EventKind.VETOED) == 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_identical_streams_have_identical_digests():
    with Ledger() as left, Ledger() as right:
        _seed(left)
        _seed(right)
        assert left.digest() == right.digest()


def test_a_changed_value_changes_the_digest():
    with Ledger() as left, Ledger() as right:
        _seed(left)
        _seed(right)
        right.append(_event(40, EventKind.STOPPED, data={"reason": "cap reached"}))

        assert left.digest() != right.digest()


def test_digest_ignores_dict_ordering():
    """Key order must not affect the hash, or reproduction is luck."""
    with Ledger() as left, Ledger() as right:
        left.append(_event(0, EventKind.DECIDED, data={"action": "RETRY_NOW", "ev": 100}))
        right.append(_event(0, EventKind.DECIDED, data={"ev": 100, "action": "RETRY_NOW"}))

        assert left.digest() == right.digest()


def test_first_divergence_points_at_the_event_that_moved():
    """A broken reproduction is useless as a diff of half a million rows."""
    with Ledger() as left, Ledger() as right:
        _seed(left)
        _seed(right)
        right.append(_event(40, EventKind.STOPPED, data={"reason": "cap reached"}))

        divergence = first_divergence(left, right)

        assert divergence is not None
        assert divergence.index == 8
        assert divergence.left is None
        assert divergence.right.kind is EventKind.STOPPED
        assert "missing on the left" in divergence.describe()


def test_identical_ledgers_do_not_diverge():
    with Ledger() as left, Ledger() as right:
        _seed(left)
        _seed(right)
        assert first_divergence(left, right) is None


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_folding_the_stream_reconstructs_the_run(ledger):
    _seed(ledger)
    state = final_state(ledger, arm="agent")

    assert state.observed == 1
    assert state.classified == 1
    assert state.decisions == 2
    assert state.vetoes == 1
    assert state.executions == 2
    assert state.recovered_count == 1
    assert state.recovered_paise == 45000
    assert state.amount_at_risk == 45000
    assert state.recovered_share == 1.0


def test_vetoes_are_counted_by_rule(ledger):
    """The refusal list depends on refusals being recorded positively."""
    _seed(ledger)
    state = final_state(ledger, arm="agent")

    assert state.veto_reasons == {"quiet_hours": 1}


def test_state_at_shows_what_was_known_then_not_now(ledger):
    """What the scrubber needs.

    Seeking to hour 3 must show a payment still open, even though the run later
    recovered it.
    """
    _seed(ledger)

    midway = state_at(ledger, arm="agent", when=START + timedelta(hours=3))

    assert midway.recovered_count == 0
    assert midway.open_payments == {"pay_000001"}
    assert midway.executions == 1

    end = final_state(ledger, arm="agent")
    assert end.recovered_count == 1
    assert end.open_payments == set()


def test_an_unclassified_failure_is_folded_separately(ledger):
    ledger.extend(
        [
            _event(0, EventKind.OBSERVED, amount=1000),
            _event(0, EventKind.CLASSIFIED, data={"cause": None}),
        ]
    )
    state = final_state(ledger, arm="agent")

    assert state.classified == 0
    assert state.unresolved == 1


def test_folding_an_empty_stream_is_safe():
    assert fold([]).observed == 0


def test_a_file_backed_ledger_survives_reopening(tmp_path):
    path = tmp_path / "ledger.db"

    with Ledger(path) as first:
        _seed(first)
        digest = first.digest()

    with Ledger(path) as reopened:
        assert reopened.count() == 8
        assert reopened.digest() == digest
