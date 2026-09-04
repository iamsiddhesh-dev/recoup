"""The replay frames behind the control room's scrubber.

A scrubber is only honest if every frame is the run as it actually stood at that
moment. The risks are all in the folding: totals that go backwards, frames that
track event volume rather than time, or a last frame that disagrees with the
scoreboard printed above it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recoup.eval.store import ledger_path, open_ledger
from recoup.ledger.events import EventKind, Ledger, LedgerEvent
from recoup.web.timeline import COLUMNS, build_frames, to_payload

ARM = "recoup_agent"
START = datetime(2026, 6, 1, 9, 0)


def _event(kind, minutes, payment="pay_1", amount=None, **data):
    return LedgerEvent(
        at=START + timedelta(minutes=minutes),
        kind=kind,
        arm=ARM,
        payment_id=payment,
        amount=amount,
        data=data,
    )


@pytest.fixture
def ledger(tmp_path):
    instance = Ledger(tmp_path / "run.db")
    yield instance
    instance.close()


def _write(ledger, events):
    for event in events:
        ledger.append(event)


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


def test_an_empty_ledger_has_no_frames(ledger):
    assert build_frames(ledger, ARM) == []


def test_totals_accumulate_and_the_last_frame_is_the_whole_run(ledger):
    _write(ledger, [
        _event(EventKind.OBSERVED, 0, "pay_1", amount=10_000),
        _event(EventKind.OBSERVED, 10, "pay_2", amount=20_000),
        _event(EventKind.RECOVERED, 20, "pay_1", amount=10_000),
        _event(EventKind.STOPPED, 30, "pay_2"),
    ])

    frames = build_frames(ledger, ARM, count=10)
    last = frames[-1]

    assert last.observed == 2
    assert last.at_risk_paise == 30_000
    assert last.recovered_count == 1
    assert last.recovered_paise == 10_000
    assert last.open_count == 0


def test_no_total_ever_goes_backwards(ledger):
    """Cumulative means cumulative. A number that dips as you drag is a bug."""
    _write(ledger, [
        _event(EventKind.OBSERVED, m, f"pay_{m}", amount=1_000 * m or 1_000)
        for m in range(0, 200, 10)
    ] + [
        _event(EventKind.RECOVERED, m + 5, f"pay_{m}", amount=1_000)
        for m in range(0, 200, 40)
    ])

    frames = build_frames(ledger, ARM, count=20)

    for field in ("observed", "recovered_count", "recovered_paise", "at_risk_paise",
                  "vetoes", "contacts", "decisions"):
        values = [getattr(f, field) for f in frames]
        assert values == sorted(values), f"{field} decreased: {values}"


def test_open_count_rises_and_falls(ledger):
    """The one figure that legitimately goes down — it is a level, not a total."""
    _write(ledger, [
        _event(EventKind.OBSERVED, 0, "pay_1", amount=1_000),
        _event(EventKind.OBSERVED, 1, "pay_2", amount=1_000),
        _event(EventKind.RECOVERED, 50, "pay_1", amount=1_000),
        _event(EventKind.STOPPED, 99, "pay_2"),
    ])

    opens = [f.open_count for f in build_frames(ledger, ARM, count=10)]

    assert max(opens) == 2
    assert opens[-1] == 0


def test_frames_are_evenly_spaced_in_time_not_in_events(ledger):
    """A quiet stretch still produces frames.

    Otherwise the slider moves fast through busy periods and stalls through calm
    ones, which misrepresents when the money actually came back.
    """
    _write(ledger, [_event(EventKind.OBSERVED, m, f"pay_{m}", amount=1_000) for m in range(10)]
           + [_event(EventKind.OBSERVED, 1000, "pay_late", amount=1_000)])

    frames = build_frames(ledger, ARM, count=20)

    # All but the final gap, which closes on the last event rather than on a
    # boundary. Compared with a tolerance because `day` is rounded for the wire.
    gaps = [b.day - a.day for a, b in zip(frames, frames[1:], strict=False)][:-1]

    assert gaps
    assert max(gaps) - min(gaps) < 0.002, f"uneven spacing: {sorted(set(gaps))}"


def test_day_starts_at_the_first_event_not_at_zero_epoch(ledger):
    _write(ledger, [
        _event(EventKind.OBSERVED, 0, amount=1_000),
        _event(EventKind.RECOVERED, 60 * 24, amount=1_000),
    ])

    frames = build_frames(ledger, ARM, count=5)

    assert frames[0].day > 0
    assert frames[-1].day == pytest.approx(1.0, abs=0.01)


def test_executions_split_by_kind(ledger):
    _write(ledger, [
        _event(EventKind.OBSERVED, 0, amount=1_000),
        _event(EventKind.EXECUTED, 1, action="RETRY_NOW", cost=100),
        _event(EventKind.EXECUTED, 2, action="NUDGE_SMS", cost=15, delivered=True),
        _event(EventKind.EXECUTED, 3, action="ESCALATE_HUMAN", cost=12_000),
    ])

    last = build_frames(ledger, ARM, count=5)[-1]

    assert last.retries == 1
    assert last.contacts == 1
    assert last.escalated == 1
    assert last.cost_paise == 12_115


def test_only_the_requested_arm_is_folded(ledger):
    _write(ledger, [_event(EventKind.OBSERVED, 0, amount=1_000)])
    other = LedgerEvent(
        at=START, kind=EventKind.OBSERVED, arm="naive_baseline",
        payment_id="pay_x", amount=99_999, data={},
    )
    ledger.append(other)

    assert build_frames(ledger, ARM, count=5)[-1].at_risk_paise == 1_000


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


def test_the_payload_is_columns_and_rows(ledger):
    _write(ledger, [
        _event(EventKind.OBSERVED, 0, amount=1_000),
        _event(EventKind.RECOVERED, 10, amount=1_000),
    ])

    payload = to_payload(build_frames(ledger, ARM, count=8))

    assert payload["columns"] == list(COLUMNS)
    assert all(len(row) == len(COLUMNS) for row in payload["rows"])
    assert payload["start"] and payload["end"]


def test_rows_carry_no_key_names(ledger):
    """Repeating thirteen key names per frame is pure weight on first paint."""
    _write(ledger, [_event(EventKind.OBSERVED, m, f"pay_{m}", amount=1_000) for m in range(50)])

    payload = to_payload(build_frames(ledger, ARM, count=200))

    assert all(isinstance(row, list) for row in payload["rows"])


def test_an_empty_payload_still_has_its_shape():
    payload = to_payload([])

    assert payload["rows"] == []
    assert payload["columns"] == list(COLUMNS)


# ---------------------------------------------------------------------------
# Against the real run
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_last_frame_matches_the_scoreboard(tmp_path):
    """The scrubber sits directly under the headline figures. They must agree."""
    from recoup.eval import run_all
    from recoup.world.config import WorldConfig

    world = WorldConfig.load()
    results, run_ledger = run_all(world, ledger_path=ledger_path(tmp_path))
    run_ledger.close()

    agent = next(m for m in results if m.arm == ARM)

    instance = open_ledger(tmp_path)
    try:
        last = build_frames(instance, ARM)[-1]
    finally:
        instance.close()

    assert last.observed == agent.observed
    assert last.recovered_count == agent.recovered_count
    assert last.recovered_paise == agent.recovered_paise
    assert last.at_risk_paise == agent.amount_at_risk
    assert last.vetoes == agent.vetoes
    assert last.contacts == agent.contacts
    assert last.cost_paise == agent.cost_paise
