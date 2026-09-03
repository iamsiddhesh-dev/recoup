"""The sensitivity sweep.

Every number this project reports rests on `config/world.yaml`, most of which is
tagged `[ASSUMPTION]`. Without a sweep, "the agent recovers 18% of money at risk"
is a claim about one hypothetical month chosen by the person making the claim,
and the obvious objection — that the numbers were picked to flatter the agent — is
unanswerable.

Running the real sweep takes minutes, so it is exercised here against a fixture
and with a single cheap axis. What is tested is the machinery and its honesty: that
axes actually change the world, that the tornado is ordered and scaled correctly,
and that `holds()` is capable of returning False.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from recoup.agent.config import PolicyConfig
from recoup.eval.sensitivity import (
    AXES,
    BY_KEY,
    Point,
    SweepResult,
    load,
    run_sweep,
    save,
    table,
)
from recoup.web.app import create_app
from recoup.world.config import WorldConfig


def _point(axis: str, level: str, judgment: int) -> Point:
    return Point(
        axis=axis,
        level=level,
        factor=1.0,
        recovered=judgment * 4,
        incremental=judgment * 3,
        judgment=judgment,
        at_risk=10_000_000,
    )


@pytest.fixture
def result() -> SweepResult:
    return SweepResult(
        base=_point("base", "base", 100_000),
        points=[
            _point("failure_rate", "low", 60_000),
            _point("failure_rate", "high", 160_000),
            _point("downtime", "low", 98_000),
            _point("downtime", "high", 103_000),
        ],
    )


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


def test_every_axis_actually_changes_the_world():
    """An axis that mutates nothing produces a zero swing that looks like a finding.

    That happened: the margin axis moved only the world's margin while the agent
    prices on its own belief, so it reported exactly ₹0 and looked meaningful.
    """
    for axis in AXES:
        world = WorldConfig.load().model_copy(deep=True)
        policy = PolicyConfig.load().model_copy(deep=True)

        before = (world.model_dump_json(), policy.model_dump_json())
        axis.apply(world, policy, axis.high)
        after = (world.model_dump_json(), policy.model_dump_json())

        assert before != after, f"{axis.key} changed nothing"


def test_axes_move_in_both_directions():
    for axis in AXES:
        assert axis.low < 1.0 < axis.high, f"{axis.key} does not bracket the baseline"


def test_every_axis_explains_why_it_matters():
    """A tornado nobody can read the axis labels of is decoration."""
    for axis in AXES:
        assert axis.label
        assert len(axis.why) > 40


def test_the_margin_axis_moves_the_agents_belief_too():
    """The bug that made it report a zero swing."""
    world = WorldConfig.load().model_copy(deep=True)
    policy = PolicyConfig.load().model_copy(deep=True)
    before = policy.assumed_margin

    BY_KEY["margin"].apply(world, policy, 0.7)

    assert policy.assumed_margin < before
    assert world.merchant_margin < WorldConfig.load().merchant_margin


# ---------------------------------------------------------------------------
# Bands and the tornado
# ---------------------------------------------------------------------------


def test_bands_are_ordered_widest_first(result):
    spans = [b.span for b in result.bands()]
    assert spans == sorted(spans, reverse=True)


def test_a_band_spans_its_two_ends(result):
    band = next(b for b in result.bands() if b.axis == "failure_rate")

    assert band.low == 60_000
    assert band.high == 160_000
    assert band.span == 100_000
    assert band.base == 100_000


def test_a_band_records_its_deltas_from_the_baseline(result):
    band = next(b for b in result.bands() if b.axis == "failure_rate")

    assert band.low_delta == -40_000
    assert band.high_delta == 60_000
    assert band.worst == 60_000


def test_the_metric_is_selectable(result):
    """Judgment is the default because it is the claim about the policy engine.

    Total recovery moves mechanically with the size of the pool, so a tornado on
    it would largely measure the arithmetic of its own axes.
    """
    assert result.bands("judgment")[0].span == 100_000
    assert result.bands("recovered")[0].span == 400_000


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------


def test_holds_is_true_when_every_point_is_positive(result):
    assert result.holds() is True


def test_holds_is_capable_of_being_false():
    """A check that cannot fail proves nothing."""
    flipped = SweepResult(
        base=_point("base", "base", 100_000),
        points=[_point("failure_rate", "low", -5_000)],
    )

    assert flipped.holds() is False


def test_the_worst_case_is_identified(result):
    worst = result.worst_case()

    assert worst.axis == "failure_rate"
    assert worst.level == "low"
    assert worst.judgment == 60_000


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_sweep_round_trips_through_disk(result, tmp_path):
    """Committed so a reader never has to re-run five minutes of it."""
    path = save(result, tmp_path / "sensitivity.json")
    restored = load(path)

    assert restored.base.judgment == result.base.judgment
    assert len(restored.points) == len(result.points)
    assert restored.bands()[0].span == result.bands()[0].span


def test_a_missing_sweep_is_not_an_error(tmp_path):
    assert load(tmp_path / "nothing.json") is None


def test_the_saved_file_is_readable(result, tmp_path):
    path = save(result, tmp_path / "sensitivity.json")
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["base"]["judgment"] == 100_000
    assert len(raw["points"]) == 4


def test_the_terminal_table_reports_whether_the_claim_holds(result):
    rendered = table(result)

    assert "claim holds everywhere: yes" in rendered
    assert "worst case" in rendered


# ---------------------------------------------------------------------------
# End to end, on one cheap axis
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_real_sweep_produces_a_real_band():
    """One axis, two extra runs. Enough to prove the machinery moves the result."""
    seen: list[tuple[str, int, int]] = []
    result = run_sweep(
        axes=[BY_KEY["saved_instruments"]],
        on_progress=lambda label, done, total: seen.append((label, done, total)),
    )

    assert len(seen) == 3, "expected a baseline plus both ends"
    assert result.base.judgment != 0

    band = result.bands()[0]
    assert band.span > 0, "halving standing authorisation should move the result"


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(create_app(data_dir=tmp_path / "empty"))


def test_the_experiment_page_survives_no_sweep(client):
    """The sweep is optional; the rest of the screen must not depend on it."""
    response = client.get("/experiment")

    assert response.status_code == 200
    assert "Does it survive the assumptions being wrong?" not in response.text


def test_the_committed_sweep_renders_when_present():
    """Against the real reports/sensitivity.json in the repo."""
    from pathlib import Path

    if not Path("reports/sensitivity.json").exists():
        pytest.skip("no committed sweep")

    sweep = load()

    assert sweep is not None
    assert sweep.holds(), "the committed sweep should show the claim holding"
    assert len(sweep.bands()) == len(AXES)
