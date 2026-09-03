"""The experiment screen, and the decomposition it exists for.

"+268% over naive retry" is true and mostly uninteresting. The baseline only
retries, and three quarters of the money sits behind failures no retry can touch,
so most of that gap is money it structurally cannot reach rather than evidence of
a clever agent.

Presenting the total as though it were the latter would be the single most
misleading thing this project could do with an honest number, so the split is
computed, tested, and put on the screen rather than left in a caveat.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recoup.eval import run_all
from recoup.eval.store import ledger_path, open_ledger, save_summary
from recoup.web.app import create_app
from recoup.web.views import build_experiment
from recoup.world.config import WorldConfig

BASELINE = "naive_baseline"
CONTACT = "contact_only"
AGENT = "recoup_agent"
ABLATION = "recoup_agent_no_llm"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("experiment")
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
def experiment(run_dir):
    ledger = open_ledger(run_dir)
    view = build_experiment(ledger)
    ledger.close()
    return view


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


def test_the_pool_splits_cleanly(experiment):
    assert experiment.retryable_amount + experiment.customer_amount == experiment.at_risk
    assert experiment.retryable_count + experiment.customer_count == experiment.observed


def test_most_of_the_money_needs_the_customer(experiment):
    """The structural fact the whole decomposition rests on.

    Razorpay cannot re-run a failed payment without standing authorisation, so
    the majority of the pool is unreachable by retrying however hard you try.
    """
    assert experiment.customer_amount > experiment.retryable_amount


def test_the_pool_is_measured_once_not_per_arm(experiment):
    """Every arm sees the same failures; counting per arm would multiply it."""
    assert experiment.observed < 3000, "pool looks like it was counted more than once"


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


def test_coverage_and_judgment_account_for_the_whole_lift(experiment):
    assert experiment.coverage_gain + experiment.judgment_gain == experiment.total_gain
    assert experiment.coverage_share + experiment.judgment_share == pytest.approx(1.0)


def test_most_of_the_lift_is_coverage_not_cleverness(experiment):
    """The finding the screen exists to state plainly."""
    assert experiment.coverage_share > 0.5


def test_judgment_is_still_a_real_contribution(experiment):
    """Coverage dominating does not mean the policy engine does nothing."""
    assert experiment.judgment_gain > 0
    assert experiment.judgment_share > 0.15


# ---------------------------------------------------------------------------
# Per-arm behaviour
# ---------------------------------------------------------------------------


def test_each_arms_recovery_splits_by_where_it_came_from(experiment):
    for arm in experiment.arms:
        assert arm.from_retryable + arm.from_customer == arm.recovered_paise


def test_the_naive_baseline_recovers_nothing_from_customers(experiment):
    """It never contacts anyone, so every rupee must come from a retry."""
    base = experiment.arm(BASELINE)

    assert base.from_customer == 0
    assert base.from_retryable == base.recovered_paise


def test_the_contact_only_arm_recovers_almost_nothing_by_retrying(experiment):
    reach = experiment.arm(CONTACT)

    assert reach.from_customer > reach.from_retryable


def test_the_agent_works_both_halves_of_the_pool(experiment):
    agent = experiment.arm(AGENT)

    assert agent.from_retryable > 0
    assert agent.from_customer > 0
    assert agent.recovered_paise > experiment.arm(BASELINE).recovered_paise


def test_recovery_is_attributed_to_causes(experiment):
    agent = experiment.arm(AGENT)

    assert agent.by_cause
    assert sum(agent.by_cause.values()) == agent.recovered_paise


def test_recovery_is_attributed_to_actions(experiment):
    agent = experiment.arm(AGENT)

    assert agent.by_action
    assert sum(agent.by_action.values()) == agent.recovered_paise


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def test_the_ablation_is_reported_whatever_it_says(experiment):
    """Measured, not tuned until it looked useful.

    On this seed the model's contribution is within noise and slightly negative.
    That is the finding; the test asserts it is small rather than asserting it is
    positive, because asserting a direction would be writing the answer down.
    """
    agent = experiment.arm(AGENT)

    assert abs(experiment.llm_gain) < agent.recovered_paise * 0.01


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@pytest.fixture
def client(run_dir) -> TestClient:
    return TestClient(create_app(data_dir=run_dir))


def test_the_experiment_page_renders(client):
    response = client.get("/experiment")

    assert response.status_code == 200
    assert "Where the lift actually comes from" in response.text


def test_the_page_says_which_number_to_lead_with(client):
    """The caveat belongs on the screen, not only in a PR description."""
    html = client.get("/experiment").text

    assert "Lead with the second number" in html
    assert "is coverage" in html


def test_the_page_shows_the_pool_split(client):
    html = client.get("/experiment").text

    assert "Needs the customer" in html
    assert "no retry can touch" in html


def test_the_page_reports_the_model_honestly(client):
    html = client.get("/experiment").text

    assert "measured, not assumed" in html
    assert "not worth much" in html


def test_the_experiment_page_without_a_run_explains_itself(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path / "empty"))
    response = client.get("/experiment")

    assert response.status_code == 200
    assert "No run to compare yet" in response.text
