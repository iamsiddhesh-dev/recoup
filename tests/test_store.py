"""Persisting a run, and serving screens from it.

The reason this layer exists is a latency mismatch: an evaluation takes ~35
seconds and a page load cannot. The resolution is to treat a finished run as an
artifact — compute once, write to disk, query it.

That works here specifically because the data is deterministic, identical for
every viewer, and frozen once produced. Those three properties between them
remove cache invalidation as a problem, which is the part of caching that is
actually hard. These tests pin the behaviour that depends on it.
"""

from __future__ import annotations

import json

import pytest

from recoup.eval import run_all
from recoup.eval.metrics import ArmMetrics
from recoup.eval.store import (
    RunSummary,
    ledger_path,
    load_summary,
    open_ledger,
    save_summary,
)
from recoup.ledger.events import EventKind
from recoup.world.config import WorldConfig


@pytest.fixture(scope="module")
def world() -> WorldConfig:
    return WorldConfig.load()


@pytest.fixture(scope="module")
def persisted(tmp_path_factory, world):
    directory = tmp_path_factory.mktemp("run")
    path = ledger_path(directory)

    results, ledger = run_all(world, ledger_path=path)
    ledger.close()

    save_summary(
        results,
        seed=world.run.seed,
        horizon_days=world.run.horizon_days,
        batch_size=world.run.batch_size,
        margin=world.merchant_margin,
        directory=directory,
    )
    return directory, results


def test_nothing_persisted_is_not_an_error(tmp_path):
    """A server started before any run exists should say so, not refuse to boot."""
    assert load_summary(tmp_path) is None
    assert open_ledger(tmp_path) is None


def test_a_run_survives_the_process_that_made_it(persisted):
    directory, results = persisted

    summary = load_summary(directory)

    assert isinstance(summary, RunSummary)
    assert summary.seed == WorldConfig.load().run.seed
    assert len(summary.arms) == len(results)


def test_the_summary_carries_derived_figures(persisted):
    """Computed once, on write.

    A number shown to a merchant should be calculated in one place, not
    re-derived in whichever template happens to need it.
    """
    directory, _ = persisted
    arm = load_summary(directory).arms[-1]

    for key in ("net_paise", "margin_recovered", "money_recovery_rate", "contacts_per_recovery"):
        assert key in arm

    assert arm["net_paise"] == arm["margin_recovered"] - arm["cost_paise"]


def test_the_summary_matches_what_was_scored(persisted):
    directory, results = persisted
    summary = load_summary(directory)

    for metrics, stored in zip(results, summary.arms, strict=True):
        assert isinstance(metrics, ArmMetrics)
        assert stored["arm"] == metrics.arm
        assert stored["recovered_paise"] == metrics.recovered_paise
        assert stored["vetoes"] == metrics.vetoes


def test_the_summary_is_readable_json(persisted):
    """Reviewable on its own, without running anything to interpret it."""
    directory, _ = persisted
    raw = json.loads((directory / "run.json").read_text(encoding="utf-8"))

    assert raw["seed"]
    assert raw["generated_at"]
    assert raw["arms"]


# ---------------------------------------------------------------------------
# The reason the ledger shape matters
# ---------------------------------------------------------------------------


def test_a_persisted_ledger_answers_screen_queries(persisted):
    """Every screen is a query against indexes built for auditability.

    The ledger was designed so a decision could be explained, not so a page could
    render quickly. It turned out to be the right shape for both, which is why
    persisting it was a one-line change rather than a rewrite.
    """
    directory, _ = persisted
    ledger = open_ledger(directory)

    assert ledger is not None

    recovered = list(ledger.events(arm="recoup_agent", kind=EventKind.RECOVERED))
    assert recovered

    # Case Detail: one payment's whole story, in order.
    story = ledger.story_of(recovered[0].payment_id, arm="recoup_agent")
    assert story
    assert story[0].kind is EventKind.OBSERVED

    # Audit & Refusals: filter to vetoes.
    assert list(ledger.events(arm="recoup_agent", kind=EventKind.VETOED))

    ledger.close()


def test_reopening_gives_the_same_digest(persisted):
    """A persisted run is frozen. Reading it must not change it."""
    directory, _ = persisted

    first = open_ledger(directory)
    digest = first.digest()
    first.close()

    second = open_ledger(directory)
    assert second.digest() == digest
    second.close()
