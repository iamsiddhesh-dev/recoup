"""Checking the checker.

`recoup reproduce` exists to catch the case where the code stops producing the
numbers the README claims. The risk with that kind of command is that it passes
because it compared nothing — so most of what is tested here is that it notices.
"""

from __future__ import annotations

import json

import pytest

from recoup.eval.metrics import ArmMetrics
from recoup.eval.reproduce import claims, compare, load, report, save


def _metrics(arm: str, **overrides) -> ArmMetrics:
    values = {
        "arm": arm,
        "description": arm,
        "observed": 1605,
        "amount_at_risk": 300_285_621,
        "recovered_count": 100,
        "recovered_paise": 10_000_000,
        "contacts": 500,
        "cost_paise": 400_000,
        "vetoes": 50,
        "unresolved": 5,
    }
    values.update(overrides)
    return ArmMetrics(**values)


def _run(**overrides) -> dict:
    results = [
        _metrics("naive_baseline", recovered_paise=14_693_348, contacts=0),
        _metrics("contact_only", recovered_paise=41_436_153),
        _metrics("recoup_agent_no_llm", recovered_paise=54_196_291),
        _metrics("recoup_agent", recovered_paise=54_172_446, unresolved=0),
    ]
    for arm, fields in overrides.items():
        target = next(m for m in results if m.arm == arm)
        for key, value in fields.items():
            setattr(target, key, value)

    return claims(
        results,
        seed=20260902,
        batch_size=5000,
        horizon_days=30,
        digests={m.arm: f"digest-{m.arm}" for m in results},
    )


# ---------------------------------------------------------------------------
# What gets recorded
# ---------------------------------------------------------------------------


def test_the_headline_figures_are_recorded_not_recomputed():
    """The README's derived numbers are claims in their own right.

    Recording them means a change in *how* incremental is derived also fails,
    not just a change in the underlying totals.
    """
    recorded = _run()

    assert recorded["headline.incremental_paise"] == 54_172_446 - 14_693_348
    assert recorded["headline.coverage_paise"] == 41_436_153 - 14_693_348
    assert recorded["headline.judgment_paise"] == 54_172_446 - 41_436_153
    assert recorded["headline.llm_delta_paise"] == 54_172_446 - 54_196_291


def test_every_arm_records_a_digest():
    recorded = _run()

    for arm in ("naive_baseline", "contact_only", "recoup_agent", "recoup_agent_no_llm"):
        assert recorded[f"{arm}.digest"] == f"digest-{arm}"


def test_claims_are_sorted_so_a_diff_is_readable():
    recorded = _run()

    assert list(recorded) == sorted(recorded)


# ---------------------------------------------------------------------------
# Noticing
# ---------------------------------------------------------------------------


def test_an_identical_run_reproduces():
    checks = compare(_run(), _run())

    assert checks, "comparing nothing must not count as passing"
    assert all(check.ok for check in checks)


def test_a_changed_total_is_caught():
    checks = compare(_run(), _run(recoup_agent={"recovered_paise": 54_172_447}))

    changed = [c.key for c in checks if not c.ok]
    assert "recoup_agent.recovered_paise" in changed
    assert "headline.incremental_paise" in changed


def test_a_changed_digest_is_caught_even_when_the_totals_match():
    """The case the digest exists for.

    Two runs can recover identical amounts while having made different decisions
    — a reordered schedule, a different channel with the same outcome. Totals
    alone would call that reproduced.
    """
    produced = _run()
    produced["recoup_agent.digest"] = "something-else"

    checks = compare(_run(), produced)

    assert [c.key for c in checks if not c.ok] == ["recoup_agent.digest"]


def test_a_removed_figure_is_a_change():
    produced = _run()
    del produced["recoup_agent.vetoes"]

    check = next(c for c in compare(_run(), produced) if c.key == "recoup_agent.vetoes")

    assert not check.ok
    assert check.actual is None


def test_an_added_figure_is_a_change():
    produced = _run()
    produced["recoup_agent.something_new"] = 1

    check = next(c for c in compare(_run(), produced) if c.key == "recoup_agent.something_new")

    assert not check.ok
    assert check.expected is None


@pytest.mark.parametrize("seed", [20260902, 1])
def test_the_seed_itself_is_checked(seed):
    recorded = _run()
    produced = dict(recorded, **{"run.seed": seed})

    checks = compare(recorded, produced)

    assert all(c.ok for c in checks) == (seed == 20260902)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_report_lists_passes_too():
    """A successful run must be distinguishable from one that checked nothing."""
    text = report(compare(_run(), _run()))

    assert "recoup_agent.recovered_paise" in text
    assert "all 37 recorded figures reproduce exactly." in text


def test_the_report_shows_what_it_expected():
    checks = compare(_run(), _run(recoup_agent={"recovered_count": 999}))

    text = report(checks)

    assert "CHANGED" in text
    assert "999" in text
    assert "recorded 100" in text
    assert "1 of" in text


def test_money_is_shown_as_money_and_seeds_are_not():
    text = report(compare(_run(), _run()))

    assert "₹5,41,724" in text
    assert "20260902" in text
    assert "20,260,902" not in text, "a seed is an identifier, not a quantity"


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_saved_claims_load_back_identically(tmp_path):
    path = tmp_path / "claims.json"
    recorded = _run()

    save(recorded, path)

    assert load(path) == recorded
    assert json.loads(path.read_text(encoding="utf-8")) == recorded


def test_loading_a_missing_file_returns_none(tmp_path):
    """A fresh clone that has not recorded a baseline must get a message, not a
    traceback."""
    assert load(tmp_path / "nothing.json") is None
