"""Rupees, written the way rupees are written.

Indian grouping puts the last three digits together and then groups in twos:
₹5,41,724, not ₹541,724. The two conventions agree below a lakh, which is exactly
why this was wrong for so long — every small test amount formatted identically
under both, and the divergence only shows up on the headline figures.
"""

from __future__ import annotations

import pytest

from recoup.money import group, rupees

# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("digits", "expected"),
    [
        ("0", "0"),
        ("7", "7"),
        ("99", "99"),
        ("100", "100"),
        ("999", "999"),
        ("1000", "1,000"),
        ("12041", "12,041"),
        ("99999", "99,999"),
        # A lakh is where the conventions part company.
        ("100000", "1,00,000"),
        ("541724", "5,41,724"),
        ("3002856", "30,02,856"),
        ("10000000", "1,00,00,000"),
        ("1234567890", "1,23,45,67,890"),
    ],
)
def test_indian_grouping(digits, expected):
    assert group(digits) == expected


def test_the_conventions_agree_below_a_lakh():
    """Which is why this bug survived: no small fixture could catch it."""
    for value in (0, 1, 99, 100, 999, 1000, 12041, 99999):
        assert group(str(value)) == f"{value:,}"


def test_they_disagree_at_a_lakh_and_above():
    for value in (100_000, 541_724, 3_002_856):
        assert group(str(value)) != f"{value:,}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_paise_are_converted_to_rupees():
    assert rupees(54_172_446) == "₹5,41,724"


def test_the_headline_matches_the_readme():
    assert rupees(30_02_856_21) == "₹30,02,856"
    assert rupees(39_479_098) == "₹3,94,791"


def test_none_is_a_dash_not_a_zero():
    """An absent figure and a figure of zero are different claims."""
    assert rupees(None) == "—"
    assert rupees(0) == "₹0"


def test_the_sign_leads():
    """A negative amount of money, not an amount of negative money.

    The LLM ablation is the first place a reader meets one of these.
    """
    assert rupees(-23_845) == "-₹238"


# ---------------------------------------------------------------------------
# Small amounts
# ---------------------------------------------------------------------------


def test_small_amounts_keep_their_paise_when_asked():
    """A WhatsApp message costs 35 paise. Rounded to ₹0 the EV sum stops adding up."""
    assert rupees(35, precise_below=10_000) == "₹0.35"
    assert rupees(120, precise_below=10_000) == "₹1.20"


def test_large_amounts_drop_the_decimals_even_when_precision_is_offered():
    assert rupees(1_204_100, precise_below=10_000) == "₹12,041"


def test_zero_stays_whole():
    """It appears in entire columns of costs that were never incurred."""
    assert rupees(0, precise_below=10_000) == "₹0"


def test_precision_is_off_by_default():
    assert rupees(35) == "₹0"


def test_negative_small_amounts_keep_their_paise_too():
    assert rupees(-35, precise_below=10_000) == "-₹0.35"


# ---------------------------------------------------------------------------
# Bare numbers
# ---------------------------------------------------------------------------


def test_the_symbol_can_be_dropped_for_a_numeric_column():
    """The tornado's columns are headed once, not per row."""
    assert rupees(54_172_446, symbol=False) == "5,41,724"
    assert rupees(-23_845, symbol=False) == "-238"


# ---------------------------------------------------------------------------
# Everything that displays money uses this
# ---------------------------------------------------------------------------


def test_the_eval_table_and_the_web_filter_agree():
    """They were separate implementations, and they disagreed."""
    from recoup.eval.metrics import rupees as metrics_rupees
    from recoup.web.app import _rupees as web_rupees

    assert metrics_rupees(54_172_446) == "₹5,41,724"
    assert web_rupees(54_172_446) == "₹5,41,724"


def test_customer_copy_uses_the_same_formatting():
    from recoup.agent.llm.copywriter import Copywriter
    from recoup.domain import Channel, FailureCause, Language

    text, _ = Copywriter(use_llm=False).render(
        FailureCause.AUTH_ABANDONED, Language.ENGLISH, Channel.SMS, amount_paise=54_172_446
    )

    assert "₹5,41,724" in text
