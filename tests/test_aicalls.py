"""The AI calls screen.

This page is the project's answer to "show me your tracing". Its whole value is
that the counts on it are recomputed from the committed cache on every load rather
than remembered from a run — so the tests here are mostly about that: that a bad
response is still caught when it is read back, and that the page's claims about
the real cache are true.
"""

from __future__ import annotations

import json

from recoup.agent.llm.explainer import CaseFacts
from recoup.web.aicalls import DEFAULT_CACHE, build_ai_calls


def _write(directory, name, **overrides):
    payload = {
        "purpose": name.rsplit("-", 1)[0],
        "provider": "gemini",
        "model": "gemini-3.7-flash",
        "system": "sys",
        "prompt": "prompt",
        "response": "{}",
    }
    payload.update(overrides)
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _copy(**overrides) -> str:
    variant = {
        "cause": "AUTH_ABANDONED",
        "language": "english",
        "sms": "Your payment of {amount} was not completed. Finish here: {link}",
        "whatsapp": "Your payment of {amount} was not completed. Finish here: {link}",
        "voice": "Your payment of {amount} was not completed. Please try again.",
        "email": "Your payment of {amount} was not completed. Finish here: {link}",
    }
    variant.update(overrides)
    return json.dumps({"variants": [variant]})


def _facts(**overrides) -> CaseFacts:
    values = {
        "payment_id": "pay_1",
        "amount_paise": 5_00_000,
        "method": "upi",
        "reason": "payment_timeout",
        "outcome": "stopped",
        "cause": "AUTH_ABANDONED",
        "contacts": 1,
        "channels": ["whatsapp"],
    }
    values.update(overrides)
    return CaseFacts(**values)


# ---------------------------------------------------------------------------
# Reading the cache
# ---------------------------------------------------------------------------


def test_no_cache_is_an_empty_page_not_a_crash(tmp_path):
    view = build_ai_calls(tmp_path / "missing")

    assert view.calls == []
    assert view.checked == 0


def test_a_corrupt_cache_file_is_skipped(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    _write(tmp_path, "nudge-copy-english-aaa", response=_copy())

    view = build_ai_calls(tmp_path)

    assert len(view.calls) == 1


def test_calls_are_ordered_stably(tmp_path):
    """Filenames are content hashes, so rows must not move when a prompt changes."""
    _write(tmp_path, "nudge-copy-hindi-zzz", response=_copy(language="hindi"))
    _write(tmp_path, "classify-unresolved-aaa", response='{"mappings": []}')

    purposes = [c.purpose for c in build_ai_calls(tmp_path).calls]

    assert purposes == sorted(purposes)


def test_tokens_are_estimated_from_the_whole_exchange(tmp_path):
    _write(tmp_path, "case-explanations-aaa", system="s" * 400, prompt="p" * 400,
           response='{"cases": []}')

    call = build_ai_calls(tmp_path).calls[0]

    assert 200 <= call.tokens <= 250


# ---------------------------------------------------------------------------
# Re-validating, which is the point
# ---------------------------------------------------------------------------


def test_good_copy_is_counted_as_accepted(tmp_path):
    _write(tmp_path, "nudge-copy-english-aaa", response=_copy())

    view = build_ai_calls(tmp_path)

    assert view.checked == 4
    assert view.accepted == 4
    assert view.rejected == 0


def test_copy_that_would_be_refused_is_refused_on_read_back(tmp_path):
    """The check is live. Weakening the validator moves this number."""
    _write(tmp_path, "nudge-copy-english-aaa", response=_copy(sms="Share your OTP"))

    view = build_ai_calls(tmp_path)
    call = view.calls[0]

    assert view.rejected == 1
    assert call.rejected[0].reason == "asks for a credential"
    assert "sms" in call.rejected[0].label


def test_an_unparseable_response_is_itself_the_finding(tmp_path):
    _write(tmp_path, "nudge-copy-english-aaa", response="not json")

    call = build_ai_calls(tmp_path).calls[0]

    assert call.rejected[0].reason == "invalid JSON"


def test_explanations_are_checked_against_their_own_case(tmp_path):
    _write(
        tmp_path,
        "case-explanations-aaa",
        response=json.dumps(
            {"cases": [{"payment_id": "pay_1", "text": "We retried it 9 times."}]}
        ),
    )

    call = build_ai_calls(tmp_path, facts_by_id={"pay_1": _facts()}).calls[0]

    assert "not in the record" in call.rejected[0].reason


def test_with_no_run_loaded_an_explanation_is_unchecked_not_refused(tmp_path):
    """Absence of evidence, not evidence against the model.

    On a clean clone the ledger an explanation would be checked against does not
    exist yet. Counting that as a refusal would blame the model for a missing file
    and put a false "4 refused" on the page.
    """
    _write(
        tmp_path,
        "case-explanations-aaa",
        response=json.dumps({"cases": [{"payment_id": "pay_x", "text": "Hello."}]}),
    )

    view = build_ai_calls(tmp_path, facts_by_id={})

    assert view.rejected == 0
    assert view.unchecked == 1
    assert view.checked == 0
    assert "no run loaded" in view.calls[0].unchecked[0].reason


def test_an_unknown_payment_is_refused_once_a_run_is_loaded(tmp_path):
    """With records present, an id that is not among them is a real anomaly."""
    _write(
        tmp_path,
        "case-explanations-aaa",
        response=json.dumps({"cases": [{"payment_id": "pay_x", "text": "Hello."}]}),
    )

    view = build_ai_calls(tmp_path, facts_by_id={"pay_1": _facts()})

    assert view.rejected == 1
    assert "not a case we asked about" in view.calls[0].rejected[0].reason


def test_proposed_mappings_are_listed(tmp_path):
    _write(
        tmp_path,
        "classify-unresolved-aaa",
        response=json.dumps(
            {"mappings": [{"id": 1, "cause": "TECHNICAL_GATEWAY", "confidence": 0.5}]}
        ),
    )

    call = build_ai_calls(tmp_path).calls[0]

    assert call.accepted == 1
    assert "TECHNICAL_GATEWAY" in call.checks[0].label


# ---------------------------------------------------------------------------
# What the page tells the reader
# ---------------------------------------------------------------------------


def test_every_call_says_what_it_was_checked_against(tmp_path):
    """A count of zero refusals means nothing without the list it is zero of."""
    _write(tmp_path, "nudge-copy-english-aaa", response=_copy())
    _write(tmp_path, "case-explanations-bbb", response='{"cases": []}')
    _write(tmp_path, "classify-unresolved-ccc", response='{"mappings": []}')

    for call in build_ai_calls(tmp_path).calls:
        assert call.rules, f"{call.purpose} lists no checks"


def test_purposes_are_rendered_as_english(tmp_path):
    _write(tmp_path, "nudge-copy-hinglish-aaa", response=_copy(language="hinglish"))

    assert build_ai_calls(tmp_path).calls[0].what.endswith("hinglish")


def test_the_counterfactual_is_the_argument(tmp_path):
    """Five calls only means something beside the number it replaced."""
    view = build_ai_calls(tmp_path)

    assert view.per_payment_calls > 1000
    assert view.over_budget > 50


# ---------------------------------------------------------------------------
# The committed cache
# ---------------------------------------------------------------------------


def test_the_real_cache_is_small_and_entirely_valid():
    """The claim the README makes, checked against what is actually committed."""
    view = build_ai_calls(DEFAULT_CACHE)

    assert view.calls, "cache/llm is committed and must not be empty"
    assert len(view.calls) <= 10, "a per-payment design would show hundreds"
    assert view.checked > 50, "expected the copy matrix to dominate the count"
    assert view.rejected == 0, (
        "a committed response that no longer validates means the validator changed "
        "without the cache being regenerated"
    )


def test_every_committed_call_records_its_provider_and_model():
    for call in build_ai_calls(DEFAULT_CACHE).calls:
        assert call.provider != "unknown"
        assert call.model != "unknown"
        assert call.system and call.prompt and call.response
