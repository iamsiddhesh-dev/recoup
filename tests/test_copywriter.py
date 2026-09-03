"""The message a customer actually receives.

Most of the money in a run sits behind failures no retry can fix, so for the
majority of payments the message *is* the recovery attempt. That makes it the one
place a model's output reaches a real person, which is why the tests here are
mostly about what gets refused rather than what gets generated.

Two failure modes matter more than the rest. Copy stating the wrong amount is
removed by construction — the model writes `{amount}` and never sees a number.
Copy asking for an OTP or a CVV is phishing regardless of intent, and a payments
company sending one is an incident, so it is checked rather than trusted.
"""

from __future__ import annotations

import json

import pytest

from recoup.agent.llm.client import LLMClient
from recoup.agent.llm.copywriter import (
    CONTACTABLE,
    FALLBACKS,
    LANGUAGES,
    LIMITS,
    Copywriter,
    validate,
)
from recoup.domain import Channel, FailureCause, Language


class StubProvider:
    name = "stub"
    model = "stub-1"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        return self.response


def _matrix(**overrides) -> str:
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


def _writer(response: str, tmp_path) -> Copywriter:
    return Copywriter(client=LLMClient(cache_dir=tmp_path, providers=[StubProvider(response)]))


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


def test_good_copy_passes():
    assert validate("Your payment of {amount} failed. Retry: {link}", "sms") is None


@pytest.mark.parametrize(
    "text",
    [
        "Share your OTP to complete the payment of {amount}",
        "Please confirm your CVV and retry {link}",
        "Reply with your UPI PIN to finish",
        "Enter your card number here: {link}",
        "Send the verification code to complete {amount}",
    ],
)
def test_copy_asking_for_credentials_is_refused(text):
    """The failure mode that would make this system indistinguishable from fraud."""
    assert validate(text, "sms") == "asks for a credential"


def test_literal_numbers_are_refused():
    """The model never sees an amount, so it must never write one.

    A message stating the wrong figure is the most damaging thing this could send,
    and templating removes the possibility rather than checking for it after.
    """
    assert validate("Your payment of Rs 1200 failed. Retry: {link}", "sms") == (
        "contains a literal number instead of a placeholder"
    )


def test_a_single_digit_is_allowed():
    """Only runs of digits look like amounts; "1 step" is fine."""
    assert validate("Just 1 step left: {link} for {amount}", "sms") is None


def test_invented_urls_are_refused():
    """A model-authored link is a phishing vector with extra steps."""
    assert "URL" in validate("Pay at https://pay-now.example for {amount}", "sms")


def test_unknown_placeholders_are_refused():
    problem = validate("Hello {first_name}, your {amount} failed: {link}", "sms")
    assert "first_name" in problem


def test_length_is_measured_after_substitution():
    """A template that fits and a message that fits are different things."""
    long_sms = "Your payment of {amount} to us could not be completed today. " * 3
    assert "too long" in validate(long_sms + "{link}", "sms")


def test_channel_limits_differ():
    text = "Your payment of {amount} was not completed, please use this link: {link}. " * 3
    assert validate(text, "sms") is not None
    assert validate(text, "email") is None


def test_empty_copy_is_refused():
    assert validate("   ", "sms") == "empty"


# ---------------------------------------------------------------------------
# Our own fallbacks must pass our own rules
# ---------------------------------------------------------------------------


def test_every_contactable_cause_has_a_fallback():
    """The run must never depend on a model being reachable."""
    for cause in CONTACTABLE:
        assert str(cause) in FALLBACKS, f"no fallback for {cause}"
        for language in LANGUAGES:
            assert FALLBACKS[str(cause)].get(language), f"{cause}/{language} missing"


def test_hand_written_fallbacks_pass_the_validator():
    """Holding our own copy to the standard we hold the model's.

    Easy to write a fallback with a literal number in it and never notice, since
    fallbacks skip the validation path in normal operation.
    """
    for cause, by_language in FALLBACKS.items():
        for language, text in by_language.items():
            for channel in LIMITS:
                problem = validate(text, channel)
                assert problem is None, f"{cause}/{language}/{channel}: {problem}"


def test_risk_blocked_has_no_copy_at_all():
    """Compliance forbids contacting them, so there is nothing to write."""
    assert FailureCause.RISK_BLOCKED not in CONTACTABLE
    assert "RISK_BLOCKED" not in FALLBACKS


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_rendering_substitutes_the_real_amount():
    writer = Copywriter(use_llm=False)

    text, source = writer.render(
        FailureCause.AUTH_ABANDONED,
        Language.ENGLISH,
        Channel.SMS,
        amount_paise=1_204_100,
        link="https://rzp.io/l/abc",
    )

    assert "₹12,041" in text
    assert "https://rzp.io/l/abc" in text
    assert "{amount}" not in text
    assert source == "fallback"


def test_regional_maps_to_a_concrete_language():
    """`REGIONAL` says "not English or Hinglish" without saying which."""
    writer = Copywriter(use_llm=False)

    text, _ = writer.render(
        FailureCause.AUTH_ABANDONED, Language.REGIONAL, Channel.SMS, amount_paise=50_000
    )

    assert any("ऀ" <= ch <= "ॿ" for ch in text), "expected Devanagari"


def test_an_unclassified_failure_still_gets_a_message():
    writer = Copywriter(use_llm=False)

    text, _ = writer.render(None, Language.ENGLISH, Channel.SMS, amount_paise=50_000)

    assert "₹500" in text


def test_with_no_model_everything_is_a_fallback():
    writer = Copywriter(use_llm=False)

    assert writer.warm() == 0
    _, source = writer.render(
        FailureCause.INSUFFICIENT_FUNDS, Language.ENGLISH, Channel.SMS, amount_paise=1000
    )
    assert source == "fallback"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generated_copy_is_used_when_it_validates(tmp_path):
    writer = _writer(_matrix(), tmp_path)

    # The stub returns the same English variant for all three language calls, so
    # four channels are accepted three times over the same keys. A real run gets
    # a distinct language per call.
    assert writer.warm() == 4 * len(LANGUAGES)
    assert len(writer._copy) == 4

    text, source = writer.render(
        FailureCause.AUTH_ABANDONED, Language.ENGLISH, Channel.SMS, amount_paise=100_000
    )

    assert source == "generated"
    assert "₹1,000" in text


def test_one_call_per_language_not_per_message(tmp_path):
    """84 strings does not fit in one free-tier response.

    Groq returned `json_validate_failed` with a visibly correct but truncated
    body, so the matrix is chunked by language. Three calls, not one, and
    certainly not one per payment.
    """
    provider = StubProvider(_matrix())
    writer = Copywriter(client=LLMClient(cache_dir=tmp_path, providers=[provider]))

    writer.warm()

    assert writer.calls == len(LANGUAGES)
    assert provider.calls == len(LANGUAGES)


def test_copy_that_fails_validation_is_rejected_and_falls_back(tmp_path):
    writer = _writer(_matrix(sms="Share your OTP for {amount}"), tmp_path)
    writer.warm()

    text, source = writer.render(
        FailureCause.AUTH_ABANDONED, Language.ENGLISH, Channel.SMS, amount_paise=100_000
    )

    assert source == "fallback"
    assert "OTP" not in text
    assert any(r.reason == "asks for a credential" for r in writer.rejections)


def test_a_rejection_records_what_was_wrong(tmp_path):
    writer = _writer(_matrix(sms="Your payment of Rs 500 failed"), tmp_path)
    writer.warm()

    rejection = next(r for r in writer.rejections if r.channel == "sms")

    assert rejection.cause == "AUTH_ABANDONED"
    assert "literal number" in rejection.reason


def test_one_bad_channel_does_not_discard_the_others(tmp_path):
    writer = _writer(_matrix(sms="Send your CVV"), tmp_path)
    writer.warm()

    _, sms_source = writer.render(
        FailureCause.AUTH_ABANDONED, Language.ENGLISH, Channel.SMS, amount_paise=1000
    )
    _, email_source = writer.render(
        FailureCause.AUTH_ABANDONED, Language.ENGLISH, Channel.EMAIL, amount_paise=1000
    )

    assert sms_source == "fallback"
    assert email_source == "generated"


def test_unparseable_output_leaves_the_fallbacks_in_place(tmp_path):
    writer = _writer("not json", tmp_path)

    assert writer.warm() == 0
    _, source = writer.render(
        FailureCause.AUTH_ABANDONED, Language.ENGLISH, Channel.SMS, amount_paise=1000
    )
    assert source == "fallback"
