"""The narrative layer, and what stops it inventing things.

The explainer describes decisions that have already been made, from facts that
have already been recorded. It cannot change an outcome, so the only way it can do
damage is by being confidently wrong on the one screen whose purpose is showing
that the numbers add up — which is why nearly everything here is about refusal.
"""

from __future__ import annotations

import json

from recoup.agent.llm.client import LLMClient
from recoup.agent.llm.explainer import (
    MAX_CHARS,
    CaseFacts,
    Explainer,
    summarise,
    validate,
)


def _facts(**overrides) -> CaseFacts:
    values = {
        "payment_id": "pay_000123",
        "amount_paise": 5_00_000,
        "method": "upi",
        "reason": "payment_timeout",
        "outcome": "stopped",
        "cause": "AUTH_ABANDONED",
        "recovered_paise": 0,
        "cost_paise": 35,
        "attempts": 0,
        "contacts": 1,
        "actions": ["NUDGE_WHATSAPP"],
        "channels": ["whatsapp"],
        "vetoes": [],
        "decisions": [],
    }
    values.update(overrides)
    return CaseFacts(**values)


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


def _batch(*cases: dict) -> str:
    return json.dumps({"cases": list(cases)})


def _explainer(response: str, tmp_path) -> Explainer:
    return Explainer(client=LLMClient(cache_dir=tmp_path, providers=[StubProvider(response)]))


# ---------------------------------------------------------------------------
# Grounding: the rule that matters
# ---------------------------------------------------------------------------


def test_an_explanation_using_only_given_numbers_passes():
    facts = _facts()

    text = "A ₹5,000 UPI payment timed out. We sent 1 WhatsApp message and it was not paid."

    assert validate(text, facts) is None


def test_a_number_that_is_not_in_the_record_is_refused():
    """The failure this exists to catch: plausible, specific, and invented."""
    facts = _facts(attempts=1)

    problem = validate("We retried the payment 3 times before giving up.", facts)

    assert "not in the record" in problem
    assert "3" in problem


def test_grouping_is_not_treated_as_a_different_number():
    """₹5,41,724 and 541724 are the same figure written two ways."""
    facts = _facts(amount_paise=5_41_724_00)

    assert validate("The ₹541724 payment failed.", facts) is None
    assert validate("The ₹5,41,724 payment failed.", facts) is None


def test_a_total_the_model_worked_out_itself_is_refused():
    """Arithmetic is not the model's job here, and a wrong sum is worse than none."""
    facts = _facts(amount_paise=5_00_000, cost_paise=35)

    assert validate("We spent ₹35 chasing ₹5,000, netting ₹4,965.", facts) is not None


# ---------------------------------------------------------------------------
# Channels and outcomes
# ---------------------------------------------------------------------------


def test_a_channel_that_was_not_used_is_refused():
    facts = _facts(channels=["whatsapp"])

    problem = validate("We sent an SMS and then called them.", facts)

    assert "channel that was not used" in problem


def test_the_channel_that_was_used_is_fine():
    assert validate("A WhatsApp message was sent.", _facts(channels=["whatsapp"])) is None


def test_claiming_a_recovery_that_did_not_happen_is_refused():
    facts = _facts(outcome="stopped")

    problem = validate("The payment was recovered the following day.", facts)

    assert "recovered when it was not" in problem


def test_the_same_claim_is_fine_when_it_is_true():
    facts = _facts(outcome="recovered", recovered_paise=5_00_000)

    assert validate("The payment was recovered the following day.", facts) is None


def test_explaining_why_nothing_was_recovered_is_not_a_false_claim():
    """"Recover" appears legitimately in sentences about not recovering."""
    facts = _facts(outcome="stopped")

    text = "We stopped once no further attempt was worth what it could recover."

    assert validate(text, facts) is None


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_empty_is_refused():
    assert validate("   ", _facts()) == "empty"


def test_an_essay_is_refused():
    assert "too long" in validate("word " * MAX_CHARS, _facts())


# ---------------------------------------------------------------------------
# The deterministic explanation
# ---------------------------------------------------------------------------


def test_every_case_gets_an_explanation_without_a_model():
    text = summarise(_facts())

    assert text
    assert "₹5,000" in text


def test_the_deterministic_explanation_passes_the_same_validator():
    """Holding our own prose to the standard we hold the model's.

    Easy to compose a summary that states a number the facts do not contain, and
    the deterministic path never goes through validation in normal operation.
    """
    cases = [
        _facts(),
        _facts(outcome="recovered", recovered_paise=5_00_000, attempts=2, contacts=0,
               channels=[], actions=["RETRY_NOW", "RETRY_SCHEDULED"]),
        _facts(actions=[], channels=[], contacts=0, cost_paise=0),
        _facts(cause=None, vetoes=["contact:quiet_hours", "attempts:cooling_off"]),
        _facts(contacts=3, channels=["sms", "voice"], actions=["NUDGE_SMS"] * 3),
    ]

    for facts in cases:
        problem = validate(summarise(facts), facts)
        assert problem is None, f"{facts.payment_id}: {problem}"


def test_a_case_nobody_touched_says_so():
    text = summarise(_facts(actions=[], channels=[], contacts=0, cost_paise=0))

    assert "no action" in text


def test_a_refused_case_names_the_rule():
    text = summarise(_facts(actions=[], channels=[], contacts=0, vetoes=["contact:quiet_hours"]))

    assert "quiet_hours" in text


def test_an_undiagnosed_case_does_not_invent_a_cause():
    text = summarise(_facts(cause=None))

    assert "could not be diagnosed" in text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_a_valid_explanation_is_used(tmp_path):
    facts = _facts()
    explainer = _explainer(
        _batch({"payment_id": "pay_000123", "text": "A ₹5,000 UPI payment timed out."}),
        tmp_path,
    )

    assert explainer.warm([facts]) == 1

    result = explainer.explain(facts)
    assert result.source == "generated"
    assert result.text == "A ₹5,000 UPI payment timed out."


def test_one_call_covers_every_case(tmp_path):
    """A call per case would be a call per page view, which is the whole problem."""
    provider = StubProvider(_batch())
    explainer = Explainer(client=LLMClient(cache_dir=tmp_path, providers=[provider]))

    explainer.warm([_facts(payment_id=f"pay_{n}") for n in range(5)])

    assert explainer.calls == 1
    assert provider.calls == 1


def test_a_failing_explanation_falls_back_and_is_recorded(tmp_path):
    facts = _facts(attempts=0)
    explainer = _explainer(
        _batch({"payment_id": "pay_000123", "text": "We retried it 7 times."}), tmp_path
    )
    explainer.warm([facts])

    result = explainer.explain(facts)

    assert result.source == "deterministic"
    assert "7 times" not in result.text
    assert explainer.rejections[0].payment_id == "pay_000123"
    assert "not in the record" in explainer.rejections[0].reason


def test_an_explanation_for_a_case_we_did_not_ask_about_is_dropped(tmp_path):
    """Nothing to validate it against, so there is no way to show it responsibly."""
    explainer = _explainer(
        _batch({"payment_id": "pay_999999", "text": "Something happened."}), tmp_path
    )

    assert explainer.warm([_facts()]) == 0
    assert explainer.rejections[0].reason == "not a case we asked about"


def test_one_bad_case_does_not_discard_the_others(tmp_path):
    good, bad = _facts(payment_id="pay_good"), _facts(payment_id="pay_bad")
    explainer = _explainer(
        _batch(
            {"payment_id": "pay_good", "text": "A ₹5,000 UPI payment timed out."},
            {"payment_id": "pay_bad", "text": "We tried 9 times."},
        ),
        tmp_path,
    )

    explainer.warm([good, bad])

    assert explainer.explain(good).source == "generated"
    assert explainer.explain(bad).source == "deterministic"


def test_unparseable_output_leaves_every_case_deterministic(tmp_path):
    explainer = _explainer("not json", tmp_path)

    assert explainer.warm([_facts()]) == 0
    assert explainer.explain(_facts()).source == "deterministic"


def test_with_no_model_nothing_is_generated():
    explainer = Explainer(use_llm=False)

    assert explainer.warm([_facts()]) == 0
    assert explainer.explain(_facts()).source == "deterministic"


def test_the_brief_is_what_grounds_the_check(tmp_path):
    """The model is validated against exactly the text it was shown."""
    facts = _facts()

    assert "pay_000123" in facts.brief()
    assert "₹5,000" in facts.brief()
    assert "AUTH_ABANDONED" in facts.brief()
