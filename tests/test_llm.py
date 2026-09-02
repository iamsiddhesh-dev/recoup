"""The model layer.

No test here touches a network. That is not only for speed: the committed cache in
`cache/llm/` is the mechanism that makes this project's numbers reproducible
without an API key, so exercising the cached path *is* exercising the thing that
matters.

The behaviour worth pinning down is what happens when the model is unavailable or
wrong. Every use of a model in Recoup is a fallback for something the
deterministic path could not do, so a bad response must degrade to the
deterministic behaviour rather than take down a run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recoup.agent.classify import Classifier, Resolution
from recoup.agent.llm.classifier import MAX_FALLBACK_CONFIDENCE, LLMFallbackClassifier
from recoup.agent.llm.client import (
    LLMClient,
    LLMUnavailable,
    TokenBucket,
    estimate_tokens,
)
from recoup.domain import FailureCause

CACHE = Path("cache/llm")


class StubProvider:
    """A provider that returns whatever it was handed."""

    name = "stub"
    model = "stub-1"

    def __init__(self, response: str = "{}", fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider exploded")
        return self.response


class UnavailableProvider(StubProvider):
    name = "unavailable"

    def available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class FakeClock:
    """A clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def test_the_bucket_allows_traffic_within_budget():
    clock = FakeClock()
    bucket = TokenBucket(requests_per_minute=5, tokens_per_minute=10_000, now=clock)

    for _ in range(5):
        bucket.take(100, sleep=clock.sleep)

    assert clock.t == 0.0, "nothing should have had to wait"


def test_the_bucket_waits_rather_than_failing_when_tokens_run_out():
    """Groq allows 8K tokens a minute; one batched request can approach that.

    Waiting is cheaper than discovering the limit by being refused, because a
    rejected call still spends from the daily request budget.
    """
    clock = FakeClock()
    bucket = TokenBucket(requests_per_minute=60, tokens_per_minute=1_000, now=clock)

    bucket.take(900, sleep=clock.sleep)
    bucket.take(900, sleep=clock.sleep)

    assert clock.t > 0.0, "expected the second call to wait for a refill"


def test_the_bucket_gives_up_rather_than_waiting_forever():
    """A request larger than the whole per-minute budget can never clear."""
    clock = FakeClock()
    bucket = TokenBucket(requests_per_minute=60, tokens_per_minute=1_000, now=clock)

    with pytest.raises(LLMUnavailable, match="did not clear"):
        bucket.take(50_000, sleep=clock.sleep)


def test_token_estimation_is_conservative():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("x" * 300) >= 100


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_a_response_is_cached_and_replayed(tmp_path):
    provider = StubProvider('{"ok": true}')
    client = LLMClient(cache_dir=tmp_path, providers=[provider])

    first = client.complete("test", "system", "prompt")
    second = client.complete("test", "system", "prompt")

    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert provider.calls == 1


def test_a_changed_prompt_is_a_cache_miss(tmp_path):
    provider = StubProvider('{"ok": true}')
    client = LLMClient(cache_dir=tmp_path, providers=[provider])

    client.complete("test", "system", "prompt one")
    client.complete("test", "system", "prompt two")

    assert provider.calls == 2


def test_a_different_model_is_a_cache_miss(tmp_path):
    """Switching models must not silently reuse another model's answer."""
    first = StubProvider('{"a": 1}')
    second = StubProvider('{"b": 2}')
    second.model = "stub-2"

    LLMClient(cache_dir=tmp_path, providers=[first]).complete("test", "s", "p")
    LLMClient(cache_dir=tmp_path, providers=[second]).complete("test", "s", "p")

    assert first.calls == 1
    assert second.calls == 1


def test_offline_serves_cache_and_refuses_to_call_out(tmp_path):
    """How a clean clone reproduces the reported numbers with no key."""
    provider = StubProvider('{"ok": true}')
    LLMClient(cache_dir=tmp_path, providers=[provider]).complete("test", "s", "p")

    offline = LLMClient(cache_dir=tmp_path, providers=[provider], offline=True)

    assert offline.complete("test", "s", "p").cached is True
    with pytest.raises(LLMUnavailable, match="offline"):
        offline.complete("test", "s", "different prompt")


# ---------------------------------------------------------------------------
# Provider chain
# ---------------------------------------------------------------------------


def test_an_unavailable_provider_is_skipped(tmp_path):
    working = StubProvider('{"ok": true}')
    client = LLMClient(cache_dir=tmp_path, providers=[UnavailableProvider(), working])

    client.complete("test", "s", "p")

    assert working.calls == 1


def test_a_failing_provider_falls_through_to_the_next(tmp_path):
    broken = StubProvider(fail=True)
    working = StubProvider('{"ok": true}')
    client = LLMClient(cache_dir=tmp_path, providers=[broken, working])

    assert client.complete("test", "s", "p").text == '{"ok": true}'
    assert broken.calls == 1
    assert working.calls == 1


def test_no_providers_at_all_is_not_a_crash(tmp_path):
    client = LLMClient(cache_dir=tmp_path, providers=[])
    with pytest.raises(LLMUnavailable):
        client.complete("test", "s", "p")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_prose_wrapped_json_is_salvaged(tmp_path):
    """Models wrap JSON in fences despite being told not to.

    A formatting preference is not a wrong answer, and discarding a correct
    classification over one would be the tool being precious rather than careful.
    """
    from pydantic import BaseModel

    class Shape(BaseModel):
        value: int

    provider = StubProvider('Here you go:\n```json\n{"value": 7}\n```')
    client = LLMClient(cache_dir=tmp_path, providers=[provider])

    assert client.structured("test", "s", "p", Shape).value == 7


def test_unparseable_output_returns_none_rather_than_raising(tmp_path):
    """A recovery agent that stops recovering because of bad JSON is backwards."""
    from pydantic import BaseModel

    class Shape(BaseModel):
        value: int

    provider = StubProvider("not json at all")
    client = LLMClient(cache_dir=tmp_path, providers=[provider])

    assert client.structured("test", "s", "p", Shape) is None


def test_an_unavailable_model_returns_none_rather_than_raising(tmp_path):
    from pydantic import BaseModel

    class Shape(BaseModel):
        value: int

    client = LLMClient(cache_dir=tmp_path, providers=[])
    assert client.structured("test", "s", "p", Shape) is None


# ---------------------------------------------------------------------------
# The fallback classifier
# ---------------------------------------------------------------------------


SYMPTOMS = [
    {
        "method": "netbanking",
        "source": "issuer_bank",
        "step": "payment_authentication",
        "reason": "bank_unavailable",
    },
    {
        "method": "wallet",
        "source": "customer",
        "step": "payment_eligibility_check",
        "reason": "wallet_not_linked",
    },
]


def _fallback(tmp_path, response: str) -> LLMFallbackClassifier:
    return LLMFallbackClassifier(
        client=LLMClient(cache_dir=tmp_path, providers=[StubProvider(response)]),
        proposals_path=tmp_path / "proposed_rules.json",
    )


def test_a_batch_of_symptoms_resolves_in_one_call(tmp_path):
    """One request per run, not one per payment. The free tiers require it."""
    response = json.dumps(
        {
            "mappings": [
                {"id": 0, "cause": "TECHNICAL_GATEWAY", "rationale": "bank is down"},
                {"id": 1, "cause": "INSTRUMENT_INVALID", "rationale": "needs relinking"},
            ]
        }
    )
    fallback = _fallback(tmp_path, response)

    assert fallback.warm(SYMPTOMS) == 2
    assert fallback.calls == 1
    assert fallback(SYMPTOMS[0]).cause is FailureCause.TECHNICAL_GATEWAY
    assert fallback(SYMPTOMS[1]).cause is FailureCause.INSTRUMENT_INVALID


def test_a_bare_array_is_accepted(tmp_path):
    """What the model actually returned the first time it was asked.

    Requiring `{"mappings": [...]}` and rejecting a bare array discarded three
    correct classifications.
    """
    response = json.dumps(
        [
            {"cause": "TECHNICAL_GATEWAY", "rationale": "down"},
            {"cause": "INSTRUMENT_INVALID", "rationale": "relink"},
        ]
    )
    fallback = _fallback(tmp_path, response)

    assert fallback.warm(SYMPTOMS) == 2


def test_an_invented_cause_is_discarded(tmp_path):
    """Not a near miss to repair — acting on it routes a payment nowhere."""
    response = json.dumps(
        {"mappings": [{"id": 0, "cause": "BANK_HAVING_A_BAD_DAY", "rationale": "made up"}]}
    )
    fallback = _fallback(tmp_path, response)

    assert fallback.warm(SYMPTOMS) == 0
    assert fallback(SYMPTOMS[0]) is None


def test_confidence_is_capped_below_the_deterministic_rules(tmp_path):
    """A guess must never outrank a documented rule."""
    mapping = {"id": 0, "cause": "TECHNICAL_GATEWAY", "confidence": 0.99, "rationale": "x"}
    fallback = _fallback(tmp_path, json.dumps({"mappings": [mapping]}))
    fallback.warm(SYMPTOMS)

    assert fallback(SYMPTOMS[0]).confidence <= MAX_FALLBACK_CONFIDENCE


def test_an_out_of_range_id_is_ignored(tmp_path):
    mapping = {"id": 99, "cause": "TECHNICAL_GATEWAY", "rationale": "x"}
    fallback = _fallback(tmp_path, json.dumps({"mappings": [mapping]}))

    assert fallback.warm(SYMPTOMS) == 0


def test_proposals_are_written_for_human_review(tmp_path):
    """The deterministic ruleset is meant to grow, by a person promoting these."""
    response = json.dumps(
        {"mappings": [{"id": 0, "cause": "TECHNICAL_GATEWAY", "rationale": "bank is down"}]}
    )
    fallback = _fallback(tmp_path, response)
    fallback.warm(SYMPTOMS)

    written = json.loads((tmp_path / "proposed_rules.json").read_text(encoding="utf-8"))

    assert written[0]["cause"] == "TECHNICAL_GATEWAY"
    assert written[0]["symptoms"]["reason"] == "bank_unavailable"


def test_the_fallback_plugs_into_the_deterministic_classifier(tmp_path):
    """It sees only what the rules could not resolve, and marks itself as fallback."""
    response = json.dumps(
        {"mappings": [{"id": 0, "cause": "TECHNICAL_GATEWAY", "rationale": "down"}]}
    )
    fallback = _fallback(tmp_path, response)
    fallback.warm([SYMPTOMS[0]])

    classifier = Classifier(fallback=fallback)

    from recoup.domain import PaymentEntity, PaymentMethod, PaymentStatus

    payment = PaymentEntity(
        id="pay_x",
        amount=100000,
        status=PaymentStatus.FAILED,
        order_id="order_x",
        method=PaymentMethod.NETBANKING,
        error_source="issuer_bank",
        error_step="payment_authentication",
        error_reason="bank_unavailable",
    )

    result = classifier.classify(payment)
    assert result.cause is FailureCause.TECHNICAL_GATEWAY
    assert result.resolution is Resolution.FALLBACK


# ---------------------------------------------------------------------------
# The committed cache
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CACHE.exists(), reason="cache not populated")
def test_the_committed_cache_holds_a_real_classification():
    """What lets a clean clone reproduce the reported numbers with no API key."""
    entries = list(CACHE.glob("classify-unresolved-*.json"))
    assert entries, "expected a committed classification response"

    payload = json.loads(entries[0].read_text(encoding="utf-8"))

    assert payload["provider"]
    assert payload["model"]
    assert "TECHNICAL_GATEWAY" in payload["response"]
