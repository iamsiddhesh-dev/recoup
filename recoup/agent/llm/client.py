"""Talking to a model, without breaking reproducibility.

## The tension this file resolves

Recoup claims its numbers are reproducible from a seed on a clean clone. Language
models are not reproducible: same prompt, different day, different answer. Those
two facts are incompatible unless something sits between them.

That something is a **content-addressed cache on disk, committed to the repo**.
Every call is keyed by a hash of (purpose, model, system, prompt). The first run
populates it; every run after replays from it. So a judge cloning this repo
reproduces the exact reported numbers **with no API key at all**, and the LLM
becomes a build-time input rather than a runtime dependency.

This is not a workaround. A recovery agent whose reported results move because a
provider silently changed a checkpoint has not measured anything, and "we ran it
again and got a different number" is not an acceptable answer about money.

## Why the calls are batched

The free-tier budgets make per-payment calls impossible, not merely wasteful.
Gemini Flash allows 20 requests/day; Groq allows 200K tokens/day. A 5,000-payment
run has ~1,600 failures, so one call each would be 75× over one limit and 6× over
the other. Everything here is therefore shaped as *one request covering the whole
run* — see `agent/llm/classifier.py`, where hundreds of unresolved payments
collapse to three distinct symptom combinations and a single request.

## Provider order

Gemini Flash for the few real calls, Groq when Gemini errors or its daily budget
is spent. Missing credentials are not an error — the chain skips that provider,
and with none configured the client serves cache only.

There is deliberately no local-model provider. Its only job would have been
offline safety for a live demo, and the cache already guarantees that more
strongly: a cached run touches no network at all. Installing a multi-gigabyte
model server to duplicate what a few kilobytes of committed JSON already
guarantees is not a tradeoff worth making.

`RECOUP_LLM_DEV=1` swaps Gemini to the Gemma model, which allows 14,400
requests/day against Flash's 20 — enough to iterate on a prompt without spending
the budget reserved for the run whose cache gets committed. The model is part of
the cache key, so a dev answer is never served in place of a Flash one.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

DEFAULT_CACHE = Path("cache/llm")
T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """No provider could serve the call and the cache had no answer."""


@dataclass
class Completion:
    text: str
    model: str
    cached: bool
    latency_ms: int = 0


class Provider(Protocol):
    name: str
    model: str

    def available(self) -> bool: ...

    def complete(self, system: str, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Requests and tokens per minute, refilled continuously.

    Groq's free tier allows 8K tokens/minute, which one batched classification
    request can approach on its own. A plain retry-on-429 would burn the daily
    request budget discovering the same limit repeatedly, so the cost is paid in
    waiting rather than in failed calls.
    """

    requests_per_minute: int
    tokens_per_minute: int
    # Injectable so the refill behaviour can be tested without waiting a minute.
    now: Callable[[], float] = time.monotonic
    _requests: float = field(default=0.0, init=False)
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._requests = float(self.requests_per_minute)
        self._tokens = float(self.tokens_per_minute)
        self._last = self.now()

    def _refill(self) -> None:
        now = self.now()
        elapsed = now - self._last
        self._last = now
        self._requests = min(
            self.requests_per_minute, self._requests + elapsed * self.requests_per_minute / 60
        )
        self._tokens = min(
            self.tokens_per_minute, self._tokens + elapsed * self.tokens_per_minute / 60
        )

    def take(self, estimated_tokens: int, *, sleep=time.sleep) -> None:
        for _ in range(120):
            self._refill()
            if self._requests >= 1 and self._tokens >= estimated_tokens:
                self._requests -= 1
                self._tokens -= estimated_tokens
                return
            sleep(0.5)
        raise LLMUnavailable("rate limiter did not clear within 60s")


def estimate_tokens(text: str) -> int:
    """Roughly four characters per token. Deliberately conservative."""
    return max(1, len(text) // 3)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class GeminiProvider:
    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or _gemini_model()
        self._key = api_key or os.environ.get("GEMINI_API_KEY", "")

        # Flash: 5 RPM / 250K TPM / 20 RPD. Gemma: 30 RPM / 15K TPM / 14,400 RPD.
        # The narrower per-minute budget is used for both — waiting a little is
        # cheaper than discovering a limit by being refused.
        self._bucket = TokenBucket(requests_per_minute=5, tokens_per_minute=15_000)

    def available(self) -> bool:
        return bool(self._key)

    def complete(self, system: str, prompt: str) -> str:
        self._bucket.take(estimate_tokens(system + prompt))

        response = httpx.post(
            f"{self.BASE}/{self.model}:generateContent",
            params={"key": self._key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": prompt}]}],
                # Temperature zero is not determinism — providers still vary — but
                # it removes the sampling noise the cache would otherwise freeze.
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
            },
            timeout=90.0,
        )
        response.raise_for_status()
        body = response.json()
        return body["candidates"][0]["content"]["parts"][0]["text"]


class GroqProvider:
    name = "groq"
    BASE = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self._key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._bucket = TokenBucket(requests_per_minute=30, tokens_per_minute=8_000)

    def available(self) -> bool:
        return bool(self._key)

    def complete(self, system: str, prompt: str) -> str:
        self._bucket.take(estimate_tokens(system + prompt))

        response = httpx.post(
            self.BASE,
            headers={"Authorization": f"Bearer {self._key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=90.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE,
        providers: list[Provider] | None = None,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.providers = providers if providers is not None else default_providers()

        self.calls_made = 0
        self.cache_hits = 0

    @staticmethod
    def _key(purpose: str, model: str, system: str, prompt: str) -> str:
        digest = hashlib.sha256(
            "\x00".join((purpose, model, system, prompt)).encode()
        ).hexdigest()
        return f"{purpose}-{digest[:24]}"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def complete(self, purpose: str, system: str, prompt: str) -> Completion:
        """Answer from cache if possible, otherwise from the first live provider.

        The cache is keyed on the model too, so switching models is a cache miss
        rather than a silent reuse of another model's answer.
        """
        model = self.providers[0].model if self.providers else "none"
        key = self._key(purpose, model, system, prompt)

        cached = self._read_cache(key)
        if cached is not None:
            self.cache_hits += 1
            return Completion(text=cached["response"], model=cached["model"], cached=True)

        if self.offline:
            raise LLMUnavailable(
                f"offline and no cached response for {purpose}. Run once with "
                f"credentials to populate cache/llm, or check the prompt has not "
                f"changed."
            )

        last_error: Exception | None = None
        for provider in self.providers:
            if not provider.available():
                continue

            text = None
            for attempt in range(2):
                try:
                    started = time.monotonic()
                    text = provider.complete(system, prompt)
                    latency = int((time.monotonic() - started) * 1000)
                    break
                except Exception as exc:  # noqa: BLE001 — any failure falls through
                    last_error = exc
                    # A 503 means the model is busy, not that it will not answer.
                    # Gemini returned one mid-build with "spikes in demand are
                    # usually temporary"; falling straight through to a weaker
                    # provider on that would be giving up early.
                    if attempt == 0 and _is_transient(exc):
                        time.sleep(2.0)
                        continue
                    break

            if text is None:
                continue

            self.calls_made += 1
            self._write_cache(
                key,
                {
                    "purpose": purpose,
                    "provider": provider.name,
                    "model": provider.model,
                    "system": system,
                    "prompt": prompt,
                    "response": text,
                },
            )
            return Completion(text=text, model=provider.model, cached=False, latency_ms=latency)

        raise LLMUnavailable(
            f"no provider could serve {purpose}"
            + (f" (last error: {last_error})" if last_error else " (none configured)")
        )

    def structured(
        self, purpose: str, system: str, prompt: str, model: type[T]
    ) -> T | None:
        """A completion parsed into a schema, or None if it will not parse.

        Returning None rather than raising is deliberate. Every use of a model in
        this project is a *fallback* — something the deterministic path could not
        do — so a malformed answer must degrade to the deterministic behaviour
        rather than take down the run. A recovery agent that stops recovering
        because a model returned bad JSON has the failure mode backwards.
        """
        try:
            completion = self.complete(purpose, system, prompt)
        except LLMUnavailable:
            return None

        try:
            return model.model_validate_json(completion.text)
        except ValidationError:
            pass

        # Providers sometimes wrap JSON in prose or a fenced block despite being
        # asked not to. One cheap salvage attempt, then give up.
        text = completion.text.strip()
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return model.model_validate_json(text[start : end + 1])
            except ValidationError:
                return None
        return None


def _is_transient(exc: Exception) -> bool:
    """Whether an error is worth trying the same provider again for.

    Server-side capacity and rate limiting; not a malformed request, which will
    fail identically however many times it is sent.
    """
    response = getattr(exc, "response", None)
    return response is not None and response.status_code in (429, 500, 502, 503, 504)


def _gemini_model() -> str:
    """Flash for real runs, Gemma while iterating.

    Flash allows 20 requests a day. Changing a prompt four times exhausts a fifth
    of that, so development runs against Gemma (14,400/day on the same key) and the
    committed cache is regenerated with Flash once the prompt has settled.
    """
    if os.environ.get("RECOUP_LLM_DEV", "").strip().lower() in ("1", "true", "yes"):
        return os.environ.get("GEMINI_DEV_MODEL", "gemma-4-26b")
    return os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")


def default_providers() -> list[Provider]:
    return [GeminiProvider(), GroqProvider()]
