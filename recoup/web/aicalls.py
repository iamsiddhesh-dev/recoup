"""Every model call this project makes, and what was done with the answer.

The plan called for Langfuse here. Then the response cache turned out to be a
better trace than a trace: `cache/llm/` holds the exact system prompt, the exact
user prompt and the exact response for every call, content-addressed and committed
to the repository. A reader who clones this can see what the model was asked and
what it said, offline, with no account and no network, three years from now. A
hosted dashboard behind a login is worse evidence than a file they already have.

So this screen reads that directory. Two things make it more than a log viewer:

**It re-runs the validators live.** Every accepted and rejected count on this page
is computed by putting the cached response back through the same `validate()` the
run used — not remembered from when the run happened. If a validator is weakened,
the numbers here move on the next page load. Showing a stored "0 rejected" would
prove nothing; recomputing it proves the check exists and still runs.

**It shows the counterfactual.** The interesting fact about this system's model use
is how little of it there is. Five calls, where the obvious design would make one
per failure. That comparison is the point, so it is on the page rather than in a
README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from recoup.agent.llm.classifier import MappingBatch
from recoup.agent.llm.copywriter import LIMITS, CopyMatrix
from recoup.agent.llm.copywriter import validate as validate_copy
from recoup.agent.llm.explainer import ExplanationBatch
from recoup.agent.llm.explainer import validate as validate_explanation

DEFAULT_CACHE = Path("cache/llm")

# Roughly four characters to a token. Good enough to say "this run cost about two
# thousand tokens, not a million", which is the only claim being made.
CHARS_PER_TOKEN = 4

# What a per-payment design would have cost on the committed seed. The comparison
# is the argument, so the numbers behind it are named rather than asserted.
FAILURES_IN_RUN = 1605
GEMINI_FLASH_DAILY_REQUESTS = 20


# An output can be accepted, refused, or not checkable at all. The third is not a
# shade of the second: on a clean clone with no run generated, the case records an
# explanation would be checked against do not exist yet, and reporting that as
# "refused" would blame the model for a missing file.
ACCEPTED, REFUSED, UNCHECKED = "accepted", "refused", "unchecked"


@dataclass
class Check:
    """One item from a response, and what validation made of it."""

    label: str
    status: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == ACCEPTED


@dataclass
class Call:
    """One cached model call."""

    purpose: str
    provider: str
    model: str
    system: str
    prompt: str
    response: str
    checks: list[Check] = field(default_factory=list)

    @property
    def what(self) -> str:
        """Plain-language purpose. The cache key is a slug, not a sentence."""
        if self.purpose.startswith("nudge-copy-"):
            return f"Recovery message templates — {self.purpose.removeprefix('nudge-copy-')}"
        return {
            "classify-unresolved": "Root causes for symptoms the rules could not map",
            "case-explanations": "Plain-language explanations of selected cases",
        }.get(self.purpose, self.purpose)

    @property
    def rules(self) -> list[str]:
        """What each output was checked against.

        Named on the page because "0 refused" is otherwise indistinguishable from
        "nothing was checked", and the second is what a sceptical reader should
        assume by default.
        """
        if self.purpose.startswith("nudge-copy-"):
            return [
                "no literal digits — the model writes {amount}, code fills it in",
                "no request for an OTP, CVV, PIN or card number",
                "no URL the model invented",
                "no placeholder we do not substitute",
                "fits the channel once the real values are substituted",
            ]
        if self.purpose == "case-explanations":
            return [
                "every number appears in that case's own record",
                "no channel that was not used",
                "no claim of a recovery that did not happen",
                "short enough to be an explanation rather than a report",
            ]
        if self.purpose == "classify-unresolved":
            return [
                "parses, and names a cause in the taxonomy",
                "confidence capped below any documented rule, so it cannot outrank one",
                "written to reports/proposed_rules.json for a human to approve",
            ]
        return []

    @property
    def tokens(self) -> int:
        return (len(self.system) + len(self.prompt) + len(self.response)) // CHARS_PER_TOKEN

    @property
    def accepted(self) -> int:
        return sum(1 for c in self.checks if c.status == ACCEPTED)

    @property
    def rejected(self) -> list[Check]:
        return [c for c in self.checks if c.status == REFUSED]

    @property
    def unchecked(self) -> list[Check]:
        return [c for c in self.checks if c.status == UNCHECKED]


@dataclass
class AICallsView:
    calls: list[Call] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.calls)

    @property
    def providers(self) -> list[str]:
        return sorted({c.provider for c in self.calls})

    @property
    def models(self) -> list[str]:
        return sorted({c.model for c in self.calls})

    @property
    def checked(self) -> int:
        return sum(len(c.checks) - len(c.unchecked) for c in self.calls)

    @property
    def accepted(self) -> int:
        return sum(c.accepted for c in self.calls)

    @property
    def rejected(self) -> int:
        return sum(len(c.rejected) for c in self.calls)

    @property
    def unchecked(self) -> int:
        return sum(len(c.unchecked) for c in self.calls)

    @property
    def per_payment_calls(self) -> int:
        """What one call per failure would have meant."""
        return FAILURES_IN_RUN

    @property
    def over_budget(self) -> int:
        """How many times over the daily allowance that design would run."""
        return round(FAILURES_IN_RUN / GEMINI_FLASH_DAILY_REQUESTS)


# ---------------------------------------------------------------------------
# Re-validating each kind of response
# ---------------------------------------------------------------------------


def _check_copy(response: str) -> list[Check]:
    """Every generated message, back through the copy validator."""
    try:
        matrix = CopyMatrix.model_validate_json(response)
    except Exception:  # noqa: BLE001 — an unparseable response is itself the finding
        return [Check(label="response did not parse", status=REFUSED, reason="invalid JSON")]

    checks: list[Check] = []
    for variant in matrix.variants:
        for channel in LIMITS:
            text = variant.for_channel(channel).strip()
            problem = validate_copy(text, channel)
            checks.append(
                Check(
                    label=f"{variant.cause} · {channel}",
                    status=ACCEPTED if problem is None else REFUSED,
                    reason=problem or "",
                )
            )
    return checks


def _check_explanations(response: str, facts_by_id: dict) -> list[Check]:
    """Every explanation, back through the grounding check against its own case."""
    try:
        batch = ExplanationBatch.model_validate_json(response)
    except Exception:  # noqa: BLE001
        return [Check(label="response did not parse", status=REFUSED, reason="invalid JSON")]

    checks: list[Check] = []
    for item in batch.cases:
        facts = facts_by_id.get(item.payment_id)
        if facts is None:
            # No run loaded is an absence of evidence; a run loaded that does not
            # contain this payment is evidence of a problem.
            checks.append(
                Check(
                    label=item.payment_id,
                    status=UNCHECKED if not facts_by_id else REFUSED,
                    reason=(
                        "no run loaded, so there is no record to check it against"
                        if not facts_by_id
                        else "not a case we asked about, so nothing to check it against"
                    ),
                )
            )
            continue
        problem = validate_explanation(item.text, facts)
        checks.append(
            Check(
                label=item.payment_id,
                status=ACCEPTED if problem is None else REFUSED,
                reason=problem or "",
            )
        )
    return checks


def _check_mappings(response: str) -> list[Check]:
    """Proposed causes are advisory: a human reviews them before they become rules.

    So the check here is narrow by design — that the answer parses and names a
    cause. The classifier caps their confidence so a proposal can never outrank a
    documented rule, which is the control that actually matters.
    """
    try:
        batch = MappingBatch.model_validate_json(response)
    except Exception:  # noqa: BLE001
        return [Check(label="response did not parse", status=REFUSED, reason="invalid JSON")]

    return [
        Check(
            label=f"symptom {m.id} → {m.cause}",
            status=ACCEPTED if m.cause else REFUSED,
        )
        for m in batch.mappings
    ]


def build_ai_calls(
    cache_dir: str | Path = DEFAULT_CACHE, facts_by_id: dict | None = None
) -> AICallsView:
    """Read the committed cache and re-check everything in it.

    Sorted by purpose so the order is stable between loads — the filenames are
    content hashes, and a page whose rows move when a prompt changes is harder to
    read than one that does not.
    """
    directory = Path(cache_dir)
    if not directory.exists():
        return AICallsView()

    calls: list[Call] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        call = Call(
            purpose=payload.get("purpose", path.stem),
            provider=payload.get("provider", "unknown"),
            model=payload.get("model", "unknown"),
            system=payload.get("system", ""),
            prompt=payload.get("prompt", ""),
            response=payload.get("response", ""),
        )

        if call.purpose.startswith("nudge-copy-"):
            call.checks = _check_copy(call.response)
        elif call.purpose == "case-explanations":
            call.checks = _check_explanations(call.response, facts_by_id or {})
        elif call.purpose == "classify-unresolved":
            call.checks = _check_mappings(call.response)

        calls.append(call)

    calls.sort(key=lambda c: c.purpose)
    return AICallsView(calls=calls)
