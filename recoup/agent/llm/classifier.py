"""Classifying the failures the rules could not.

This is the clearest case in the project for using a model, and it is worth being
precise about why, because the same reasoning rules a model *out* everywhere else.

Razorpay enumerates `source` and `step` exhaustively per payment method but
publishes `reason` only by example. So a ruleset written from the documentation is
structurally complete about *where* a payment broke and necessarily incomplete
about *why* — and the gap lands on vendor-specific reason strings nobody wrote
down. Those strings are natural language, they are unbounded, and new ones appear
without warning. Reading unfamiliar text and mapping it onto a known taxonomy is
exactly what a language model is for.

Two properties make it affordable and safe.

**It is one request per run, not one per payment.** Hundreds of unresolved
payments collapse to a handful of distinct (method, source, step, reason)
combinations — three, on the committed seed. Sending those three together fits any
free tier; sending 1,600 individually fits none.

**It proposes rules, not verdicts.** The model returns a mapping plus a rationale,
which is written to disk for review and can be promoted into
`config/classifier.yaml` by a person. The deterministic path grows; the model does
not silently become the classifier.

Nothing here is trusted blindly: a returned cause must be a real `FailureCause`,
confidence is capped below the deterministic rules, and anything unparseable
leaves the failure unresolved rather than guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from recoup.agent.classify import Classification, Resolution
from recoup.agent.llm.client import LLMClient
from recoup.domain import FailureCause

# A model's guess must never outrank a documented rule. Capped below the lowest
# deterministic confidence so that if a rule is ever added for the same symptoms,
# the rule wins without anyone having to remember to re-tune this.
MAX_FALLBACK_CONFIDENCE = 0.6

SYSTEM = """You classify failed Indian online payments for a revenue recovery system.

You are given payment failure symptoms that a deterministic rules engine could not
map. The rules were written from Razorpay's public documentation, which enumerates
`source` and `step` exhaustively but publishes `reason` only by example — so what
reaches you is almost always a vendor-specific reason string.

Map each to exactly one cause from this taxonomy. The taxonomy is about what to do
next, not about who is at fault:

- SOFT_ISSUER_DECLINE: the bank refused, but might not next time. Worth retrying later.
- INSUFFICIENT_FUNDS: not enough money or balance. Worth retrying after a salary credit.
- AUTH_ABANDONED: the customer started and did not finish — OTP not entered, session
  timed out, UPI collect request ignored. The instrument is fine.
- TECHNICAL_GATEWAY: infrastructure failed — gateway, network, PSP or bank outage.
  Retry once it recovers.
- INSTRUMENT_INVALID: the instrument cannot succeed as-is — expired card, invalid
  VPA, unlinked wallet. Only a new instrument can work.
- RISK_BLOCKED: blocked by a risk or fraud control. Never retried.
- MANDATE_PROBLEM: e-mandate revoked, paused, expired, or the debit exceeds its cap.
- CUSTOMER_INTENT: the customer deliberately cancelled.

Rules:
- Choose the single best cause. Do not invent causes.
- If genuinely ambiguous, prefer the more conservative reading: an unnecessary
  retry costs money and an attempt from a capped budget.
- Give a one-sentence rationale a payments engineer would accept.
- Return only JSON matching the requested shape."""


class ProposedMapping(BaseModel):
    """One answer, tied to its question by id.

    Matching by id rather than by position or by echoed input fields. The first
    version required the model to repeat `method`/`source`/`step`/`reason` back and
    then matched on those; the model answered correctly but returned only `cause`
    and `rationale`, so every mapping was discarded. Demanding an echo of the input
    is asking for something that adds no information and gives the response an
    extra way to fail.
    """

    id: int
    cause: str
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MappingBatch(BaseModel):
    mappings: list[ProposedMapping]


class LLMFallbackClassifier:
    """A batched fallback for symptoms the rules do not cover."""

    def __init__(
        self,
        client: LLMClient | None = None,
        proposals_path: str | Path = "reports/proposed_rules.json",
    ) -> None:
        self._client = client or LLMClient()
        self._proposals_path = Path(proposals_path)
        self._resolved: dict[tuple, Classification] = {}
        self.calls = 0

    @staticmethod
    def _key(fields: dict[str, str | None]) -> tuple:
        return tuple(fields.get(k) for k in ("method", "source", "step", "reason"))

    def warm(self, unresolved: list[dict[str, str | None]]) -> int:
        """Classify every distinct unresolved symptom in one request.

        Called once before a run. Returns how many combinations were resolved.
        """
        pending = [f for f in unresolved if self._key(f) not in self._resolved]
        if not pending:
            return 0

        prompt = json.dumps(
            {
                "instruction": (
                    "Classify each failure. Return one object per input with the "
                    'same "id", the chosen "cause", a "rationale", and a '
                    '"confidence" between 0 and 1. Wrap them in {"mappings": [...]}.'
                ),
                "failures": [{"id": index, **fields} for index, fields in enumerate(pending)],
            },
            indent=2,
        )

        batch = self._structured_batch(prompt)
        self.calls += 1

        if batch is None:
            return 0

        resolved = 0
        for mapping in batch.mappings:
            if not 0 <= mapping.id < len(pending):
                continue

            try:
                cause = FailureCause(mapping.cause)
            except ValueError:
                # A cause outside the taxonomy is not a near miss to be repaired.
                # It means the model invented one, and acting on it would send a
                # payment down a recovery path that does not exist.
                continue

            self._resolved[self._key(pending[mapping.id])] = Classification(
                cause=cause,
                confidence=min(mapping.confidence, MAX_FALLBACK_CONFIDENCE),
                resolution=Resolution.FALLBACK,
                rule_id="llm-fallback",
                note=mapping.rationale,
            )
            resolved += 1

        self._write_proposals(batch, pending)
        return resolved

    def _structured_batch(self, prompt: str) -> MappingBatch | None:
        """Accept the two shapes a model actually returns.

        Asked for `{"mappings": [...]}`, models frequently return the bare array
        instead. That is a formatting preference, not a wrong answer, and
        discarding a correct classification over it would be the tool being
        precious rather than careful.
        """
        batch = self._client.structured("classify-unresolved", SYSTEM, prompt, MappingBatch)
        if batch is not None:
            return batch

        try:
            completion = self._client.complete("classify-unresolved", SYSTEM, prompt)
            payload = json.loads(completion.text)
        except Exception:  # noqa: BLE001 — any failure here means no fallback, which is safe
            return None

        if isinstance(payload, list):
            for index, item in enumerate(payload):
                item.setdefault("id", index)
            return MappingBatch.model_validate({"mappings": payload})

        return None

    def _write_proposals(
        self, batch: MappingBatch, pending: list[dict[str, str | None]]
    ) -> None:
        """Write what the model proposed, for a human to promote into the rules.

        The deterministic ruleset is meant to grow. A mapping that holds up gets
        moved into `config/classifier.yaml` by a person, at which point the model
        stops being consulted for it — which is the direction this should travel.

        Written with the symptoms alongside the answer so the file is reviewable
        on its own, without re-running anything to find out what was asked.
        """
        proposals = [
            {"symptoms": pending[m.id], "cause": m.cause, "rationale": m.rationale}
            for m in batch.mappings
            if 0 <= m.id < len(pending)
        ]

        self._proposals_path.parent.mkdir(parents=True, exist_ok=True)
        self._proposals_path.write_text(
            json.dumps(proposals, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def __call__(self, fields: dict[str, str | None]) -> Classification | None:
        """The `Fallback` seam the deterministic classifier already expects."""
        return self._resolved.get(self._key(fields))
