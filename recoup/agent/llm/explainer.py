"""Explaining one payment's handling to the merchant who lost the money.

Case Detail already shows the arithmetic — every candidate action, its expected
value as a checkable sum, and the compliance rules that fired. That is the right
artifact for someone auditing a decision. It is the wrong one for someone asking
"so what happened to this payment, and why did nobody chase it?"

This writes the second thing. The model narrates a decision that has *already been
made*, from facts that have already been recorded. It has no influence on any
action, cannot change an outcome, and is not in the ablation — it is a reading of
the ledger, not a participant in it.

## The rule that makes it safe

**The model may not use a number it was not given.** Every digit run in the
generated text has to appear in the brief it was handed. That is a stronger and
much simpler check than asking whether a claim is true: a narrative that invents
"three retry attempts" for a payment that had one is worse than no narrative,
because it is wrong in the confident register of a system report, on the screen
whose whole purpose is showing that the numbers add up.

The same reasoning as the copywriter, arrived at from the other side. There the
model never sees a number; here it may only echo the ones it was shown.

Channels and outcomes are checked the same way — an explanation cannot mention a
WhatsApp message on a payment that only got an SMS, and cannot say a payment was
recovered when it was not.

## What happens when it fails

`summarise()` produces a deterministic explanation from the same facts, and it is
used whenever there is no model, no cached answer, or the generated text fails
validation. It is not a stub: the case screen always has something to say, and the
model's version is an improvement on it rather than a prerequisite for it.

## Budget

One batched call covers every case in the selection. Explaining on demand would be
one call per page view — a judge clicking through ten cases would spend half the
daily Gemini allowance on a read-only screen, and two visits to the same case
could disagree with each other.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from recoup.agent.llm.client import LLMClient
from recoup.money import rupees

# Longer than this stops being an explanation and starts being a report. Four
# sentences is the working limit; the cap is generous enough not to fire on
# reasonable prose and tight enough to catch a model that will not stop.
MAX_CHARS = 700

CHANNELS = ("sms", "whatsapp", "voice", "email")

ESCALATE = "ESCALATE_HUMAN"

# Claims that are simply false when the payment was not recovered. Deliberately
# specific phrases rather than the word "recovered", which appears legitimately in
# sentences like "we stopped before spending more than the payment could recover".
RECOVERY_CLAIMS = re.compile(
    r"\b(was recovered|were recovered|successfully recovered|payment succeeded"
    r"|the payment was collected|we recovered it)\b",
    re.IGNORECASE,
)

DIGITS = re.compile(r"\d+")

SYSTEM = (
    "You explain payment recovery decisions to the merchant who lost the money. "
    "You are given the complete record of one failed payment: what failed, what the "
    "agent decided, what it was refused, and how it ended.\n\n"
    "Write two to four plain sentences. Say what went wrong, what was done about it "
    "and why, and how it ended. Where the agent chose not to act, say so and give "
    "the reason — a deliberate refusal is a result, not an absence of one.\n\n"
    "Absolute rules:\n"
    "- Use ONLY numbers that appear in the brief. Never estimate, round differently, "
    "total two figures together, or introduce a new one.\n"
    "- Never mention a channel that was not used.\n"
    "- Never say a payment was recovered unless the brief says it was.\n"
    "- No greeting, no sign-off, no bullet points, no markdown. Prose only.\n"
    "- Write for a merchant, not an engineer: no enum names, no field names."
)


@dataclass(frozen=True)
class CaseFacts:
    """Everything the model is shown, and the only thing it may draw on.

    Built by the caller from the ledger rather than read from it here, which keeps
    this module unaware of both the ledger and the web layer — it takes facts and
    returns prose.
    """

    payment_id: str
    amount_paise: int
    method: str
    reason: str
    outcome: str
    cause: str | None = None
    recovered_paise: int = 0
    cost_paise: int = 0
    attempts: int = 0
    contacts: int = 0
    actions: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        return self.outcome == "recovered"

    def brief(self) -> str:
        """The case as the model sees it.

        Also the grounding corpus: validation checks generated numbers against
        this exact text, so anything not stated here cannot legitimately appear in
        the explanation.
        """
        lines = [
            f"payment: {self.payment_id}",
            f"amount: {rupees(self.amount_paise)}",
            f"method: {self.method or 'unknown'}",
            f"reported failure: {self.reason or 'not reported'}",
            f"diagnosed cause: {self.cause or 'could not be determined'}",
            f"outcome: {self.outcome}",
            f"retry attempts: {self.attempts}",
            f"messages sent: {self.contacts}",
            # Precise, because a single WhatsApp send costs 35 paise and a brief
            # saying "spent ₹0" invites the model to report that nothing was spent.
            f"spent on recovery: {rupees(self.cost_paise, precise_below=10_000)}",
        ]
        if self.recovered:
            lines.append(f"amount recovered: {rupees(self.recovered_paise)}")
        if self.actions:
            # Counted rather than listed. One payment escalated twelve times
            # produced twelve identical entries, which cost tokens, read badly,
            # and — because the *total* appeared nowhere — left "acted 12 times"
            # ungrounded even though it was true.
            tally = Counter(self.actions)
            detail = ", ".join(f"{a} ×{n}" if n > 1 else a for a, n in tally.items())
            lines.append(f"actions taken: {len(self.actions)} in total — {detail}")
        if self.decisions:
            # Deduplicated: the same decision reached twelve times is one piece of
            # information, and repeating it invites the model to narrate a
            # sequence of distinct events that did not happen.
            lines.append("reasoning recorded at the time:")
            lines += [f"  - {d}" for d in dict.fromkeys(self.decisions)]
        if self.vetoes:
            lines.append("actions compliance refused:")
            lines += [f"  - {v}" for v in dict.fromkeys(self.vetoes)]
        return "\n".join(lines)


@dataclass(frozen=True)
class Explanation:
    text: str
    source: str  # "generated" or "deterministic"


@dataclass(frozen=True)
class Rejection:
    payment_id: str
    reason: str
    text: str


class GeneratedExplanation(BaseModel):
    payment_id: str
    text: str


class ExplanationBatch(BaseModel):
    cases: list[GeneratedExplanation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _digit_runs(text: str) -> set[str]:
    """Digit sequences, with grouping separators removed first.

    ₹5,41,724 and 541724 are the same number written two ways, and a validator
    that treated them as different would reject correct prose for reformatting.
    """
    return set(DIGITS.findall(text.replace(",", "")))


def validate(text: str, facts: CaseFacts) -> str | None:
    """Why this explanation cannot be shown, or None if it can."""
    text = text.strip()
    if not text:
        return "empty"

    if len(text) > MAX_CHARS:
        return f"too long: {len(text)} > {MAX_CHARS}"

    ungrounded = _digit_runs(text) - _digit_runs(facts.brief())
    if ungrounded:
        return f"uses a number not in the record: {', '.join(sorted(ungrounded))}"

    lowered = text.lower()
    used = {c.lower() for c in facts.channels}
    invented = [c for c in CHANNELS if c in lowered and c not in used]
    if invented:
        return f"mentions a channel that was not used: {', '.join(invented)}"

    if not facts.recovered and RECOVERY_CLAIMS.search(text):
        return "claims the payment was recovered when it was not"

    return None


# ---------------------------------------------------------------------------
# The deterministic explanation
# ---------------------------------------------------------------------------


def _times(n: int) -> str:
    """"1 times" is the kind of thing that makes a report look generated."""
    return f"{n} time" if n == 1 else f"{n} times"


def summarise(facts: CaseFacts) -> str:
    """An explanation composed from the facts, with no model involved.

    Always available, and the reason the case screen never depends on a model
    being reachable. Assembled rather than templated per cause, so a case with an
    unusual shape — refused everything, recovered on the first retry, never
    classified — still reads as a sentence about that case.
    """
    amount = rupees(facts.amount_paise)
    cause = facts.cause.replace("_", " ").lower() if facts.cause else None

    opening = (
        f"A {amount} {facts.method or 'payment'} failed"
        + (f", diagnosed as {cause}." if cause else " and could not be diagnosed.")
    )

    escalated = ESCALATE in facts.actions

    if facts.actions:
        did = []
        if facts.attempts:
            did.append(f"retried it {_times(facts.attempts)}")
        if facts.contacts:
            channels = " and ".join(dict.fromkeys(facts.channels)) or "the customer"
            did.append(f"contacted the customer {_times(facts.contacts)} by {channels}")
        if escalated:
            did.append("handed it to human review")
        spent = rupees(facts.cost_paise, precise_below=10_000)
        middle = (
            f" The agent {' and '.join(did)}, spending {spent}."
            if did
            else f" The agent acted {_times(len(facts.actions))}, spending {spent}."
        )
    else:
        middle = " The agent took no action on it."

    if facts.recovered:
        closing = f" It was recovered, returning {rupees(facts.recovered_paise)}."
    elif escalated:
        # Whether a person then resolved it is outside this system, and the run
        # does not claim credit for it either way.
        closing = " The outcome of that review is outside this record."
    elif facts.vetoes:
        rules = ", ".join(dict.fromkeys(facts.vetoes))
        closing = f" It was not recovered. Compliance refused further action under {rules}."
    else:
        closing = " It was not recovered, and the agent stopped when no option was worth its cost."

    return opening + middle + closing


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _build_prompt(cases: list[CaseFacts]) -> str:
    briefs = "\n\n".join(f"=== case {c.payment_id} ===\n{c.brief()}" for c in cases)
    return (
        f"Explain each of the following {len(cases)} payments.\n\n"
        f"{briefs}\n\n"
        "Return JSON only:\n"
        '{"cases": [{"payment_id": "...", "text": "..."}]}\n\n'
        "One entry per case, using the payment id exactly as given. "
        "Two to four sentences each. Only numbers that appear in that case's brief."
    )


class Explainer:
    """Generates explanations for a set of cases in one call, serves them per case."""

    def __init__(self, client: LLMClient | None = None, use_llm: bool = True) -> None:
        self._client = client if client is not None else (LLMClient() if use_llm else None)
        self._texts: dict[str, str] = {}
        self.rejections: list[Rejection] = []
        self.calls = 0

    def warm(self, cases: list[CaseFacts]) -> int:
        """Generate explanations for every case at once. Returns how many were kept.

        One call regardless of how many cases: they are short, the whole selection
        fits comfortably inside a single request, and sending them together means
        the model describes them in a consistent voice.
        """
        if self._client is None or not cases:
            return 0

        batch = self._client.structured(
            "case-explanations", SYSTEM, _build_prompt(cases), ExplanationBatch
        )
        self.calls += 1

        if batch is None:
            return 0

        by_id = {c.payment_id: c for c in cases}
        kept = 0

        for item in batch.cases:
            facts = by_id.get(item.payment_id)
            if facts is None:
                # A payment id we did not ask about. Nothing to validate it
                # against, so there is no way to show it responsibly.
                self.rejections.append(
                    Rejection(item.payment_id, "not a case we asked about", item.text)
                )
                continue

            problem = validate(item.text, facts)
            if problem:
                self.rejections.append(Rejection(item.payment_id, problem, item.text))
                continue

            self._texts[item.payment_id] = item.text.strip()
            kept += 1

        return kept

    def explain(self, facts: CaseFacts) -> Explanation:
        """The generated explanation if one survived validation, else the composed one."""
        text = self._texts.get(facts.payment_id)
        if text:
            return Explanation(text=text, source="generated")
        return Explanation(text=summarise(facts), source="deterministic")
