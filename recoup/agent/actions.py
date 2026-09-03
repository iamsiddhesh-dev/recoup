"""What the agent can do, and the record of why it chose to.

The action set is short on purpose. Every entry is something the adapter can
actually execute against real Razorpay — see `supports_silent_retry` for the
constraint that shapes it, and `config/policy.yaml` for the one action
deliberately excluded.

A `Candidate` carries its own arithmetic. That is not diagnostics: the case-detail
screen renders it, and "why did you charge this customer at 4am" has to be
answerable in numbers a merchant can check, not a summary the system asserts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from recoup.domain import Channel
from recoup.money import rupees


class ActionKind(StrEnum):
    RETRY_NOW = "RETRY_NOW"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    NUDGE_SMS = "NUDGE_SMS"
    NUDGE_WHATSAPP = "NUDGE_WHATSAPP"
    NUDGE_EMAIL = "NUDGE_EMAIL"
    NUDGE_VOICE = "NUDGE_VOICE"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"
    STOP = "STOP"

    @property
    def is_retry(self) -> bool:
        return self in (ActionKind.RETRY_NOW, ActionKind.RETRY_SCHEDULED)

    @property
    def is_contact(self) -> bool:
        return self.channel is not None

    @property
    def channel(self) -> Channel | None:
        return {
            ActionKind.NUDGE_SMS: Channel.SMS,
            ActionKind.NUDGE_WHATSAPP: Channel.WHATSAPP,
            ActionKind.NUDGE_EMAIL: Channel.EMAIL,
            ActionKind.NUDGE_VOICE: Channel.VOICE,
        }.get(self)


@dataclass(frozen=True)
class Candidate:
    """One option, priced.

    `ev` is in paise and may be negative — negative candidates are kept rather
    than discarded so the audit trail can show what was considered and rejected,
    not only what was chosen.
    """

    action: ActionKind
    at: datetime
    ev: int
    probability: float
    breakdown: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def delay_hours(self) -> float:
        return self.breakdown.get("delay_hours", 0.0)


def explain(
    action: ActionKind,
    breakdown: dict,
    at: datetime,
    ev: int,
    probability: float,
    cause: str | None = None,
) -> str:
    """One sentence a merchant could check against the numbers.

    A pure function of values that are already stored, which is the point: it is
    called at write time to describe a decision in the terminal, and again at read
    time to render the same sentence on the case screen. Keeping the prose *out*
    of the ledger removed roughly 7% of it, and keeping the function shared means
    the two paths cannot drift into describing the same decision differently.

    `at` is when the decision was *made*. The scheduled moment is derived from it
    and `delay_hours`, rather than passed in — the first version took the
    scheduled time directly, and the read path had only the decision time, so the
    sentence read "in 6h (Thu 12:05)" with the two halves disagreeing.
    """
    amount = rupees(breakdown.get("amount", 0))
    worth = rupees(ev)
    subject = cause or "an unclassified failure"
    delay = breakdown.get("delay_hours", 0.0)

    if action.is_retry:
        scheduled = at + timedelta(hours=delay)
        when = "now" if delay < 1 else f"in {delay:.0f}h ({scheduled:%a %H:%M})"
        return (
            f"Retry {when}: {subject} on {amount} has an estimated "
            f"{probability:.0%} chance of clearing, worth {worth} net."
        )

    if action.is_contact:
        return (
            f"Contact by {action.channel}: {subject} needs the customer to act on "
            f"{amount}, estimated {probability:.0%} to recover, worth "
            f"{worth} net after "
            f"{breakdown.get('prior_contacts', 0)} prior contacts."
        )

    if action is ActionKind.ESCALATE_HUMAN:
        return (
            f"Escalate: {amount} is large enough that human review at "
            f"{rupees(breakdown.get('cost', 0))}, resolving an estimated "
            f"{probability:.0%}, still nets {worth}."
        )

    return "Stop."


@dataclass(frozen=True)
class Veto:
    """A refusal, with the rule that caused it.

    Recorded as positively as an action. The refusal list — "we deliberately did
    not touch these cases, and here is why" — is a deliverable, and it only
    exists if this object gets written down.
    """

    rule: str
    action: ActionKind
    why: str


@dataclass
class Decision:
    """What the agent decided, everything it weighed, and everything refused."""

    payment_id: str
    at: datetime
    chosen: Candidate | None
    considered: list[Candidate] = field(default_factory=list)
    vetoes: list[Veto] = field(default_factory=list)
    reason: str = ""

    @property
    def acted(self) -> bool:
        return self.chosen is not None and self.chosen.action is not ActionKind.STOP

    @property
    def action(self) -> ActionKind:
        return self.chosen.action if self.chosen else ActionKind.STOP

    def to_ledger_data(self) -> dict:
        """Flattened for the audit trail.

        Deliberately narrow. Three fields that used to be here were removed after
        measuring what the ledger actually contained, and together they were ~45%
        of the file:

        * `vetoes` — 68% of this payload, and already written as its own `vetoed`
          events. The same information stored twice.
        * `at` — duplicated the ledger row's own timestamp column.
        * `reason` — an English sentence derived entirely from `breakdown`, so it
          is regenerated at read time by `explain()` instead of stored per row.

        Candidates are truncated to the best few: the tail of negative-EV options
        is noise once the top of the ranking is kept.
        """
        return {
            "action": str(self.action),
            "ev": self.chosen.ev if self.chosen else 0,
            "probability": round(self.chosen.probability, 4) if self.chosen else 0.0,
            "breakdown": self.chosen.breakdown if self.chosen else {},
            "considered": [
                {"action": str(c.action), "ev": c.ev, "p": round(c.probability, 4)}
                for c in sorted(self.considered, key=lambda c: -c.ev)[:5]
            ],
        }
