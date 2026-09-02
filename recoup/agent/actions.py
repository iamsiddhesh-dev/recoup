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
from datetime import datetime
from enum import StrEnum

from recoup.domain import Channel


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

        Candidates are truncated to the best few: a run stores hundreds of
        thousands of decisions and the tail of negative-EV options is noise once
        the top of the ranking is preserved.
        """
        return {
            "action": str(self.action),
            "reason": self.reason,
            "ev": self.chosen.ev if self.chosen else 0,
            "probability": round(self.chosen.probability, 4) if self.chosen else 0.0,
            "at": self.at.isoformat(),
            "breakdown": self.chosen.breakdown if self.chosen else {},
            "considered": [
                {"action": str(c.action), "ev": c.ev, "p": round(c.probability, 4)}
                for c in sorted(self.considered, key=lambda c: -c.ev)[:5]
            ],
            "vetoes": [
                {"rule": v.rule, "action": str(v.action), "why": v.why}
                for v in self.vetoes
            ],
        }
