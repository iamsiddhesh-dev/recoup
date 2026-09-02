"""Replaying a run from its ledger.

Two jobs.

**Proving reproducibility.** `make reproduce` regenerates every committed figure
from fixed seeds and asserts the output is byte-identical. When it is not, the
useful question is not "did it change" but "where", and `first_divergence` answers
that by walking two streams in step and returning the first event that differs.
Without it, a broken reproduction is a diff of half a million rows.

**Reconstructing state.** The control room's scrubber seeks to a point in
simulated time and shows what was known then — not what is known now. That is only
possible because the ledger holds every intermediate state rather than a final
summary, and `state_at` folds the stream up to a timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from recoup.ledger.events import EventKind, Ledger, LedgerEvent


@dataclass
class RunState:
    """What was true at a point in a run."""

    at: datetime
    arm: str
    observed: int = 0
    classified: int = 0
    unresolved: int = 0
    decisions: int = 0
    vetoes: int = 0
    executions: int = 0
    recovered_count: int = 0
    recovered_paise: int = 0
    amount_at_risk: int = 0
    stopped: int = 0
    veto_reasons: dict[str, int] = field(default_factory=dict)
    open_payments: set[str] = field(default_factory=set)

    @property
    def recovery_rate(self) -> float:
        return self.recovered_count / self.observed if self.observed else 0.0

    @property
    def recovered_share(self) -> float:
        return self.recovered_paise / self.amount_at_risk if self.amount_at_risk else 0.0


def fold(events: list[LedgerEvent] | None = None, arm: str = "") -> RunState:
    """Reduce an event stream to a state snapshot."""
    events = events or []
    state = RunState(at=events[0].at if events else datetime.min, arm=arm)

    for event in events:
        state.at = event.at

        match event.kind:
            case EventKind.OBSERVED:
                state.observed += 1
                state.amount_at_risk += event.amount or 0
                if event.payment_id:
                    state.open_payments.add(event.payment_id)

            case EventKind.CLASSIFIED:
                if event.data.get("cause"):
                    state.classified += 1
                else:
                    state.unresolved += 1

            case EventKind.DECIDED:
                state.decisions += 1

            case EventKind.VETOED:
                state.vetoes += 1
                rule = event.data.get("rule", "unknown")
                state.veto_reasons[rule] = state.veto_reasons.get(rule, 0) + 1

            case EventKind.EXECUTED:
                state.executions += 1

            case EventKind.RECOVERED:
                state.recovered_count += 1
                state.recovered_paise += event.amount or 0
                state.open_payments.discard(event.payment_id or "")

            case EventKind.STOPPED:
                state.stopped += 1
                state.open_payments.discard(event.payment_id or "")

    return state


def state_at(ledger: Ledger, arm: str, when: datetime) -> RunState:
    """What the run looked like at `when`. Powers the scrubber."""
    return fold([e for e in ledger.events(arm=arm) if e.at <= when], arm=arm)


def final_state(ledger: Ledger, arm: str) -> RunState:
    return fold(list(ledger.events(arm=arm)), arm=arm)


@dataclass(frozen=True)
class Divergence:
    index: int
    left: LedgerEvent | None
    right: LedgerEvent | None

    def describe(self) -> str:
        if self.left is None:
            return f"event {self.index}: missing on the left, right has {self.right.kind}"
        if self.right is None:
            return f"event {self.index}: missing on the right, left has {self.left.kind}"
        return (
            f"event {self.index} differs\n"
            f"  left:  {self.left.canonical()}\n"
            f"  right: {self.right.canonical()}"
        )


def first_divergence(
    left: Ledger, right: Ledger, arm: str | None = None
) -> Divergence | None:
    """The first event where two runs stopped agreeing, or None if identical."""
    left_events = list(left.events(arm=arm))
    right_events = list(right.events(arm=arm))

    for index in range(max(len(left_events), len(right_events))):
        a = left_events[index] if index < len(left_events) else None
        b = right_events[index] if index < len(right_events) else None

        if a is None or b is None or a.canonical() != b.canonical():
            return Divergence(index=index, left=a, right=b)

    return None
