"""Shaping ledger events into what a screen renders.

The ledger stores what happened. These build what a person needs to see. Kept
separate from both so the templates contain no arithmetic and the ledger contains
no presentation.

The important one is `build_case`. Recoup's central claim is that every money
action is explainable, and the honest test of that is whether someone can open one
payment and follow the reasoning end to end without reading code — including the
decisions that were *not* taken and the rules that stopped them. That is what this
assembles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from recoup.ledger.events import EventKind, Ledger, LedgerEvent

# How each event kind presents. Tone drives colour; a veto is `refused` rather
# than `critical` because refusing is a correct outcome, not a failure.
TONE = {
    EventKind.OBSERVED: "neutral",
    EventKind.CLASSIFIED: "neutral",
    EventKind.DECIDED: "accent",
    EventKind.VETOED: "refused",
    EventKind.EXECUTED: "neutral",
    EventKind.RECOVERED: "accent",
    EventKind.STOPPED: "warn",
}

# The order the EV arithmetic is presented in, with the operator that joins each
# line to the one before. Rendering the calculation as a sum a reader can check is
# the point — a single "expected value: 208" asserts, this one shows its working.
RETRY_STEPS = [
    ("probability", "", "chance this clears"),
    ("amount", "×", "payment amount"),
    ("margin", "×", "contribution margin"),
    ("gross", "=", "expected gross"),
    ("decay", "×", "value lost to delay"),
    ("cost", "−", "cost of the attempt"),
]

CONTACT_STEPS = [
    ("response", "", "chance they act"),
    ("completion", "×", "having acted, they finish"),
    ("amount", "×", "payment amount"),
    ("margin", "×", "contribution margin"),
    ("gross", "=", "expected gross"),
    ("cost", "−", "cost of the message"),
    ("annoyance", "−", "goodwill spent on repeat contact"),
]

ESCALATION_STEPS = [
    ("probability", "", "chance a human resolves it"),
    ("amount", "×", "payment amount"),
    ("margin", "×", "contribution margin"),
    ("gross", "=", "expected gross"),
    ("cost", "−", "ops time"),
    ("scarcity_premium", "−", "the slot not spent on a larger payment"),
]

MONEY_KEYS = {"amount", "gross", "cost", "annoyance", "scarcity_premium"}


@dataclass
class Step:
    key: str
    operator: str
    label: str
    value: float
    is_money: bool


@dataclass
class Working:
    """One decision's arithmetic, laid out to be checked."""

    action: str
    at: datetime
    ev: int
    reason: str
    steps: list[Step]
    alternatives: list[dict] = field(default_factory=list)
    vetoes: list[dict] = field(default_factory=list)
    note: str = ""


@dataclass
class TimelineEntry:
    at: datetime
    kind: EventKind
    title: str
    detail: str
    tone: str
    working: Working | None = None
    data: dict = field(default_factory=dict)


@dataclass
class CaseView:
    payment_id: str
    arm: str
    amount: int = 0
    method: str = ""
    reason: str = ""
    customer_id: str | None = None

    cause: str | None = None
    confidence: float = 0.0
    rule: str | None = None
    resolution: str = ""

    outcome: str = "open"
    recovered_paise: int = 0
    cost_paise: int = 0
    attempts: int = 0
    contacts: int = 0

    entries: list[TimelineEntry] = field(default_factory=list)

    @property
    def net_paise(self) -> int:
        return self.recovered_paise - self.cost_paise

    @property
    def veto_count(self) -> int:
        return sum(1 for e in self.entries if e.kind is EventKind.VETOED)


def _steps_for(action: str, breakdown: dict) -> list[Step]:
    if action.startswith("RETRY"):
        template = RETRY_STEPS
    elif action.startswith("NUDGE"):
        template = CONTACT_STEPS
    elif action.startswith("ESCALATE"):
        template = ESCALATION_STEPS
    else:
        return []

    return [
        Step(
            key=key,
            operator=operator,
            label=label,
            value=breakdown[key],
            is_money=key in MONEY_KEYS,
        )
        for key, operator, label in template
        if key in breakdown
    ]


def _working(event: LedgerEvent) -> Working | None:
    data = event.data
    action = data.get("action", "")
    if not action or action == "STOP":
        return None

    # One entry per action, keeping the best. The policy scores the same action at
    # many candidate times, so an undeduplicated list reads as
    # "RETRY_SCHEDULED ₹486, RETRY_SCHEDULED ₹486" and looks like a glitch rather
    # than a schedule. `considered` arrives sorted by value, so first wins.
    alternatives: list[dict] = []
    seen: set[str] = {action}
    for candidate in data.get("considered", []):
        name = candidate.get("action")
        if name in seen:
            continue
        seen.add(name)
        alternatives.append(candidate)

    return Working(
        action=action,
        at=event.at,
        ev=data.get("ev", 0),
        reason=data.get("reason", ""),
        steps=_steps_for(action, data.get("breakdown", {})),
        alternatives=alternatives[:4],
        vetoes=data.get("vetoes", []),
    )


def _describe(event: LedgerEvent) -> tuple[str, str]:
    data = event.data

    match event.kind:
        case EventKind.OBSERVED:
            return "Payment failed", data.get("reason") or "no reason reported"

        case EventKind.CLASSIFIED:
            cause = data.get("cause")
            if not cause:
                return "Could not classify", "no rule matched these symptoms"
            how = data.get("resolution", "")
            rule = data.get("rule") or "—"
            return f"Classified as {cause}", f"{how} · rule {rule}"

        case EventKind.DECIDED:
            return f"Decided: {data.get('action', '')}", data.get("reason", "")

        case EventKind.VETOED:
            return f"Refused: {data.get('action', '')}", data.get("why", "")

        case EventKind.EXECUTED:
            action = data.get("action", "")
            if "succeeded" in data:
                verdict = "succeeded" if data["succeeded"] else "failed"
                extra = "" if data["succeeded"] else f" — {data.get('error_reason') or 'no reason'}"
                return f"Executed {action}", f"{verdict}{extra}"
            delivered = "delivered" if data.get("delivered") else "not delivered"
            acted = ", customer acted" if data.get("acted_on") else ""
            link = ", with payment link" if data.get("with_link") else ""
            return f"Executed {action}", f"{delivered}{link}{acted}"

        case EventKind.RECOVERED:
            return "Recovered", f"via {data.get('via', '')}"

        case EventKind.STOPPED:
            return "Stopped", data.get("reason", "")

    return str(event.kind), ""


def build_case(ledger: Ledger, payment_id: str, arm: str) -> CaseView | None:
    """One payment's whole story, in order, with the arithmetic intact."""
    events = ledger.story_of(payment_id, arm)
    if not events:
        return None

    case = CaseView(payment_id=payment_id, arm=arm)

    for event in events:
        title, detail = _describe(event)

        case.entries.append(
            TimelineEntry(
                at=event.at,
                kind=event.kind,
                title=title,
                detail=detail,
                tone=TONE.get(event.kind, "neutral"),
                working=_working(event) if event.kind is EventKind.DECIDED else None,
                data=event.data,
            )
        )

        match event.kind:
            case EventKind.OBSERVED:
                case.amount = event.amount or 0
                case.method = event.data.get("method", "")
                case.reason = event.data.get("reason", "") or ""
                case.customer_id = event.customer_id

            case EventKind.CLASSIFIED:
                case.cause = event.data.get("cause")
                case.confidence = event.data.get("confidence", 0.0)
                case.rule = event.data.get("rule")
                case.resolution = event.data.get("resolution", "")

            case EventKind.EXECUTED:
                case.cost_paise += event.data.get("cost", 0)
                if event.data.get("action", "").startswith("RETRY"):
                    case.attempts += 1
                if event.data.get("delivered"):
                    case.contacts += 1

            case EventKind.RECOVERED:
                case.recovered_paise += event.amount or 0
                case.outcome = "recovered"

            case EventKind.STOPPED:
                if case.outcome != "recovered":
                    case.outcome = "stopped"

    return case


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


@dataclass
class QueueRow:
    payment_id: str
    amount: int
    method: str
    reason: str
    cause: str | None
    outcome: str
    actions: int
    contacts: int
    cost_paise: int
    recovered_paise: int

    @property
    def net_paise(self) -> int:
        return self.recovered_paise - self.cost_paise


def build_queue(
    ledger: Ledger,
    arm: str,
    outcome: str | None = None,
    cause: str | None = None,
    limit: int = 200,
) -> list[QueueRow]:
    """Every failure in the run, folded to one row each.

    A single ordered scan rather than a query per payment. The ledger is an event
    stream, so N+1 queries would be the obvious mistake here and would turn a
    page load into thousands of round trips.
    """
    rows: dict[str, QueueRow] = {}

    for event in ledger.events(arm=arm):
        pid = event.payment_id
        if pid is None:
            continue

        row = rows.get(pid)
        if row is None:
            if event.kind is not EventKind.OBSERVED:
                continue
            row = QueueRow(
                payment_id=pid,
                amount=event.amount or 0,
                method=event.data.get("method", ""),
                reason=event.data.get("reason", "") or "",
                cause=None,
                outcome="open",
                actions=0,
                contacts=0,
                cost_paise=0,
                recovered_paise=0,
            )
            rows[pid] = row
            continue

        match event.kind:
            case EventKind.CLASSIFIED:
                row.cause = event.data.get("cause")
            case EventKind.EXECUTED:
                row.actions += 1
                row.cost_paise += event.data.get("cost", 0)
                if event.data.get("delivered"):
                    row.contacts += 1
            case EventKind.RECOVERED:
                row.recovered_paise += event.amount or 0
                row.outcome = "recovered"
            case EventKind.STOPPED:
                if row.outcome != "recovered":
                    row.outcome = "stopped"

    selected = list(rows.values())

    if outcome:
        selected = [r for r in selected if r.outcome == outcome]
    if cause:
        selected = [r for r in selected if r.cause == cause]

    # Largest first: the money is what a reader is scanning for, and a queue
    # ordered by payment id is a queue nobody reads twice.
    selected.sort(key=lambda r: -r.amount)
    return selected[:limit]


def queue_facets(ledger: Ledger, arm: str) -> dict[str, list[str]]:
    """The filter values actually present, so the UI never offers an empty one."""
    causes: set[str] = set()
    for event in ledger.events(arm=arm, kind=EventKind.CLASSIFIED):
        cause = event.data.get("cause")
        if cause:
            causes.add(cause)

    return {"causes": sorted(causes), "outcomes": ["recovered", "stopped", "open"]}
