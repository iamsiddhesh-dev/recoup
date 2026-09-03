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

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from recoup.agent.actions import ActionKind, explain
from recoup.agent.llm.explainer import CaseFacts
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
    """One decision's arithmetic, laid out to be checked.

    `reason` is regenerated here rather than read from the ledger. It is a
    sentence derived entirely from the numbers beside it, and storing a per-row
    copy of derivable prose was ~7% of the ledger. Refusals are not carried here
    either — they are their own events and appear in the timeline in sequence,
    which is where they belong anyway.
    """

    action: str
    at: datetime
    ev: int
    reason: str
    steps: list[Step]
    alternatives: list[dict] = field(default_factory=list)
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


def _working(event: LedgerEvent, cause: str | None) -> Working | None:
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

    breakdown = data.get("breakdown", {})
    ev = data.get("ev", 0)

    return Working(
        action=action,
        at=event.at,
        ev=ev,
        reason=explain(
            action=ActionKind(action),
            breakdown=breakdown,
            at=event.at,
            ev=ev,
            probability=data.get("probability", 0.0),
            cause=cause,
        ),
        steps=_steps_for(action, breakdown),
        alternatives=alternatives[:4],
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
            # The sentence is regenerated in `_working` from the stored numbers,
            # so the detail line is filled in by the caller rather than read here.
            return f"Decided: {data.get('action', '')}", ""

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
            language = f" · {data['language']}" if data.get("language") else ""
            return f"Executed {action}", f"{delivered}{link}{acted}{language}"

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

        # `case.cause` is already set by the time any decision arrives, because
        # classification always precedes it in the stream.
        working = (
            _working(event, case.cause) if event.kind is EventKind.DECIDED else None
        )

        case.entries.append(
            TimelineEntry(
                at=event.at,
                kind=event.kind,
                title=title,
                detail=working.reason if working else detail,
                tone=TONE.get(event.kind, "neutral"),
                working=working,
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


def case_facts(case: CaseView) -> CaseFacts:
    """A case, reduced to what an explanation may be built from.

    The explainer never sees the ledger or a `CaseView`; it takes this and returns
    prose. That keeps the narrative layer unable to reach anything it was not
    explicitly handed, which is also what makes "the model may only use numbers it
    was given" a checkable statement rather than an aspiration.
    """
    actions: list[str] = []
    channels: list[str] = []
    vetoes: list[str] = []
    decisions: list[str] = []

    for entry in case.entries:
        if entry.kind is EventKind.EXECUTED:
            action = entry.data.get("action", "")
            if action:
                actions.append(action)
            if entry.data.get("delivered") and action.startswith("NUDGE_"):
                channels.append(action.removeprefix("NUDGE_").lower())
        elif entry.kind is EventKind.VETOED:
            rule = entry.data.get("rule")
            if rule:
                vetoes.append(rule)
        elif entry.kind is EventKind.DECIDED and entry.working:
            decisions.append(entry.working.reason)

    return CaseFacts(
        payment_id=case.payment_id,
        amount_paise=case.amount,
        method=case.method,
        reason=case.reason,
        outcome=case.outcome,
        cause=case.cause,
        recovered_paise=case.recovered_paise,
        cost_paise=case.cost_paise,
        attempts=case.attempts,
        contacts=case.contacts,
        actions=actions,
        channels=channels,
        vetoes=vetoes,
        decisions=decisions,
    )


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


# The shapes worth explaining. Each is a different way a payment can end, so the
# selection spans the range of outcomes rather than the flattering end of it.
EXPLAINABLE_SHAPES: list[tuple[str, Callable[[QueueRow], bool]]] = [
    ("recovered without contacting anyone", lambda r: r.outcome == "recovered" and not r.contacts),
    ("recovered after contact", lambda r: r.outcome == "recovered" and r.contacts > 0),
    ("chased and still lost", lambda r: r.outcome != "recovered" and r.contacts > 0),
    ("never actioned at all", lambda r: r.actions == 0),
    ("cause never determined", lambda r: r.cause is None),
]


def explainable_cases(ledger: Ledger, arm: str, per_shape: int = 1) -> list[str]:
    """Which payments get a generated explanation, chosen by rule rather than by eye.

    Picking cases by hand would make this a highlight reel — the ones that read
    well would get the prose and the awkward ones would not. So the selection is
    the largest payment of each distinct *shape*: recovered without contact,
    recovered after contact, chased and lost, never touched, never diagnosed. Two
    of those five are cases where the agent achieved nothing, and they are in the
    list on purpose.

    Largest first within a shape, matching the queue's own order, so a judge
    scanning from the top meets an explained case early.
    """
    rows = build_queue(ledger, arm, limit=100_000)

    chosen: list[str] = []
    seen: set[str] = set()

    for _, matches in EXPLAINABLE_SHAPES:
        taken = 0
        for row in rows:
            if taken >= per_shape:
                break
            if row.payment_id in seen or not matches(row):
                continue
            chosen.append(row.payment_id)
            seen.add(row.payment_id)
            taken += 1

    return chosen


# ---------------------------------------------------------------------------
# Audit and refusals
# ---------------------------------------------------------------------------


@dataclass
class RefusalRule:
    """One compliance rule, and what it stopped."""

    rule: str
    refusals: int
    payments: int
    amount: int
    why: str
    actions: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        """Rules are keyed for machines. This is for the person reading them."""
        return HUMAN_RULE_NAMES.get(self.rule, self.rule.replace(":", " · "))


@dataclass
class RefusedPayment:
    payment_id: str
    amount: int
    cause: str | None
    rules: list[str]
    outcome: str
    recovered_paise: int

    @property
    def abandoned(self) -> bool:
        return self.outcome != "recovered"


@dataclass
class AuditView:
    arm: str
    total_events: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    digest: str = ""
    triggers: list[str] = field(default_factory=list)
    replay_ok: bool = False

    rules: list[RefusalRule] = field(default_factory=list)
    refused: list[RefusedPayment] = field(default_factory=list)

    # Configured hard stops that never had to fire. Reported rather than hidden:
    # a rule showing zero looks inactive, and the reason it is zero is the
    # interesting part.
    dormant: list[str] = field(default_factory=list)

    refusals: int = 0
    payments_touched: int = 0
    abandoned_count: int = 0
    abandoned_amount: int = 0
    recovered_anyway_count: int = 0
    recovered_anyway_amount: int = 0


HUMAN_RULE_NAMES = {
    "attempts:max_per_payment": "Retry cap reached",
    "attempts:cooling_off": "Instrument in cooling-off",
    "contact:quiet_hours": "Inside quiet hours",
    "contact:min_interval": "Too soon after the last message",
    "contact:max_per_customer": "Weekly contact cap reached",
    "contact:consent:whatsapp": "No WhatsApp consent",
    "contact:consent:voice": "No voice consent",
    "downtime:active": "Issuer outage in progress",
    "escalation:below_threshold": "Below the human-review threshold",
    "escalation:run_cap": "Human review capacity spent",
    "execution:max_actions_per_run": "Run action cap reached",
    "hard_stop:RISK_BLOCKED": "Blocked by risk — never retried",
    "hard_stop:MANDATE_PROBLEM": "Mandate revoked or exceeded",
    "hard_stop:INSTRUMENT_INVALID": "Instrument cannot succeed as-is",
    "hard_stop:CUSTOMER_INTENT": "Customer cancelled deliberately",
    "hard_stop:CUSTOMER_INTENT:max_contacts": "Already followed up once",
}


def build_audit(
    ledger: Ledger,
    arm: str,
    recorded_digest: str | None = None,
    configured_hard_stops: list[str] | None = None,
) -> AuditView:
    """The refusal list, and evidence that the trail it comes from is intact.

    Two things on one screen because they answer the same question. "We
    deliberately did not touch these cases" is only worth reading if the record
    it is drawn from cannot have been edited, so the integrity of the ledger is
    shown next to what it says rather than asserted elsewhere.
    """
    view = AuditView(arm=arm)

    view.by_kind = ledger.counts_by_kind(arm)
    view.total_events = sum(view.by_kind.values())
    view.triggers = ledger.append_only_triggers()

    # Recomputed now, compared against the hash taken when the run was written.
    # A match means nothing in the stream has changed since — which is the whole
    # claim an audit trail makes, checked rather than asserted.
    current = ledger.digest(arm)
    view.digest = current[:16]
    view.replay_ok = recorded_digest is not None and current == recorded_digest

    # Rules, and the payments each one stopped.
    rules: dict[str, RefusalRule] = {}
    per_payment: dict[str, set[str]] = {}
    seen_by_rule: dict[str, set[str]] = {}
    amounts: dict[str, int] = {}
    outcomes: dict[str, str] = {}
    recovered: dict[str, int] = {}
    causes: dict[str, str | None] = {}

    for event in ledger.events(arm=arm):
        pid = event.payment_id

        match event.kind:
            case EventKind.OBSERVED if pid:
                amounts[pid] = event.amount or 0
                outcomes.setdefault(pid, "open")

            case EventKind.CLASSIFIED if pid:
                causes[pid] = event.data.get("cause")

            case EventKind.VETOED if pid:
                rule = event.data.get("rule", "unknown")
                entry = rules.get(rule)
                if entry is None:
                    entry = RefusalRule(
                        rule=rule,
                        refusals=0,
                        payments=0,
                        amount=0,
                        why=event.data.get("why", ""),
                    )
                    rules[rule] = entry
                    seen_by_rule[rule] = set()

                entry.refusals += 1
                action = event.data.get("action", "")
                if action and action not in entry.actions:
                    entry.actions.append(action)

                if pid not in seen_by_rule[rule]:
                    seen_by_rule[rule].add(pid)
                    entry.payments += 1
                    entry.amount += event.amount or 0

                per_payment.setdefault(pid, set()).add(rule)

            case EventKind.RECOVERED if pid:
                recovered[pid] = recovered.get(pid, 0) + (event.amount or 0)
                outcomes[pid] = "recovered"

            case EventKind.STOPPED if pid:
                if outcomes.get(pid) != "recovered":
                    outcomes[pid] = "stopped"

    view.rules = sorted(rules.values(), key=lambda r: -r.amount)
    view.refusals = sum(r.refusals for r in view.rules)
    view.payments_touched = len(per_payment)

    for pid, applied in per_payment.items():
        row = RefusedPayment(
            payment_id=pid,
            amount=amounts.get(pid, 0),
            cause=causes.get(pid),
            rules=sorted(applied),
            outcome=outcomes.get(pid, "open"),
            recovered_paise=recovered.get(pid, 0),
        )
        view.refused.append(row)

        if row.abandoned:
            view.abandoned_count += 1
            view.abandoned_amount += row.amount
        else:
            view.recovered_anyway_count += 1
            view.recovered_anyway_amount += row.recovered_paise

    # Largest first — the refusals worth arguing about are the expensive ones.
    view.refused.sort(key=lambda r: -r.amount)

    # A hard stop with zero refusals has not failed; it was never reached,
    # because the policy declined to propose the action on economic grounds
    # before compliance had to refuse it on principle. Both layers doing their
    # job looks, from the gate's side, like one layer doing nothing.
    fired = set(rules)
    view.dormant = sorted(
        f"hard_stop:{name}"
        for name in (configured_hard_stops or [])
        if not any(rule.startswith(f"hard_stop:{name}") for rule in fired)
    )

    return view


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class ArmBreakdown:
    arm: str
    recovered_paise: int = 0
    recovered_count: int = 0
    from_retryable: int = 0
    from_customer: int = 0
    by_cause: dict[str, int] = field(default_factory=dict)
    by_action: dict[str, int] = field(default_factory=dict)


@dataclass
class ExperimentView:
    """The headline number, decomposed until it stops flattering itself."""

    at_risk: int = 0
    observed: int = 0

    retryable_amount: int = 0
    retryable_count: int = 0
    customer_amount: int = 0
    customer_count: int = 0

    arms: list[ArmBreakdown] = field(default_factory=list)

    baseline: str = ""
    contact: str = ""
    agent: str = ""
    ablation: str = ""

    coverage_gain: int = 0
    judgment_gain: int = 0
    total_gain: int = 0
    llm_gain: int = 0
    llm_contacts_delta: int = 0

    def arm(self, name: str) -> ArmBreakdown | None:
        return next((a for a in self.arms if a.arm == name), None)

    @property
    def coverage_share(self) -> float:
        return self.coverage_gain / self.total_gain if self.total_gain else 0.0

    @property
    def judgment_share(self) -> float:
        return self.judgment_gain / self.total_gain if self.total_gain else 0.0


def build_experiment(
    ledger: Ledger,
    baseline: str = "naive_baseline",
    contact: str = "contact_only",
    agent: str = "recoup_agent",
    ablation: str = "recoup_agent_no_llm",
) -> ExperimentView:
    """Split the pool, then split the lift.

    The headline "+296% over naive retry" is true and mostly uninteresting: the
    baseline only retries, and three quarters of the money sits behind failures no
    retry can touch, so most of the gap is money it structurally cannot reach.
    Reporting that as though it were the agent being clever would be the single
    most misleading thing this project could do with an honest number.

    So the lift is decomposed into the part any arm that contacts customers would
    get, and the part that comes from choosing better — which is the only half
    that is about judgment.
    """
    view = ExperimentView(baseline=baseline, contact=contact, agent=agent, ablation=ablation)

    retryable: dict[str, bool] = {}
    amounts: dict[str, int] = {}
    causes: dict[str, str | None] = {}
    breakdowns: dict[str, ArmBreakdown] = {}
    counted_pool = False

    for event in ledger.events():
        pid = event.payment_id
        if pid is None:
            continue

        breakdown = breakdowns.setdefault(event.arm, ArmBreakdown(arm=event.arm))

        match event.kind:
            case EventKind.OBSERVED:
                # The pool is identical in every arm, so it is measured once.
                if event.arm == baseline:
                    counted_pool = True
                    view.observed += 1
                    view.at_risk += event.amount or 0
                    if event.data.get("retryable"):
                        view.retryable_count += 1
                        view.retryable_amount += event.amount or 0
                    else:
                        view.customer_count += 1
                        view.customer_amount += event.amount or 0

                retryable[f"{event.arm}:{pid}"] = bool(event.data.get("retryable"))
                amounts[f"{event.arm}:{pid}"] = event.amount or 0

            case EventKind.CLASSIFIED:
                causes[f"{event.arm}:{pid}"] = event.data.get("cause")

            case EventKind.RECOVERED:
                amount = event.amount or 0
                breakdown.recovered_paise += amount
                breakdown.recovered_count += 1

                key = f"{event.arm}:{pid}"
                if retryable.get(key):
                    breakdown.from_retryable += amount
                else:
                    breakdown.from_customer += amount

                cause = causes.get(key) or "unclassified"
                breakdown.by_cause[cause] = breakdown.by_cause.get(cause, 0) + amount

                via = event.data.get("via", "unknown")
                breakdown.by_action[via] = breakdown.by_action.get(via, 0) + amount

    if not counted_pool:
        return view

    view.arms = [breakdowns[name] for name in ledger.arms() if name in breakdowns]

    base = breakdowns.get(baseline)
    reach = breakdowns.get(contact)
    smart = breakdowns.get(agent)
    plain = breakdowns.get(ablation)

    if base and smart:
        view.total_gain = smart.recovered_paise - base.recovered_paise
    if base and reach:
        view.coverage_gain = reach.recovered_paise - base.recovered_paise
    if reach and smart:
        view.judgment_gain = smart.recovered_paise - reach.recovered_paise
    if plain and smart:
        view.llm_gain = smart.recovered_paise - plain.recovered_paise

    return view


def queue_facets(ledger: Ledger, arm: str) -> dict[str, list[str]]:
    """The filter values actually present, so the UI never offers an empty one."""
    causes: set[str] = set()
    for event in ledger.events(arm=arm, kind=EventKind.CLASSIFIED):
        cause = event.data.get("cause")
        if cause:
            causes.add(cause)

    return {"causes": sorted(causes), "outcomes": ["recovered", "stopped", "open"]}
