"""Driving one arm through a simulated month.

Event-driven, not a loop over payments. Failures arrive when they arrive, retries
fire when they were scheduled, and outages start and clear on their own timetable
— which is the only way to represent decisions that depend on *when*. A batch loop
would collapse the schedule and quietly delete the thing being measured.

The horizon is a real boundary. A retry scheduled past the end of the run does not
happen, exactly as it would not happen if the merchant closed the books. An
implementation that let late actions settle anyway would report recoveries that
never occurred.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from recoup.adapters.base import supports_silent_retry
from recoup.adapters.simulated import SimulatedAdapter, SimulatedNotifier
from recoup.agent.classify import Classifier
from recoup.agent.config import ComplianceConfig, PolicyConfig
from recoup.agent.context import ContextBuilder
from recoup.agent.executor import Executor
from recoup.agent.llm.copywriter import Copywriter
from recoup.domain import PaymentEntity, PaymentStatus
from recoup.eval.arms import Arm
from recoup.ledger.events import EventKind, Ledger, LedgerEvent
from recoup.world.clock import Timeline
from recoup.world.config import WorldConfig
from recoup.world.customers import Population
from recoup.world.generator import Batch
from recoup.world.issuers import IssuerBook


class Task(StrEnum):
    INGEST = "ingest"
    RECONSIDER = "reconsider"
    ACT = "act"
    DOWNTIME_STARTED = "downtime_started"
    DOWNTIME_RESOLVED = "downtime_resolved"


# A payment that keeps failing must not be reconsidered forever. Compliance caps
# bound it in practice, but a bug in those caps should not turn into an unbounded
# loop, so the runner enforces its own ceiling and reports when it bites.
MAX_DECISIONS_PER_PAYMENT = 12

# Called with the fraction of the horizon simulated so far, 0.0 to 1.0.
ProgressHook = Callable[[float], None]


@dataclass
class Scheduled:
    task: Task
    payment: PaymentEntity | None = None
    payload: object = None
    decision: object = None


@dataclass
class RunResult:
    arm: str
    description: str
    ledger: Ledger
    decisions: int = 0
    actions: int = 0
    vetoes: int = 0
    expired: int = 0
    notes: dict[str, int] = field(default_factory=dict)


class Runner:
    def __init__(
        self,
        world: WorldConfig,
        policy: PolicyConfig,
        compliance: ComplianceConfig,
        batch: Batch,
        population: Population,
        issuers: IssuerBook,
        ledger: Ledger,
        classifier: Classifier | None = None,
        copywriter: Copywriter | None = None,
    ) -> None:
        self._world = world
        self._policy = policy
        self._compliance = compliance
        self._batch = batch
        self._population = population
        self._issuers = issuers
        self._ledger = ledger
        self._classifier = classifier or Classifier()

        # Shared across arms on purpose. Copy is a rendering concern, not a policy
        # one, and the simulator has no basis for modelling whether wording
        # changes outcomes — so varying it between arms would add noise to the
        # comparison without measuring anything.
        self._copywriter = copywriter or Copywriter(use_llm=False)

    def run(self, arm: Arm, on_progress: ProgressHook | None = None) -> RunResult:
        """Drive one arm through the horizon.

        `on_progress` is called with a fraction of *simulated time* elapsed, which
        is a genuinely good progress signal and free to compute: the run is
        literally walking a clock from the start of the horizon to its end, so how
        far that clock has moved is how far the run has got. Counting events would
        need a total nobody knows in advance, because decisions create more work.
        """
        arm.reset()

        # A fresh adapter and notifier per arm: attempt counters and contact
        # history must not leak between experiments. The *world* is shared, and
        # so is its randomness — that is the point.
        adapter = SimulatedAdapter(self._world, self._batch, self._issuers, self._population)
        notifier = SimulatedNotifier(
            self._world, self._population, self._issuers, self._batch
        )

        context_builder = ContextBuilder(self._policy)
        executor = Executor(
            arm=arm.name,
            adapter=adapter,
            notifier=notifier,
            context=context_builder,
            cost_of=self._policy.cost_of,
            copywriter=self._copywriter,
            merchant=self._world.run.merchant_name,
        )

        result = RunResult(arm=arm.name, description=arm.description, ledger=self._ledger)
        self._decisions_taken: dict[str, int] = {}
        timeline = Timeline(self._world.run.start_at)
        horizon = self._world.run.end_at
        events: list[LedgerEvent] = []

        self._seed_timeline(timeline, horizon)

        start = self._world.run.start_at
        span = max((horizon - start).total_seconds(), 1.0)
        reported = 0.0

        for when, item in timeline.run(until=horizon):
            if on_progress is not None:
                fraction = (when - start).total_seconds() / span
                # Only on movement worth redrawing for. A callback per event
                # would dominate the runtime of the thing it is measuring.
                if fraction - reported >= 0.01:
                    reported = fraction
                    on_progress(min(1.0, fraction))

            match item.task:
                case Task.DOWNTIME_STARTED | Task.DOWNTIME_RESOLVED:
                    context_builder.note_downtime(str(item.payload[0]), item.payload[1])

                case Task.INGEST | Task.RECONSIDER | Task.ACT:
                    self._step(
                        arm=arm,
                        item=item,
                        when=when,
                        timeline=timeline,
                        horizon=horizon,
                        context_builder=context_builder,
                        executor=executor,
                        events=events,
                        result=result,
                    )

        result.expired = sum(1 for _ in timeline.drain())
        self._ledger.extend(events)
        return result

    # -- setup ---------------------------------------------------------------

    def _seed_timeline(self, timeline: Timeline, horizon: datetime) -> None:
        for when, name, downtime in self._issuers.events():
            if when <= horizon:
                task = (
                    Task.DOWNTIME_STARTED
                    if name.endswith("started")
                    else Task.DOWNTIME_RESOLVED
                )
                timeline.schedule(when, Scheduled(task=task, payload=(name, downtime)))

        for payment in self._batch.failures:
            at = datetime.fromtimestamp(payment.created_at)
            if at <= horizon:
                timeline.schedule(at, Scheduled(task=Task.INGEST, payment=payment))

    # -- one decision --------------------------------------------------------

    def _step(
        self, *, arm, item, when, timeline, horizon, context_builder, executor, events, result
    ) -> None:
        payment = item.payment
        assert payment is not None

        # Each arm classifies with its own classifier. The ablation depends on it:
        # the two agent arms differ only in whether an LLM fallback is attached.
        classification = arm.classifier.classify(payment)

        if item.task is Task.INGEST:
            events.append(
                LedgerEvent(
                    at=when,
                    kind=EventKind.OBSERVED,
                    arm=arm.name,
                    payment_id=payment.id,
                    customer_id=payment.customer_ref,
                    amount=payment.amount,
                    data={
                        "method": str(payment.method),
                        "reason": payment.error_reason,
                        # Whether standing authorisation exists. Not a secret —
                        # the agent reads it off the payment entity — but
                        # recording it makes the pool splittable afterwards,
                        # which is what separates "recovered more because it
                        # tried harder" from "recovered more because it could
                        # reach money the baseline structurally cannot".
                        "retryable": supports_silent_retry(
                            payment.method,
                            {
                                k: v
                                for k, v in (
                                    ("token_id", payment.token_id),
                                    ("card_id", payment.card_id),
                                )
                                if v
                            },
                        ),
                    },
                )
            )
            events.append(
                LedgerEvent(
                    at=when,
                    kind=EventKind.CLASSIFIED,
                    arm=arm.name,
                    payment_id=payment.id,
                    data={
                        "cause": str(classification.cause) if classification.cause else None,
                        "confidence": classification.confidence,
                        "rule": classification.rule_id,
                        "resolution": str(classification.resolution),
                    },
                )
            )

        context = context_builder.build(payment, classification, when)
        for channel in executor.consented_channels(payment.customer_ref or ""):
            context.notes[f"consent:{channel}"] = "true"

        # An ACT carries a decision already made. Re-deriving it here would be a
        # bug and was one: the policy proposes offsets relative to *now*, so
        # asking again at the scheduled moment simply proposes another future
        # moment, and the action never fires.
        if item.task is Task.ACT and item.decision is not None:
            self._perform(
                arm, item.decision, context, when, horizon, timeline, executor, events, result
            )
            return

        seen = self._decisions_taken.get(payment.id, 0)
        if seen >= MAX_DECISIONS_PER_PAYMENT:
            result.notes["decision_cap"] = result.notes.get("decision_cap", 0) + 1
            return
        self._decisions_taken[payment.id] = seen + 1

        decision = arm.decide(context)
        result.decisions += 1

        # Every decision is recorded, not only the first. An earlier version
        # logged this on ingest alone to keep the ledger small, and the result was
        # a case history where the opening move showed its arithmetic and the
        # three actions after it appeared from nowhere. "Every money action is
        # explainable" is the whole claim; a cheaper audit trail that only
        # explains the first one is not a smaller version of that claim, it is a
        # different and false one.
        events.append(
            LedgerEvent(
                at=when,
                kind=EventKind.DECIDED,
                arm=arm.name,
                payment_id=payment.id,
                customer_id=payment.customer_ref,
                data=decision.to_ledger_data(),
            )
        )

        self._record_vetoes(decision, payment, when, arm, events, result)

        due = decision.chosen.at if decision.chosen else when
        if decision.acted and due > when:
            if due <= horizon:
                timeline.schedule(
                    due,
                    Scheduled(task=Task.ACT, payment=payment, decision=decision),
                )
            else:
                events.append(
                    LedgerEvent(
                        at=when,
                        kind=EventKind.STOPPED,
                        arm=arm.name,
                        payment_id=payment.id,
                        data={"reason": "scheduled beyond the run horizon"},
                    )
                )
            return

        self._perform(
            arm, decision, context, when, horizon, timeline, executor, events, result
        )

    def _perform(
        self, arm, decision, context, when, horizon, timeline, executor, events, result
    ) -> None:
        executed, produced = executor.execute(decision, context)
        events.extend(produced)

        if not decision.acted:
            return

        result.actions += 1
        arm.gate.note_executed(decision.chosen)

        # Failed, still open, still time: reconsider. The policy will propose the
        # next attempt or stop.
        if not executed.succeeded and when < horizon:
            timeline.schedule(
                min(when + timedelta(minutes=1), horizon),
                Scheduled(task=Task.RECONSIDER, payment=context.payment),
            )

    def _record_vetoes(self, decision, payment, when, arm, events, result) -> None:
        """One row per rule, not one per rejected candidate.

        The gate screens every retry-time variant, so a single refused payment can
        produce fifty near-identical vetoes differing only by scheduled hour. That
        makes the refusal list unreadable and inflates the count into nonsense.
        What a reader wants is which *rules* stopped this payment.
        """
        seen: set[str] = set()

        for veto in decision.vetoes:
            if veto.rule in seen:
                continue
            seen.add(veto.rule)
            result.vetoes += 1
            events.append(
                LedgerEvent(
                    at=when,
                    kind=EventKind.VETOED,
                    arm=arm.name,
                    payment_id=payment.id,
                    amount=payment.amount,
                    data={"rule": veto.rule, "action": str(veto.action), "why": veto.why},
                )
            )

def failures_of(batch: Batch) -> list[PaymentEntity]:
    return [p for p in batch.payments if p.status is PaymentStatus.FAILED]
