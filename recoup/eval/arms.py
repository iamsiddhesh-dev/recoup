"""The experiment arms.

The headline claim is *incremental* recovery, which only means something if the
comparison is fair. Two choices make it so.

**Every arm passes through the same compliance gate.** The baseline is naive, not
reckless. If it were allowed to retry revoked mandates and message people at 3am,
part of the agent's measured advantage would be "we follow the rules and they do
not", which is not a product difference — it is a rigged comparison. All arms are
compliant; what differs is judgment.

**Every arm runs against the same world with the same luck.** Outcomes are drawn
from `(seed, payment_id, attempt)`, so the same payment retried at the same time
gets the same die roll in every arm. Differences are attributable to decisions
rather than to variance, which at these sample sizes would otherwise swamp a real
few-point improvement.

The baseline is deliberately what merchants actually do: a fixed retry schedule,
no cause analysis, no timing, no customer contact. Beating a strawman proves
nothing, but this is not a strawman — it is the status quo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from recoup.agent.actions import ActionKind, Candidate, Decision
from recoup.agent.compliance import ComplianceGate
from recoup.agent.config import ComplianceConfig, PolicyConfig
from recoup.agent.context import DecisionContext
from recoup.agent.policy import PolicyEngine


class Arm(Protocol):
    name: str
    description: str

    def decide(self, context: DecisionContext) -> Decision: ...

    def reset(self) -> None: ...


@dataclass
class ArmResultRow:
    name: str
    description: str


class NaiveRetryArm:
    """What most merchants do: retry three times on a fixed schedule.

    No cause analysis, no learned timing, no customer contact. Retries only where
    standing authorisation exists, because nothing else is technically possible —
    that constraint is not a choice and applies to every arm equally.
    """

    name = "naive_baseline"
    description = "fixed retries at +1h, +24h, +72h; no cause analysis, no contact"

    OFFSETS_HOURS = (1, 24, 72)

    def __init__(self, policy: PolicyConfig, compliance: ComplianceConfig) -> None:
        self._policy = policy
        self.gate = ComplianceGate(compliance)

    def reset(self) -> None:
        self.gate.reset()

    def decide(self, context: DecisionContext) -> Decision:
        attempt = context.attempts

        if not context.can_retry_silently or attempt >= len(self.OFFSETS_HOURS):
            return _stopped(context, "fixed schedule exhausted")

        at = context.now + timedelta(hours=self.OFFSETS_HOURS[attempt])
        candidate = Candidate(
            action=ActionKind.RETRY_SCHEDULED,
            at=at,
            ev=0,
            probability=0.0,
            breakdown={"schedule_position": attempt + 1},
            note="fixed schedule",
        )

        chosen, vetoes = self.gate.screen([candidate], context)
        if chosen is None:
            return _stopped(context, "blocked by compliance", vetoes)

        return Decision(
            payment_id=context.payment.id,
            at=context.now,
            chosen=chosen,
            considered=[candidate],
            vetoes=vetoes,
            reason=f"fixed retry {attempt + 1} of {len(self.OFFSETS_HOURS)}",
        )


class ContactOnlyArm:
    """Message the customer, never retry.

    An ablation, not a strawman. It isolates how much of the agent's result comes
    from talking to people rather than from scheduling, which matters because most
    of the money at risk sits behind failures no retry can fix.
    """

    name = "contact_only"
    description = "expected-value contact decisions only; retries disabled"

    def __init__(self, policy: PolicyConfig, compliance: ComplianceConfig) -> None:
        self._engine = PolicyEngine(policy)
        self._threshold = policy.ev_threshold_paise
        self.gate = ComplianceGate(compliance)

    def reset(self) -> None:
        self.gate.reset()

    def decide(self, context: DecisionContext) -> Decision:
        decision = self._engine.decide(context)
        candidates = [
            c
            for c in decision.considered
            if (c.action.is_contact or c.action is ActionKind.ESCALATE_HUMAN)
            and c.ev >= self._threshold
        ]

        if not candidates:
            return _stopped(context, "no contact option clears the threshold")

        chosen, vetoes = self.gate.screen(candidates, context)
        if chosen is None:
            return _stopped(context, "no permitted contact option", vetoes)

        return Decision(
            payment_id=context.payment.id,
            at=context.now,
            chosen=chosen,
            considered=candidates,
            vetoes=vetoes,
            reason=decision.reason,
        )


class RecoupArm:
    """The product: rank every action by expected value, take the best permitted."""

    name = "recoup_agent"
    description = "expected-value ranking over all actions, compliance-gated"

    def __init__(self, policy: PolicyConfig, compliance: ComplianceConfig) -> None:
        self._engine = PolicyEngine(policy)
        self._threshold = policy.ev_threshold_paise
        self.gate = ComplianceGate(compliance)

    def reset(self) -> None:
        self.gate.reset()

    def decide(self, context: DecisionContext) -> Decision:
        decision = self._engine.decide(context)

        # Screen only options actually worth taking. A candidate below the
        # threshold is not a real option, and putting it in front of the gate
        # produces a refusal that reads like a compliance decision when it was
        # really an economic one — which is how a refusal list fills with noise
        # and stops being worth reading.
        worthwhile = [c for c in decision.considered if c.ev >= self._threshold]

        if not worthwhile:
            return _stopped(context, "no option clears the expected-value threshold")

        chosen, vetoes = self.gate.screen(worthwhile, context)
        decision.vetoes = vetoes

        if chosen is None:
            return _stopped(context, "every worthwhile option refused by compliance", vetoes)

        decision.chosen = chosen
        return decision


def _stopped(context: DecisionContext, reason: str, vetoes=None) -> Decision:
    return Decision(
        payment_id=context.payment.id,
        at=context.now,
        chosen=Candidate(
            action=ActionKind.STOP, at=context.now, ev=0, probability=0.0
        ),
        vetoes=list(vetoes or []),
        reason=reason,
    )


def build_arms(policy: PolicyConfig, compliance: ComplianceConfig) -> list[Arm]:
    return [
        NaiveRetryArm(policy, compliance),
        ContactOnlyArm(policy, compliance),
        RecoupArm(policy, compliance),
    ]
