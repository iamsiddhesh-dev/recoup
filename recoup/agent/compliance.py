"""The compliance gate: hard rules with veto power over the policy.

Applied *after* the expected-value calculation, never blended into it. That
ordering is the design, not an implementation detail.

A rule expressed as a large negative cost is a rule that a large enough amount
can outvote. On a ₹50,000 payment, a ₹900 goodwill penalty for a fourth 3am
message disappears into the rounding, and the optimiser will happily send it and
be right by its own arithmetic. Money is exactly the domain where "the model
decided the fine was worth paying" is not an acceptable answer, so the rules that
must not be traded away are removed from the trade entirely.

The gate walks the policy's ranked candidates from the top and takes the first
that survives. That matters: a vetoed best option should fall through to the
second-best permitted one, not collapse to doing nothing. Everything refused on
the way down is recorded — those vetoes are what the refusal list is made of, and
"we deliberately did not touch these 340 cases, and here is the rule that stopped
each one" is a deliverable rather than an error log.

No model participates in anything here.
"""

from __future__ import annotations

from datetime import timedelta

from recoup.agent.actions import ActionKind, Candidate, Veto
from recoup.agent.config import ComplianceConfig
from recoup.agent.context import DecisionContext
from recoup.domain import FailureCause
from recoup.money import rupees


class ComplianceGate:
    def __init__(self, compliance: ComplianceConfig) -> None:
        self._rules = compliance
        self._escalations = 0
        self._actions = 0

    def reset(self) -> None:
        """Per-run counters. Caps are per run, not per lifetime."""
        self._escalations = 0
        self._actions = 0

    # -- individual rules ----------------------------------------------------
    # Each returns a Veto or None. Named individually rather than expressed as a
    # table so that deleting one is a visible act.

    def _check_hard_stop(
        self, candidate: Candidate, context: DecisionContext
    ) -> Veto | None:
        stop = self._rules.hard_stop_for(context.cause)
        if stop is None:
            return None

        if candidate.action.is_retry and not stop.retry:
            return Veto(
                rule=f"hard_stop:{context.cause}",
                action=candidate.action,
                why=stop.why.strip(),
            )

        if candidate.action.is_contact:
            if not stop.contact:
                return Veto(
                    rule=f"hard_stop:{context.cause}",
                    action=candidate.action,
                    why=stop.why.strip(),
                )
            if stop.max_contacts is not None and context.contacts_in_window >= stop.max_contacts:
                return Veto(
                    rule=f"hard_stop:{context.cause}:max_contacts",
                    action=candidate.action,
                    why=(
                        f"{context.cause} permits at most {stop.max_contacts} contact(s); "
                        f"{context.contacts_in_window} already sent"
                    ),
                )

        return None

    def _check_attempt_caps(
        self, candidate: Candidate, context: DecisionContext
    ) -> Veto | None:
        if not candidate.action.is_retry:
            return None

        caps = self._rules.attempts

        if context.attempts >= caps.max_per_payment:
            return Veto(
                rule="attempts:max_per_payment",
                action=candidate.action,
                why=f"{context.attempts} attempts already made, cap is {caps.max_per_payment}",
            )

        if context.consecutive_failures >= caps.cooling_off_after_failures:
            return Veto(
                rule="attempts:cooling_off",
                action=candidate.action,
                why=(
                    f"{context.consecutive_failures} consecutive failures on this "
                    f"instrument; it needs replacing, not retrying"
                ),
            )

        return None

    def _check_contact_limits(
        self, candidate: Candidate, context: DecisionContext
    ) -> Veto | None:
        if not candidate.action.is_contact:
            return None

        limits = self._rules.contact

        if context.contacts_in_window >= limits.max_per_customer_per_7d:
            return Veto(
                rule="contact:max_per_customer",
                action=candidate.action,
                why=(
                    f"{context.contacts_in_window} contacts in the last 7 days, "
                    f"cap is {limits.max_per_customer_per_7d}"
                ),
            )

        since = context.hours_since_last_contact()
        if since is not None and since < limits.min_hours_between_contacts:
            return Veto(
                rule="contact:min_interval",
                action=candidate.action,
                why=(
                    f"last contact was {since:.0f}h ago, minimum interval is "
                    f"{limits.min_hours_between_contacts}h"
                ),
            )

        if limits.quiet_hours.covers(candidate.at.time()):
            return Veto(
                rule="contact:quiet_hours",
                action=candidate.action,
                why=(
                    f"{candidate.at:%H:%M} falls inside quiet hours "
                    f"{limits.quiet_hours.start:%H:%M}–{limits.quiet_hours.end:%H:%M}"
                ),
            )

        channel = candidate.action.channel
        if channel and str(channel) in limits.consent_required:
            if not context.notes.get(f"consent:{channel}") == "true":
                return Veto(
                    rule=f"contact:consent:{channel}",
                    action=candidate.action,
                    why=f"no recorded consent for {channel}",
                )

        return None

    def _check_downtime(
        self, candidate: Candidate, context: DecisionContext
    ) -> Veto | None:
        """Refuse to retry into a known outage.

        Not a compliance rule in the legal sense, but it belongs here for the same
        reason the others do: it is a bound on behaviour that must not be traded
        against a large amount. A retry during a severe outage is near certain to
        fail *and* consumes an attempt from a cap that cannot be refilled. The
        real cost is the lost future attempt, which no per-attempt price captures,
        so left to arithmetic a large enough payment would always justify burning
        one.

        Only retries scheduled *into* the outage are refused. A retry scheduled
        for after it clears is exactly the right response and must survive.
        """
        if not candidate.action.is_retry or context.downtime is None:
            return None

        if context.downtime.severity not in self._rules.downtime.refuse_retry_on_severity:
            return None

        if context.downtime.end and candidate.at.timestamp() >= context.downtime.end:
            return None

        return Veto(
            rule="downtime:active",
            action=candidate.action,
            why=(
                f"{context.downtime.method} is in a "
                f"{context.downtime.severity}-severity outage; retrying now would "
                f"spend an attempt from a cap that cannot be refilled"
            ),
        )

    def _check_escalation(
        self, candidate: Candidate, context: DecisionContext
    ) -> Veto | None:
        if candidate.action is not ActionKind.ESCALATE_HUMAN:
            return None

        rules = self._rules.escalation

        if context.amount < rules.human_review_above_paise and context.cause not in (
            FailureCause.RISK_BLOCKED,
            FailureCause.MANDATE_PROBLEM,
        ):
            return Veto(
                rule="escalation:below_threshold",
                action=candidate.action,
                why=(
                    f"{rupees(context.amount)} is below the "
                    f"{rupees(rules.human_review_above_paise)} human-review threshold"
                ),
            )

        if context.escalations >= rules.max_escalations_per_payment:
            return Veto(
                rule="escalation:already_escalated",
                action=candidate.action,
                why=(
                    f"already handed to human review "
                    f"{context.escalations} time(s). A second reviewer does not "
                    f"make the first one's answer different, and each slot spent "
                    f"here is one not spent on another payment."
                ),
            )

        if self._escalations >= rules.max_escalations_per_run:
            return Veto(
                rule="escalation:run_cap",
                action=candidate.action,
                why=(
                    f"{self._escalations} escalations already this run, cap is "
                    f"{rules.max_escalations_per_run}. An agent that escalates "
                    f"everything is not a product."
                ),
            )

        return None

    def _check_run_caps(self, candidate: Candidate) -> Veto | None:
        if candidate.action is ActionKind.STOP:
            return None

        if self._actions >= self._rules.execution.max_actions_per_run:
            return Veto(
                rule="execution:max_actions_per_run",
                action=candidate.action,
                why=(
                    f"{self._actions} actions already executed this run. A runaway "
                    f"loop in a recovery agent is not a crash, it is a customer "
                    f"being charged repeatedly."
                ),
            )

        return None

    # -- screening -----------------------------------------------------------

    def screen(
        self, candidates: list[Candidate], context: DecisionContext
    ) -> tuple[Candidate | None, list[Veto]]:
        """Highest-EV candidate that survives every rule, plus everything refused."""
        vetoes: list[Veto] = []

        for candidate in candidates:
            if candidate.action is ActionKind.STOP:
                continue

            veto = (
                self._check_run_caps(candidate)
                or self._check_hard_stop(candidate, context)
                or self._check_attempt_caps(candidate, context)
                or self._check_downtime(candidate, context)
                or self._check_contact_limits(candidate, context)
                or self._check_escalation(candidate, context)
            )

            if veto is not None:
                vetoes.append(veto)
                continue

            return candidate, vetoes

        return None, vetoes

    def note_executed(self, candidate: Candidate) -> None:
        self._actions += 1
        if candidate.action is ActionKind.ESCALATE_HUMAN:
            self._escalations += 1


def next_permitted_contact_time(
    context: DecisionContext, compliance: ComplianceConfig
):
    """When a contact would next be allowed.

    Quiet hours are a deferral, not a refusal: the right response to "it is 3am"
    is to send at 9am, not to abandon the money. Returning the time lets the
    executor reschedule rather than drop.
    """
    quiet = compliance.contact.quiet_hours
    when = context.now

    for _ in range(48):
        if not quiet.covers(when.time()):
            return when
        when += timedelta(hours=1)

    return None
