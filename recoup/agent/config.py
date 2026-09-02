"""Typed loading of the agent's own configuration.

`config/policy.yaml` holds beliefs and costs; `config/compliance.yaml` holds hard
rules. Both belong to the agent and both are allowed to be wrong about the world —
that is the difference between them and `config/world.yaml`, which the agent
cannot read at all.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

from recoup.domain import FailureCause, Severity


class AnnoyanceConfig(BaseModel):
    penalty_per_prior_contact: int
    window_days: int
    escalation: float


class LearningConfig(BaseModel):
    min_observations: int
    blend: str = "shrinkage"


class NudgeConfig(BaseModel):
    """A contact recovers money only if two independent things happen.

    Kept separate because they fail for different reasons, and only response
    decays with how often we have already written to this customer.
    """

    prior_response: float
    prior_completion: float
    decay_per_prior_contact: float
    channel_response: dict[str, float]

    def multiplier_for(self, channel: str) -> float:
        return self.channel_response.get(channel, 1.0)


class DowntimePolicy(BaseModel):
    """Timing only. Whether an outage blocks a retry is a compliance rule."""

    recheck_minutes: int
    max_defer_hours: int


class PolicyConfig(BaseModel):
    assumed_margin: float
    action_costs: dict[str, int]
    retry_cost_escalation: float
    annoyance: AnnoyanceConfig
    prior_recovery_probability: dict[FailureCause, float]
    learning: LearningConfig
    nudge: NudgeConfig
    escalation_success_rate: float
    escalation_scarcity_premium: int
    retry_offsets_hours: list[int]
    delay_decay_per_day: float
    payday_lookahead_days: int
    downtime: DowntimePolicy
    ev_threshold_paise: int

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.action_costs.get("STOP", 0) != 0:
            raise ValueError("stopping must be free, or restraint is penalised")
        if any(cost < 0 for cost in self.action_costs.values()):
            raise ValueError("a negative action cost would be a free lunch")
        return self

    def cost_of(self, action: str, prior_attempts: int = 0) -> int:
        """Cost in paise, escalating for repeated attempts on one instrument.

        Retries look free per attempt and are not: repeated failed authorisations
        against the same instrument degrade the issuer relationship. Modelling
        that as an escalating cost is what stops the optimiser retrying forever
        wherever the cap happens to allow it.
        """
        base = self.action_costs.get(action, 0)
        if prior_attempts and action.startswith("RETRY"):
            return int(base * (self.retry_cost_escalation**prior_attempts))
        return base

    @classmethod
    def load(cls, path: str | Path = "config/policy.yaml") -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class AttemptLimits(BaseModel):
    max_per_payment: int
    max_per_instrument_per_day: int
    max_after_hard_decline: int
    cooling_off_after_failures: int
    cooling_off_hours: int


class QuietHours(BaseModel):
    start: time
    end: time
    timezone: str

    def covers(self, at: time) -> bool:
        """Quiet hours wrap midnight, so this is a union, not a range."""
        if self.start <= self.end:
            return self.start <= at < self.end
        return at >= self.start or at < self.end


class ContactLimits(BaseModel):
    max_per_customer_per_7d: int
    min_hours_between_contacts: int
    quiet_hours: QuietHours
    consent_required: list[str]


class HardStop(BaseModel):
    retry: bool
    contact: bool
    why: str
    max_contacts: int | None = None


class MandateRules(BaseModel):
    require_prenotification_hours: int
    respect_mandate_amount_cap: bool
    respect_mandate_end_date: bool
    max_debits_per_cycle: int


class EscalationRules(BaseModel):
    human_review_above_paise: int
    max_escalations_per_run: int


class ExecutionRules(BaseModel):
    require_test_mode: bool
    max_actions_per_run: int
    require_idempotency_key: bool


class DowntimeRules(BaseModel):
    refuse_retry_on_severity: list[Severity]


class ComplianceConfig(BaseModel):
    attempts: AttemptLimits
    contact: ContactLimits
    hard_stops: dict[str, HardStop] = Field(default_factory=dict)
    downtime: DowntimeRules
    mandate: MandateRules
    escalation: EscalationRules
    execution: ExecutionRules

    @model_validator(mode="after")
    def _hard_stops_name_real_causes(self) -> Self:
        """A hard stop keyed on something that is not a cause never fires.

        This is not hypothetical: the rule against re-debiting a revoked mandate
        was written as `MANDATE_REVOKED`, which is a reason string rather than a
        FailureCause, and silently matched nothing. The most legally consequential
        rule in the file was inert, and nothing failed. See FAILURES.md.

        Failing at load time makes that class of mistake impossible rather than
        merely unlikely.
        """
        known = {str(cause) for cause in FailureCause}
        unknown = set(self.hard_stops) - known

        if unknown:
            raise ValueError(
                f"hard_stops keys must be FailureCause values; {sorted(unknown)} "
                f"match nothing and would never fire. Valid: {sorted(known)}"
            )
        return self

    def hard_stop_for(self, cause: FailureCause | None) -> HardStop | None:
        return self.hard_stops.get(str(cause)) if cause else None

    @classmethod
    def load(cls, path: str | Path = "config/compliance.yaml") -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
