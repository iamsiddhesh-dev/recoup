"""Typed loading of config/world.yaml.

The YAML is the honest record of every modelling assumption; this is the schema
that stops a typo in it from becoming a silent bias. Mixture weights are validated
at load time rather than trusted, because a `method_mix` summing to 0.98 does not
crash — it just quietly reweights the whole simulation.

tests/test_config.py checks the same invariants against the file on disk. The
duplication is deliberate: the tests catch it in CI, this catches it for anyone
who edits the YAML and runs the simulator directly.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

from recoup.domain import FailureCause, Language, PaymentMethod, Severity

TOLERANCE = 1e-6


def _check_sums_to_one(values: dict[str, float], label: str) -> None:
    total = sum(values.values())
    if abs(total - 1.0) > TOLERANCE:
        raise ValueError(f"{label} must sum to 1.0, got {total}")


class RunConfig(BaseModel):
    seed: int
    batch_size: int
    horizon_days: int
    currency: str = "INR"
    timezone: str = "Asia/Kolkata"
    start_date: date = date(2026, 6, 1)

    @property
    def start_at(self) -> datetime:
        """Midnight on the fixed start date.

        Deliberately not `today`: a run recorded months ago has to be comparable
        to one from this afternoon, and nothing in the simulator may derive from
        wall-clock time.
        """
        return datetime.combine(self.start_date, time.min)

    @property
    def end_at(self) -> datetime:
        return self.start_at + timedelta(days=self.horizon_days)


class AmountSpec(BaseModel):
    dist: str
    mu: float
    sigma: float
    min: int
    max: int


class TaxonomyEntry(BaseModel):
    """One failure archetype: how it presents, and what it really is.

    `cause` is ground truth. It is used to decide what happens next and is never
    written onto the event the agent sees.

    `code` is optional and, where set, was observed on a real Razorpay response.
    It exists because the original design derived the top-level error code from
    `source`, and a live capture disproved that: Razorpay returned
    `BAD_REQUEST_ERROR` for a `gateway`-sourced failure. The two are independent,
    so the code is carried as data where it is known rather than inferred.
    """

    weight: float
    cause: FailureCause
    source: str
    step: str
    reason: str
    code: str | None = None


class PaydaySpec(BaseModel):
    days_of_month: list[int]
    multiplier: float


class RecoveryConfig(BaseModel):
    base_probability: dict[FailureCause, float]
    attempt_decay: float
    hour_multiplier: list[float]
    payday: PaydaySpec

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if len(self.hour_multiplier) != 24:
            raise ValueError(f"hour_multiplier needs 24 entries, got {len(self.hour_multiplier)}")
        return self


class SegmentSpec(BaseModel):
    share: float
    nudge_response: float
    annoyance_sensitivity: float


class CustomersConfig(BaseModel):
    count: int
    segments: dict[str, SegmentSpec]
    consent_rate: dict[str, float]
    language: dict[Language, float]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        _check_sums_to_one({k: v.share for k, v in self.segments.items()}, "segment shares")
        _check_sums_to_one({str(k): v for k, v in self.language.items()}, "language mix")
        return self


class DurationSpec(BaseModel):
    min: int
    max: int


class DowntimeConfig(BaseModel):
    events_per_horizon: int
    duration_minutes: DurationSpec
    severity_mix: dict[Severity, float]
    success_multiplier: dict[Severity, float]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        _check_sums_to_one({str(k): v for k, v in self.severity_mix.items()}, "severity_mix")
        return self


class IssuerSpec(BaseModel):
    code: str
    share: float
    reliability: float


class WorldConfig(BaseModel):
    run: RunConfig
    traffic_by_hour: list[float] = Field(min_length=24, max_length=24)
    downtime_technical_tilt: float = 1.0
    method_mix: dict[PaymentMethod, float]
    amounts: dict[PaymentMethod, AmountSpec]
    merchant_margin: float
    saved_instrument_rate: dict[PaymentMethod, float]
    failure_rate: dict[PaymentMethod, float]
    error_taxonomy: dict[PaymentMethod, list[TaxonomyEntry]]
    recovery: RecoveryConfig
    customers: CustomersConfig
    downtime: DowntimeConfig
    issuers: list[IssuerSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        _check_sums_to_one({str(k): v for k, v in self.method_mix.items()}, "method_mix")
        _check_sums_to_one({i.code: i.share for i in self.issuers}, "issuer shares")

        methods = set(self.method_mix)
        for label, mapping in (
            ("amounts", self.amounts),
            ("failure_rate", self.failure_rate),
            ("saved_instrument_rate", self.saved_instrument_rate),
            ("error_taxonomy", self.error_taxonomy),
        ):
            if set(mapping) != methods:
                missing = methods - set(mapping)
                extra = set(mapping) - methods
                raise ValueError(f"{label} methods mismatch: missing={missing} extra={extra}")

        for method, entries in self.error_taxonomy.items():
            _check_sums_to_one(
                {f"{i}": e.weight for i, e in enumerate(entries)},
                f"{method} failure weights",
            )
            for entry in entries:
                if entry.cause not in self.recovery.base_probability:
                    raise ValueError(f"{method}/{entry.reason}: no recovery probability for "
                                     f"{entry.cause}")

        return self

    @classmethod
    def load(cls, path: str | Path = "config/world.yaml") -> WorldConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
