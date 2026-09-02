"""Issuers and their outages.

Two jobs. First, give payments a bank so that root-cause analysis has something
real to find — a run where every issuer behaves identically has no signal in it,
and an agent that finds none looks the same as an agent that cannot.

Second, and more importantly, generate downtime. Downtime is the trap the policy
engine exists to avoid: retrying into a known issuer outage is guaranteed to fail
and burns an attempt against a hard cap that cannot be refilled. An agent that
ignores the signal loses attempts it needed later, so this is where "consult the
degradation feed" stops being a nice-to-have and starts costing money.

Windows are emitted as Razorpay-shaped `payment.downtime.started` / `.resolved`
events, so the agent consumes exactly the feed it would consume in production.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from recoup.domain import DowntimeEntity, PaymentMethod, Severity
from recoup.rng import substream, weighted_choice
from recoup.world.config import WorldConfig


@dataclass(frozen=True)
class DowntimeWindow:
    id: str
    method: PaymentMethod
    issuer_code: str | None
    begin: datetime
    end: datetime
    severity: Severity

    def covers(self, when: datetime) -> bool:
        return self.begin <= when < self.end


class IssuerBook:
    """The issuer population plus the outage calendar for a run."""

    def __init__(self, config: WorldConfig) -> None:
        self._config = config
        self._shares = {issuer.code: issuer.share for issuer in config.issuers}
        self._reliability = {issuer.code: issuer.reliability for issuer in config.issuers}
        self._rng = substream(config.run.seed, "issuers")
        self.windows: list[DowntimeWindow] = self._generate_windows()

    # -- population ---------------------------------------------------------

    def pick(self, rng: random.Random) -> str:
        return weighted_choice(rng, self._shares)

    def reliability(self, issuer_code: str) -> float:
        """Multiplier on recovery probability. Some banks are just better."""
        return self._reliability.get(issuer_code, 1.0)

    # -- downtime -----------------------------------------------------------

    def _generate_windows(self) -> list[DowntimeWindow]:
        cfg = self._config.downtime
        start = self._config.run.start_at
        horizon = timedelta(days=self._config.run.horizon_days)
        method_weights = {str(k): v for k, v in self._config.method_mix.items()}
        severity_weights = {str(k): v for k, v in cfg.severity_mix.items()}

        windows: list[DowntimeWindow] = []
        for index in range(cfg.events_per_horizon):
            offset = timedelta(seconds=self._rng.uniform(0, horizon.total_seconds()))
            begin = start + offset
            minutes = self._rng.randint(cfg.duration_minutes.min, cfg.duration_minutes.max)
            method = PaymentMethod(weighted_choice(self._rng, method_weights))

            # Card and netbanking outages are usually one bank; UPI outages more
            # often sit at the network or PSP layer and hit everyone at once.
            bank_specific = method in (PaymentMethod.CARD, PaymentMethod.NETBANKING) or (
                self._rng.random() < 0.4
            )
            issuer_code = self.pick(self._rng) if bank_specific else None

            windows.append(
                DowntimeWindow(
                    id=f"down_{index:04d}",
                    method=method,
                    issuer_code=issuer_code,
                    begin=begin,
                    end=begin + timedelta(minutes=minutes),
                    severity=Severity(weighted_choice(self._rng, severity_weights)),
                )
            )

        return sorted(windows, key=lambda w: w.begin)

    def multiplier_at(
        self, when: datetime, method: PaymentMethod, issuer_code: str | None
    ) -> float:
        """How much a live outage suppresses success. 1.0 when all is well."""
        worst = 1.0
        for window in self.windows:
            if window.method != method or not window.covers(when):
                continue
            if window.issuer_code is not None and window.issuer_code != issuer_code:
                continue
            worst = min(worst, self._config.downtime.success_multiplier[window.severity])
        return worst

    def active_at(self, when: datetime) -> list[DowntimeWindow]:
        return [w for w in self.windows if w.covers(when)]

    def events(self) -> list[tuple[datetime, str, DowntimeEntity]]:
        """Downtime as (when, event_name, entity), ready to be queued.

        Both edges are emitted. The resolved event matters as much as the started
        one — an agent that defers on downtime but never learns it ended would sit
        on recoverable money until the horizon runs out.
        """
        emitted: list[tuple[datetime, str, DowntimeEntity]] = []
        for window in self.windows:
            instrument = {"bank": window.issuer_code} if window.issuer_code else {}
            base = {
                "id": window.id,
                "method": window.method,
                "begin": int(window.begin.timestamp()),
                "severity": window.severity,
                "instrument": instrument,
            }
            emitted.append(
                (
                    window.begin,
                    "payment.downtime.started",
                    DowntimeEntity(**base, status="started", end=None),
                )
            )
            emitted.append(
                (
                    window.end,
                    "payment.downtime.resolved",
                    DowntimeEntity(
                        **base, status="resolved", end=int(window.end.timestamp())
                    ),
                )
            )
        return sorted(emitted, key=lambda item: item[0])
