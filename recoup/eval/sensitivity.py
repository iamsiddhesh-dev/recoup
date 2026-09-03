"""Does the result survive the assumptions being wrong?

Every number this project reports rests on `config/world.yaml`, and most of that
file is tagged `[ASSUMPTION]` because it is a modelling choice rather than a
measurement. A single run at one setting is therefore a claim about one
hypothetical month, and the obvious objection — "you picked numbers that made your
agent look good" — is unanswerable without this.

So each load-bearing assumption is moved down and up, the whole evaluation is
re-run, and the effect on the headline is recorded. What comes out is a tornado:
axes ordered by how much they move the result, widest at the top.

Two things make it honest rather than decorative.

**The agent's configuration does not move.** Only the world does. Re-tuning the
policy at each point would be asking "could a differently configured agent still
win", which is a much easier question than the one being asked.

**The reported metric is the judgment gain**, not total recovery. Total recovery
moves mechanically with the size of the pool — make failures more common and
everyone recovers more — so a tornado on it would mostly measure the arithmetic of
its own axes. The gap between the agent and an arm that also contacts customers is
the part that is actually a claim about the policy engine, and it is the part worth
stress-testing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from recoup.agent.config import PolicyConfig
from recoup.eval import run_all
from recoup.eval.metrics import ArmMetrics
from recoup.money import rupees
from recoup.world.config import WorldConfig

BASELINE = "naive_baseline"
CONTACT = "contact_only"
AGENT = "recoup_agent"

DEFAULT_REPORT = Path("reports/sensitivity.json")


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


def _scale_failure_rate(world: WorldConfig, policy: PolicyConfig, factor: float) -> None:
    for method in world.failure_rate:
        world.failure_rate[method] = min(0.95, world.failure_rate[method] * factor)


def _scale_saved_instruments(world: WorldConfig, policy: PolicyConfig, factor: float) -> None:
    for method in world.saved_instrument_rate:
        world.saved_instrument_rate[method] = min(
            1.0, world.saved_instrument_rate[method] * factor
        )


def _scale_recovery(world: WorldConfig, policy: PolicyConfig, factor: float) -> None:
    for cause in world.recovery.base_probability:
        world.recovery.base_probability[cause] = min(
            1.0, world.recovery.base_probability[cause] * factor
        )


def _scale_nudge_response(world: WorldConfig, policy: PolicyConfig, factor: float) -> None:
    for segment in world.customers.segments.values():
        segment.nudge_response = min(1.0, segment.nudge_response * factor)


def _scale_downtime(world: WorldConfig, policy: PolicyConfig, factor: float) -> None:
    world.downtime.events_per_horizon = max(0, round(world.downtime.events_per_horizon * factor))


def _scale_margin(world: WorldConfig, policy: PolicyConfig, factor: float) -> None:
    """Move the true margin *and* the agent's belief about it.

    The first version scaled only the world's margin and produced exactly zero
    swing — which looked like a finding and was a bug. The agent prices every
    decision on `policy.assumed_margin`, so changing the world's margin alone
    moves nothing it can see; it only rescales the reported net. A merchant whose
    margin is genuinely different would know that, so the belief moves with it.
    """
    world.merchant_margin = min(1.0, world.merchant_margin * factor)
    policy.assumed_margin = min(1.0, policy.assumed_margin * factor)


def _scale_annoyance(world: WorldConfig, policy: PolicyConfig, factor: float) -> None:
    for segment in world.customers.segments.values():
        segment.annoyance_sensitivity = segment.annoyance_sensitivity * factor


@dataclass(frozen=True)
class Axis:
    key: str
    label: str
    why: str
    low: float
    high: float
    # Takes the policy too, because a few assumptions are things the merchant
    # would know about themselves. Most axes ignore it.
    apply: Callable[[WorldConfig, PolicyConfig, float], None]


AXES: list[Axis] = [
    Axis(
        key="failure_rate",
        label="How often payments fail",
        why="Sets the size of the recoverable pool. The single most load-bearing "
        "number in the model.",
        low=0.7,
        high=1.3,
        apply=_scale_failure_rate,
    ),
    Axis(
        key="saved_instruments",
        label="Share with standing authorisation",
        why="Decides how much of the pool can be retried at all rather than "
        "needing the customer. Drives the coverage/judgment split.",
        low=0.5,
        high=1.5,
        apply=_scale_saved_instruments,
    ),
    Axis(
        key="recovery_probability",
        label="How recoverable failures are",
        why="If almost nothing recovers, no policy can distinguish itself.",
        low=0.7,
        high=1.3,
        apply=_scale_recovery,
    ),
    Axis(
        key="nudge_response",
        label="How responsive customers are",
        why="Most of the money needs a customer to act, so this governs the "
        "ceiling on the contact path.",
        low=0.7,
        high=1.3,
        apply=_scale_nudge_response,
    ),
    Axis(
        key="annoyance",
        label="How much repeat contact costs",
        why="The agent's restraint is only worth something if over-contacting "
        "genuinely hurts.",
        low=0.5,
        high=2.0,
        apply=_scale_annoyance,
    ),
    Axis(
        key="downtime",
        label="How often issuers go down",
        why="The trap the policy avoids. Thin exposure here would mean the "
        "downtime logic is untested by the result.",
        low=0.3,
        high=3.0,
        apply=_scale_downtime,
    ),
    Axis(
        key="margin",
        label="Contribution margin",
        why="Every expected-value decision is priced on margin, so it moves what "
        "is worth doing at all.",
        low=0.7,
        high=1.3,
        apply=_scale_margin,
    ),
]

BY_KEY = {axis.key: axis for axis in AXES}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Point:
    axis: str
    level: str  # base | low | high
    factor: float
    recovered: int = 0
    incremental: int = 0
    judgment: int = 0
    contacts: int = 0
    at_risk: int = 0

    @property
    def rate(self) -> float:
        return self.recovered / self.at_risk if self.at_risk else 0.0


@dataclass
class Band:
    """One axis's effect: where the metric lands at each end."""

    axis: str
    label: str
    why: str
    low: int
    high: int
    base: int

    # Filled in by the view so every bar sits on one shared scale. Kept here
    # rather than computed in the template, which should not do arithmetic.
    left_pct: float = 0.0
    width_pct: float = 0.0

    @property
    def span(self) -> int:
        return abs(self.high - self.low)

    @property
    def low_delta(self) -> int:
        return self.low - self.base

    @property
    def high_delta(self) -> int:
        return self.high - self.base

    @property
    def worst(self) -> int:
        return min(self.low, self.high)


@dataclass
class SweepResult:
    base: Point
    points: list[Point] = field(default_factory=list)

    def bands(self, metric: str = "judgment") -> list[Band]:
        """Axes ordered by how much they move the metric — a tornado."""
        base_value = getattr(self.base, metric)
        by_axis: dict[str, dict[str, int]] = {}

        for point in self.points:
            by_axis.setdefault(point.axis, {})[point.level] = getattr(point, metric)

        bands = [
            Band(
                axis=key,
                label=BY_KEY[key].label,
                why=BY_KEY[key].why,
                low=levels.get("low", base_value),
                high=levels.get("high", base_value),
                base=base_value,
            )
            for key, levels in by_axis.items()
            if key in BY_KEY
        ]

        return sorted(bands, key=lambda b: -b.span)

    def holds(self, metric: str = "judgment") -> bool:
        """Whether the claim survives every point in the sweep.

        The claim being tested is directional — the agent beats an arm that also
        contacts customers — not that it beats it by any particular amount. A
        result that flips sign under a plausible assumption is a result that
        should not be reported as a finding.
        """
        return all(getattr(point, metric) > 0 for point in [self.base, *self.points])

    def worst_case(self, metric: str = "judgment") -> Point:
        return min([self.base, *self.points], key=lambda p: getattr(p, metric))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _measure(results: list[ArmMetrics], axis: str, level: str, factor: float) -> Point:
    def find(name: str) -> ArmMetrics | None:
        return next((m for m in results if m.arm == name), None)

    base, contact, agent = find(BASELINE), find(CONTACT), find(AGENT)
    if agent is None:
        return Point(axis=axis, level=level, factor=factor)

    return Point(
        axis=axis,
        level=level,
        factor=factor,
        recovered=agent.recovered_paise,
        incremental=agent.recovered_paise - (base.recovered_paise if base else 0),
        judgment=agent.recovered_paise - (contact.recovered_paise if contact else 0),
        contacts=agent.contacts,
        at_risk=agent.amount_at_risk,
    )


def run_sweep(
    axes: list[Axis] | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> SweepResult:
    """Re-run the whole evaluation at each end of each axis.

    The LLM fallback is switched off throughout. It was measured as contributing
    approximately nothing, it costs an arm and a model call per run, and holding
    it constant keeps every point in the sweep comparable.
    """
    axes = axes if axes is not None else AXES
    total = 1 + 2 * len(axes)
    done = 0

    def step(label: str) -> None:
        nonlocal done
        done += 1
        if on_progress:
            on_progress(label, done, total)

    results, ledger = run_all(WorldConfig.load(), use_llm=False)
    ledger.close()
    base = _measure(results, "base", "base", 1.0)
    step("baseline")

    points: list[Point] = []
    for axis in axes:
        for level, factor in (("low", axis.low), ("high", axis.high)):
            world = WorldConfig.load().model_copy(deep=True)
            policy = PolicyConfig.load().model_copy(deep=True)
            axis.apply(world, policy, factor)

            results, ledger = run_all(world, use_llm=False, policy=policy)
            ledger.close()

            points.append(_measure(results, axis.key, level, factor))
            step(f"{axis.key} {level}")

    return SweepResult(base=base, points=points)


def save(result: SweepResult, path: str | Path = DEFAULT_REPORT) -> Path:
    """Commit the sweep so a reader does not have to re-run five minutes of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "base": asdict(result.base),
                "points": [asdict(p) for p in result.points],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load(path: str | Path = DEFAULT_REPORT) -> SweepResult | None:
    path = Path(path)
    if not path.exists():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))
    return SweepResult(
        base=Point(**raw["base"]),
        points=[Point(**p) for p in raw["points"]],
    )


def table(result: SweepResult, metric: str = "judgment") -> str:
    """The tornado, for a terminal."""
    bands = result.bands(metric)
    base = getattr(result.base, metric)
    widest = max((b.span for b in bands), default=1) or 1

    lines = [
        f"{'assumption':<34}{'low':>12}{'high':>12}{'swing':>12}",
        "-" * 70,
    ]

    for band in bands:
        bar = "█" * max(1, round(band.span / widest * 18))
        lines.append(
            f"{band.label:<34}"
            f"{rupees(band.low, symbol=False):>11}"
            f"{rupees(band.high, symbol=False):>12}"
            f"{rupees(band.span, symbol=False):>12}  {bar}"
        )

    worst = result.worst_case(metric)
    lines += [
        "-" * 70,
        f"baseline {metric}: {rupees(base)}",
        f"worst case:        {rupees(getattr(worst, metric))} "
        f"({worst.axis} {worst.level})",
        f"claim holds everywhere: {'yes' if result.holds(metric) else 'NO'}",
    ]
    return "\n".join(lines)
