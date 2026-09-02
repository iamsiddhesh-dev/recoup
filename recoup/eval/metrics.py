"""Scoring a run.

The headline is **incremental** recovery — what the agent earned above the naive
baseline — not gross. Gross recovery is a number any arm can produce by retrying
everything, and it says nothing about whether the decisions were good.

Everything is reported in paise internally and converted only for display, so no
rounding creeps into the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recoup.ledger.events import EventKind, Ledger


@dataclass
class ArmMetrics:
    arm: str
    description: str = ""

    observed: int = 0
    amount_at_risk: int = 0

    recovered_count: int = 0
    recovered_paise: int = 0

    actions: int = 0
    contacts: int = 0
    cost_paise: int = 0

    vetoes: int = 0
    veto_amount: int = 0
    unresolved: int = 0

    actions_by_kind: dict[str, int] = field(default_factory=dict)
    veto_by_rule: dict[str, int] = field(default_factory=dict)
    recovered_by_action: dict[str, int] = field(default_factory=dict)

    margin: float = 0.28

    @property
    def recovery_rate(self) -> float:
        return self.recovered_count / self.observed if self.observed else 0.0

    @property
    def money_recovery_rate(self) -> float:
        return self.recovered_paise / self.amount_at_risk if self.amount_at_risk else 0.0

    @property
    def margin_recovered(self) -> int:
        """What the merchant actually keeps. The only figure worth optimising."""
        return int(self.recovered_paise * self.margin)

    @property
    def net_paise(self) -> int:
        return self.margin_recovered - self.cost_paise

    @property
    def contacts_per_recovery(self) -> float:
        return self.contacts / self.recovered_count if self.recovered_count else 0.0

    @property
    def cost_per_recovery(self) -> int:
        return int(self.cost_paise / self.recovered_count) if self.recovered_count else 0


def score(ledger: Ledger, arm: str, description: str, margin: float) -> ArmMetrics:
    metrics = ArmMetrics(arm=arm, description=description, margin=margin)

    for event in ledger.events(arm=arm):
        match event.kind:
            case EventKind.OBSERVED:
                metrics.observed += 1
                metrics.amount_at_risk += event.amount or 0

            case EventKind.CLASSIFIED:
                if not event.data.get("cause"):
                    metrics.unresolved += 1

            case EventKind.EXECUTED:
                action = event.data.get("action", "UNKNOWN")
                metrics.actions += 1
                metrics.actions_by_kind[action] = metrics.actions_by_kind.get(action, 0) + 1
                metrics.cost_paise += event.data.get("cost", 0)
                if event.data.get("delivered"):
                    metrics.contacts += 1

            case EventKind.RECOVERED:
                metrics.recovered_count += 1
                metrics.recovered_paise += event.amount or 0
                via = event.data.get("via", "UNKNOWN")
                metrics.recovered_by_action[via] = (
                    metrics.recovered_by_action.get(via, 0) + 1
                )

            case EventKind.VETOED:
                metrics.vetoes += 1
                metrics.veto_amount += event.amount or 0
                rule = event.data.get("rule", "unknown")
                metrics.veto_by_rule[rule] = metrics.veto_by_rule.get(rule, 0) + 1

    return metrics


@dataclass
class Comparison:
    """One arm measured against the baseline."""

    arm: ArmMetrics
    baseline: ArmMetrics

    @property
    def incremental_paise(self) -> int:
        return self.arm.recovered_paise - self.baseline.recovered_paise

    @property
    def incremental_count(self) -> int:
        return self.arm.recovered_count - self.baseline.recovered_count

    @property
    def incremental_net(self) -> int:
        return self.arm.net_paise - self.baseline.net_paise

    @property
    def lift(self) -> float:
        if not self.baseline.recovered_paise:
            return 0.0
        return self.incremental_paise / self.baseline.recovered_paise


def rupees(paise: int) -> str:
    return f"₹{paise / 100:,.0f}"


def table(results: list[ArmMetrics], baseline_name: str) -> str:
    """The arms table, for the terminal.

    Deliberately shows cost and contacts alongside recovery. An arm that recovers
    more money by messaging everyone twice as often has not necessarily done
    better, and a table that hides the cost invites exactly that conclusion.
    """
    baseline = next(m for m in results if m.arm == baseline_name)

    header = (
        f"{'arm':<18}{'recovered':>13}{'rate':>8}{'incremental':>14}"
        f"{'cost':>10}{'net':>13}{'contacts':>10}{'vetoes':>8}"
    )
    lines = [header, "-" * len(header)]

    for metrics in results:
        comparison = Comparison(arm=metrics, baseline=baseline)
        incremental = (
            "—"
            if metrics.arm == baseline_name
            else f"{rupees(comparison.incremental_paise)}"
        )
        lines.append(
            f"{metrics.arm:<18}"
            f"{rupees(metrics.recovered_paise):>13}"
            f"{metrics.money_recovery_rate:>7.1%} "
            f"{incremental:>14}"
            f"{rupees(metrics.cost_paise):>10}"
            f"{rupees(metrics.net_paise):>13}"
            f"{metrics.contacts:>10,}"
            f"{metrics.vetoes:>8,}"
        )

    lines.append("")
    lines.append(
        f"at risk: {rupees(baseline.amount_at_risk)} across {baseline.observed:,} "
        f"failures · margin {baseline.margin:.0%} · net = margin recovered − cost"
    )
    return "\n".join(lines)
