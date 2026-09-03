"""Checking that the numbers in the README are still the numbers the code produces.

A README asserting a result is a claim. `recoup reproduce` turns it into a test:
re-run the evaluation from the committed seed and compare, figure by figure,
against what was recorded when the claim was written.

The strongest check here is the ledger digest — a hash over every event in an
arm's stream, in order. Two runs with the same digest made every identical
decision, in the same sequence, at the same simulated times. Nothing weaker
catches a change that happens to leave the totals alone, and "the money came out
the same by coincidence" is exactly the kind of drift that makes a reproducible
claim quietly stop being one.

The individual figures are recorded anyway, even though the digest subsumes them,
because a digest tells you *that* something changed and never *what*. A failing
reproduce should point at the line that moved.

The claims file is committed. `data/` is not — it is regenerable — so a fresh
clone has no baseline of its own, and without one this command could only report
that a run agrees with itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from recoup.eval.metrics import ArmMetrics, rupees

DEFAULT_CLAIMS = Path("reports/claims.json")

BASELINE = "naive_baseline"
CONTACT_ONLY = "contact_only"
AGENT = "recoup_agent"
AGENT_NO_LLM = "recoup_agent_no_llm"

# Recorded per arm. Deliberately not everything in ArmMetrics: these are the
# figures the README and the screens actually assert. Adding a field here makes
# reproduce stricter; the digest is already maximally strict, so the point of
# this list is diagnosis rather than coverage.
ARM_FIELDS = (
    "recovered_paise",
    "recovered_count",
    "contacts",
    "cost_paise",
    "vetoes",
    "unresolved",
)

MONEY_FIELDS = {"recovered_paise", "cost_paise", "incremental_paise", "coverage_paise",
                "judgment_paise", "llm_delta_paise", "amount_at_risk"}

# Identifiers rather than quantities. A seed rendered as "20,260,902" invites the
# reader to compare its magnitude to something, which is meaningless.
OPAQUE_KEYS = {"run.seed"}


@dataclass(frozen=True)
class Check:
    """One recorded figure, and whether it still holds."""

    key: str
    expected: object
    actual: object

    @property
    def ok(self) -> bool:
        return self.expected == self.actual

    def _show(self, value: object) -> str:
        if value is None:
            return "—"
        if self.key in OPAQUE_KEYS:
            return str(value)
        if isinstance(value, int) and self.key.rsplit(".", 1)[-1] in MONEY_FIELDS:
            return rupees(value)
        if isinstance(value, str) and len(value) == 64:
            return value[:12] + "…"
        return f"{value:,}" if isinstance(value, int) else str(value)

    def line(self) -> str:
        mark = "ok" if self.ok else "CHANGED"
        if self.ok:
            return f"  {mark:<8} {self.key:<44} {self._show(self.actual)}"
        return (
            f"  {mark:<8} {self.key:<44} {self._show(self.actual)}"
            f"   (recorded {self._show(self.expected)})"
        )


def claims(
    results: list[ArmMetrics],
    *,
    seed: int,
    batch_size: int,
    horizon_days: int,
    digests: dict[str, str],
) -> dict:
    """The canonical shape of a recorded run.

    Flat, string-keyed and sorted, so a diff of two claims files is readable by a
    person rather than only by this module.
    """
    by_arm = {m.arm: m for m in results}

    recorded: dict[str, object] = {
        "run.seed": seed,
        "run.batch_size": batch_size,
        "run.horizon_days": horizon_days,
        "run.observed": by_arm[BASELINE].observed if BASELINE in by_arm else None,
        "run.amount_at_risk": by_arm[BASELINE].amount_at_risk if BASELINE in by_arm else None,
    }

    for name, metrics in sorted(by_arm.items()):
        for field in ARM_FIELDS:
            recorded[f"{name}.{field}"] = getattr(metrics, field)
        recorded[f"{name}.digest"] = digests.get(name)

    # The derived figures the README leads with. Recorded rather than recomputed
    # at comparison time so that a change in how they are derived is itself a
    # change reproduce reports.
    baseline = by_arm.get(BASELINE)
    contact = by_arm.get(CONTACT_ONLY)
    agent = by_arm.get(AGENT)
    no_llm = by_arm.get(AGENT_NO_LLM)

    if agent and baseline:
        recorded["headline.incremental_paise"] = (
            agent.recovered_paise - baseline.recovered_paise
        )
    if agent and contact and baseline:
        recorded["headline.coverage_paise"] = (
            contact.recovered_paise - baseline.recovered_paise
        )
        recorded["headline.judgment_paise"] = agent.recovered_paise - contact.recovered_paise
    if agent and no_llm:
        recorded["headline.llm_delta_paise"] = (
            agent.recovered_paise - no_llm.recovered_paise
        )

    return dict(sorted(recorded.items()))


def compare(recorded: dict, produced: dict) -> list[Check]:
    """Every key in either file, so an added or removed figure is also a change."""
    return [
        Check(key=key, expected=recorded.get(key), actual=produced.get(key))
        for key in sorted(set(recorded) | set(produced))
    ]


def save(recorded: dict, path: str | Path = DEFAULT_CLAIMS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    return path


def load(path: str | Path = DEFAULT_CLAIMS) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def report(checks: list[Check]) -> str:
    """The whole comparison, passes included.

    Printing only failures would make a successful reproduce indistinguishable
    from one that checked nothing — which is the failure mode this command exists
    to rule out.
    """
    changed = [c for c in checks if not c.ok]
    lines = [c.line() for c in checks]
    lines.append("")
    if changed:
        lines.append(
            f"{len(changed)} of {len(checks)} recorded figures changed. "
            "Either the code moved or the claim is stale — fix one of them."
        )
    else:
        lines.append(f"all {len(checks)} recorded figures reproduce exactly.")
    return "\n".join(lines)
