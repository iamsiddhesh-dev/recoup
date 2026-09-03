"""Persisting a run so the control room can read it.

A full evaluation takes ~35 seconds. A page load cannot. The resolution is to
treat a completed run as an artifact: compute it once, write it to disk, and serve
every screen as a query against that.

This works cleanly here because of what the data *is* — deterministic given a
seed, identical for every viewer, and frozen once produced. Those three properties
between them delete the hardest problem in caching, which is knowing when to throw
something away. There is never a reason to.

The ledger was already SQLite with indexes on `(arm, payment_id, seq)` and
`(arm, kind, seq)`, built for auditability rather than for this. Every screen turns
out to be a query against those: the case view is `story_of`, the scrubber is
`state_at`, the refusal list is a filter on vetoes. Pointing it at a file instead
of memory was the whole change.

Scored metrics are written alongside as JSON so the server does not have to fold
the entire event stream at boot just to render a headline number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from recoup.eval.metrics import ArmMetrics
from recoup.ledger.events import Ledger

DEFAULT_DIR = Path("data")
LEDGER_NAME = "run.db"
SUMMARY_NAME = "run.json"


@dataclass
class RunSummary:
    """Everything the control room needs before touching the ledger."""

    seed: int
    generated_at: str
    horizon_days: int
    batch_size: int
    margin: float
    arms: list[dict]

    @property
    def at_risk(self) -> int:
        return self.arms[0]["amount_at_risk"] if self.arms else 0

    @property
    def observed(self) -> int:
        return self.arms[0]["observed"] if self.arms else 0

    def arm(self, name: str) -> dict | None:
        return next((a for a in self.arms if a["arm"] == name), None)


def _metrics_to_dict(metrics: ArmMetrics) -> dict:
    """Flatten, including the derived figures.

    The properties are computed here rather than left to the template. A number
    shown to a merchant should be calculated in one place, not re-derived in
    whichever view happens to need it.
    """
    return {
        **asdict(metrics),
        "recovery_rate": metrics.recovery_rate,
        "money_recovery_rate": metrics.money_recovery_rate,
        "margin_recovered": metrics.margin_recovered,
        "net_paise": metrics.net_paise,
        "contacts_per_recovery": metrics.contacts_per_recovery,
        "cost_per_recovery": metrics.cost_per_recovery,
    }


def save_summary(
    results: list[ArmMetrics],
    seed: int,
    horizon_days: int,
    batch_size: int,
    margin: float,
    directory: str | Path = DEFAULT_DIR,
) -> Path:
    path = Path(directory) / SUMMARY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = RunSummary(
        seed=seed,
        generated_at=datetime.now(UTC).isoformat(),
        horizon_days=horizon_days,
        batch_size=batch_size,
        margin=margin,
        arms=[_metrics_to_dict(m) for m in results],
    )

    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    return path


def load_summary(directory: str | Path = DEFAULT_DIR) -> RunSummary | None:
    path = Path(directory) / SUMMARY_NAME
    if not path.exists():
        return None
    return RunSummary(**json.loads(path.read_text(encoding="utf-8")))


def ledger_path(directory: str | Path = DEFAULT_DIR) -> Path:
    return Path(directory) / LEDGER_NAME


def open_ledger(directory: str | Path = DEFAULT_DIR) -> Ledger | None:
    """Open a persisted run, or None if nothing has been generated yet.

    Returning None rather than raising: a server started before any run exists
    should say so on the page, not refuse to boot.
    """
    path = ledger_path(directory)
    return Ledger(path) if path.exists() else None


def clear(directory: str | Path = DEFAULT_DIR) -> None:
    for name in (LEDGER_NAME, SUMMARY_NAME):
        (Path(directory) / name).unlink(missing_ok=True)
