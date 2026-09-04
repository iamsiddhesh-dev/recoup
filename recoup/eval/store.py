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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from recoup.eval.metrics import ArmMetrics
from recoup.ledger.events import Ledger

DEFAULT_DIR = Path("data")
LEDGER_NAME = "run.db"
SUMMARY_NAME = "run.json"
EXPLANATIONS_NAME = "explanations.json"
FRAMES_NAME = "frames.json"


@dataclass
class RunSummary:
    """Everything the control room needs before touching the ledger."""

    seed: int
    generated_at: str
    horizon_days: int
    batch_size: int
    margin: float
    arms: list[dict]

    # Per-arm hash of the event stream, taken when the run was written. Verifying
    # against it later is what turns "this is an audit trail" from a claim in a
    # README into a property the audit screen can check in front of you.
    digests: dict[str, str] = field(default_factory=dict)

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
    digests: dict[str, str] | None = None,
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
        digests=digests or {},
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


def save_explanations(explanations: dict[str, dict], directory: str | Path = DEFAULT_DIR) -> Path:
    """Case narratives, written beside the run they describe.

    Kept out of `run.json` because they belong to a different lifecycle: the
    summary is arithmetic over the ledger and always present, while these depend
    on a model and may legitimately be absent. A screen that needs the first
    should not have to parse the second.
    """
    path = Path(directory) / EXPLANATIONS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(explanations, indent=2) + "\n", encoding="utf-8")
    return path


def load_explanations(directory: str | Path = DEFAULT_DIR) -> dict[str, dict]:
    """Empty rather than None: no explanations is a normal state, not an error."""
    path = Path(directory) / EXPLANATIONS_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_frames(payload: dict, directory: str | Path = DEFAULT_DIR) -> Path:
    """The scrubber's frames, precomputed with the run they describe.

    Folding the event stream into frames takes about 400ms. That is nothing once,
    and far too much on every page load of the busiest screen — so it happens here,
    with the rest of the run, and the control room reads the answer.
    """
    path = Path(directory) / FRAMES_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def load_frames(directory: str | Path = DEFAULT_DIR) -> dict | None:
    """None rather than empty: the caller can rebuild from the ledger."""
    path = Path(directory) / FRAMES_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def clear(directory: str | Path = DEFAULT_DIR) -> None:
    for name in (LEDGER_NAME, SUMMARY_NAME, EXPLANATIONS_NAME, FRAMES_NAME):
        (Path(directory) / name).unlink(missing_ok=True)
