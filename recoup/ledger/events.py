"""The audit trail.

Every observation, classification, decision, veto, execution and outcome lands
here in order. It is the answer to "why did you charge this customer at 4am on a
Tuesday", and it is what the control room reads to replay a run.

Two properties are enforced rather than promised.

**Append-only, in the database.** SQLite triggers abort any UPDATE or DELETE. A
ledger that is append-only by convention stops being append-only the first time
someone is debugging at 1am and finds it convenient. Making the storage engine
refuse costs four lines and removes the question.

**Digestible.** `digest()` hashes the whole ordered stream, so two runs of the
same seed can be compared in one comparison rather than row by row. `reproduce`
depends on this: a run is reproducible when its digest matches, and when it does
not, `first_divergence` says which event moved.

Vetoes are recorded as their own event kind, not as an absence of a decision.
That is what makes the refusal list possible — "we deliberately did not touch
these 340 cases, and here is the rule that stopped each one" is a deliverable,
and it only exists if refusals are written down as positively as actions.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    arm         TEXT    NOT NULL,
    payment_id  TEXT,
    customer_id TEXT,
    amount      INTEGER,
    data        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ledger_payment ON ledger (arm, payment_id, seq);
CREATE INDEX IF NOT EXISTS ledger_kind    ON ledger (arm, kind, seq);

-- Append-only, enforced by the storage engine rather than by good intentions.
CREATE TRIGGER IF NOT EXISTS ledger_no_update
BEFORE UPDATE ON ledger
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;

CREATE TRIGGER IF NOT EXISTS ledger_no_delete
BEFORE DELETE ON ledger
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""


class EventKind(StrEnum):
    OBSERVED = "observed"          # a failure arrived
    CLASSIFIED = "classified"      # a cause was assigned, or could not be
    DECIDED = "decided"            # the policy chose an action, with its arithmetic
    VETOED = "vetoed"              # compliance refused it, with the rule that fired
    EXECUTED = "executed"          # the action ran, with the adapter's answer
    RECOVERED = "recovered"        # money came back
    STOPPED = "stopped"            # gave up, with a reason


@dataclass(frozen=True)
class LedgerEvent:
    at: datetime
    kind: EventKind
    arm: str
    payment_id: str | None = None
    customer_id: str | None = None
    amount: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    seq: int | None = None

    def canonical(self) -> str:
        """Stable serialisation for hashing.

        `sort_keys` because dict ordering must not affect the digest; `seq` is
        excluded because it is assigned by the database and says nothing about
        content.
        """
        return json.dumps(
            {
                "at": self.at.isoformat(),
                "kind": str(self.kind),
                "arm": self.arm,
                "payment_id": self.payment_id,
                "customer_id": self.customer_id,
                "amount": self.amount,
                "data": self.data,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


class Ledger:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- writing -------------------------------------------------------------

    def append(self, event: LedgerEvent) -> int:
        cursor = self._conn.execute(
            "INSERT INTO ledger (at, kind, arm, payment_id, customer_id, amount, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.at.isoformat(),
                str(event.kind),
                event.arm,
                event.payment_id,
                event.customer_id,
                event.amount,
                json.dumps(event.data, sort_keys=True, default=str),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def extend(self, events: list[LedgerEvent]) -> None:
        """Write a batch in one transaction.

        A per-event commit costs a disk sync each; a run writes hundreds of
        thousands of events across arms.
        """
        self._conn.executemany(
            "INSERT INTO ledger (at, kind, arm, payment_id, customer_id, amount, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    e.at.isoformat(),
                    str(e.kind),
                    e.arm,
                    e.payment_id,
                    e.customer_id,
                    e.amount,
                    json.dumps(e.data, sort_keys=True, default=str),
                )
                for e in events
            ],
        )
        self._conn.commit()

    # -- reading -------------------------------------------------------------

    def events(
        self,
        arm: str | None = None,
        payment_id: str | None = None,
        kind: EventKind | None = None,
        limit: int | None = None,
    ) -> Iterator[LedgerEvent]:
        clauses, params = [], []
        if arm is not None:
            clauses.append("arm = ?")
            params.append(arm)
        if payment_id is not None:
            clauses.append("payment_id = ?")
            params.append(payment_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(str(kind))

        sql = "SELECT * FROM ledger"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        for row in self._conn.execute(sql, params):
            yield _row_to_event(row)

    def story_of(self, payment_id: str, arm: str) -> list[LedgerEvent]:
        """Everything that happened to one payment, in order.

        This is what the case-detail screen renders, and what a judge reads when
        asking why a particular decision was made.
        """
        return list(self.events(arm=arm, payment_id=payment_id))

    def count(self, arm: str | None = None, kind: EventKind | None = None) -> int:
        return sum(1 for _ in self.events(arm=arm, kind=kind))

    def arms(self) -> list[str]:
        return [row["arm"] for row in self._conn.execute(
            "SELECT DISTINCT arm FROM ledger ORDER BY arm"
        )]

    def counts_by_kind(self, arm: str | None = None) -> dict[str, int]:
        sql = "SELECT kind, COUNT(*) n FROM ledger"
        params: list[str] = []
        if arm is not None:
            sql += " WHERE arm = ?"
            params.append(arm)
        sql += " GROUP BY kind"
        return {row["kind"]: row["n"] for row in self._conn.execute(sql, params)}

    def append_only_triggers(self) -> list[str]:
        """The triggers that make this an audit trail rather than a log.

        Read back from the schema rather than asserted, so the audit screen can
        show that the guarantee is enforced by the storage engine and not merely
        claimed in a docstring.
        """
        return [
            row["name"]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'ledger_no_%' ORDER BY name"
            )
        ]

    # -- reproducibility -----------------------------------------------------

    def digest(self, arm: str | None = None) -> str:
        """A single hash over the ordered event stream."""
        hasher = hashlib.sha256()
        for event in self.events(arm=arm):
            hasher.update(event.canonical().encode())
        return hasher.hexdigest()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _row_to_event(row: sqlite3.Row) -> LedgerEvent:
    return LedgerEvent(
        seq=row["seq"],
        at=datetime.fromisoformat(row["at"]),
        kind=EventKind(row["kind"]),
        arm=row["arm"],
        payment_id=row["payment_id"],
        customer_id=row["customer_id"],
        amount=row["amount"],
        data=json.loads(row["data"]),
    )
