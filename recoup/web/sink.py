"""Where received webhooks land.

Deliberately dumb: append to a JSONL file, keep the last few in memory for the
dashboard. Durability matters more than structure at this stage — the point of
this file is that when a live webhook arrives at 11pm through a tunnel that is
about to expire, the evidence survives the process exiting.

Razorpay retries webhooks it does not get a 2xx for, and can deliver the same
event more than once, so events are deduplicated on `x-razorpay-event-id`. An
agent that treats a duplicate delivery as a second failure would double-count the
money at risk.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from recoup.domain import WebhookEvent

DEFAULT_PATH = Path("data/webhooks.jsonl")


class WebhookSink:
    """Append-only record of everything Razorpay sent us."""

    def __init__(self, path: Path | str = DEFAULT_PATH, keep: int = 50) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recent: deque[dict] = deque(maxlen=keep)
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def record(self, event: WebhookEvent, event_id: str | None = None) -> bool:
        """Store an event. Returns False if it was a duplicate delivery."""
        with self._lock:
            if event_id and event_id in self._seen:
                return False
            if event_id:
                self._seen.add(event_id)

            entry = {
                "received_at": datetime.now(UTC).isoformat(),
                "event_id": event_id,
                "event": event.event,
                "payload": event.model_dump(mode="json"),
            }
            self._recent.appendleft(entry)

            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

        return True

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._recent)[:limit]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle)
