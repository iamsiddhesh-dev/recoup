"""Simulated time.

The world runs on an event queue, not on wall-clock time. That buys three things
the project actually needs:

* **A thirty-day horizon replays in seconds.** Recovery is a game played over days
  — retries scheduled for tomorrow morning, nudges deferred past quiet hours,
  payday waits. None of that is demonstrable against a real clock.
* **The control room gets a scrubber.** Because every state change is a queued
  event with a timestamp, the UI can seek to any point in the run and replay it.
* **Runs are reproducible.** Ties are broken by insertion order rather than by
  whatever the heap happens to do, so identical seeds produce identical
  timelines down to event ordering.

Nothing here knows about payments. It is a scheduler.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass(order=True)
class _Scheduled:
    at: datetime
    seq: int
    payload: Any = field(compare=False)


class Timeline:
    """An event queue with a notion of now.

    `now` only ever moves forward, and only when an event is popped — so the
    world cannot accidentally observe a future it has not simulated yet.
    """

    def __init__(self, start: datetime) -> None:
        self._now = start
        self._start = start
        self._queue: list[_Scheduled] = []
        self._seq = 0

    @property
    def now(self) -> datetime:
        return self._now

    @property
    def start(self) -> datetime:
        return self._start

    def __len__(self) -> int:
        return len(self._queue)

    def schedule(self, at: datetime, payload: Any) -> None:
        """Queue something for a point in time.

        Scheduling into the past is a bug — it means something computed a delay
        against a stale clock — so it raises rather than silently firing now.
        """
        if at < self._now:
            raise ValueError(
                f"cannot schedule into the past: {at.isoformat()} < now {self._now.isoformat()}"
            )
        self._seq += 1
        heapq.heappush(self._queue, _Scheduled(at=at, seq=self._seq, payload=payload))

    def schedule_after(self, delay: timedelta, payload: Any) -> None:
        self.schedule(self._now + delay, payload)

    def peek(self) -> datetime | None:
        return self._queue[0].at if self._queue else None

    def run(self, until: datetime | None = None) -> Iterator[tuple[datetime, Any]]:
        """Drain the queue in time order, advancing `now` as it goes.

        Handlers may schedule further events while iterating; they are picked up
        in the right order. Yields (when, payload).
        """
        while self._queue:
            if until is not None and self._queue[0].at > until:
                self._now = until
                return
            item = heapq.heappop(self._queue)
            self._now = item.at
            yield item.at, item.payload

    def drain(self, until: datetime | None = None) -> list[tuple[datetime, Any]]:
        return list(self.run(until))
