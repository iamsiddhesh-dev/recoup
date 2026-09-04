"""Thirty days of a run, as frames a slider can move through.

The ledger holds every intermediate state rather than a final summary, so "what
did this look like on day nine?" is answerable. `replay.state_at` answers it for
one moment by folding the stream up to a timestamp.

Dragging a slider does not want one moment. It wants two hundred of them, in
order, faster than a person can move their hand — and `state_at` is O(events) per
call, so asking it per drag position would re-scan the whole run each time.

So the whole sequence is computed **once**, in a single pass, and handed to the
page as data. Scrubbing is then an array lookup with no network in it: no request
per frame, no loading state, no way for the slider to feel laggy. The run is
frozen and deterministic, which is what makes precomputing it safe.

That is the same argument as `eval/store.py` — compute once, read many — applied
at a smaller scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from recoup.ledger.events import EventKind, Ledger

# Enough that dragging feels continuous, few enough that the payload stays small.
# 200 frames over a 30-day horizon is roughly one every 3.6 hours, and the whole
# series serialises to about 20KB.
FRAMES = 200


@dataclass
class Frame:
    """The run as it stood at one moment."""

    at: str
    day: float

    observed: int = 0
    resolved: int = 0
    decisions: int = 0
    vetoes: int = 0
    contacts: int = 0
    retries: int = 0
    recovered_count: int = 0
    recovered_paise: int = 0
    at_risk_paise: int = 0
    cost_paise: int = 0
    open_count: int = 0
    escalated: int = 0


def build_frames(ledger: Ledger, arm: str, count: int = FRAMES) -> list[Frame]:
    """Every frame of the run, in one pass over the event stream.

    Events are already ordered by `seq`, and `seq` is monotonic in simulated time,
    so this walks the stream once and closes a frame each time the clock crosses
    the next boundary. A frame carries cumulative totals, which is what makes the
    slider read as accumulation rather than as a series of unrelated snapshots.
    """
    events = list(ledger.events(arm=arm))
    if not events:
        return []

    start, end = events[0].at, events[-1].at
    span = (end - start) or timedelta(seconds=1)
    step = span / count

    frames: list[Frame] = []
    running = Frame(at=start.isoformat(), day=0.0)
    open_payments: set[str] = set()
    boundary = start + step

    def snapshot(at: datetime) -> Frame:
        running.at = at.isoformat()
        running.day = round((at - start).total_seconds() / 86400, 3)
        running.open_count = len(open_payments)
        return Frame(**asdict(running))

    for event in events:
        # Close every boundary the clock has passed. A quiet stretch still
        # produces frames, so the slider moves at a constant rate through
        # simulated time rather than through event volume.
        while event.at > boundary and len(frames) < count:
            frames.append(snapshot(boundary))
            boundary += step

        match event.kind:
            case EventKind.OBSERVED:
                running.observed += 1
                running.at_risk_paise += event.amount or 0
                if event.payment_id:
                    open_payments.add(event.payment_id)

            case EventKind.CLASSIFIED:
                if event.data.get("cause"):
                    running.resolved += 1

            case EventKind.DECIDED:
                running.decisions += 1

            case EventKind.VETOED:
                running.vetoes += 1

            case EventKind.EXECUTED:
                action = event.data.get("action", "")
                running.cost_paise += event.data.get("cost", 0)
                if event.data.get("delivered"):
                    running.contacts += 1
                if action.startswith("RETRY"):
                    running.retries += 1
                elif action == "ESCALATE_HUMAN":
                    running.escalated += 1

            case EventKind.RECOVERED:
                running.recovered_count += 1
                running.recovered_paise += event.amount or 0
                open_payments.discard(event.payment_id or "")

            case EventKind.STOPPED:
                open_payments.discard(event.payment_id or "")

    frames.append(snapshot(end))
    return frames


# Order matters: the payload sends these once as a header and every frame as a
# bare list of values in this order. Sending 200 copies of thirteen JSON key names
# costs about 45KB for no information, and this page embeds its data rather than
# fetching it — so that weight lands on first paint.
COLUMNS = (
    "day",
    "observed",
    "resolved",
    "decisions",
    "vetoes",
    "contacts",
    "retries",
    "recovered_count",
    "recovered_paise",
    "at_risk_paise",
    "cost_paise",
    "open_count",
    "escalated",
)


def to_payload(frames: list[Frame]) -> dict:
    """Frames as columns and rows, for embedding in a page."""
    if not frames:
        return {"start": "", "end": "", "columns": list(COLUMNS), "rows": []}

    return {
        "start": frames[0].at,
        "end": frames[-1].at,
        "columns": list(COLUMNS),
        "rows": [[getattr(f, c) for c in COLUMNS] for f in frames],
    }
