"""The web surface: the control room, and the webhook receiver.

Screens read a **completed run** rather than computing one. A full evaluation
takes ~35 seconds and a page load cannot, so `recoup demo` writes the run to
`data/run.db` and every screen is a query against it. See `eval/store.py` for why
that is the right shape for this particular data rather than merely the fast one.

Three things about webhook handling that are easy to get wrong:

* **Verify before parsing.** The raw bytes are the signed material. Anything that
  round-trips through a JSON decode first is verifying a different string than the
  one Razorpay signed.
* **Return 2xx fast.** Razorpay retries anything else, so slow or failing
  processing turns into duplicate deliveries and a growing backlog.
* **A bad signature is a 400, not a 500.** It means someone POSTed to a public URL
  without the secret, which is expected background noise on the internet.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recoup.adapters.webhooks import SignatureError, parse
from recoup.eval.store import load_summary
from recoup.web.sink import WebhookSink

EVENT_ID_HEADER = "x-razorpay-event-id"

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

# `built` marks what exists. A link that 404s is worse than a link that says it
# is not ready — the first looks broken, the second looks in progress.
NAV = [
    {"key": "control", "label": "Control Room", "href": "/", "icon": "◉", "built": True},
    {"key": "queue", "label": "Recovery Queue", "href": "/queue", "icon": "≡", "built": False},
    {"key": "case", "label": "Case Detail", "href": "/case", "icon": "⊙", "built": False},
    {"key": "studio", "label": "Policy Studio", "href": "/studio", "icon": "⚙", "built": False},
    {"key": "audit", "label": "Audit & Refusals", "href": "/audit", "icon": "⊘", "built": False},
    {
        "key": "experiment",
        "label": "Experiment",
        "href": "/experiment",
        "icon": "⇄",
        "built": False,
    },
]

BASELINE = "naive_baseline"
AGENT = "recoup_agent"
CONTACT = "contact_only"


def _rupees(paise) -> str:
    """Money is displayed in rupees and stored in paise, always.

    Formatting lives here rather than in templates so a figure shown to a
    merchant is rendered one way everywhere.
    """
    if paise is None:
        return "—"
    return f"₹{paise / 100:,.0f}"


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _pct_signed(value) -> str:
    return "—" if value is None else f"{value:+.1%}"


def create_app(sink: WebhookSink | None = None, data_dir: str | Path = "data") -> FastAPI:
    resolved = sink or WebhookSink()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.sink = resolved
        yield

    app = FastAPI(
        title="Recoup",
        description="Revenue recovery agent for Indian payments",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters["rupees"] = _rupees
    templates.env.filters["pct"] = _pct
    templates.env.filters["pct_signed"] = _pct_signed

    # ------------------------------------------------------------------ screens

    @app.get("/")
    async def control_room(request: Request):
        summary = load_summary(data_dir)

        context: dict = {
            "request": request,
            "nav": NAV,
            "active": "control",
            "summary": summary,
        }

        if summary is not None:
            agent = summary.arm(AGENT) or summary.arms[-1]
            baseline = summary.arm(BASELINE) or summary.arms[0]
            contact = summary.arm(CONTACT) or baseline

            incremental = agent["recovered_paise"] - baseline["recovered_paise"]
            lift = (
                incremental / baseline["recovered_paise"]
                if baseline["recovered_paise"]
                else 0.0
            )

            by_action = sorted(
                agent["recovered_by_action"].items(), key=lambda kv: -kv[1]
            )
            vetoes = sorted(agent["veto_by_rule"].items(), key=lambda kv: -kv[1])[:6]

            context |= {
                "arms": summary.arms,
                "agent": agent,
                "baseline": baseline,
                "contact": contact,
                "incremental": incremental,
                "lift": lift,
                "recovered_by_action": by_action,
                "max_recovered_action": max((c for _, c in by_action), default=1),
                "top_vetoes": vetoes,
                "max_veto": max((c for _, c in vetoes), default=1),
            }

        return templates.TemplateResponse(request, "control_room.html", context)

    # ------------------------------------------------------------------ health

    @app.get("/healthz")
    async def healthz() -> dict:
        summary = load_summary(data_dir)
        return {
            "status": "ok",
            "webhooks_received": resolved.count(),
            "run_loaded": summary is not None,
            "seed": summary.seed if summary else None,
        }

    # ---------------------------------------------------------------- webhooks

    @app.post("/webhooks/razorpay")
    async def receive_webhook(
        request: Request,
        x_razorpay_signature: str | None = Header(default=None),
        x_razorpay_event_id: str | None = Header(default=None),
    ) -> JSONResponse:
        raw = await request.body()

        try:
            event = parse(raw, x_razorpay_signature)
        except SignatureError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        stored = resolved.record(event, x_razorpay_event_id)

        print(
            f"[webhook] {event.event}"
            f"{'' if stored else '  (duplicate delivery, ignored)'}"
        )

        return JSONResponse(status_code=200, content={"status": "ok", "stored": stored})

    @app.get("/webhooks/recent")
    async def recent_webhooks(limit: int = 20) -> dict:
        return {"count": resolved.count(), "events": resolved.recent(limit)}

    return app


app = create_app()


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    print(f"Recoup listening on http://{host}:{port}")
    if load_summary("data") is None:
        print("  no run found — generate one with `python -m recoup demo`")
    if not os.environ.get("RAZORPAY_WEBHOOK_SECRET"):
        print("  note: RAZORPAY_WEBHOOK_SECRET unset — webhooks will be refused")

    uvicorn.run(app, host=host, port=port, log_level="warning")
