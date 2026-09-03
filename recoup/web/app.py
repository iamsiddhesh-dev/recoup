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

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recoup.adapters.webhooks import SignatureError, parse
from recoup.agent.config import ComplianceConfig
from recoup.eval import run_all
from recoup.eval.store import load_summary, open_ledger
from recoup.web.jobs import JobRegistry
from recoup.web.sink import WebhookSink
from recoup.web.studio import KNOBS
from recoup.web.studio import apply as studio_apply
from recoup.web.studio import clean as studio_clean
from recoup.web.studio import defaults as studio_defaults
from recoup.web.views import build_audit, build_case, build_queue, queue_facets

EVENT_ID_HEADER = "x-razorpay-event-id"

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

# `built` marks what exists. A link that 404s is worse than a link that says it
# is not ready — the first looks broken, the second looks in progress.
NAV = [
    {"key": "control", "label": "Control Room", "href": "/", "icon": "◉", "built": True},
    {"key": "queue", "label": "Recovery Queue", "href": "/queue", "icon": "≡", "built": True},
    {"key": "studio", "label": "Policy Studio", "href": "/studio", "icon": "⚙", "built": True},
    {"key": "audit", "label": "Audit & Refusals", "href": "/audit", "icon": "⊘", "built": True},
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

    Small amounts keep two decimals. A WhatsApp message costs 35 paise, and
    rounding it to "₹0" makes the expected-value sum look like it does not add
    up — the one thing that screen exists to demonstrate. Large amounts drop the
    decimals, because ₹12,041.00 is noise in a column of them.
    """
    if paise is None:
        return "—"
    if 0 < abs(paise) < 10_000:
        return f"₹{paise / 100:,.2f}"
    return f"₹{paise / 100:,.0f}"


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _pct_signed(value) -> str:
    return "—" if value is None else f"{value:+.1%}"


def _int_comma(value) -> str:
    return "—" if value is None else f"{int(value):,}"


def create_app(
    sink: WebhookSink | None = None,
    data_dir: str | Path = "data",
    evaluate=run_all,
) -> FastAPI:
    """Build the app.

    `evaluate` is injectable so tests can exercise the studio's job plumbing —
    progress, completion, failure handling — without running a real twelve-second
    evaluation for each one. The seam is small and it is the difference between
    those paths being tested and being hoped about.
    """
    resolved = sink or WebhookSink()
    jobs = JobRegistry(Path(data_dir) / "studio")

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
    templates.env.filters["int_comma"] = _int_comma

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

    @app.get("/queue")
    async def queue(
        request: Request,
        arm: str = AGENT,
        outcome: str | None = None,
        cause: str | None = None,
    ):
        summary = load_summary(data_dir)
        ledger = open_ledger(data_dir)

        context: dict = {
            "request": request,
            "nav": NAV,
            "active": "queue",
            "summary": summary,
            "arm": arm,
            "outcome": outcome,
            "cause": cause,
            "rows": [],
            "arms": [],
            "facets": {"causes": [], "outcomes": []},
        }

        if ledger is not None:
            try:
                context |= {
                    "rows": build_queue(ledger, arm, outcome or None, cause or None),
                    "arms": ledger.arms(),
                    "facets": queue_facets(ledger, arm),
                }
            finally:
                ledger.close()

        return templates.TemplateResponse(request, "queue.html", context)

    @app.get("/case/{payment_id}")
    async def case_detail(request: Request, payment_id: str, arm: str = AGENT):
        ledger = open_ledger(data_dir)
        if ledger is None:
            raise HTTPException(404, "No run has been generated yet")

        try:
            case = build_case(ledger, payment_id, arm)
        finally:
            ledger.close()

        if case is None:
            raise HTTPException(404, f"No events for {payment_id} in arm {arm}")

        return templates.TemplateResponse(
            request,
            "case.html",
            {
                "request": request,
                "nav": NAV,
                "active": "queue",
                "summary": load_summary(data_dir),
                "case": case,
            },
        )

    @app.get("/audit")
    async def audit(request: Request, arm: str = AGENT):
        summary = load_summary(data_dir)
        ledger = open_ledger(data_dir)

        view = None
        if ledger is not None:
            try:
                view = build_audit(
                    ledger,
                    arm,
                    (summary.digests if summary else {}).get(arm),
                    configured_hard_stops=list(ComplianceConfig.load().hard_stops),
                )
            finally:
                ledger.close()

        return templates.TemplateResponse(
            request,
            "audit.html",
            {
                "request": request,
                "nav": NAV,
                "active": "audit",
                "summary": summary,
                "audit": view,
            },
        )

    # ------------------------------------------------------------------ studio

    @app.get("/studio")
    async def studio_page(request: Request):
        summary = load_summary(data_dir)
        job = jobs.latest()

        return templates.TemplateResponse(
            request,
            "studio.html",
            {
                "request": request,
                "nav": NAV,
                "active": "studio",
                "summary": summary,
                "knobs": KNOBS,
                "values": (job.overrides if job else None) or studio_defaults(),
                "job": job.to_dict() if job else None,
                "baseline_arms": summary.arms if summary else [],
            },
        )

    @app.post("/studio/run")
    async def studio_run(request: Request) -> JSONResponse:
        """Start a real evaluation with the submitted configuration.

        A full run, not an approximation. Studio's whole value is that the number
        it produces is the same kind of number the control room shows — a
        cheaper estimate would be a different quantity wearing the same label.
        """
        # JSON rather than a form: multipart parsing needs an extra dependency
        # for a payload that is seven numbers, and the values arrive typed.
        overrides = studio_clean(await request.json())

        def work(job) -> None:
            policy, compliance = studio_apply(overrides)

            def progress(stage: str, fraction: float) -> None:
                job.stage = stage
                job.progress = fraction

            results, ledger = evaluate(
                ledger_path=job.ledger_path,
                policy=policy,
                compliance=compliance,
                on_progress=progress,
            )
            ledger.close()
            job.results = [
                {
                    "arm": m.arm,
                    "recovered_paise": m.recovered_paise,
                    "cost_paise": m.cost_paise,
                    "net_paise": m.net_paise,
                    "contacts": m.contacts,
                    "vetoes": m.vetoes,
                    "recovered_count": m.recovered_count,
                }
                for m in results
            ]

        job = jobs.submit(overrides, work)
        return JSONResponse({"job": job.id})

    @app.get("/studio/status/{job_id}")
    async def studio_status(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No such run")
        return JSONResponse(job.to_dict())

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
