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
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recoup.adapters.webhooks import SignatureError, parse
from recoup.agent.config import ComplianceConfig
from recoup.agent.llm.explainer import Explanation, summarise
from recoup.eval import run_all
from recoup.eval.sensitivity import load as load_sweep
from recoup.eval.store import (
    load_explanations,
    load_frames,
    load_summary,
    open_ledger,
)
from recoup.money import rupees
from recoup.web.aicalls import build_ai_calls
from recoup.web.jobs import JobRegistry
from recoup.web.sink import WebhookSink
from recoup.web.studio import KNOBS
from recoup.web.studio import apply as studio_apply
from recoup.web.studio import clean as studio_clean
from recoup.web.studio import defaults as studio_defaults
from recoup.web.timeline import build_frames, to_payload
from recoup.web.views import (
    build_audit,
    build_case,
    build_experiment,
    case_facts,
    neighbours,
    queue_facets,
    search_queue,
)

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
        "built": True,
    },
    {"key": "ai", "label": "AI Calls", "href": "/ai", "icon": "◇", "built": True},
]

# Rows per queue page. Large enough that scrolling is the normal way to read it,
# small enough that the page stays under a hundred kilobytes.
QUEUE_PAGE = 100

BASELINE = "naive_baseline"
AGENT = "recoup_agent"
CONTACT = "contact_only"


def _rupees(paise) -> str:
    """Money is displayed in rupees and stored in paise, always.

    Amounts under ₹100 keep two decimals: a WhatsApp message costs 35 paise, and
    rounding it to "₹0" makes the expected-value sum look like it does not add
    up — the one thing that screen exists to demonstrate.
    """
    return rupees(paise, precise_below=10_000)


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _pct_signed(value) -> str:
    return "—" if value is None else f"{value:+.1%}"


def _int_comma(value) -> str:
    return "—" if value is None else f"{int(value):,}"


def _when(value) -> str:
    """An ISO timestamp as a date a person reads.

    Shown beside the seed because reproducibility has two halves: which inputs,
    and when they were last run. A stale run that still says "verified" is the
    thing this pairing prevents.
    """
    if not value:
        return "—"
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return "—"
    # Built rather than formatted: `%-d` is POSIX-only and raises on Windows,
    # which is where this is developed.
    return f"{when.day} {when:%b}"


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
    templates.env.filters["when"] = _when

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

            # Precomputed by `recoup demo`. Rebuilt here only if that file is
            # missing, so an older run still scrubs rather than showing nothing.
            frames = load_frames(data_dir)
            if frames is None:
                ledger = open_ledger(data_dir)
                if ledger is not None:
                    try:
                        frames = to_payload(build_frames(ledger, AGENT))
                    finally:
                        ledger.close()

            context["frames"] = frames

        return templates.TemplateResponse(request, "control_room.html", context)

    @app.get("/queue")
    async def queue(
        request: Request,
        arm: str = AGENT,
        outcome: str | None = None,
        cause: str | None = None,
        q: str | None = None,
        offset: int = 0,
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
            "q": q or "",
            "page": None,
            "arms": [],
            "facets": {"causes": [], "outcomes": []},
        }

        if ledger is not None:
            try:
                context |= {
                    "page": search_queue(
                        ledger,
                        arm,
                        outcome or None,
                        cause or None,
                        query=q,
                        offset=max(0, offset),
                        limit=QUEUE_PAGE,
                    ),
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

        # One open, both reads. Opening twice leaked a handle per request.
        try:
            case = build_case(ledger, payment_id, arm)
            previous_id, next_id = (
                neighbours(ledger, arm, payment_id) if case else (None, None)
            )
        finally:
            ledger.close()

        if case is None:
            raise HTTPException(404, f"No events for {payment_id} in arm {arm}")

        # A generated narrative if this case was in the explained selection,
        # otherwise one composed from the same facts. Never a model call on a page
        # load: a read-only screen must not spend quota or vary between visits.
        stored = load_explanations(data_dir).get(payment_id)
        explanation = (
            Explanation(text=stored["text"], source=stored["source"])
            if stored
            else Explanation(text=summarise(case_facts(case)), source="deterministic")
        )

        return templates.TemplateResponse(
            request,
            "case.html",
            {
                "request": request,
                "nav": NAV,
                "active": "queue",
                "summary": load_summary(data_dir),
                "case": case,
                "explanation": explanation,
                "previous_id": previous_id,
                "next_id": next_id,
            },
        )

    @app.get("/ai")
    async def ai_calls(request: Request):
        """Every model call in the run, re-checked on load.

        The accept and reject counts are recomputed here rather than read from a
        record of the run, so this page shows whether the validators still hold
        rather than whether they once did.
        """
        ledger = open_ledger(data_dir)

        facts_by_id: dict = {}
        if ledger is not None:
            try:
                for payment_id in load_explanations(data_dir):
                    case = build_case(ledger, payment_id, AGENT)
                    if case is not None:
                        facts_by_id[payment_id] = case_facts(case)
            finally:
                ledger.close()

        return templates.TemplateResponse(
            request,
            "ai.html",
            {
                "request": request,
                "nav": NAV,
                "active": "ai",
                "summary": load_summary(data_dir),
                "view": build_ai_calls(facts_by_id=facts_by_id),
            },
        )

    @app.get("/experiment")
    async def experiment(request: Request):
        ledger = open_ledger(data_dir)

        view = None
        if ledger is not None:
            try:
                view = build_experiment(ledger)
            finally:
                ledger.close()

        sweep = load_sweep()
        bands, base_pct, inverted = [], 50.0, []

        if sweep is not None:
            bands = sweep.bands()

            # Position every bar on one shared scale, so the bars are comparable
            # to each other rather than each filling its own row.
            values = [b.low for b in bands] + [b.high for b in bands] + [sweep.base.judgment]
            low, high = min(values), max(values)
            span = (high - low) or 1

            for band in bands:
                left = min(band.low, band.high)
                band.left_pct = round((left - low) / span * 100, 2)
                band.width_pct = round(band.span / span * 100, 2)

            base_pct = round((sweep.base.judgment - low) / span * 100, 2)

            # Axes where a higher assumption produces a *smaller* gap. Called out
            # because it reads as an error and is in fact the interesting part.
            inverted = [b for b in bands if b.high < b.low and b.span > 0]

        return templates.TemplateResponse(
            request,
            "experiment.html",
            {
                "request": request,
                "nav": NAV,
                "active": "experiment",
                "summary": load_summary(data_dir),
                "experiment": view,
                "sweep": sweep,
                "bands": bands,
                "base_pct": base_pct,
                "inverted": inverted,
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
