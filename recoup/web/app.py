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
from recoup.web.aicalls import DEFAULT_CACHE as LLM_CACHE
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
    queue_totals,
    search_queue,
)

EVENT_ID_HEADER = "x-razorpay-event-id"

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

# Drawn rather than typed. The rail used to carry Unicode glyphs — ◉ ≡ ⚙ ⇄ ⊘ ◇ —
# which come from four different Unicode blocks and therefore resolve to four
# different fallback fonts. At one font-size ⚙ rendered half again as large as
# ≡, so no single size made the set look like a set. These are one 16×16 box and
# one stroke weight each, which is the only way "the same size" is actually true.
ICONS = {
    "control": (
        '<circle cx="8" cy="8" r="6"/>'
        '<circle cx="8" cy="8" r="2.2" fill="currentColor" stroke="none"/>'
    ),
    "queue": '<path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h7"/>',
    "studio": (
        '<path d="M3 4.5h10M3 11.5h10"/>'
        '<circle cx="6" cy="4.5" r="1.8" fill="var(--surface)"/>'
        '<circle cx="10.5" cy="11.5" r="1.8" fill="var(--surface)"/>'
    ),
    "experiment": '<path d="M2.5 5.5h9l-2.2-2.2M13.5 10.5h-9l2.2 2.2"/>',
    "audit": '<circle cx="8" cy="8" r="6"/><path d="M4 12 12 4"/>',
    "ai": '<path d="M8 2.2 13.8 8 8 13.8 2.2 8Z"/>',
}

# `built` marks what exists. A link that 404s is worse than a link that says it
# is not ready — the first looks broken, the second looks in progress.
#
# `group` splits the rail in two, because the screens do two different jobs and
# a flat list of six hides that. The first three are the product; the last three
# are the argument that the product's number is real. A judge who only has two
# minutes should be able to see which half to open.
NAV = [
    {"key": "control", "label": "Control Room", "href": "/control",
     "group": "Operate", "built": True},
    {"key": "queue", "label": "Recovery Queue", "href": "/queue",
     "group": "Operate", "built": True},
    {"key": "studio", "label": "Policy Studio", "href": "/studio",
     "group": "Operate", "built": True},
    {"key": "experiment", "label": "Experiment", "href": "/experiment",
     "group": "Prove it", "built": True},
    {"key": "audit", "label": "Audit & Refusals", "href": "/audit",
     "group": "Prove it", "built": True},
    {"key": "ai", "label": "AI Calls", "href": "/ai",
     "group": "Prove it", "built": True},
]

# The one-line answer to "what is this screen for?", under the title. Written as
# what the screen lets you conclude rather than what it contains — "Every model
# call in the run, re-checked" beats "AI Calls".
PURPOSE = {
    "control": "What the run earned, and how much of it counts as judgment",
    "queue": "Every failure the agent saw, largest first",
    "case": "One payment end to end, with the arithmetic shown",
    "studio": "Change what the agent believes, re-run, watch the number move",
    "experiment": "Four arms, one world — and what happens when the assumptions move",
    "audit": "The trail, verified in front of you, and what we refused to do",
    "ai": "Every model call in the run, re-validated on this page load",
}


# Rows per queue page. Large enough that scrolling is the normal way to read it,
# small enough that the page stays under a hundred kilobytes.
QUEUE_PAGE = 100

BASELINE = "naive_baseline"
AGENT = "recoup_agent"
CONTACT = "contact_only"
ABLATION = "recoup_agent_no_llm"


def _nav_for(summary) -> list[dict]:
    """The rail, with the counts that make each screen's size visible.

    A number beside a link is not decoration: "Recovery Queue 1,605" tells you
    the queue is worth opening, and "Audit & Refusals 903" is the whole argument
    for that screen existing, made before you click it.
    """
    counts: dict[str, str] = {}
    if summary is not None:
        counts["queue"] = f"{summary.observed:,}"
        counts["experiment"] = f"{len(summary.arms)} arms"
        agent = summary.arm(AGENT)
        if agent:
            counts["audit"] = f"{agent['vetoes']:,}"
    calls = len(list(LLM_CACHE.glob("*.json"))) if LLM_CACHE.is_dir() else 0
    if calls:
        counts["ai"] = str(calls)
    return [
        dict(item, count=counts.get(item["key"]), icon=ICONS.get(item["key"], ""))
        for item in NAV
    ]


# The arm switch, as it reads in the top bar. Shortened because the control is
# four items wide and the full names do not fit; the full name is on every row
# of every table underneath it.
ARM_LABELS = [
    (AGENT, "agent"),
    (ABLATION, "no‑llm"),
    (CONTACT, "contact"),
    (BASELINE, "naive"),
]


def _arm_options(path: str, summary) -> list[dict]:
    """The arm switch, or nothing at all.

    Only offered on screens that actually take an `arm` — a segmented control
    that changes nothing when pressed is worse than no control, and the
    experiment screen shows every arm at once so choosing between them there
    would mean nothing.
    """
    if summary is None:
        return []

    present = {a["arm"] for a in summary.arms}
    return [
        {"name": name, "label": label, "href": f"{path}?arm={name}"}
        for name, label in ARM_LABELS
        if name in present
    ]


def _shell(
    request: Request,
    active: str,
    summary,
    *,
    purpose: str | None = None,
    arm: str | None = None,
    arm_path: str | None = None,
) -> dict:
    """The context every screen shares: navigation, run identity, and a subtitle.

    Assembled in one place because it is easy to add a screen and forget one of
    the three, and the failure is silent — a rail with no counts on it still
    renders.
    """
    return {
        "request": request,
        "nav": _nav_for(summary),
        "active": active,
        "purpose": PURPOSE.get(purpose or active),
        "summary": summary,
        "arm": arm,
        "arms_available": _arm_options(arm_path, summary) if arm_path else [],
    }


def _asset_stamp() -> str:
    """A token that changes when the stylesheets do.

    Appended to every `/static` URL. Without it a browser holds the previous
    CSS after an edit and renders the new markup with the old rules, which looks
    exactly like a broken layout and wastes the time it takes to work out that
    nothing is actually wrong.

    Evaluated per render rather than once at startup. Computed at startup it is
    correct in production and wrong in development, because the reloader watches
    Python and not CSS — so every stylesheet edit served the old file under the
    old stamp, which is the exact failure this exists to prevent. Three `stat`
    calls per page is not a cost worth optimising.
    """
    newest = max(
        (path.stat().st_mtime for path in STATIC.glob("*.css")),
        default=0.0,
    )
    return f"{int(newest)}"


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

    # Available to every template, so the shell and the landing page cannot get
    # out of step with each other. Passed as the function, not its result.
    templates.env.globals["assets"] = _asset_stamp

    # ------------------------------------------------------------------ screens

    @app.get("/")
    async def landing(request: Request):
        """The argument, before the product.

        A judge opening this link has not agreed yet that the number means
        anything, and dropping them straight into a dashboard asks them to take
        the framing on trust. So the front door states the claim, decomposes it,
        names what it is not, and then offers the control room.

        Every figure on it is read from the committed run rather than typed into
        the copy. A landing page with hand-written numbers goes stale the first
        time the seed changes, and then the most public surface of the project is
        the one lying about it.
        """
        summary = load_summary(data_dir)
        ledger = open_ledger(data_dir)

        view = None
        headline_case = None
        if ledger is not None:
            try:
                view = build_experiment(ledger)

                # The largest failure in the run, so the front door can offer a
                # real case rather than a link to the list it lives in. Case
                # Detail is the screen that does the most work in a pitch, and it
                # was two clicks from here. The queue is ordered largest-first,
                # so the first row is both the biggest number and the story worth
                # the most.
                top = search_queue(ledger, AGENT, limit=1).rows
                if top:
                    headline_case = top[0]
            finally:
                ledger.close()

        # The three assumptions that move the answer most, scaled against the
        # widest of them. Three rather than all seven: this is the front door
        # making the point that a sweep exists, not the sweep itself.
        sweep = load_sweep()
        bands = []
        if sweep is not None:
            bands = sweep.bands()[:3]
            widest = max((b.span for b in bands), default=1) or 1
            for band in bands:
                band.width_pct = round(band.span / widest * 100, 1)

        # Money columns come from the run summary; the decomposition comes from
        # the experiment view. They are different objects because they answer
        # different questions — one is what each arm did, the other is what the
        # difference between them is attributable to.
        arms = summary.arms if summary else []
        by_name = {a["arm"]: a for a in arms}

        return templates.TemplateResponse(
            request,
            "landing.html",
            {
                "request": request,
                "summary": summary,
                "experiment": view,
                "arms": arms,
                "agent": by_name.get(AGENT),
                "headline_case": headline_case,
                "contact": by_name.get(CONTACT),
                "ablation": by_name.get(ABLATION),
                "sweep": sweep,
                "bands": bands,
            },
        )

    @app.get("/control")
    async def control_room(request: Request, arm: str = AGENT):
        summary = load_summary(data_dir)

        context: dict = _shell(
            request, "control", summary, arm=arm, arm_path="/control"
        )

        if summary is not None:
            # The hero follows the switch. The coverage/judgment split does not:
            # it is a property *of* the agent, measured against the other arms,
            # so it is shown only when the agent is the one on screen.
            agent = summary.arm(arm) or summary.arm(AGENT) or summary.arms[-1]
            baseline = summary.arm(BASELINE) or summary.arms[0]
            contact = summary.arm(CONTACT) or baseline

            incremental = agent["recovered_paise"] - baseline["recovered_paise"]
            lift = (
                incremental / baseline["recovered_paise"]
                if baseline["recovered_paise"]
                else 0.0
            )

            # The headline, taken apart, on the screen that states it.
            #
            # Coverage is what any arm that contacts customers gets for free —
            # three quarters of failures cannot be retried at all, so a baseline
            # that only retries cannot reach them. Judgment is what the policy
            # engine earns on top of that. Quoting the total as evidence of a
            # smart agent would be the most misleading thing this project could
            # do with an honest number, so the split is shown beside it rather
            # than left for the experiment screen.
            coverage = contact["recovered_paise"] - baseline["recovered_paise"]
            judgment = agent["recovered_paise"] - contact["recovered_paise"]
            share = (
                (coverage / incremental, judgment / incremental)
                if incremental
                else (0.0, 0.0)
            )

            by_action = sorted(
                agent["recovered_by_action"].items(), key=lambda kv: -kv[1]
            )
            vetoes = sorted(agent["veto_by_rule"].items(), key=lambda kv: -kv[1])[:6]

            # Same order as the experiment screen: what merchants do, then what
            # contacting adds, then judgment, then the ablation. Two tables of
            # the same four arms in two different orders is a small thing that
            # makes a reader distrust both.
            rank = {name: i for i, name in enumerate((BASELINE, CONTACT, AGENT, ABLATION))}

            context |= {
                "is_agent": agent["arm"] == AGENT,
                "AGENT_ARM": AGENT,
                "arms": sorted(
                    summary.arms, key=lambda a: (rank.get(a["arm"], 9), a["arm"])
                ),
                "agent": agent,
                "baseline": baseline,
                "contact": contact,
                "incremental": incremental,
                "lift": lift,
                "coverage": coverage,
                "judgment": judgment,
                "coverage_share": share[0],
                "judgment_share": share[1],
                "recovered_by_action": by_action,
                "max_recovered_action": max((c for _, c in by_action), default=1),
                "top_vetoes": vetoes,
                "max_veto": max((c for _, c in vetoes), default=1),
            }

            # Precomputed by `recoup demo` — for the agent, which is the arm
            # this screen shows by default. Any other arm has to be walked out of
            # the ledger here, and it is worth the scan: the alternative is a
            # curve plotting one arm underneath a scoreboard reporting another,
            # which is not a slower version of the truth, it is a false screen.
            frames = load_frames(data_dir) if agent["arm"] == AGENT else None
            if frames is None:
                ledger = open_ledger(data_dir)
                if ledger is not None:
                    try:
                        frames = to_payload(build_frames(ledger, agent["arm"]))
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

        context: dict = _shell(
            request, "queue", summary, arm=arm, arm_path="/queue"
        ) | {
            "arm": arm,
            "outcome": outcome,
            "cause": cause,
            "q": q or "",
            "page": None,
            "arms": [],
            "facets": {"causes": [], "outcomes": []},
            "totals": None,
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
                    "totals": queue_totals(ledger, arm),
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
                **_shell(
                    request,
                    "queue",
                    load_summary(data_dir),
                    purpose="case",
                    arm=arm,
                ),
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
                **_shell(request, "ai", load_summary(data_dir)),
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

        summary = load_summary(data_dir)

        # Cost, net and contacts live on the run summary; the recovery
        # decomposition lives on the experiment view. The arms table needs both,
        # so they are joined by arm name here rather than in the template.
        totals = {a["arm"]: a for a in (summary.arms if summary else [])}

        return templates.TemplateResponse(
            request,
            "experiment.html",
            {
                **_shell(request, "experiment", summary),
                "totals": totals,
                # How many failures the model actually resolved. Read off the
                # ablation arm, not the agent: `unresolved` counts what the rules
                # alone could not map, and on the agent that number is zero
                # precisely because the model already answered it.
                "unresolved": (totals.get(ABLATION) or {}).get("unresolved", 0),
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
                **_shell(request, "audit", summary, arm=arm, arm_path="/audit"),
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
                **_shell(request, "studio", summary),
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

    # Both entry points, named. `/` is the argument and `/control` is the
    # product, and someone who lands on the first without being told the second
    # exists will read a page and close the tab.
    print(f"Recoup listening on http://{host}:{port}")
    print(f"  the case for it   http://{host}:{port}/")
    print(f"  the product       http://{host}:{port}/control")
    if load_summary("data") is None:
        print("  no run found — generate one with `python -m recoup demo`")
    if not os.environ.get("RAZORPAY_WEBHOOK_SECRET"):
        print("  note: RAZORPAY_WEBHOOK_SECRET unset — webhooks will be refused")

    uvicorn.run(app, host=host, port=port, log_level="warning")
