"""The web surface. For now: a webhook endpoint and enough to prove it works.

The control room is built on this app later; this is the skeleton plus the one
route that has to exist before anything can be demonstrated end to end against
real Razorpay.

Three things about webhook handling that are easy to get wrong:

* **Verify before parsing.** The raw bytes are the signed material. Anything that
  round-trips through a JSON decode first is verifying a different string than the
  one Razorpay signed.
* **Return 2xx fast.** Razorpay retries anything else, so slow or failing
  processing turns into duplicate deliveries and a growing backlog. Record and
  acknowledge; do the thinking afterwards.
* **A bad signature is a 400, not a 500.** It means someone POSTed to a public URL
  without the secret, which is expected background noise on the internet, not an
  incident.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from recoup.adapters.webhooks import SignatureError, parse
from recoup.web.sink import WebhookSink

EVENT_ID_HEADER = "x-razorpay-event-id"


def create_app(sink: WebhookSink | None = None) -> FastAPI:
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

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "webhooks_received": resolved.count()}

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
            # Expected traffic on a public URL. Say so precisely and move on.
            return JSONResponse(status_code=400, content={"error": str(exc)})

        stored = resolved.record(event, x_razorpay_event_id)

        print(
            f"[webhook] {event.event}"
            f"{'' if stored else '  (duplicate delivery, ignored)'}"
        )

        return JSONResponse(status_code=200, content={"status": "ok", "stored": stored})

    @app.get("/webhooks/recent")
    async def recent_webhooks(limit: int = 20) -> dict:
        """What has actually arrived. The proof, viewable in a browser."""
        return {"count": resolved.count(), "events": resolved.recent(limit)}

    return app


app = create_app()


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    secret_set = bool(os.environ.get("RAZORPAY_WEBHOOK_SECRET"))
    print(f"Recoup listening on http://{host}:{port}")
    print("  webhook endpoint   POST /webhooks/razorpay")
    print("  what arrived        GET /webhooks/recent")
    if not secret_set:
        print("  WARNING: RAZORPAY_WEBHOOK_SECRET is unset — webhooks will be refused")

    uvicorn.run(app, host=host, port=port, log_level="info")
