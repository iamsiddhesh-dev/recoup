"""Webhook receipt and verification.

A webhook endpoint is an unauthenticated POST route on the public internet that
tells your system a payment succeeded. Verifying the signature is the entire
security model, so it is done first, on the raw bytes, before anything is parsed.

Razorpay signs the exact request body with the secret configured on the webhook
and sends it in `X-Razorpay-Signature` as a hex HMAC-SHA256 digest. Two rules that
are easy to get wrong and expensive to get wrong:

* **Verify the raw body, not a re-serialised object.** `json.loads` followed by
  `json.dumps` will not reproduce the original bytes — key order, separators and
  unicode escaping all differ — and the signature will never match.
* **Compare in constant time.** `hmac.compare_digest`, not `==`, so the comparison
  does not leak the expected digest through timing.

https://razorpay.com/docs/webhooks/validate-test/
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os

from recoup.domain import WebhookEvent

SIGNATURE_HEADER = "X-Razorpay-Signature"


class SignatureError(ValueError):
    """The payload did not come from Razorpay, or was tampered with in transit."""


def expected_signature(raw_body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()


def verify(raw_body: bytes, signature: str | None, secret: str | None = None) -> None:
    """Raise unless `signature` is a valid Razorpay signature for `raw_body`.

    Raises rather than returning a bool so that a caller cannot accidentally
    ignore the result — the common form of this bug is a truthy check on a
    function that was actually returning None.
    """
    secret = secret or os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")

    if not secret:
        raise SignatureError(
            "RAZORPAY_WEBHOOK_SECRET is not set. Webhooks cannot be verified, so "
            "they are refused rather than trusted."
        )

    if not signature:
        raise SignatureError(f"missing {SIGNATURE_HEADER} header")

    if not hmac.compare_digest(expected_signature(raw_body, secret), signature):
        raise SignatureError("signature mismatch")


def parse(raw_body: bytes, signature: str | None, secret: str | None = None) -> WebhookEvent:
    """Verify, then parse. In that order, always."""
    verify(raw_body, signature, secret)
    return WebhookEvent.model_validate(json.loads(raw_body))
