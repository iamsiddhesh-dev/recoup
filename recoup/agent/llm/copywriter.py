"""Writing the message a customer actually receives.

Most of the money in this run sits behind failures no retry can fix, so for the
majority of payments the *message* is the product. Until now every customer got
the same hardcoded sentence, which is not a recovery attempt so much as a
notification.

## Two rules that make a model safe to use here

**The model writes templates, never numbers.** It produces
`"Your payment of {amount} didn't go through"`; code substitutes the real value at
send time. A model that never sees an amount cannot state the wrong one, which
removes the single most damaging failure mode by construction rather than by
checking for it afterwards. The validator enforces it: literal digit runs are
rejected outright.

**The validator refuses phishing-shaped copy.** A message about a failed payment
that asks someone to share an OTP, a CVV or a card number is indistinguishable
from a scam, and a payment company sending one is a serious incident. Any copy
mentioning credentials is discarded and the deterministic fallback used instead —
the model is not asked politely, its output is checked.

## What this does *not* claim

The simulator has no basis for modelling whether better copy recovers more money —
nudge outcomes depend on channel, cause and contact history, not on wording. So
this changes what a customer receives and does not move the headline number, and
it is deliberately kept out of the ablation for that reason. Claiming a lift from
copy would be inventing an effect to justify a feature.

One batched call generates the whole matrix — causes × languages, each with four
channel forms — which is then cached and reused. Per-message generation would be
roughly 900 calls per run against a 20-per-day budget, and would produce a
different message for two identical situations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from recoup.agent.llm.client import LLMClient
from recoup.domain import Channel, FailureCause, Language
from recoup.money import rupees

# Longest a message may be, per channel. SMS is the binding one: a single GSM
# segment is 160 characters, and a template that overflows after substitution
# silently becomes two billed messages.
LIMITS: dict[str, int] = {
    "sms": 160,
    "whatsapp": 400,
    "voice": 300,
    "email": 800,
}

# Copy asking for any of these is discarded. A payment-failure message requesting
# credentials is phishing whether or not it was meant that way.
CREDENTIAL_WORDS = re.compile(
    r"\b(otp|cvv|pin|password|passcode|card\s*number|expiry|net\s*banking\s*login"
    r"|upi\s*pin|mpin|verification\s*code)\b",
    re.IGNORECASE,
)

# Two or more consecutive digits. Amounts, dates and reference numbers all belong
# to the caller, not the model.
LITERAL_NUMBER = re.compile(r"\d{2,}")

# Only the placeholder link is permitted; a model-invented URL is a phishing
# vector with extra steps.
BARE_URL = re.compile(r"https?://|www\.", re.IGNORECASE)

PLACEHOLDER = re.compile(r"\{(\w+)\}")
ALLOWED_PLACEHOLDERS = {"amount", "link", "merchant"}

# Causes a customer is ever contacted about. RISK_BLOCKED is absent on purpose:
# compliance forbids contacting the customer at all, so there is nothing to write.
CONTACTABLE = [
    FailureCause.AUTH_ABANDONED,
    FailureCause.INSTRUMENT_INVALID,
    FailureCause.INSUFFICIENT_FUNDS,
    FailureCause.SOFT_ISSUER_DECLINE,
    FailureCause.TECHNICAL_GATEWAY,
    FailureCause.MANDATE_PROBLEM,
    FailureCause.CUSTOMER_INTENT,
]

# `Language.REGIONAL` says a customer prefers something other than English or
# Hinglish, but not which. Hindi is the concrete choice with the widest reach;
# a real deployment would carry the actual locale.
LANGUAGES = ["english", "hinglish", "hindi"]

LANGUAGE_OF = {
    Language.ENGLISH: "english",
    Language.HINGLISH: "hinglish",
    Language.REGIONAL: "hindi",
}

SYSTEM = """You write short recovery messages for customers whose Indian online
payment has just failed. The merchant is sending these; you are writing on their
behalf.

Absolute rules, in order of importance:

1. NEVER ask for an OTP, CVV, PIN, password, card number, expiry date, UPI PIN or
   any login credential. A payment message asking for these is indistinguishable
   from fraud. There is no exception.
2. NEVER write a literal number. Use the placeholder {amount} for the payment
   amount. Do not invent dates, reference numbers, deadlines or counts.
3. NEVER write a URL. Use the placeholder {link} where the customer should be sent
   to complete payment.
4. Do not promise anything the merchant has not committed to — no extensions, no
   waivers, no guarantees, no threats of account closure.

Style:

- Short, plain, and calm. A failed payment is mildly embarrassing; do not make it
  dramatic.
- Say what happened and what the single next step is. One action, not a menu.
- `english` is plain Indian English. `hinglish` is romanised Hindi-English mix as
  people actually message in India — natural, not translated. `hindi` is
  Devanagari script.
- SMS must be very short. Voice is read aloud, so write it as speech with no
  formatting. Email may be a little longer and can open with a greeting.

Available placeholders: {amount}, {link}, {merchant}. Nothing else.

Return only JSON matching the requested shape."""


class Variant(BaseModel):
    cause: str
    language: str
    sms: str = ""
    whatsapp: str = ""
    voice: str = ""
    email: str = ""

    def for_channel(self, channel: str) -> str:
        return getattr(self, channel, "")


class CopyMatrix(BaseModel):
    variants: list[Variant] = Field(default_factory=list)


@dataclass(frozen=True)
class Rejection:
    cause: str
    language: str
    channel: str
    reason: str


def validate(text: str, channel: str) -> str | None:
    """Return the reason this copy is unusable, or None if it passes.

    Returning a reason rather than a boolean because rejections are written to a
    report — "the model produced something we refused to send" is worth being able
    to read.
    """
    if not text or not text.strip():
        return "empty"

    if CREDENTIAL_WORDS.search(text):
        return "asks for a credential"

    if LITERAL_NUMBER.search(text):
        return "contains a literal number instead of a placeholder"

    if BARE_URL.search(text):
        return "contains a URL instead of the {link} placeholder"

    unknown = set(PLACEHOLDER.findall(text)) - ALLOWED_PLACEHOLDERS
    if unknown:
        return f"unknown placeholder(s): {', '.join(sorted(unknown))}"

    limit = LIMITS.get(channel, 400)
    # Measured after substitution, using a generous stand-in, because a template
    # that fits and a message that fits are different things.
    rendered = _substitute(text, amount=rupees(1_234_500), link="https://rzp.io/l/xxxxxxxx")
    if len(rendered) > limit:
        return f"too long for {channel}: {len(rendered)} > {limit}"

    return None


def _substitute(text: str, *, amount: str, link: str, merchant: str = "the merchant") -> str:
    return (
        text.replace("{amount}", amount)
        .replace("{link}", link)
        .replace("{merchant}", merchant)
    )


# ---------------------------------------------------------------------------
# Deterministic fallbacks
# ---------------------------------------------------------------------------
# Used whenever the model is unavailable or its copy fails validation. Hand
# written, short, and safe. The run never depends on a model being reachable.

FALLBACKS: dict[str, dict[str, str]] = {
    "AUTH_ABANDONED": {
        "english": "Your payment of {amount} wasn't completed. Finish it here: {link}",
        "hinglish": "Aapka {amount} ka payment complete nahi hua. Yahan pura karein: {link}",
        "hindi": "आपका {amount} का भुगतान पूरा नहीं हुआ। यहाँ पूरा करें: {link}",
    },
    "INSTRUMENT_INVALID": {
        "english": "Your saved card or UPI ID didn't work for {amount}. Add another here: {link}",
        "hinglish": "Aapka saved card ya UPI {amount} ke liye kaam nahi kiya. "
        "Dusra add karein: {link}",
        "hindi": "आपका सेव किया गया कार्ड या UPI {amount} के लिए काम नहीं किया। दूसरा जोड़ें: {link}",
    },
    "INSUFFICIENT_FUNDS": {
        "english": "Your payment of {amount} didn't go through. Try again when ready: {link}",
        "hinglish": "Aapka {amount} ka payment nahi hua. Ready hone par try karein: {link}",
        "hindi": "आपका {amount} का भुगतान नहीं हुआ। तैयार होने पर पुनः प्रयास करें: {link}",
    },
    "SOFT_ISSUER_DECLINE": {
        "english": "Your bank declined a payment of {amount}. You can retry here: {link}",
        "hinglish": "Aapke bank ne {amount} ka payment decline kiya. Dobara try karein: {link}",
        "hindi": "आपके बैंक ने {amount} का भुगतान अस्वीकार किया। पुनः प्रयास करें: {link}",
    },
    "TECHNICAL_GATEWAY": {
        "english": "A technical issue stopped your payment of {amount}. Try again here: {link}",
        "hinglish": "Technical issue ki wajah se {amount} ka payment ruk gaya. Try karein: {link}",
        "hindi": "तकनीकी समस्या के कारण {amount} का भुगतान रुक गया। पुनः प्रयास करें: {link}",
    },
    "MANDATE_PROBLEM": {
        "english": "We couldn't collect {amount} on your saved mandate. Set it up again: {link}",
        "hinglish": "Aapke mandate se {amount} collect nahi ho paya. Dobara set karein: {link}",
        "hindi": "आपके मैंडेट से {amount} नहीं लिया जा सका। दोबारा सेट करें: {link}",
    },
    "CUSTOMER_INTENT": {
        "english": "Your payment of {amount} was cancelled. If that was a mistake: {link}",
        "hinglish": "Aapka {amount} ka payment cancel hua. Galti se hua ho to: {link}",
        "hindi": "आपका {amount} का भुगतान रद्द हुआ। गलती से हुआ हो तो: {link}",
    },
}

UNKNOWN_FALLBACK = {
    "english": "Your payment of {amount} didn't go through. Complete it here: {link}",
    "hinglish": "Aapka {amount} ka payment nahi hua. Yahan pura karein: {link}",
    "hindi": "आपका {amount} का भुगतान नहीं हुआ। यहाँ पूरा करें: {link}",
}


class Copywriter:
    """Generates the copy matrix once, renders per message."""

    def __init__(self, client: LLMClient | None = None, use_llm: bool = True) -> None:
        self._client = client if client is not None else (LLMClient() if use_llm else None)
        self._copy: dict[tuple[str, str, str], str] = {}
        self.rejections: list[Rejection] = []
        self.generated = 0
        self.calls = 0

    # -- generation ----------------------------------------------------------

    def warm(self) -> int:
        """Generate the matrix, one request per language. Returns variants accepted.

        Chunked by language rather than sent as one call, because the whole matrix
        is 84 strings and does not fit. Groq's first attempt returned
        `json_validate_failed` with a `failed_generation` that was visibly correct
        and simply truncated — it ran out of output tokens mid-array.

        Three calls instead of one is still nothing against a twenty-per-day
        budget, and chunking by language keeps each request coherent: one language,
        every cause, so tone stays consistent within a locale.
        """
        if self._client is None:
            return 0

        accepted = 0
        for language in LANGUAGES:
            wanted = [{"cause": str(cause), "language": language} for cause in CONTACTABLE]

            matrix = self._client.structured(
                f"nudge-copy-{language}", SYSTEM, _build_prompt(wanted), CopyMatrix
            )
            self.calls += 1

            if matrix is None:
                continue

            accepted += self._absorb(matrix, language)

        self.generated = accepted
        return accepted

    def _absorb(self, matrix: CopyMatrix, language: str) -> int:
        """Validate every channel form and keep only what is safe to send."""
        accepted = 0

        for variant in matrix.variants:
            for channel in ("sms", "whatsapp", "voice", "email"):
                text = variant.for_channel(channel).strip()
                problem = validate(text, channel)

                if problem is not None:
                    self.rejections.append(
                        Rejection(
                            cause=variant.cause,
                            language=variant.language or language,
                            channel=channel,
                            reason=problem,
                        )
                    )
                    continue

                self._copy[(variant.cause, variant.language or language, channel)] = text
                accepted += 1

        return accepted

    # -- rendering -----------------------------------------------------------

    def template(
        self, cause: FailureCause | str | None, language: Language | str, channel: Channel | str
    ) -> tuple[str, str]:
        """The template to use, and where it came from."""
        cause_key = str(cause) if cause else "UNKNOWN"
        language_key = (
            LANGUAGE_OF.get(language, "english")
            if isinstance(language, Language)
            else str(language)
        )
        channel_key = str(channel)

        generated = self._copy.get((cause_key, language_key, channel_key))
        if generated:
            return generated, "generated"

        table = FALLBACKS.get(cause_key, UNKNOWN_FALLBACK)
        return table.get(language_key, table["english"]), "fallback"

    def render(
        self,
        cause: FailureCause | str | None,
        language: Language | str,
        channel: Channel | str,
        *,
        amount_paise: int,
        link: str = "",
        merchant: str = "the merchant",
    ) -> tuple[str, str]:
        """The message to send, and its source. Numbers are filled in here."""
        template, source = self.template(cause, language, channel)

        rendered = _substitute(
            template,
            amount=rupees(amount_paise),
            link=link or "your payment link",
            merchant=merchant,
        )
        return rendered, source


def _build_prompt(wanted: list[dict[str, str]]) -> str:
    import json

    guidance = {
        "AUTH_ABANDONED": "They started paying and did not finish. Nothing is wrong "
        "with their card or account — they just need to complete it.",
        "INSTRUMENT_INVALID": "Their saved card or UPI ID cannot work any more. They "
        "must supply a different one; retrying the same thing will fail.",
        "INSUFFICIENT_FUNDS": "There was not enough money. They already know this. Be "
        "brief and do not lecture or shame them.",
        "SOFT_ISSUER_DECLINE": "Their bank refused it, often for no visible reason. It "
        "may well work later. Do not blame the customer.",
        "TECHNICAL_GATEWAY": "Something on the payment infrastructure broke. This is "
        "not their fault at all and the message should make that clear.",
        "MANDATE_PROBLEM": "An automatic payment mandate is revoked, expired, or the "
        "amount exceeds its limit. They need to authorise it again.",
        "CUSTOMER_INTENT": "They cancelled deliberately. Be light, do not push, and "
        "make it easy to ignore.",
    }

    return json.dumps(
        {
            "instruction": (
                "Write recovery copy for each (cause, language) pair below. For each "
                "one give four channel forms: sms, whatsapp, voice, email. Return "
                '{"variants": [...]} with the same cause and language echoed back.'
            ),
            "limits": LIMITS,
            "cause_guidance": guidance,
            "wanted": wanted,
        },
        ensure_ascii=False,
        indent=2,
    )
