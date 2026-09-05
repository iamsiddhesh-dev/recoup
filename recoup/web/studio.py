"""The knobs Policy Studio exposes, and how they reach the engine.

Defined as data rather than as form fields so the screen, the validation and the
override application all read from one list. Adding a knob is one entry here.

Only the agent's own configuration is adjustable — costs, caps, thresholds. The
world is deliberately fixed. If a viewer could change how often payments fail or
how likely customers are to respond, they could produce any number they liked and
the comparison would mean nothing. Studio answers "what would a differently
configured agent have earned against the same month?", which is a question with
an honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recoup.agent.config import ComplianceConfig, PolicyConfig
from recoup.money import rupees


@dataclass(frozen=True)
class Knob:
    key: str
    label: str
    help: str
    minimum: float
    maximum: float
    step: float
    unit: str = ""

    # Which heading this sits under in the studio. Seven undifferentiated
    # sliders read as a settings page; grouped, they read as three separate
    # arguments — what the agent thinks money is worth, what reaching a customer
    # costs, and what it is not allowed to do regardless of either.
    group: str = "Economics"

    def value_from(self, policy: PolicyConfig, compliance: ComplianceConfig) -> float:
        return KNOB_READERS[self.key](policy, compliance)

    def display(self, value: float) -> str:
        if self.unit == "₹":
            return rupees(value, precise_below=10_000)
        return f"{value:g}{self.unit}"


KNOBS: list[Knob] = [
    Knob(
        key="ev_threshold_paise",
        label="Act above",
        help="Minimum expected value before the agent will do anything at all. "
        "Raise it and the agent gets pickier, spending less and recovering less.",
        minimum=0,
        maximum=5_000,
        step=100,
        unit="₹",
    ),
    Knob(
        key="annoyance",
        label="Cost of a repeat contact",
        help="What writing to someone again is assumed to cost in goodwill. This "
        "is the number that stops the optimiser messaging everyone constantly.",
        minimum=0,
        maximum=5_000,
        step=100,
        unit="₹",
    ),
    Knob(
        key="escalation_scarcity",
        label="Value of a human slot",
        help="The opportunity cost of spending one of a limited number of "
        "escalations here rather than on a larger payment.",
        minimum=0,
        maximum=300_000,
        step=10_000,
        unit="₹",
    ),
    Knob(
        key="whatsapp_cost",
        label="WhatsApp cost",
        help="Per message. Cheap enough that reach usually wins.",
        minimum=0,
        maximum=500,
        step=5,
        unit="₹",
        group="Channel cost",
    ),
    Knob(
        key="voice_cost",
        label="Voice call cost",
        help="The most effective channel and the most expensive. Where the two "
        "cross over is the interesting part.",
        minimum=0,
        maximum=1_000,
        step=10,
        unit="₹",
        group="Channel cost",
    ),
    Knob(
        key="max_contacts",
        label="Contacts per customer / 7d",
        help="A hard compliance cap. Lower it and the refusal list grows.",
        minimum=0,
        maximum=6,
        step=1,
        group="Compliance caps",
    ),
    Knob(
        key="max_attempts",
        label="Retries per payment",
        help="A hard cap on charge attempts, which cannot be refilled once spent.",
        minimum=1,
        maximum=8,
        step=1,
        group="Compliance caps",
    ),
]

KNOB_READERS = {
    "ev_threshold_paise": lambda p, c: p.ev_threshold_paise,
    "annoyance": lambda p, c: p.annoyance.penalty_per_prior_contact,
    "whatsapp_cost": lambda p, c: p.action_costs["NUDGE_WHATSAPP"],
    "voice_cost": lambda p, c: p.action_costs["NUDGE_VOICE"],
    "escalation_scarcity": lambda p, c: p.escalation_scarcity_premium,
    "max_contacts": lambda p, c: c.contact.max_per_customer_per_7d,
    "max_attempts": lambda p, c: c.attempts.max_per_payment,
}

BY_KEY = {knob.key: knob for knob in KNOBS}


def defaults() -> dict[str, float]:
    policy, compliance = PolicyConfig.load(), ComplianceConfig.load()
    return {knob.key: knob.value_from(policy, compliance) for knob in KNOBS}


def clean(raw: dict[str, Any]) -> dict[str, float]:
    """Keep only known knobs, clamped to their range.

    Values arrive from a browser, so they are untrusted input. Clamping rather
    than rejecting because a slider that silently refuses is worse than one that
    stops at its end, and nothing here can be out of range and still meaningful.
    """
    cleaned: dict[str, float] = {}

    for key, value in raw.items():
        knob = BY_KEY.get(key)
        if knob is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        cleaned[key] = max(knob.minimum, min(knob.maximum, number))

    return cleaned


def apply(overrides: dict[str, float]) -> tuple[PolicyConfig, ComplianceConfig]:
    """Fresh configs with the overrides applied.

    Deep copies, so a studio run cannot mutate the configuration every other
    screen is reading.
    """
    policy = PolicyConfig.load().model_copy(deep=True)
    compliance = ComplianceConfig.load().model_copy(deep=True)

    for key, value in overrides.items():
        match key:
            case "ev_threshold_paise":
                policy.ev_threshold_paise = int(value)
            case "annoyance":
                policy.annoyance.penalty_per_prior_contact = int(value)
            case "whatsapp_cost":
                policy.action_costs["NUDGE_WHATSAPP"] = int(value)
            case "voice_cost":
                policy.action_costs["NUDGE_VOICE"] = int(value)
            case "escalation_scarcity":
                policy.escalation_scarcity_premium = int(value)
            case "max_contacts":
                compliance.contact.max_per_customer_per_7d = int(value)
            case "max_attempts":
                compliance.attempts.max_per_payment = int(value)

    return policy, compliance


def changed_from_default(overrides: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Which knobs actually moved, as (default, current)."""
    base = defaults()
    return {
        key: (base[key], value)
        for key, value in overrides.items()
        if key in base and base[key] != value
    }
