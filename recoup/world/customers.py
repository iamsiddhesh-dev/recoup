"""The customer population.

Customers exist in the world because recovery is not purely a retry problem. A
large share of failures need a *person* to do something — re-enter an OTP, top up
a balance, update an expired card — and whether they do depends on who they are,
how reachable they are, and how many times they have already been chased this
week.

Three properties drive everything downstream:

* **nudge_response** — the probability a contact produces action. This is what
  makes nudging a real decision rather than a free action.
* **annoyance_sensitivity** — how much goodwill each additional contact costs.
  Without it an optimiser will message everyone constantly, because each
  individual message looks cheap and mildly positive.
* **consent** — whether WhatsApp and voice are permitted at all. Absent consent
  the compliance gate vetoes those channels outright, no matter the expected
  value, which is the correct behaviour and occasionally an expensive one.

The agent sees none of these. It sees a customer id on the payment's notes and
whatever it has observed that customer do.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from recoup.domain import Channel, Language
from recoup.rng import substream, weighted_choice
from recoup.world.config import WorldConfig


@dataclass(frozen=True)
class Customer:
    id: str
    segment: str
    nudge_response: float
    annoyance_sensitivity: float
    language: Language
    consent: dict[Channel, bool] = field(default_factory=dict)
    contact: str = ""
    email: str = ""

    def may_contact(self, channel: Channel) -> bool:
        return self.consent.get(channel, False)


class Population:
    """A deterministic set of customers for one run."""

    def __init__(self, config: WorldConfig) -> None:
        self._config = config
        rng = substream(config.run.seed, "customers")
        self.customers: list[Customer] = self._generate(rng)
        self._by_id = {customer.id: customer for customer in self.customers}

    def __len__(self) -> int:
        return len(self.customers)

    def _generate(self, rng: random.Random) -> list[Customer]:
        cfg = self._config.customers
        segment_weights = {name: spec.share for name, spec in cfg.segments.items()}
        language_weights = {str(k): v for k, v in cfg.language.items()}

        customers: list[Customer] = []
        for index in range(cfg.count):
            segment_name = weighted_choice(rng, segment_weights)
            segment = cfg.segments[segment_name]

            consent = {
                channel: rng.random() < cfg.consent_rate.get(str(channel), 0.0)
                for channel in Channel
            }

            # Synthetic, non-routable contact details. These are fabricated
            # identifiers for a simulation and must never be real: the 555 mobile
            # block and example.invalid are reserved for exactly this.
            customers.append(
                Customer(
                    id=f"cust_{index:05d}",
                    segment=segment_name,
                    nudge_response=segment.nudge_response,
                    annoyance_sensitivity=segment.annoyance_sensitivity,
                    language=Language(weighted_choice(rng, language_weights)),
                    consent=consent,
                    contact=f"+9155500{index:05d}",
                    email=f"cust{index:05d}@example.invalid",
                )
            )

        return customers

    def pick(self, rng: random.Random) -> Customer:
        return self.customers[rng.randrange(len(self.customers))]

    def get(self, customer_id: str) -> Customer | None:
        return self._by_id.get(customer_id)

    def responds_to_nudge(
        self, customer: Customer, rng: random.Random, prior_contacts: int
    ) -> bool:
        """Whether a nudge lands.

        Response decays with prior contacts in the window: the fourth message
        about the same failed payment is not as persuasive as the first, and for
        an annoyance-sensitive customer it is close to worthless. This is the
        world's half of the annoyance model — the agent pays a modelled penalty,
        and here that penalty turns out to be real.
        """
        decay = 1.0 / (1.0 + customer.annoyance_sensitivity * prior_contacts)
        return rng.random() < customer.nudge_response * decay
