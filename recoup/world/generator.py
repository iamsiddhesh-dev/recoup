"""Batch generation — where the recoverable pool comes from.

Produces a run's worth of payments over the horizon, decides which of them fail,
and dresses each failure in Razorpay-shaped error fields. Two records come out of
every payment:

* a `PaymentEntity`, which is everything the agent will ever see, and
* a `PaymentTruth`, which stays inside the world.

`PaymentTruth` carries the failure's real cause. The entity carries only source,
step and reason — the observable symptoms. Recovering the cause from the symptoms
is the classification problem the agent has to solve, and keeping the answer on
the other side of this boundary is what stops the whole evaluation from being
circular.

The one correlation worth knowing about: during an active outage, failures skew
hard toward `TECHNICAL_GATEWAY` (`downtime_technical_tilt`). That is the pattern
the agent is supposed to discover — a spike in technical failures means stop
retrying and wait, not retry harder — and it is planted here rather than handed
over as a labelled feature.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from recoup.domain import (
    FailureCause,
    PaymentEntity,
    PaymentMethod,
    PaymentStatus,
)
from recoup.rng import lognormal_paise, substream, weighted_choice
from recoup.world.config import TaxonomyEntry, WorldConfig
from recoup.world.customers import Customer, Population
from recoup.world.issuers import IssuerBook

# Razorpay groups errors by who is at fault. Mapping the taxonomy's `source` onto
# the real top-level codes keeps generated events parseable by code written
# against the live API. https://razorpay.com/docs/errors/
_ERROR_CODE_BY_SOURCE = {
    "customer": "BAD_REQUEST_ERROR",
    "business": "BAD_REQUEST_ERROR",
    "internal": "SERVER_ERROR",
    "gateway": "GATEWAY_ERROR",
    "network": "GATEWAY_ERROR",
    "issuer_bank": "GATEWAY_ERROR",
    "issuer": "GATEWAY_ERROR",
    "bank": "GATEWAY_ERROR",
    "customer_psp": "GATEWAY_ERROR",
    "beneficiary_bank": "GATEWAY_ERROR",
}

# Human-readable text for known reasons. Anything absent falls through to a
# generic string, which is deliberate: novel, unmapped descriptions are exactly
# the case the LLM classifier fallback exists to handle, and the simulator should
# produce some rather than pretending the taxonomy is closed.
_DESCRIPTIONS = {
    "payment_failed": "Payment was unsuccessful as the bank did not authorise it",
    "insufficient_funds": "Payment failed due to insufficient funds in the account",
    "invalid_otp": "Authentication failed due to incorrect otp",
    "payment_timeout": "Payment was not completed within the permitted time",
    "gateway_technical_error": "Payment processing failed due to a technical error at the gateway",
    "card_expired": "Payment failed as the card has expired",
    "payment_blocked_risk": "Payment was blocked due to a risk assessment",
    "payment_cancelled": "Payment was cancelled by the customer",
    "npci_unavailable": "Payment failed as the UPI network was unavailable",
    "payment_declined": "Payment was declined by the issuing bank",
    "psp_unavailable": "Payment failed as the customer's UPI app was unreachable",
    "invalid_vpa": "Payment failed as the UPI ID could not be resolved",
    "bank_unavailable": "Payment failed as the bank's netbanking service was unavailable",
    "insufficient_balance": "Payment failed due to insufficient wallet balance",
    "wallet_unavailable": "Payment failed as the wallet service was unreachable",
    "wallet_not_linked": "Payment failed as the wallet is not linked to this account",
    "mandate_revoked": "Debit failed as the mandate has been revoked by the customer",
    "mandate_limit_exceeded": "Debit failed as it exceeds the mandate's permitted amount",
}

_WALLETS = ["payzapp", "phonepe", "amazonpay", "freecharge", "mobikwik"]
_UPI_HANDLES = {
    "HDFC": "okhdfcbank",
    "ICIC": "okicici",
    "SBIN": "oksbi",
    "UTIB": "okaxis",
    "KKBK": "okkotak",
    "PUNB": "okpnb",
    "YESB": "okyesbank",
}


@dataclass(frozen=True)
class PaymentTruth:
    """Ground truth. Never leaves the world, never serialised onto an event."""

    payment_id: str
    customer_id: str
    issuer_code: str
    method: PaymentMethod
    amount: int
    created_at: datetime
    failed: bool
    cause: FailureCause | None


@dataclass
class Batch:
    payments: list[PaymentEntity]
    truths: dict[str, PaymentTruth]
    config: WorldConfig

    @property
    def failures(self) -> list[PaymentEntity]:
        return [p for p in self.payments if p.status is PaymentStatus.FAILED]

    @property
    def amount_at_risk(self) -> int:
        """Total paise sitting in failed payments — the recoverable pool."""
        return sum(p.amount for p in self.failures)

    def truth_for(self, payment_id: str) -> PaymentTruth:
        return self.truths[payment_id]


class Generator:
    """Builds one deterministic batch from a seed."""

    def __init__(
        self,
        config: WorldConfig,
        population: Population,
        issuers: IssuerBook,
    ) -> None:
        self._config = config
        self._population = population
        self._issuers = issuers

        seed = config.run.seed
        self._arrival = substream(seed, "arrivals")
        self._selection = substream(seed, "selection")
        self._amounts = substream(seed, "amounts")
        self._outcomes = substream(seed, "outcomes")

    # -- arrival times ------------------------------------------------------

    def _arrival_times(self, count: int) -> list[datetime]:
        """Sample creation times, weighted by the diurnal traffic curve.

        Days are uniform; hours are not. Sorting at the end means the batch is in
        chronological order, which is what a real event feed looks like.
        """
        run = self._config.run
        hour_weights = {str(h): w for h, w in enumerate(self._config.traffic_by_hour)}

        times: list[datetime] = []
        for _ in range(count):
            day = self._arrival.randrange(run.horizon_days)
            hour = int(weighted_choice(self._arrival, hour_weights))
            times.append(
                run.start_at
                + timedelta(
                    days=day,
                    hours=hour,
                    minutes=self._arrival.randrange(60),
                    seconds=self._arrival.randrange(60),
                )
            )
        return sorted(times)

    # -- failure decision ---------------------------------------------------

    def _fails(self, method: PaymentMethod, issuer_code: str, when: datetime) -> bool:
        """Whether the first attempt fails.

        Base rate per method, made better by a reliable issuer and worse by a live
        outage. Composing on the success side rather than the failure side keeps
        the result bounded without a clamp doing the real work.
        """
        success = (1.0 - self._config.failure_rate[method]) * self._issuers.reliability(
            issuer_code
        )
        success *= self._issuers.multiplier_at(when, method, issuer_code)
        return self._outcomes.random() >= max(0.0, min(1.0, success))

    def _pick_failure(
        self, method: PaymentMethod, issuer_code: str, when: datetime
    ) -> TaxonomyEntry:
        """Choose how the failure presents.

        During an outage the technical causes are tilted up. This is the only
        place the downtime signal is wired into failure *shape* rather than
        failure *rate*, and it is what makes the downtime feed worth consulting.
        """
        entries = self._config.error_taxonomy[method]
        degraded = self._issuers.multiplier_at(when, method, issuer_code) < 1.0
        tilt = self._config.downtime_technical_tilt if degraded else 1.0

        weights = {
            str(index): entry.weight
            * (tilt if entry.cause is FailureCause.TECHNICAL_GATEWAY else 1.0)
            for index, entry in enumerate(entries)
        }
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        return entries[int(weighted_choice(self._selection, weights))]

    # -- instrument ---------------------------------------------------------

    def _instrument(
        self, payment_id: str, method: PaymentMethod, issuer_code: str, customer: Customer
    ) -> dict[str, str | None]:
        match method:
            case PaymentMethod.CARD:
                return {"card_id": f"card_{payment_id[4:]}", "bank": issuer_code}
            case PaymentMethod.UPI:
                handle = _UPI_HANDLES.get(issuer_code, "okbank")
                return {"vpa": f"{customer.id}@{handle}", "bank": issuer_code}
            case PaymentMethod.NETBANKING:
                return {"bank": issuer_code}
            case PaymentMethod.WALLET:
                return {"wallet": self._selection.choice(_WALLETS)}
            case PaymentMethod.EMANDATE:
                return {"token_id": f"token_{payment_id[4:]}", "bank": issuer_code}
        return {}

    # -- generation ---------------------------------------------------------

    def generate(self) -> Batch:
        run = self._config.run
        method_weights = {str(k): v for k, v in self._config.method_mix.items()}

        payments: list[PaymentEntity] = []
        truths: dict[str, PaymentTruth] = {}

        for index, when in enumerate(self._arrival_times(run.batch_size)):
            payment_id = f"pay_{index:06d}"
            customer = self._population.pick(self._selection)
            method = PaymentMethod(weighted_choice(self._selection, method_weights))
            issuer_code = self._issuers.pick(self._selection)

            spec = self._config.amounts[method]
            amount = lognormal_paise(
                self._amounts, spec.mu, spec.sigma, spec.min, spec.max
            )

            failed = self._fails(method, issuer_code, when)
            entry = self._pick_failure(method, issuer_code, when) if failed else None

            payment = PaymentEntity(
                id=payment_id,
                amount=amount,
                currency=run.currency,
                status=PaymentStatus.FAILED if failed else PaymentStatus.CAPTURED,
                captured=not failed,
                order_id=f"order_{index:06d}",
                method=method,
                contact=customer.contact,
                email=customer.email,
                created_at=int(when.timestamp()),
                notes={"customer_id": customer.id},
                acquirer_data={"bank_transaction_id": f"txn_{index:08d}"},
                **self._instrument(payment_id, method, issuer_code, customer),
            )

            if entry is not None:
                payment.error_code = _ERROR_CODE_BY_SOURCE.get(entry.source, "BAD_REQUEST_ERROR")
                payment.error_source = entry.source
                payment.error_step = entry.step
                payment.error_reason = entry.reason
                payment.error_description = _DESCRIPTIONS.get(
                    entry.reason, f"Payment failed ({entry.reason})"
                )

            payments.append(payment)
            truths[payment_id] = PaymentTruth(
                payment_id=payment_id,
                customer_id=customer.id,
                issuer_code=issuer_code,
                method=method,
                amount=amount,
                created_at=when,
                failed=failed,
                cause=entry.cause if entry else None,
            )

        return Batch(payments=payments, truths=truths, config=self._config)


def build_batch(config: WorldConfig) -> tuple[Batch, Population, IssuerBook]:
    """Assemble a run: population, issuers with their outage calendar, payments."""
    population = Population(config)
    issuers = IssuerBook(config)
    batch = Generator(config, population, issuers).generate()
    return batch, population, issuers


def random_stream(config: WorldConfig, name: str) -> random.Random:
    """Convenience for callers needing another independent stream off the seed."""
    return substream(config.run.seed, name)
