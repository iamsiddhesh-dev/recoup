# Recoup

**A revenue recovery agent for Indian payments.** It ingests payment, mandate and invoice
failures, decides and executes a recovery action for each one, and then measures how much
money that decision actually earned — against a holdout.

> **Status: in development.** Day 1 of 16. Headline results are not in yet; this README will
> lead with the measured number once the evaluation harness lands. No results are claimed
> here until they are reproducible via `make reproduce`.

---

## The problem

A failed payment is not a lost payment. Issuer declines soften, balances get topped up, OTP
sessions get abandoned and retried, outages resolve. But most merchants respond with a fixed
retry schedule — three attempts, fixed intervals, no regard for *why* it failed — which
recovers some money, burns customer goodwill on the rest, and cannot tell you which is which.

Recoup treats each failure as a decision under uncertainty with a cost attached.

## What it does

For every failure it classifies a root cause from Razorpay's documented error taxonomy, then
picks the action with the highest expected value:

```
EV(action) = P(recover | cause, issuer, attempt_n, hour, context)
             × amount × margin
           − cost(action)
           − annoyance_penalty(customer_contact_history)
```

Actions: retry now · retry at a modelled better time · switch payment method · switch route ·
nudge the customer · escalate to a human · **stop**.

Every choice passes a compliance gate with veto power — attempt caps, quiet hours, consent
checks, mandate and risk hard-stops — and every decision, including every veto, lands in an
append-only ledger that replays deterministically.

## Three design decisions

**1. A hard wall between the world and the agent.** The simulator holds all ground truth. The
agent sees only webhook-shaped events, exactly as Razorpay would deliver them. Enforced by a
test that scans agent code for imports of the world module and fails the build.

**2. An adapter seam, so the agent is portable.** `SimulatedAdapter` and `RazorpayTestAdapter`
implement the same protocol — the agent does not know which world it is in. The test-mode
adapter refuses to initialise against a live key, so a live money action is structurally
impossible.

**3. A holdout, not a highlight reel.** A control arm runs the naive fixed-retry schedule that
merchants actually use. Results are reported as **incremental** recovery over that baseline,
with an ablation arm measuring what the LLM itself contributes.

## Where AI is used, and where it deliberately is not

| LLM **yes** | LLM **no** — deterministic by design |
|---|---|
| Unmapped free-text error → taxonomy + a proposed mapping rule for human review | Retry timing → empirical success model (cause × issuer × hour) |
| Nudge copy in Hinglish/English, behind a template guardrail and a validator that checks amounts and claims | Action selection → expected-value argmax |
| Merchant-facing "why did we do this?" narrative over the audit trail | Compliance → hard rules, no model in the loop |
| Inbound receivables reply → structured promise-to-pay | All money math → arithmetic |
| Run digest and anomaly narration | Root-cause mapping for known error codes → lookup table |

This split is not stylistic. Razorpay documents `source` and `step` exhaustively per payment
method but leaves `reason` open-ended, so the deterministic classifier keys on what is
enumerable and the model handles only what is genuinely novel.

The free-tier token budget pushed the same way: a per-payment LLM call would need roughly 1,500
calls per run, which exceeds every available quota by between 6× and 75×. So model calls are
**batched** — the whole run's unmapped errors go up in a single request — costing about five to
ten calls per run in total. The constraint produced a better architecture than an unconstrained
budget would have.

## Quickstart

```
make setup      # install
make demo       # run a batch end to end, open the control room
make eval       # print the arms table: gross, incremental, cost, net, refusals
make reproduce  # regenerate every committed figure from fixed seeds
```

*Not yet wired — landing across days 1–14.*

## License

MIT — see [LICENSE](LICENSE).
