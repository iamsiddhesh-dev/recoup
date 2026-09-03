# Recoup

**A revenue recovery agent for Indian payments.** It ingests payment, mandate and invoice
failures, decides and executes a recovery action for each one, and measures how much money
that decision actually earned — against a holdout.

> **On a 5,000-payment batch — 1,605 failures, ₹30,02,856 at risk — Recoup recovers
> ₹5,41,724, of which ₹3,94,791 is incremental over the naive fixed-retry schedule
> merchants actually run. It spends ₹8,336 on 926 contacts to do it, and refuses 903
> actions on compliance grounds.**
>
> Seed `20260902`. Reproduce with `make eval` — no API key required.

The interesting half of that number is the second half. Anyone can report gross recovery;
gross recovery includes every payment that would have succeeded on a dumb retry anyway.
Incremental recovery is what the agent is actually worth.

---

## The result

| Arm | What it does | Recovered | Contacts | Cost | Net margin |
|---|---|---:|---:|---:|---:|
| `naive_baseline` | fixed retry schedule, no regard for cause | ₹1,46,933 | 0 | ₹2,509 | ₹38,632 |
| `contact_only` | always contact, never reason about it | ₹4,14,362 | 975 | ₹6,386 | ₹1,09,636 |
| **`recoup_agent`** | **expected-value policy + compliance gate** | **₹5,41,724** | **926** | **₹8,336** | **₹1,43,347** |
| `recoup_agent_no_llm` | ablation — deterministic classifier only | ₹5,41,963 | 927 | ₹8,338 | ₹1,43,412 |

All four arms run against the same world, the same failures, in the same order, with the
same underlying luck. Outcomes are drawn from `(seed, payment_id, attempt)`, so two arms
that make the same decision get the same result and the arms differ *only* by their
decisions.

### Where the lift comes from, honestly

`contact_only` exists to make the agent's contribution decomposable, and the decomposition
is not especially flattering:

| Source | Amount | Share |
|---|---:|---:|
| **Coverage** — contacting at all, versus retrying blindly | ₹2,67,428 | **67.7%** |
| **Judgment** — which channel, when, to whom, and when to stop | ₹1,27,363 | **32.3%** |

Two thirds of the lift is available to anyone who realises that most failed payments
cannot be retried and have to be *asked* about. That is worth saying out loud, because a
single headline number would have quietly claimed all of it for the policy engine.

The remaining third is what the expected-value machinery earns: the agent recovers **more
money from fewer contacts** than the arm that contacts everybody — 319 recoveries from 926
contacts against 222 from 975.

### What the LLM contributes: −₹238

Measured, not assumed. The ablation arm removes only the model and changes nothing else:

| | Recovered | Contacts |
|---|---:|---:|
| `recoup_agent_no_llm` | ₹5,41,963 | 927 |
| `recoup_agent` | ₹5,41,724 | 926 |

The model is not wrong — it classifies every unresolved symptom correctly, checked against
ground truth it cannot see. It is simply not worth much *here*, because 46 of the 51
failures it resolves are bank and wallet outages where the correct action is "do not
chase", which is close to what the pessimistic unknown-cause default already produced.

This is reported rather than fixed. The obvious way to manufacture a win would be to
degrade the default until the model looks necessary; the number that produced would mean
nothing. [FAILURES.md](FAILURES.md#the-llm-added-exactly-nothing-twice--for-two-different-reasons)
sets out the conditions under which this number would be different, as a testable claim.

### Does the claim survive its own assumptions?

`make sweep` moves each load-bearing assumption to both ends of a plausible band and
re-runs the entire evaluation — 14 full runs plus the base case.

**The agent beats both other arms at all 15 points.** Incremental recovery ranges from
₹2,36,219 to ₹4,66,620; the narrowest margin over `contact_only` is ₹43,045. Ordered by
swing, the result depends most on the failure rate (₹2,06,191) and issuer downtime
(₹1,62,871), and least on the annoyance penalty (₹5,047).

Committed as [`reports/sensitivity.json`](reports/sensitivity.json) so nobody has to spend
twenty minutes reproducing it before they can read it.

---

## The problem

A failed payment is not a lost payment. Issuer declines soften, balances get topped up,
OTP sessions get abandoned and retried, outages resolve. But most merchants respond with a
fixed retry schedule — three attempts, fixed intervals, no regard for *why* it failed —
which recovers some money, burns customer goodwill on the rest, and cannot tell you which
is which.

There is also a constraint most retry logic ignores: **Razorpay cannot re-run a failed
payment.** Charging a customer who is not present requires standing authorisation — an
e-mandate, a tokenised card, a UPI Autopay mandate. In this batch that covers 23.7% of
failures. For the other 76.3% the only recovery is to *ask the customer again*, which
makes the message the recovery attempt rather than a notification about one.

Recoup treats each failure as a decision under uncertainty with a cost attached.

## How it decides

For every failure it classifies a root cause from Razorpay's documented error taxonomy,
then picks the action with the highest expected value:

```
EV(action) = P(recover | cause, issuer, attempt_n, hour, context)
             × amount × margin
           − cost(action)
           − annoyance_penalty(customer_contact_history)
```

Actions: retry now · retry at a modelled better time · switch payment method · nudge the
customer on a chosen channel in a chosen language · escalate to a human · **stop**.

The arithmetic is shown on screen, line by line, for every decision — including the
options that lost. A recovery agent that cannot explain a refusal is not auditable.

Every choice then passes a compliance gate with **veto power**: attempt caps, quiet hours,
contact intervals, consent checks, mandate and risk hard-stops. The gate runs *after* the
policy and can only subtract. Compliance is never a term inside the expected-value sum,
because a rule that can be outbid by a large enough number is not a rule.

## Four design decisions

**1. A hard wall between the world and the agent.** The simulator holds all ground truth.
The agent sees only webhook-shaped events, exactly as Razorpay would deliver them. A test
AST-scans every module under `recoup/agent/` for imports of `recoup.world` and fails the
build. The wall is the reason the result means anything.

**2. An adapter seam, so the agent is portable.** `SimulatedAdapter` and
`RazorpayTestAdapter` implement one protocol — the agent does not know which world it is
in. The test-mode adapter refuses to initialise against a key that is not `rzp_test_`, so a
live money action is structurally impossible rather than merely unlikely. A real
test-mode payment has been run through a tunnel end to end; the captured webhook is
committed as a fixture, and it disproved two of our assumptions (see
[FAILURES.md](FAILURES.md)).

**3. A holdout, and then a second holdout.** `naive_baseline` is what merchants do.
`contact_only` isolates how much of the lift is coverage rather than judgment. Both exist
to make the headline number smaller and truer.

**4. An append-only ledger.** Every observation, decision, veto, execution and outcome is
one row, ordered, immutable — SQLite triggers reject `UPDATE` and `DELETE` outright. Each
arm's stream is hashed when written, and the audit screen re-hashes it in front of you, so
"this is an audit trail" is a property you can check rather than a claim in a README.

## Where AI is used, and where it deliberately is not

| LLM **yes** | LLM **no** — deterministic by design |
|---|---|
| Unmapped free-text error → taxonomy + a proposed mapping rule for human review | Retry timing → empirical success model (cause × issuer × hour) |
| Recovery copy, generated as **templates** in English, Hinglish and Hindi, behind a validator | Action selection → expected-value argmax |
| | Compliance → hard rules, no model in the loop |
| | All money math → arithmetic |
| | Root-cause mapping for known error codes → lookup table |

This split is not stylistic. Razorpay documents `source` and `step` exhaustively per
payment method but leaves `reason` open-ended, so the deterministic classifier keys on
what is enumerable and the model handles only what is genuinely novel. On this seed the
rules resolve **96.8%** of failures; the model is asked about the remaining 3.2%, which is
9.4% of the money.

Two rules make the model safe to put in front of a customer:

- **It writes templates, never numbers.** It produces `"Your payment of {amount} did not go
  through"`; code substitutes the real value at send time. A model that never sees an
  amount cannot state the wrong one. The validator enforces it — any run of two or more
  digits is rejected.
- **Phishing-shaped copy is refused.** A payment-failure message asking for an OTP, CVV or
  card number is indistinguishable from fraud. Copy mentioning a credential is discarded
  and a hand-written fallback used instead, with the rejection recorded.

### Four model calls per run

The free-tier budget settled the architecture before taste could. A per-payment call would
need ~1,500 requests per run, which exceeds Gemini Flash's daily allowance by 75× and
Groq's token budget by 6×. **A per-payment LLM call here is not merely poor judgment; it
is impossible.**

So the calls are batched: every unmapped error in the run goes up in one request, and the
copy matrix — 7 causes × 3 languages × 4 channels — in three more, one per language.
Four calls per run, against a twenty-per-day quota.

Responses are content-addressed and **committed to `cache/llm/`**, so the reported numbers
reproduce on a clean clone with no key and no network. Keys are only needed to regenerate
the cache after changing a prompt.

## The product

Six screens, server-rendered, hand-authored design system, no build step.

| Screen | What it is for |
|---|---|
| **Control Room** | The scoreboard: at risk, recovered, incremental, cost, net — with the coverage/judgment split stated rather than hidden |
| **Recovery Queue** | Every failure, filterable by cause, arm and outcome |
| **Case Detail** | One payment end to end: the timeline, each decision's EV arithmetic as a checkable sum, the options it beat, and the actual message the customer received |
| **Policy Studio** | Change a cost, a cap or a quiet-hour window, re-run the real evaluation, see the number move |
| **Audit & Refusals** | The ledger, its digest verified live, and the 903 actions compliance refused, grouped by rule |
| **Experiment** | Arms table, the ablation, and the sensitivity tornado |

## Quickstart

```bash
make setup      # install
make eval       # the arms table — ~35s, no API key needed
make demo       # run a batch, then serve the control room at :8000
make sweep      # re-run the evaluation at 14 perturbed assumptions (~20 min)
make test       # the wall, every compliance rule, the EV math
```

`data/` is generated and gitignored, so run `make demo` once before browsing the UI.
`reports/` is committed, because a twenty-minute sweep should be readable without being
re-run.

Everything is deterministic given a seed. `--seed N` on any command; the seed is displayed
in the top bar of every screen.

Optional, and not needed to reproduce anything: copy `.env.example` to `.env` for a
Razorpay test key (to exercise the live adapter with `python -m recoup probe`) or an LLM
key (to regenerate the cache).

## Reading the code

```
recoup/
├─ world/       ground truth — generator, issuers, customers, outcomes, clock
│               ↓ webhook-shaped events only ── THE WALL, enforced by test
├─ adapters/    one protocol, two implementations: simulated, razorpay_test
├─ agent/       classify → context → policy → compliance → executor
│  └─ llm/      client with a fallback chain · classifier · copywriter
├─ ledger/      append-only events, deterministic replay
├─ eval/        arms · runner · metrics · sensitivity · store
└─ web/         FastAPI + Jinja2, six screens
```

- [FAILURES.md](FAILURES.md) — eight things this project got wrong, how each was caught,
  and what changed. The most useful file in the repo.
- [DECISIONS.md](DECISIONS.md) — why it is built this way, and what was deliberately left
  out.
- [`config/world.yaml`](config/world.yaml) — every simulator assumption, each tagged
  `[OBSERVED]` or `[ASSUMPTION]`.

## What this is not

It is a measured claim about a simulated world, not a production result. One error
archetype has been verified against live Razorpay test mode; twenty are assumptions, and
the first one checked was half wrong. The simulator cannot model whether better copy
recovers more money, so the copywriter is excluded from the ablation rather than credited
with a lift it has no evidence for.

The honest summary is that the *method* — holdout, ablation, sensitivity sweep, and a wall
that makes the agent unable to cheat — would survive being pointed at real data. The
numbers would move.

## License

MIT — see [LICENSE](LICENSE).
