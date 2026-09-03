# Decisions

Why this is built the way it is, and — more usefully — what was deliberately left out.

Most of these were decided against a real alternative that a reasonable person would have
picked. Where that is true, the alternative is named. A decision record that only lists
what was chosen is a changelog.

---

## 1. The world is simulated, and the agent is forbidden from seeing it

**Chosen:** a generative simulator holding all ground truth, emitting webhook-shaped events
to an agent that cannot import it. A test AST-scans every module under `recoup/agent/` for
an import of `recoup.world` and fails the build.

**Rejected:** using a public payments dataset. There isn't one for Indian failure recovery,
and a dataset without counterfactuals cannot answer the only question that matters here —
*what would have happened if we had acted differently?* Retry outcomes are unobservable for
the arm you did not run.

**The cost, stated plainly.** A simulated evaluation grades its own homework. Four things
push back on that, in descending order of strength: the wall makes cheating structurally
impossible rather than merely discouraged; the adapter seam means the same agent runs
against real Razorpay test mode; every assumption in `world.yaml` is tagged `[OBSERVED]` or
`[ASSUMPTION]` with a source; and the sensitivity sweep re-runs the whole evaluation at
both ends of every load-bearing assumption.

It is still a simulation. [FAILURES.md](FAILURES.md) records that the first archetype
checked against live test mode was half wrong, and that is the honest prior for the twenty
that have not been checked.

## 2. Expected value, not a learned policy

**Chosen:** an explicit arithmetic policy — `EV = P(recover) × amount × margin − cost −
annoyance`, argmax over the candidate actions.

**Rejected:** a contextual bandit or an RL policy over the same action space. It would have
been more fashionable and, on this data, meaningless: it would learn the simulator's
generative parameters and report them back as insight. Fitting a model to a world whose
parameters you wrote yourself measures nothing.

The deciding argument is not accuracy though — it is that this is a system that moves other
people's money. Every decision has to be defensible line by line to a merchant, a
compliance reviewer, and eventually a regulator. `argmax` over a printed sum is auditable;
a learned policy is auditable only in aggregate.

That is why Case Detail renders the arithmetic as a checkable sum including the options
that *lost*. The explanation is the computation, not a narration of it after the fact.

## 3. Compliance vetoes after the policy, and is never a term inside it

**Chosen:** the policy proposes; a separate gate disposes. The gate can only subtract, and
every veto is written to the ledger with the rule that fired.

**Rejected:** folding compliance into the EV sum as a large negative cost. This is the
common shape and it is wrong: a rule expressed as a penalty is a rule with a price, and a
sufficiently large payment will outbid it. Quiet hours are not worth ₹40,000. Re-debiting a
revoked mandate is not a trade.

The structural benefit is that refusals become a first-class output. 903 vetoes on this run
group cleanly by rule, which is what makes "here is what we deliberately did not touch, and
why" a deliverable rather than an absence.

## 4. An append-only ledger, enforced by the database

**Chosen:** every observation, decision, veto, execution and outcome is one immutable row.
SQLite triggers reject `UPDATE` and `DELETE` outright. Each arm's event stream is hashed
when written, and the audit screen re-hashes it live.

**Rejected:** mutable per-payment state rows with a separate log table. Faster to query,
and it makes the log decorative — the state is the truth and the log is a description of
it, which can drift. Here the events *are* the state; every screen is a fold over them.

**Rejected:** trusting application code not to mutate. The trigger costs four lines and
turns a convention into a guarantee. Guarantees survive refactors.

## 5. The run is a precomputed artifact, not a live computation

**Chosen:** `make demo` runs the evaluation once (~35s), writes `data/run.db` and
`data/run.json`, and every screen is a query against those. Policy Studio, which genuinely
needs a fresh run, does it as a background job with progress.

**Rejected:** computing on request with a cache. Caching's hard problem is invalidation,
and this data deletes it: a run is deterministic given a seed, identical for every viewer,
and frozen once produced. There is never a reason to throw it away, so there is no
invalidation logic to get wrong.

The ledger's indexes on `(arm, payment_id, seq)` and `(arm, kind, seq)` were built for
auditability. Every screen turned out to be a query against them — the case view is
`story_of`, the refusal list is a filter on vetoes. Pointing the ledger at a file instead of
memory was the entire change.

## 6. Arms share their randomness

**Chosen:** outcomes are drawn from `(seed, payment_id, attempt)`. Two arms that make the
same decision get the same result.

**Rejected:** an independent random stream per arm, which is the default and is subtly
wrong. It makes the arms differ by luck as well as by policy, so a lift of ₹3,94,791 could
be a policy difference or a sampling difference, and you cannot tell which without many
more runs. Common random numbers remove the variance you do not care about instead of
averaging over it.

## 7. Two holdouts, because one flatters

**Chosen:** `naive_baseline` (what merchants do) and `contact_only` (contact everything,
reason about nothing).

The second arm exists solely to make the headline smaller. Without it the whole ₹3,94,791
reads as the policy engine's work. With it, 67.7% is coverage — available to anyone who
notices that 76.3% of failures cannot be retried at all — and only 32.3% is judgment.

Reporting a number that shrinks your own result by two thirds is the cheapest available
signal that the remaining third is real.

## 8. Model calls are batched, and their responses are committed

**Chosen:** four LLM calls per run. Every unmapped error goes up in one request; the copy
matrix in three, one per language. Responses are content-addressed and committed to
`cache/llm/`.

**Forced, not chosen.** A per-payment call needs ~1,500 requests per run — 75× Gemini
Flash's daily allowance, 6× Groq's token budget. The free tier made the naive design
impossible, and the design it forced is better: batching gives the model the whole
population at once, which is when a classifier is most consistent.

Committing the cache is what makes the README's number reproducible on a clean clone with
no key and no network. It also means the LLM cannot silently change the reported result
between two readers.

## 9. The model writes templates, never numbers

**Chosen:** the model produces `"Your payment of {amount} did not go through"`; code
substitutes the value at send time. The validator rejects any run of two or more digits,
any credential word, and any URL it did not template.

**Rejected:** generating the finished message per payment, then checking the number is
right. That is the same design with a test bolted on, and it fails open — a checker that
misses one case sends a customer the wrong amount. A model that never sees an amount cannot
state one. The failure mode is removed by construction rather than detected afterwards.

The credential rule is the one that would end the project: a payment-failure SMS asking for
an OTP is indistinguishable from a scam, and a payments company sending one has an incident,
not a bug.

## 10. Server-rendered HTML with a hand-authored design system

**Chosen:** FastAPI + Jinja2, ~1,000 lines of CSS tokens and components, two small inline
scripts. No npm, no build step, no framework.

**Rejected:** React — two days of setup for a read-mostly application whose one interactive
surface is a form that kicks off a background job. **Rejected:** Tailwind — its defaults
produce the generic admin-panel look, and this is meant to read as an instrument.
**Dropped:** HTMX, Alpine and ECharts, all in the original plan and none of them needed once
the pages existed. The charts are CSS and inline SVG; the two behaviours that need
JavaScript have it.

A judge clones this and runs it. Anything between `git clone` and a running product is
a cost with no benefit.

---

# Deliberately not built

Each of these was planned. Each was dropped for a reason better than running out of time.

## Langfuse tracing

The plan called for it under "explainability". Then the LLM cache turned out to be a better
trace than a trace: `cache/llm/` holds the exact prompt and the exact response for every
call, content-addressed, committed to the repository, readable offline by anyone who clones
it, three years from now, with no account.

A hosted dashboard behind a login that a reader cannot open is worse evidence than a JSON
file they already have. Adding Langfuse would have meant an external dependency, a network
call in the hot path, and a second source of truth about what the model was asked.

**What is genuinely lost:** aggregate latency and token dashboards across many runs. At four
calls per run that is not a problem this project has.

## Resend, and any real outbound message

Every channel is mocked. Sending real email to simulated customers proves nothing about
recovery and creates a live side effect in a project whose central safety claim is that it
cannot take one.

The seam is what matters and it is real: `Notifier` is a protocol, the simulated
implementation records the send, and a production implementation would substitute at that
boundary without touching the agent.

## LangGraph, or any multi-agent supervisor/worker framework

The strongest temptation, because "agent" in 2026 is widely assumed to mean a graph of LLMs
passing messages.

The work here is a decision per payment: classify, price the options, check the rules, act.
It is a pipeline with one branch point, and the branch point is an `argmax` over arithmetic.
Wrapping that in a supervisor that dispatches to workers would add a scheduler, a message
format and a failure surface on top of a function call, and would make the system *less*
explainable — the thing it is optimised for.

There is a real version of this argument: if the recovery loop grew a negotiation with the
customer over several turns, or had to hold state across channels, a graph would start
earning its cost. It does not do that today, and building the framework first in
anticipation is how projects acquire abstractions that never pay for themselves.

Calling this multi-agent would be dressing. It is one agent with a policy.

## Promise-to-pay parsing

Planned as an LLM call: parse an inbound "I'll pay on the 5th" into a structured commitment
and schedule against it.

There is no inbound channel. The simulator does not model customer replies, so the only
text available to parse would be text this project wrote — a round trip that measures the
model's ability to read our own generated strings. That is a demo, not a result.

The genuinely interesting version needs B2B receivables data with real replies, and would
have to be evaluated against whether the promised payment actually arrived.

## SWITCH_ROUTE

In the action space in the original plan; not implemented.

Route switching means sending a payment through a different PSP when one is degraded. Two
problems. Razorpay *is* the PSP here, so the action is not one the agent can take through
the adapter. And modelling it would require comparative success rates across acquirers,
which is exactly the data nobody publishes — so the simulator's route model would be pure
invention, and any lift it produced would be invention too.

`SWITCH_METHOD` — asking a customer whose card failed to try UPI — is retained, because
that is a real, exposed, modellable action.

## Learned retry timing

Retry timing is an empirical table: cause × issuer × hour. Fitting a model to it on
simulated data would recover the parameters we wrote and dress them as a finding. On real
data it is a good idea, and the table is the right place to plug it in.

## WhatsApp Business API, and Sarvam voice

WhatsApp Business API approval takes weeks and does not begin without a verified business.
Voice recovery in Hinglish is genuinely compelling — it is one of Razorpay's own listed
ideas — but it is a production integration, not an evaluation, and the copywriter already
generates the voice script.

Both channels are modelled with distinct costs and response rates, which is what the policy
needs. Neither is wired to a provider.

## Live mode

`RazorpayTestAdapter` raises unless the key ID starts with `rzp_test_`. Not a configuration
default — a constructor that refuses to exist. KYC was deliberately left incomplete so live
keys cannot be issued at all.

Two independent barriers for a system whose entire purpose is taking actions that move
money.
