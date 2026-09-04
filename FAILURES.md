# What broke

A running record of things this project got wrong, how they were caught, and what
changed. Kept because the interesting question about a simulated evaluation is not
whether it produced a number, but how many of its assumptions survived contact
with the real API.

---

## Ticket sizes were generated a hundred times too small

**Found by:** inspecting generated output before writing any tests against it.

`config/world.yaml` specifies amounts as a lognormal with `mu` = ln(median). The
amounts are in paise, but the values written were ln(median in **rupees**) — so
`mu: 6.4` produced a median UPI ticket of e^6.4 ≈ 600 paise, or **₹6**.

Nothing failed. The simulator ran, the distributions were internally consistent,
every mixture summed to 1.0. It would simply have reported that recovering a
payment was worth about six rupees, and every expected-value decision downstream
would have been computed against a pool a hundred times too small — which would
have made *every* action look negative-EV and the agent look correctly cautious.

**Fixed:** corrected to log-paise. `test_median_ticket_size_is_sane` now asserts
realised medians land in a plausible band per method, because this class of error
is invisible to type checks and to mixture validation.

---

## The simulator allowed retries that Razorpay cannot perform

**Found by:** writing the live adapter, which had to answer "which endpoint does a
retry actually call?"

There isn't one. Razorpay cannot re-run a failed payment. A charge with the
customer absent is only possible where standing authorisation exists — an
e-mandate, a tokenised card-on-file, a UPI Autopay mandate. Everything else needs
the customer to act again.

The simulated adapter would retry anything, and the world generated zero tokenised
instruments. So the agent would have been trained and measured on a retry policy
that is impossible in production, and the simulated result would have looked
*better* than reality precisely because it ignored the constraint. This is the
failure mode a simulated evaluation is most likely to hide, and it survived
undetected until real API surface forced the question.

**Fixed:** added `saved_instrument_rate` to the world, and enforced
`supports_silent_retry` identically in both adapters, with a test asserting they
refuse the same charges. The split is now 23.7% silently retryable versus 76.3%
requiring customer action — which changes what kind of product this is.

---

## Two invented error fields were wrong, and the first real payment proved it

**Found by:** running one live test-mode payment through a cloudflared tunnel and
comparing the webhook against the taxonomy.

Razorpay documents `source` and `step` exhaustively but publishes only examples of
`reason`, so most of `error_taxonomy` was plausible invention. The first genuine
`payment.failed` we captured disagreed on two of four fields for the one archetype
it covered:

| Field | Assumed | Actual |
|---|---|---|
| `error_reason` | `payment_failed` | `payment_failed` ✅ |
| `error_step` | `payment_authorization` | `payment_authorization` ✅ |
| `error_source` | `issuer_bank` | **`gateway`** |
| `error_code` | derived → `GATEWAY_ERROR` | **`BAD_REQUEST_ERROR`** |

The second one is the more useful finding. The generator derived the top-level
error code from `source` on the assumption that they encode the same thing. They
do not: Razorpay returned `BAD_REQUEST_ERROR` for a `gateway`-sourced failure. A
classifier keying on that derivation would have keyed on a relationship that does
not exist.

**Fixed:** `code` is now carried explicitly on taxonomy entries where observed
rather than inferred, the verified archetype is tagged `[OBSERVED]`, and the
capture is committed as `tests/fixtures/live_payment_failed.json` with a test
suite parsing it — including one asserting that code is *not* derivable from
source, so the assumption cannot quietly return.

**What this implies about the rest.** One archetype has been verified against live
test mode. Twenty are still `[ASSUMPTION]`. The first one checked was half wrong,
which is the honest prior for the others, and the reason those rows are tagged in
the config rather than presented as fact.

---

## The rule against re-debiting a revoked mandate was silently inert

**Found by:** the first test written against the compliance gate.

`config/compliance.yaml` keyed the hard stop as `MANDATE_REVOKED`. That is a
Razorpay *reason* string. The agent's taxonomy calls the cause `MANDATE_PROBLEM`.
So `hard_stop_for(MANDATE_PROBLEM)` looked up a key that did not exist, returned
`None`, and the gate waved the retry through.

The rule read correctly. It was in the right file, with a prose justification
explaining why re-debiting a withdrawn authorisation is unauthorised regardless of
outcome. It parsed. It validated. It did nothing.

This is the worst shape a bug can take in this project. Every other failure here
made something visibly wrong — amounts too small, retries that could not execute,
error fields that did not match. This one made a compliance control *look* present
while being absent, and it was the single most legally consequential rule in the
file. Nothing failed. Coverage did not drop. A demo would have looked fine.

**Fixed:** corrected the key, and added a load-time validator asserting every
`hard_stops` key is a real `FailureCause` — a rule that names something the system
has never heard of now refuses to load rather than quietly matching nothing.

**What it changed about how the rest is written.** Config that is looked up by
string is config that can miss silently. Everywhere a key has to correspond to
something in code, that correspondence is now checked at load time rather than
assumed, and the same reasoning is why `world.yaml` mixtures are validated on load
rather than trusted.

---

## The baseline recovered nothing, because it never executed anything

**Found by:** the first evaluation run, which reported ₹0 for the naive arm.

The runner scheduled a chosen action for later, and when that moment arrived it
called `decide()` again instead of carrying out the decision already made. The
policy proposes offsets relative to *now*, so asking again at the scheduled time
simply produced another future time. The naive arm deferred its first retry
forever and executed nothing across a thirty-day horizon.

The failure was loud — a zero is hard to miss — but it would have been easy to
paper over. Any lift measured against a zero baseline is infinite, and a table
showing the agent recovering money while the baseline recovers none reads like a
spectacular result rather than a broken control arm.

**Fixed:** scheduled actions carry their decision. `ACT` executes what was
decided; `RECONSIDER` is a separate task that decides afresh. The runner also caps
decisions per payment, so a bug in the compliance caps cannot turn into an
unbounded loop.

---

## The agent sent 1,582 emails and nothing else

**Found by:** reading the per-action breakdown rather than the summary table.

Every channel was priced with the same response probability, so expected value
differed only by cost, and email is the cheapest by an order of magnitude. The
agent emailed everyone, forever, and the arithmetic was correct — the model was
wrong. In India that is close to the worst available choice: transactional email
goes unopened where WhatsApp gets read.

The summary table showed a healthy recovery number and gave no hint of this. It
was visible only in `actions_by_kind`.

**Fixed:** per-channel response multipliers, in the world and in the agent's
priors (deliberately not identical). The agent now uses a real mix — voice,
WhatsApp, SMS — and recovers more from *fewer* contacts than a contact-only arm.

---

## The refusal list was 80,945 rows of noise

**Found by:** the veto count being larger than the number of decisions.

Two causes. The gate screens every retry-time variant, so one refused payment
produced fifty near-identical vetoes differing only by scheduled hour. And the
policy proposed human escalation for nearly every payment, which compliance
refused ~1,450 times, because the arithmetic said escalation was worth it above
about ₹1,200 while the operational cap was ₹50,000.

Both made the refusal list unreadable, which matters: "here is what we
deliberately did not touch, and why" is a deliverable, and a deliverable nobody
can read is not one.

**Fixed:** vetoes are deduplicated by rule per decision, options below the
expected-value threshold are never put in front of the gate, and the policy skips
channels the customer has not consented to. Escalation now carries a **scarcity
premium** — fifty human slots against sixteen hundred failures means spending one
on a small payment forecloses spending it on a large one, and that opportunity
cost is real even though nothing invoices for it. Refusals fell from 80,945 to
1,030, and now read as a compliance report.

---

## The LLM added exactly nothing, twice — for two different reasons

**Found by:** running the ablation instead of assuming.

**First time, it could not have helped.** The ablation arms came back byte-identical.
The cause was a gap in the policy, not the model: `_nudge_candidate` priced a
contact using amount, channel and prior contacts, and *not* the failure cause. So
for any payment that could not be retried — 76% of them — knowing why it failed
changed no decision. A classifier can only be worth something if something
downstream reads its output.

Fixed by adding per-cause response multipliers. Asking someone to complete a
payment while their bank is down is close to useless; asking someone who abandoned
an OTP usually works. Now the cause moves the arithmetic.

**Second time, it genuinely did not help.** With the gap closed, the measured
result on the committed seed:

| | Recovered | Contacts |
|---|---|---|
| `recoup_agent_no_llm` | 3 (₹9,521) | 29 |
| `recoup_agent` | 2 (₹9,282) | 28 |

Across the whole run: −₹239 and −1 contact. Noise.

The model is not wrong — it classified all three unresolved symptom types
correctly, verified against ground truth the classifier cannot see. It is simply
not worth much *here*, because 46 of the 51 failures it resolves are bank and
wallet outages, where the correct action is "do not chase", and that is close to
what the pessimistic unknown-cause default already produced. It bought precision
on a population where the default was already approximately right.

**This is reported rather than fixed.** The obvious way to manufacture a win would
be to lower `unknown_cause_response` until the default is wrong enough for the
model to look useful. That would be tuning the baseline to flatter the result, and
the number it produced would mean nothing.

**When it would matter,** and this is a testable claim rather than a hedge: the
fallback's value scales with how *mixed* the unresolved population is. Here it is
90% outages, so one guess fits nearly all of it. A population split evenly between
"abandoned, chase them" and "outage, leave them alone" would punish a single
default badly, and that is the workload where classification earns its cost. The
sensitivity sweep can vary that share directly.

---

## A sensitivity axis that measured nothing, and reported it as a finding

**Found by:** reading the first tornado and noticing one bar was exactly zero.

The sweep moves each load-bearing assumption to either end of a plausible band and
re-runs the whole evaluation. Contribution margin came back with a swing of
**exactly ₹0** — identical results at 0.7× and 1.3×.

Zero is a suspicious number. A real insensitivity would wobble.

The axis scaled `world.merchant_margin`. The agent prices every decision on
`policy.assumed_margin`, which is a different field — its *belief* about margin,
not the truth. So the axis changed what the scoreboard multiplied by at the end
and nothing the agent could see, and the decisions were byte-identical by
construction.

Left alone this would have appeared in the writeup as "the result is completely
insensitive to margin", which is a strong and entirely false claim. It is exactly
the kind of error a sensitivity analysis is supposed to catch, arriving inside the
sensitivity analysis.

**Fixed:** axes now receive the policy as well as the world, and the margin axis
moves both — a merchant whose margin genuinely differs would know it. The swing is
₹18,255, and margin sits fifth of seven rather than last. A test now asserts every
axis actually mutates something, so an inert axis fails the build instead of
producing a confident zero.

**The general shape.** A measurement that reports "no effect" deserves more
scrutiny than one reporting a large effect, not less. A large effect is usually
real; no effect is often the instrument being disconnected.

---

## Four payments quietly consumed the entire human-review budget

**Found by:** reading a case brief while building the explainer, and noticing it
listed `ESCALATE_HUMAN` twelve times for one payment.

Escalation hands a case to a person. The executor records it, writes a `STOPPED`
event, and returns an `Execution` — with `succeeded=False`, because no money came
back. The runner's rule for what to do next was:

```python
if not executed.succeeded and when < horizon:
    reschedule(RECONSIDER)
```

So every escalated payment came back a minute later. The policy proposed
escalation again — nothing about the situation had changed — the gate allowed it,
and the loop ran until the per-payment decision cap of twelve stopped it.

The result on the committed seed: **48 escalations across just four payments,
twelve each, against a run cap of fifty.** ₹5,280 spent on forty-four handovers of
cases that had already been handed over. Two more repeats on any one of them and
the pool would have been exhausted, and every subsequent payment that genuinely
warranted a human would have been refused one.

**Why nothing caught it.** Every individual piece behaved correctly. The executor
did record the escalation and did stop touching the payment. The compliance gate
did enforce its run cap — 48 is under 50, so it never fired. The policy did price
escalation correctly each time it was asked. `actions_by_kind` showed
`ESCALATE_HUMAN: 48` and 48 looks like forty-eight escalated payments, which would
have been a reasonable number. The failure was only visible in the *distribution*,
and nothing summarised that.

It also cost nothing detectable in recovery, because escalation never recovers
money in this simulator — so the totals moved by ₹5,280 of cost and not one rupee
of revenue. A bug that only makes you slightly poorer is much harder to notice
than one that breaks something.

**Fixed in two places, deliberately.**

`Execution` now carries `terminal` alongside `succeeded`. They are different
questions — handing a case to a human did not recover the money and is also not a
failed attempt worth retrying — and collapsing them into one boolean is what
created the loop. That is the root cause.

The compliance gate separately refuses a second escalation on the same payment,
via `max_escalations_per_payment`. This is defence in depth and it is the more
important of the two: "one payment, one human" is a *policy*, and a policy that
holds only because of the shape of a scheduling loop somewhere else is not
enforced, it is coincidental. The previous entry in this file is about a
compliance rule that looked present and did nothing; the lesson generalises to
rules that are absent because another layer happens to make them unnecessary.

That rule does not fire on the committed seed — with the runner fixed, a second
escalation is never proposed. A rule that never fires is exactly the shape of the
mandate bug above, so it is covered by unit tests that construct the condition
directly and assert the veto, rather than being trusted because the run looks
clean.

**What changed in the numbers.** Recovery, contacts and refusals are identical to
the rupee: escalation never recovered anything, so removing forty-four of them
removed only cost. Agent spend fell from ₹8,336 to ₹3,056 and net margin rose from
₹1,43,347 to ₹1,48,627. The headline incremental figure of ₹3,94,791 is unchanged,
because it is measured on recovery.

**The general shape.** A cap on a shared resource is not a cap on any one consumer
of it. `max_escalations_per_run` was doing exactly what it said and was still the
wrong rule on its own, because the interesting failure was not "too many
escalations" but "too few payments receiving them".
