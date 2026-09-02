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
