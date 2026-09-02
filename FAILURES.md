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
