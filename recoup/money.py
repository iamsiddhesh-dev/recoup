"""Money, formatted the way the merchants reading it write it.

Two rules, and both were being broken in seven different files.

**Indian digit grouping.** ₹5,41,724 — the last three digits, then twos. Python's
`,` format specifier groups in threes and produces ₹541,724, which is not how a
rupee figure is written anywhere this product operates. It is the sort of detail
that costs nothing and, got wrong, tells an Indian reader in the first second that
nobody thought about them.

**One implementation.** The formatting was inlined at every call site as
`f"₹{paise / 100:,.0f}"`, which meant the README, the terminal tables, the web
screens, the decision reasons and the customer-facing copy could disagree about
the same amount — and did. The store module already states the principle for
computation: a number shown to a merchant should be produced in one place. This is
that place for display.

Amounts are stored in paise everywhere and converted only here.
"""

from __future__ import annotations

RUPEE = "₹"
DASH = "—"


def group(digits: str) -> str:
    """Indian digit grouping over a string of digits: last three, then twos.

    Takes and returns a string rather than a number so the caller decides how the
    value was rounded. Doing it in here would hide a rounding choice inside a
    formatting function.
    """
    if len(digits) <= 3:
        return digits

    head, tail = digits[:-3], digits[-3:]

    pairs: list[str] = []
    while len(head) > 2:
        pairs.append(head[-2:])
        head = head[:-2]
    if head:
        pairs.append(head)

    return ",".join(reversed(pairs)) + "," + tail


def rupees(paise: float | int | None, *, precise_below: int = 0, symbol: bool = True) -> str:
    """Paise in, a rupee string out.

    `precise_below` keeps two decimals for amounts under that many paise. A
    WhatsApp message costs 35 paise, and rounding it to ₹0 makes the
    expected-value sum on Case Detail look like it does not add up — the one thing
    that screen exists to demonstrate. Large amounts drop the decimals, because
    ₹12,041.00 is noise in a column of them.

    The sign leads: -₹238, not ₹-238. It is a negative amount of money, not an
    amount of negative money, and the ablation result is the first place a reader
    meets one.
    """
    if paise is None:
        return DASH

    magnitude = abs(paise)
    prefix = "-" if paise < 0 else ""
    unit = RUPEE if symbol else ""

    # Exactly zero stays "₹0". It is the one small amount where the decimals say
    # nothing, and it appears in whole columns of costs that were never incurred.
    if precise_below and 0 < magnitude < precise_below:
        whole, _, fraction = f"{magnitude / 100:.2f}".partition(".")
        return f"{prefix}{unit}{group(whole)}.{fraction}"

    return f"{prefix}{unit}{group(f'{magnitude / 100:.0f}')}"
