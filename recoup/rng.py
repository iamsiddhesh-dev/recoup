"""Deterministic random streams.

Every number this project reports has to be reproducible from a seed on a clean
clone — `recoup reproduce` asserts byte-identical output. Two things make that
harder than calling `random.seed()` once:

1. **Python's `hash()` is salted per process.** Deriving a substream seed from
   `hash(name)` gives different results on every run. `hashlib` does not have that
   problem, so it is used here even though it looks heavier than necessary.

2. **A single shared stream couples everything to draw order.** If customers and
   payments pull from one generator, adding a field to customer generation shifts
   every subsequent payment amount, and a diff that should have changed nothing
   changes every number in the report. Named substreams keep each concern's draws
   independent, so the simulator stays extensible without invalidating past runs.
"""

from __future__ import annotations

import hashlib
import random


def substream(seed: int, name: str) -> random.Random:
    """A named, independent random stream derived from the run seed.

    Deterministic across processes, machines and Python versions.
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    """Pick a key in proportion to its weight.

    Iterates in sorted key order rather than dict order so the result depends only
    on the seed and the weights, never on how the mapping happened to be built.
    """
    keys = sorted(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def lognormal_paise(rng: random.Random, mu: float, sigma: float, lo: int, hi: int) -> int:
    """Draw a ticket size in paise, clipped to a plausible band.

    `mu` is ln(median in paise). Clipping rather than resampling keeps the draw
    count fixed, which keeps the stream aligned when parameters are swept.
    """
    return max(lo, min(hi, int(rng.lognormvariate(mu, sigma))))
