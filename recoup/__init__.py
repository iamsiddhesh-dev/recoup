"""Recoup — a revenue recovery agent for Indian payments.

Package layout, and the one rule that matters:

    world/      the simulated world. Holds ALL ground truth.
    adapters/   the seam. One protocol, two implementations (simulated, Razorpay
                test mode) so the agent cannot tell which world it is in.
    agent/      the product. Sees only webhook-shaped events.
    ledger/     append-only decision log, deterministically replayable.
    eval/       arms, incremental recovery, ablation, sensitivity.
    web/        the control room.

`agent/` must never import from `world/`. That is the whole basis of the claim
that measured recovery is not self-graded, and it is enforced by
tests/test_no_ground_truth_leak.py rather than left to discipline.
"""

__version__ = "0.1.0"
