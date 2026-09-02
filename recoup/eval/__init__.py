"""Arms, incremental recovery, ablation and sensitivity."""

from __future__ import annotations

from pathlib import Path

from recoup.agent.config import ComplianceConfig, PolicyConfig
from recoup.eval.arms import build_arms
from recoup.eval.metrics import ArmMetrics, score
from recoup.eval.runner import Runner
from recoup.ledger.events import Ledger
from recoup.world.config import WorldConfig
from recoup.world.generator import build_batch

BASELINE = "naive_baseline"


def run_all(
    world: WorldConfig | None = None,
    ledger_path: str | Path = ":memory:",
) -> tuple[list[ArmMetrics], Ledger]:
    """Run every arm against one shared world and score them.

    The world is built once. Every arm sees the same failures in the same order
    with the same underlying luck, which is what makes the difference between them
    attributable to their decisions.
    """
    world = world or WorldConfig.load()
    policy = PolicyConfig.load()
    compliance = ComplianceConfig.load()

    batch, population, issuers = build_batch(world)
    ledger = Ledger(ledger_path)

    runner = Runner(
        world=world,
        policy=policy,
        compliance=compliance,
        batch=batch,
        population=population,
        issuers=issuers,
        ledger=ledger,
    )

    results: list[ArmMetrics] = []
    for arm in build_arms(policy, compliance):
        outcome = runner.run(arm)
        results.append(
            score(ledger, arm.name, outcome.description, world.merchant_margin)
        )

    return results, ledger
