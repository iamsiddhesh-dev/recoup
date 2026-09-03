"""Arms, incremental recovery, ablation and sensitivity."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from recoup.agent.classify import Classifier
from recoup.agent.config import ComplianceConfig, PolicyConfig
from recoup.agent.llm.classifier import LLMFallbackClassifier
from recoup.agent.llm.client import LLMClient, LLMUnavailable
from recoup.agent.llm.copywriter import Copywriter
from recoup.eval.arms import build_arms
from recoup.eval.metrics import ArmMetrics, score
from recoup.eval.runner import Runner
from recoup.ledger.events import Ledger
from recoup.world.config import WorldConfig
from recoup.world.generator import build_batch

BASELINE = "naive_baseline"
AGENT = "recoup_agent"
AGENT_NO_LLM = "recoup_agent_no_llm"


def _llm_classifier(batch, offline: bool) -> Classifier | None:
    """Build the LLM-backed classifier, or None if no answer is available.

    Warmed once, before any arm runs: every distinct symptom the rules could not
    resolve goes up in a single request. On the committed seed that is three
    combinations for 51 payments, which is why this fits in a free tier at all.

    Returns None rather than raising when there is no key and no cache. The
    ablation then simply does not appear, and the run still produces its number —
    a missing model must not be able to stop the evaluation.
    """
    rules = Classifier()
    unresolved = rules.unresolved_fields(batch.failures)
    if not unresolved:
        return None

    fallback = LLMFallbackClassifier(client=LLMClient(offline=offline))

    try:
        if fallback.warm(unresolved) == 0:
            return None
    except LLMUnavailable:
        return None

    return Classifier(fallback=fallback)


def run_all(
    world: WorldConfig | None = None,
    ledger_path: str | Path = ":memory:",
    *,
    offline: bool = False,
    use_llm: bool = True,
    policy: PolicyConfig | None = None,
    compliance: ComplianceConfig | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> tuple[list[ArmMetrics], Ledger]:
    """Run every arm against one shared world and score them.

    The world is built once. Every arm sees the same failures in the same order
    with the same underlying luck, which is what makes the difference between them
    attributable to their decisions.

    `policy` and `compliance` can be supplied rather than loaded from disk, which
    is what Policy Studio does: change a cost or a cap, re-run the real
    evaluation, compare. The world is untouched, so the comparison stays honest —
    only the agent's beliefs and rules moved.
    """
    world = world or WorldConfig.load()
    policy = policy or PolicyConfig.load()
    compliance = compliance or ComplianceConfig.load()

    batch, population, issuers = build_batch(world)
    ledger = Ledger(ledger_path)

    llm = _llm_classifier(batch, offline) if use_llm else None

    # Warmed once for the whole experiment, and shared by every arm. Generating
    # copy is a rendering concern rather than a policy one, so holding it constant
    # keeps the arms comparable.
    copywriter = Copywriter(client=LLMClient(offline=offline)) if use_llm else Copywriter(
        use_llm=False
    )
    if use_llm:
        try:
            copywriter.warm()
        except LLMUnavailable:
            pass  # falls back to the hand-written templates

    runner = Runner(
        world=world,
        policy=policy,
        compliance=compliance,
        batch=batch,
        population=population,
        issuers=issuers,
        ledger=ledger,
        copywriter=copywriter,
    )

    arms = build_arms(policy, compliance, llm)

    results: list[ArmMetrics] = []
    for index, arm in enumerate(arms):

        def report(fraction: float, index=index, name=arm.name) -> None:
            if on_progress is not None:
                on_progress(name, (index + fraction) / len(arms))

        outcome = runner.run(arm, on_progress=report)
        results.append(
            score(ledger, arm.name, outcome.description, world.merchant_margin)
        )

    if on_progress is not None:
        on_progress("scoring", 1.0)

    return results, ledger
