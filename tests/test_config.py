"""The config files are the model. If they are wrong, everything downstream is.

world.yaml carries every modelling assumption in the project, deliberately in one
place so it can be audited and swept. That only works if the file is actually
consistent — mixtures that sum to one, causes that exist in every table that
references them. A typo in a weight is a silent bias in the results, so it fails
here instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
TOLERANCE = 1e-9


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def world() -> dict:
    return _load("world.yaml")


@pytest.fixture(scope="module")
def policy() -> dict:
    return _load("policy.yaml")


@pytest.fixture(scope="module")
def compliance() -> dict:
    return _load("compliance.yaml")


def test_all_configs_parse(world, policy, compliance):
    for name, cfg in (("world", world), ("policy", policy), ("compliance", compliance)):
        assert isinstance(cfg, dict) and cfg, f"{name}.yaml did not parse into a mapping"


def test_method_mix_is_a_distribution(world):
    total = sum(world["method_mix"].values())
    assert abs(total - 1.0) < TOLERANCE, f"method_mix sums to {total}, not 1.0"


def test_every_method_has_amounts_and_failure_rate(world):
    methods = set(world["method_mix"])
    assert set(world["amounts"]) == methods
    assert set(world["failure_rate"]) == methods
    assert set(world["error_taxonomy"]) == methods


@pytest.mark.parametrize("method", ["upi", "card", "netbanking", "wallet", "emandate"])
def test_error_taxonomy_weights_sum_to_one(world, method):
    total = sum(entry["weight"] for entry in world["error_taxonomy"][method])
    assert abs(total - 1.0) < 1e-9, f"{method} failure weights sum to {total}, not 1.0"


def test_every_cause_has_a_recovery_probability(world):
    causes = {
        entry["cause"]
        for entries in world["error_taxonomy"].values()
        for entry in entries
    }
    known = set(world["recovery"]["base_probability"])
    assert causes <= known, f"causes with no recovery probability: {sorted(causes - known)}"


def test_agent_prior_covers_the_same_causes(world, policy):
    world_causes = set(world["recovery"]["base_probability"])
    prior_causes = set(policy["prior_recovery_probability"])
    assert world_causes == prior_causes, (
        "the agent's prior and the world's truth must cover the same causes; "
        f"world-only={sorted(world_causes - prior_causes)} "
        f"prior-only={sorted(prior_causes - world_causes)}"
    )


def test_agent_prior_does_not_secretly_match_the_world(world, policy):
    """The agent must start with a wrong prior, or the evaluation proves nothing."""
    truth = world["recovery"]["base_probability"]
    prior = policy["prior_recovery_probability"]
    differing = [c for c in truth if abs(truth[c] - prior[c]) > TOLERANCE]
    assert differing, (
        "the agent's prior is identical to the world's ground truth — it would be "
        "starting with the answer, and measured recovery would be meaningless"
    )


def test_hour_multiplier_covers_the_day(world):
    hours = world["recovery"]["hour_multiplier"]
    assert len(hours) == 24, f"hour_multiplier has {len(hours)} entries, expected 24"
    assert all(m > 0 for m in hours)


def test_customer_segments_are_a_distribution(world):
    total = sum(seg["share"] for seg in world["customers"]["segments"].values())
    assert abs(total - 1.0) < TOLERANCE, f"customer segment shares sum to {total}, not 1.0"


def test_issuer_shares_are_a_distribution(world):
    total = sum(issuer["share"] for issuer in world["issuers"])
    assert abs(total - 1.0) < 1e-9, f"issuer shares sum to {total}, not 1.0"


def test_language_mix_is_a_distribution(world):
    total = sum(world["customers"]["language"].values())
    assert abs(total - 1.0) < TOLERANCE, f"language mix sums to {total}, not 1.0"


def test_every_policy_action_has_a_cost(policy):
    costs = policy["action_costs"]
    assert costs["STOP"] == 0, "stopping must be free, or the agent is penalised for restraint"
    assert all(v >= 0 for v in costs.values()), "negative action cost would be a free lunch"


def test_risk_blocked_is_never_retried(compliance):
    assert compliance["hard_stops"]["RISK_BLOCKED"]["retry"] is False
    assert compliance["hard_stops"]["RISK_BLOCKED"]["contact"] is False


def test_execution_requires_test_mode(compliance):
    assert compliance["execution"]["require_test_mode"] is True
    assert compliance["execution"]["require_idempotency_key"] is True
