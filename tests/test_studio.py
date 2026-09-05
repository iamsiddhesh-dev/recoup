"""Policy Studio: the write path.

Every other screen reads a finished run in milliseconds. This one asks for a new
one, which takes about twelve seconds, so it gets a different mechanism — a
background job with progress and a poll. Reads and writes rarely want the same
thing.

The parts worth testing are not the happy path. They are: untrusted numbers from
a browser being clamped rather than trusted, a failed run surfacing as state
rather than vanishing into a dead thread, and old runs being evicted along with
their files rather than leaking disk.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from recoup.agent.config import ComplianceConfig, PolicyConfig
from recoup.web.app import create_app
from recoup.web.jobs import JobRegistry
from recoup.web.studio import BY_KEY, KNOBS, apply, changed_from_default, clean, defaults


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------


def test_every_knob_can_be_read_from_the_live_config():
    """A knob whose reader is wrong silently shows a default nobody configured."""
    policy, compliance = PolicyConfig.load(), ComplianceConfig.load()

    for knob in KNOBS:
        value = knob.value_from(policy, compliance)
        assert value is not None
        assert knob.minimum <= value <= knob.maximum, f"{knob.key} default is out of range"


def test_defaults_match_the_configuration_on_disk():
    policy = PolicyConfig.load()
    values = defaults()

    assert values["ev_threshold_paise"] == policy.ev_threshold_paise
    assert values["whatsapp_cost"] == policy.action_costs["NUDGE_WHATSAPP"]


def test_unknown_keys_are_dropped():
    """The payload comes from a browser. Nothing in it is trusted."""
    assert clean({"ev_threshold_paise": 100, "drop_all_tables": 1}) == {
        "ev_threshold_paise": 100.0
    }


def test_values_are_clamped_rather_than_refused():
    """A slider that silently rejects is worse than one that stops at its end."""
    knob = BY_KEY["max_attempts"]

    assert clean({"max_attempts": 9999})["max_attempts"] == knob.maximum
    assert clean({"max_attempts": -5})["max_attempts"] == knob.minimum


def test_unparseable_values_are_ignored():
    assert clean({"max_attempts": "not a number", "ev_threshold_paise": "700"}) == {
        "ev_threshold_paise": 700.0
    }


def test_changed_knobs_are_identified():
    base = defaults()
    moved = {**base, "ev_threshold_paise": base["ev_threshold_paise"] + 1000}

    changed = changed_from_default(moved)

    assert set(changed) == {"ev_threshold_paise"}


# ---------------------------------------------------------------------------
# Applying overrides
# ---------------------------------------------------------------------------


def test_overrides_reach_the_right_configuration_fields():
    policy, compliance = apply(
        {
            "ev_threshold_paise": 4200,
            "whatsapp_cost": 99,
            "max_contacts": 5,
            "max_attempts": 7,
            "escalation_scarcity": 12345,
        }
    )

    assert policy.ev_threshold_paise == 4200
    assert policy.action_costs["NUDGE_WHATSAPP"] == 99
    assert policy.escalation_scarcity_premium == 12345
    assert compliance.contact.max_per_customer_per_7d == 5
    assert compliance.attempts.max_per_payment == 7


def test_applying_overrides_does_not_mutate_the_shared_config():
    """A studio run must not change what every other screen is reading."""
    before = PolicyConfig.load().action_costs["NUDGE_WHATSAPP"]

    apply({"whatsapp_cost": 999})

    assert PolicyConfig.load().action_costs["NUDGE_WHATSAPP"] == before


def test_the_world_is_not_adjustable():
    """Studio changes the agent, never the month it is measured against.

    If a viewer could change how often payments fail or how customers respond,
    they could produce any number they liked and the comparison would mean
    nothing.
    """
    world_ish = {"failure_rate", "method_mix", "nudge_response", "seed", "batch_size"}
    assert world_ish.isdisjoint(BY_KEY)


# ---------------------------------------------------------------------------
# The job registry
# ---------------------------------------------------------------------------


def test_a_job_runs_and_reports_completion(tmp_path):
    registry = JobRegistry(tmp_path)
    ran = []

    job = registry.submit({}, lambda j: ran.append(j.id))

    assert _wait(lambda: job.finished)
    assert job.status == "done"
    assert job.progress == 1.0
    assert ran == [job.id]


def test_a_failing_job_records_the_error_rather_than_disappearing(tmp_path):
    """A traceback on a background thread reaches nobody.

    Without this the browser polls a job that never finishes and the user learns
    nothing at all.
    """
    registry = JobRegistry(tmp_path)

    def explode(job):
        raise ValueError("issuer feed unavailable")

    job = registry.submit({}, explode)

    assert _wait(lambda: job.finished)
    assert job.status == "failed"
    assert "issuer feed unavailable" in job.error


def test_progress_is_visible_while_a_job_runs(tmp_path):
    registry = JobRegistry(tmp_path)
    release = time.monotonic() + 0.4

    def slow(job):
        job.progress = 0.5
        job.stage = "recoup_agent"
        while time.monotonic() < release:
            time.sleep(0.01)

    job = registry.submit({}, slow)

    assert _wait(lambda: job.progress == 0.5 and job.status == "running")
    assert job.stage == "recoup_agent"
    assert _wait(lambda: job.finished)


def test_old_jobs_are_evicted_with_their_ledgers(tmp_path):
    """Forgetting the job but keeping a 10MB file is how a cache becomes an incident."""
    registry = JobRegistry(tmp_path, max_jobs=2)

    def touch(job):
        from pathlib import Path

        Path(job.ledger_path).write_bytes(b"x" * 128)

    first = registry.submit({}, touch)
    assert _wait(lambda: first.finished)

    for _ in range(2):
        later = registry.submit({}, touch)
        assert _wait(lambda: later.finished)  # noqa: B023 — awaited before reassignment

    assert registry.get(first.id) is None

    from pathlib import Path

    assert not Path(first.ledger_path).exists()


def test_the_latest_job_is_retrievable(tmp_path):
    registry = JobRegistry(tmp_path)

    registry.submit({}, lambda j: None)
    second = registry.submit({}, lambda j: None)

    assert _wait(lambda: second.finished)
    assert registry.latest().id == second.id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _fake_evaluate(**kwargs):
    """Stand in for a real evaluation, so route tests do not take twelve seconds."""

    class FakeLedger:
        def close(self):
            pass

    class FakeMetrics:
        def __init__(self, arm):
            self.arm = arm
            self.recovered_paise = 100_000
            self.cost_paise = 5_000
            self.net_paise = 23_000
            self.contacts = 12
            self.vetoes = 3
            self.recovered_count = 4

    if kwargs.get("on_progress"):
        kwargs["on_progress"]("recoup_agent", 0.5)

    return [FakeMetrics("naive_baseline"), FakeMetrics("recoup_agent")], FakeLedger()


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(create_app(data_dir=tmp_path, evaluate=_fake_evaluate))


def test_the_studio_page_renders_every_knob(client):
    response = client.get("/studio")

    assert response.status_code == 200
    for knob in KNOBS:
        assert knob.label in response.text


def test_the_page_says_the_world_is_fixed(client):
    """The honesty of the comparison should be on the screen, not only in a doc.

    Asserting on the reason rather than the heading: the wording of the heading
    has changed twice, and both times this test failed while the claim it exists
    to protect was still on the page.
    """
    text = client.get("/studio").text

    assert "adjustable" in text
    assert "any number you liked" in text


def test_starting_a_run_returns_a_job_id(client):
    response = client.post("/studio/run", json=defaults())

    assert response.status_code == 200
    assert response.json()["job"]


def test_a_started_run_can_be_polled_to_completion(client):
    job_id = client.post("/studio/run", json=defaults()).json()["job"]

    for _ in range(200):
        body = client.get(f"/studio/status/{job_id}").json()
        if body["status"] in ("done", "failed"):
            break
        time.sleep(0.02)

    assert body["status"] == "done"
    assert body["progress"] == 1.0
    assert [r["arm"] for r in body["results"]] == ["naive_baseline", "recoup_agent"]


def test_overrides_survive_the_round_trip(client):
    payload = {**defaults(), "max_attempts": 7}
    job_id = client.post("/studio/run", json=payload).json()["job"]

    body = client.get(f"/studio/status/{job_id}").json()

    assert body["overrides"]["max_attempts"] == 7


def test_polling_an_unknown_job_is_a_404(client):
    assert client.get("/studio/status/does-not-exist").status_code == 404


def test_garbage_in_the_payload_does_not_break_the_run(client):
    """Clamped and filtered, not trusted — and never a 500."""
    response = client.post(
        "/studio/run", json={"max_attempts": 10_000, "nonsense": "x", "seed": 1}
    )

    assert response.status_code == 200
    job_id = response.json()["job"]
    overrides = client.get(f"/studio/status/{job_id}").json()["overrides"]

    assert overrides == {"max_attempts": BY_KEY["max_attempts"].maximum}
