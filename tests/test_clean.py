"""`clean` must not destroy anything it cannot cheaply rebuild.

The whole design of this command is the default set. `data/` is a 35-second
re-run; `cache/llm/` is committed, needs API keys to rebuild, and Gemini Flash
allows twenty requests a day, so a careless `clean` can cost a day of work and
break the repo's offline reproducibility at the same time.
"""

from __future__ import annotations

from pathlib import Path

from recoup.clean import human, remove, targets


def _project(root: Path) -> Path:
    (root / "data").mkdir()
    (root / "data" / "run.db").write_bytes(b"x" * 100)
    (root / "cache" / "llm").mkdir(parents=True)
    (root / "cache" / "llm" / "nudge-copy.json").write_text("{}", encoding="utf-8")
    (root / "reports").mkdir()
    (root / "reports" / "sensitivity.json").write_text("{}", encoding="utf-8")
    (root / "recoup" / "__pycache__").mkdir(parents=True)
    (root / "recoup" / "__pycache__" / "a.pyc").write_bytes(b"x")
    (root / "recoup" / "keep.py").write_text("# source", encoding="utf-8")
    return root


def _paths(root: Path, **kwargs) -> set[str]:
    return {t.path.relative_to(root).as_posix() for t in targets(root, **kwargs)}


# ---------------------------------------------------------------------------
# What the default removes, and what it refuses to
# ---------------------------------------------------------------------------


def test_the_default_removes_only_what_is_free_to_rebuild(tmp_path):
    found = _paths(_project(tmp_path))

    assert "data" in found
    assert "recoup/__pycache__" in found


def test_the_llm_cache_survives_a_plain_clean(tmp_path):
    """Committed, quota-limited, and the reason the numbers reproduce offline."""
    assert "cache/llm" not in _paths(_project(tmp_path))


def test_reports_survive_a_plain_clean(tmp_path):
    """Twenty minutes of sweeping is not something to lose by default."""
    assert "reports" not in _paths(_project(tmp_path))


def test_the_expensive_directories_are_removable_when_asked(tmp_path):
    found = _paths(_project(tmp_path), llm_cache=True, reports=True)

    assert "cache/llm" in found
    assert "reports" in found


def test_source_is_never_a_target(tmp_path):
    root = _project(tmp_path)

    for target in targets(root, llm_cache=True, reports=True):
        assert target.path.name != "keep.py"
        assert "recoup/keep.py" not in target.path.as_posix()


def test_every_target_says_why_it_is_safe(tmp_path):
    """The output is the only place a reader learns what they are about to lose."""
    for target in targets(_project(tmp_path), llm_cache=True, reports=True):
        assert target.why.strip()


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------


def test_caches_are_found_at_any_depth(tmp_path):
    root = _project(tmp_path)
    deep = root / "recoup" / "agent" / "llm" / "__pycache__"
    deep.mkdir(parents=True)

    assert "recoup/agent/llm/__pycache__" in _paths(root)


def test_the_walk_does_not_descend_into_git_or_the_virtualenv(tmp_path):
    """Both contain thousands of directories and neither is ours to clean."""
    root = _project(tmp_path)
    (root / ".venv" / "lib" / "__pycache__").mkdir(parents=True)
    (root / ".git" / "__pycache__").mkdir(parents=True)

    found = _paths(root)

    assert not any(path.startswith((".venv", ".git")) for path in found)


def test_nothing_to_clean_is_not_an_error(tmp_path):
    assert targets(tmp_path) == []


# ---------------------------------------------------------------------------
# Removing and reporting
# ---------------------------------------------------------------------------


def test_removing_a_target_removes_its_contents(tmp_path):
    root = _project(tmp_path)
    data = next(t for t in targets(root) if t.path.name == "data")

    remove(data)

    assert not (root / "data").exists()
    assert (root / "cache" / "llm" / "nudge-copy.json").exists()


def test_size_counts_the_whole_tree(tmp_path):
    root = _project(tmp_path)
    data = next(t for t in targets(root) if t.path.name == "data")

    assert data.size == 100


def test_sizes_are_reported_in_units_a_person_reads():
    assert human(512) == "512B"
    assert human(2048) == "2.0KB"
    assert human(5 * 1024 * 1024) == "5.0MB"
