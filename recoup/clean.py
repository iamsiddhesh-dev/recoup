"""Removing generated state, with the expensive things behind a flag.

`clean` is usually a one-line `rm -rf` over everything not tracked. That is wrong
here, because two of the generated directories are *not* cheap to regenerate and
one of them is committed.

The split is by cost of recreation, not by whether something is generated:

- **Free.** `data/` is a 35-second evaluation and the Python caches cost nothing.
  Removed by default.
- **Expensive.** `reports/sensitivity.json` is a twenty-minute sweep — fifteen
  full evaluations. Behind `--reports`.
- **Possibly unrecoverable.** `cache/llm/` needs API keys the cloner may not have,
  and Gemini Flash allows twenty requests a day, so deleting it can mean waiting
  until tomorrow. It is also committed, and it is what makes the reported numbers
  reproducible offline — removing it locally is a step toward removing it from the
  repository. Behind `--llm-cache`.

A `clean` that silently destroys a day of quota is a `clean` people stop running.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

# Directories whose *contents* are generated, matched anywhere under the root.
CACHE_DIRS = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")

SKIP_WALK = {".git", ".venv", "venv", "node_modules"}


@dataclass(frozen=True)
class Target:
    path: Path
    why: str

    @property
    def size(self) -> int:
        if self.path.is_file():
            return self.path.stat().st_size
        return sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())


def _cache_dirs(root: Path) -> list[Path]:
    """Walk once, pruning the directories that would dominate the walk."""
    found: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name in SKIP_WALK:
                continue
            if entry.name in CACHE_DIRS:
                found.append(entry)
            else:
                stack.append(entry)
    return sorted(found)


def targets(
    root: str | Path = ".",
    *,
    llm_cache: bool = False,
    reports: bool = False,
) -> list[Target]:
    root = Path(root)
    found: list[Target] = []

    data = root / "data"
    if data.exists():
        found.append(Target(data, "a run, regenerable in ~35s with `recoup demo`"))

    for path in _cache_dirs(root):
        found.append(Target(path, "interpreter and tool cache"))

    if reports:
        directory = root / "reports"
        if directory.exists():
            found.append(Target(directory, "~20 min to regenerate with `recoup sweep`"))

    if llm_cache:
        directory = root / "cache" / "llm"
        if directory.exists():
            found.append(
                Target(directory, "committed; needs API keys and daily quota to rebuild")
            )

    return found


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def remove(target: Target) -> None:
    if target.path.is_dir():
        shutil.rmtree(target.path, ignore_errors=True)
    else:
        target.path.unlink(missing_ok=True)
