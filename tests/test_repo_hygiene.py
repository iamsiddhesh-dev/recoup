"""Hygiene checks that run from the first commit.

This repo is public and handles real Razorpay test-mode credentials for the length of
the build. These tests are a cheap backstop behind GitHub's push protection: push
protection catches a credential on its way to GitHub, this catches one that reached the
working tree at all.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Razorpay key IDs are a fixed prefix followed by an alphanumeric body.
KEY_PATTERN = re.compile(r"rzp_" + r"(?:live|test)_" + r"[A-Za-z0-9]{10,}")

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".db", ".sqlite3", ".duckdb"}


def _candidate_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if SKIP_DIRS & set(path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path


def test_no_api_keys_in_working_tree():
    offenders = []
    for path in _candidate_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if KEY_PATTERN.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"Razorpay API key found in: {offenders}"


def test_gitignore_covers_secrets_and_local_notes():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", "PLAN.md"):
        assert entry in gitignore, f"{entry!r} must be gitignored"
