"""Hygiene checks that run from the first commit.

This repo is public and handles real Razorpay test-mode credentials for the length of
the build. These tests are a cheap backstop behind GitHub's push protection: push
protection catches a credential on its way to GitHub, this catches one that reached the
working tree at all.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Razorpay key IDs are a fixed prefix followed by an alphanumeric body.
KEY_PATTERN = re.compile(r"rzp_" + r"(?:live|test)_" + r"[A-Za-z0-9]{10,}")

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".db", ".sqlite3", ".duckdb"}


def _candidate_files():
    """Everything git would actually publish: tracked, plus untracked-not-ignored.

    Deliberately *not* a walk of the working tree. `.env` holds live test-mode
    credentials by design — that is what the file is for — and it is gitignored,
    so it can never reach the repository. Scanning it would make this test fail
    for anyone who configured the project correctly, and a test that fails when
    you do the right thing gets deleted rather than fixed.

    `--exclude-standard` applies .gitignore, so a secret in a new untracked file
    is still caught, which is the case that actually matters.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        path = REPO_ROOT / line
        if not path.is_file():
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
