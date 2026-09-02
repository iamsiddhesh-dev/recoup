"""The wall: agent code may not reach into the simulated world.

Recoup's central claim is that its measured recovery is not self-graded — the agent
sees only what Razorpay would actually send it, never the simulator's ground truth
about whether a payment was always going to succeed.

A claim resting on discipline is worth nothing over a sixteen-day build, so it is
enforced mechanically: any import of `recoup.world` from anywhere under
`recoup/agent/` fails the build.

`recoup/adapters/` is deliberately exempt. The simulated adapter is the seam and
must import the world; that is its job. The agent talks to the adapter protocol.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / "recoup" / "agent"

FORBIDDEN_ROOT = "recoup.world"


def _module_name(path: Path) -> str:
    """Dotted module name for a file inside the package, e.g. recoup.agent.policy."""
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(module: str | None, level: int, importer: str) -> str:
    """Resolve a possibly-relative import to an absolute dotted module name."""
    if level == 0:
        return module or ""
    package_parts = importer.split(".")[:-1]
    base = package_parts[: len(package_parts) - (level - 1)] if level > 1 else package_parts
    return ".".join([*base, module]) if module else ".".join(base)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    importer = _module_name(path)
    found: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(_resolve(node.module, node.level, importer))

    return found


def test_agent_never_imports_the_world():
    if not AGENT_DIR.exists():
        return  # nothing to guard yet

    violations: list[str] = []
    for path in sorted(AGENT_DIR.rglob("*.py")):
        for imported in _imports(path):
            if imported == FORBIDDEN_ROOT or imported.startswith(FORBIDDEN_ROOT + "."):
                violations.append(f"{path.relative_to(REPO_ROOT)} imports {imported}")

    assert not violations, (
        "The agent reached into the simulated world, which invalidates every "
        "recovery number this project reports:\n  " + "\n  ".join(violations)
    )
