"""Recoup command line entry point.

`make` is the interface documented in the README, but it is a thin wrapper over
this module: make is not installed by default on Windows, where this is developed,
so `python -m recoup <command>` is the canonical path and make simply calls it.

Commands are wired up as their components land. Anything not yet implemented says
so plainly rather than failing obscurely.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from recoup import __version__

COMMANDS: dict[str, str] = {
    "demo": "run a batch end to end and serve the control room",
    "eval": "print the arms table: gross, incremental, cost, net, refusals",
    "reproduce": "regenerate every committed figure from fixed seeds",
    "clean": "remove generated state (ledger, reports, caches)",
}


def _not_yet(name: str) -> Callable[[argparse.Namespace], int]:
    def run(_: argparse.Namespace) -> int:
        print(f"recoup {name}: not implemented yet", file=sys.stderr)
        return 1

    return run


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    name: _not_yet(name) for name in COMMANDS
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recoup",
        description="Revenue recovery agent for Indian payments.",
    )
    parser.add_argument("--version", action="version", version=f"recoup {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, help_text in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "--config",
            default="config",
            help="directory holding world.yaml, policy.yaml and compliance.yaml",
        )
        sub.add_argument(
            "--seed",
            type=int,
            default=None,
            help="override the seed in world.yaml; runs are deterministic given a seed",
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
