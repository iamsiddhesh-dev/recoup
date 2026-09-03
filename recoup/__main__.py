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

from dotenv import load_dotenv

from recoup import __version__

COMMANDS: dict[str, str] = {
    "demo": "run a batch end to end and serve the control room",
    "eval": "print the arms table: gross, incremental, cost, net, refusals",
    "reproduce": "regenerate every committed figure from fixed seeds",
    "serve": "run the web server and webhook receiver",
    "probe": "create a real test-mode payment link, to prove the live adapter works",
    "clean": "remove generated state (ledger, reports, caches)",
}


def _not_yet(name: str) -> Callable[[argparse.Namespace], int]:
    def run(_: argparse.Namespace) -> int:
        print(f"recoup {name}: not implemented yet", file=sys.stderr)
        return 1

    return run


def _eval(args: argparse.Namespace) -> int:
    from recoup.eval import BASELINE, run_all
    from recoup.eval.metrics import Comparison, rupees, table
    from recoup.world.config import WorldConfig

    world = WorldConfig.load()
    if args.seed is not None:
        world.run.seed = args.seed

    results, ledger = run_all(world)
    print(table(results, BASELINE))

    baseline = next(m for m in results if m.arm == BASELINE)
    agent = next((m for m in results if m.arm == "recoup_agent"), None)

    if agent is not None:
        comparison = Comparison(arm=agent, baseline=baseline)
        print()
        print(
            f"{rupees(agent.recovered_paise)} recovered, of which "
            f"{rupees(comparison.incremental_paise)} is incremental over naive retry "
            f"({comparison.lift:+.1%}), at a cost of {agent.contacts:,} contacts, "
            f"with {agent.vetoes:,} actions refused by compliance."
        )

    ledger.close()
    return 0


def _demo(args: argparse.Namespace) -> int:
    """Generate a run, persist it, then serve the control room from it.

    Deliberately two phases. The evaluation takes ~35 seconds and a page load
    cannot, so the run is written to disk once and every screen reads it back.
    """
    from recoup.eval import run_all
    from recoup.eval.metrics import table
    from recoup.eval.store import ledger_path, save_summary
    from recoup.web.app import serve
    from recoup.world.config import WorldConfig

    world = WorldConfig.load()
    if args.seed is not None:
        world.run.seed = args.seed

    path = ledger_path("data")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    print(f"generating a run (seed {world.run.seed})…")
    results, ledger = run_all(world, ledger_path=path)
    ledger.close()

    save_summary(
        results,
        seed=world.run.seed,
        horizon_days=world.run.horizon_days,
        batch_size=world.run.batch_size,
        margin=world.merchant_margin,
    )

    print()
    print(table(results, "naive_baseline"))
    print()

    serve(host=args.host, port=args.port)
    return 0


def _serve(args: argparse.Namespace) -> int:
    from recoup.web.app import serve

    serve(host=args.host, port=args.port)
    return 0


def _probe(args: argparse.Namespace) -> int:
    """Prove the live adapter works, end to end, against real Razorpay test mode.

    Creates a payment link for a small amount. Opening it and paying with failing
    credentials produces a genuine `payment.failed` webhook — which is the whole
    point: the seam is only credible if the live side has actually been exercised.
    """
    from recoup.adapters.base import LinkRequest, TestModeViolation
    from recoup.adapters.razorpay_test import RazorpayTestAdapter

    try:
        adapter = RazorpayTestAdapter()
    except TestModeViolation as exc:
        print(f"cannot probe: {exc}", file=sys.stderr)
        return 1

    with adapter:
        link = adapter.create_recovery_link(
            LinkRequest(
                payment_id="probe",
                order_id="probe",
                amount=args.amount,
                customer_ref="probe",
                description="Recoup live-adapter probe",
                idempotency_key=f"probe-{args.amount}-{args.tag}",
            )
        )

    print(f"payment link created: {link.url}")
    print()
    print("Open it and pay. To produce a failure webhook:")
    print("  UPI   use  failure@razorpay")
    print("  card  use  4100 2800 0000 1007, then an OTP of FEWER than 4 digits")
    print()
    print("Then check  http://127.0.0.1:8000/webhooks/recent")
    return 0


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    name: _not_yet(name) for name in COMMANDS
}
HANDLERS["serve"] = _serve
HANDLERS["probe"] = _probe
HANDLERS["eval"] = _eval
HANDLERS["demo"] = _demo


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

        if name in ("serve", "demo"):
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8000)

        if name == "probe":
            sub.add_argument(
                "--amount", type=int, default=10000, help="in paise (default ₹100)"
            )
            sub.add_argument(
                "--tag", default="1", help="change to create a fresh link"
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
