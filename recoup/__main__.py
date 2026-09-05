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
from pathlib import Path

from dotenv import load_dotenv

from recoup import __version__

COMMANDS: dict[str, str] = {
    "demo": "run a batch end to end and serve the product",
    "eval": "print the arms table: gross, incremental, cost, net, refusals",
    "reproduce": "regenerate every committed figure from fixed seeds",
    "sweep": "re-run the evaluation with each assumption moved, and report the swing",
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

    # Hash each arm's stream before closing, so the audit screen can later prove
    # the file has not changed since it was written.
    digests = {m.arm: ledger.digest(m.arm) for m in results}
    ledger.close()

    save_summary(
        results,
        seed=world.run.seed,
        horizon_days=world.run.horizon_days,
        batch_size=world.run.batch_size,
        margin=world.merchant_margin,
        digests=digests,
    )

    _write_frames(path)
    _write_explanations(path)

    print()
    print(table(results, "naive_baseline"))
    print()

    serve(host=args.host, port=args.port)
    return 0


def _sweep(args: argparse.Namespace) -> int:
    """Re-run the evaluation with each load-bearing assumption moved.

    Slow on purpose — it is doing the work rather than estimating it — and the
    result is committed so a reader never has to.
    """
    import time

    from recoup.eval.sensitivity import run_sweep, save, table

    started = time.monotonic()

    def progress(label: str, done: int, total: int) -> None:
        elapsed = time.monotonic() - started
        eta = (elapsed / done) * (total - done) if done else 0
        print(f"  [{done:>2}/{total}] {label:<28} {eta / 60:>4.1f}m left", flush=True)

    print("sweeping assumptions — this re-runs the whole evaluation at each point")
    result = run_sweep(on_progress=progress)

    path = save(result)
    print()
    print(table(result, args.metric))
    print()
    print(f"written to {path}")
    return 0


def _write_frames(ledger_file: Path, arm: str = "recoup_agent") -> None:
    """Precompute the scrubber's frames beside the run.

    Folding the stream costs ~400ms. Once here, never on a page load.
    """
    from recoup.eval.store import save_frames
    from recoup.ledger.events import Ledger
    from recoup.web.timeline import build_frames, to_payload

    with Ledger(ledger_file) as ledger:
        frames = build_frames(ledger, arm)

    if frames:
        save_frames(to_payload(frames))
        print(f"replay: {len(frames)} frames across {frames[-1].day:.0f} days")


def _write_explanations(ledger_file: Path, arm: str = "recoup_agent") -> None:
    """Narrate a handful of representative cases, once, after the run is written.

    Done here rather than lazily on a page view for the same reason every other
    model call is batched: one request covers the whole selection, so a judge
    clicking through cases spends no quota and two visits to the same case cannot
    disagree with each other.

    Failure is silent by design. Every case already has a deterministic
    explanation, so a missing model costs prose, not information — and `demo` must
    not fall over on the last step because a free tier was busy.
    """
    from recoup.agent.llm.client import LLMUnavailable
    from recoup.agent.llm.explainer import Explainer
    from recoup.eval.store import save_explanations
    from recoup.ledger.events import Ledger
    from recoup.web.views import build_case, case_facts, explainable_cases

    with Ledger(ledger_file) as ledger:
        wanted = explainable_cases(ledger, arm)
        facts = [
            case_facts(case)
            for pid in wanted
            if (case := build_case(ledger, pid, arm)) is not None
        ]

    if not facts:
        return

    explainer = Explainer()
    try:
        explainer.warm(facts)
    except LLMUnavailable:
        pass

    written = {
        item.payment_id: {
            "text": (result := explainer.explain(item)).text,
            "source": result.source,
        }
        for item in facts
    }
    save_explanations(written)

    generated = sum(1 for e in written.values() if e["source"] == "generated")
    print(f"explained {len(written)} cases ({generated} generated)")
    for rejection in explainer.rejections:
        print(f"  rejected {rejection.payment_id}: {rejection.reason}")


def _reproduce(args: argparse.Namespace) -> int:
    """Re-run the committed seed and check the README's numbers still hold.

    Writes its ledger to a temporary file rather than `data/`, so verifying a
    claim never disturbs the run the web server is serving.
    """
    import tempfile

    from recoup.eval import run_all
    from recoup.eval.reproduce import claims, compare, load, report, save
    from recoup.world.config import WorldConfig

    recorded = load(args.claims)
    if recorded is None and not args.update:
        print(
            f"no recorded claims at {args.claims}.\n"
            "Run `recoup reproduce --update` once to record the current run as the "
            "baseline, then commit it.",
            file=sys.stderr,
        )
        return 1

    world = WorldConfig.load()
    if args.seed is not None:
        world.run.seed = args.seed
    elif recorded is not None:
        # The recorded seed wins over whatever world.yaml currently says. A claim
        # is about a specific run, and silently reproducing a different one would
        # be worse than failing.
        world.run.seed = recorded.get("run.seed", world.run.seed)

    print(f"re-running the evaluation (seed {world.run.seed})…")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "reproduce.db"
        results, ledger = run_all(world, ledger_path=path)
        digests = {m.arm: ledger.digest(m.arm) for m in results}
        ledger.close()

    produced = claims(
        results,
        seed=world.run.seed,
        batch_size=world.run.batch_size,
        horizon_days=world.run.horizon_days,
        digests=digests,
    )

    if args.update:
        written = save(produced, args.claims)
        print(f"recorded {len(produced)} figures to {written}")
        return 0

    checks = compare(recorded, produced)
    print()
    print(report(checks))

    return 0 if all(check.ok for check in checks) else 1


def _clean(args: argparse.Namespace) -> int:
    """Remove generated state, keeping the expensive things unless asked."""
    from recoup.clean import human, remove, targets

    found = targets(".", llm_cache=args.llm_cache, reports=args.reports)
    if not found:
        print("nothing to clean")
        return 0

    total = 0
    for target in found:
        size = target.size
        total += size
        verb = "would remove" if args.dry_run else "removed"
        print(f"  {verb:<13} {str(target.path):<28} {human(size):>9}  ({target.why})")
        if not args.dry_run:
            remove(target)

    print()
    print(f"{'would free' if args.dry_run else 'freed'} {human(total)}")

    if not (args.llm_cache and args.reports):
        kept = []
        if not args.reports:
            kept.append("reports/ (--reports)")
        if not args.llm_cache:
            kept.append("cache/llm/ (--llm-cache)")
        print(f"kept: {', '.join(kept)}")

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
HANDLERS["sweep"] = _sweep
HANDLERS["reproduce"] = _reproduce
HANDLERS["clean"] = _clean


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

        if name == "sweep":
            sub.add_argument(
                "--metric",
                default="judgment",
                choices=("judgment", "incremental", "recovered"),
                help="which headline to tornado (default: the gap over contact_only, "
                "which is the part that is a claim about the policy engine)",
            )

        if name == "probe":
            sub.add_argument(
                "--amount", type=int, default=10000, help="in paise (default ₹100)"
            )
            sub.add_argument(
                "--tag", default="1", help="change to create a fresh link"
            )

        if name == "reproduce":
            sub.add_argument(
                "--claims",
                # The literal rather than recoup.eval.reproduce.DEFAULT_CLAIMS:
                # importing it here would pull the whole eval package in just to
                # build `--help`, and every other command imports lazily for the
                # same reason.
                default="reports/claims.json",
                help="the recorded run to check against (default: %(default)s)",
            )
            sub.add_argument(
                "--update",
                action="store_true",
                help="record this run as the new baseline instead of checking against "
                "one. Commit the result: it is what makes the claim verifiable.",
            )

        if name == "clean":
            sub.add_argument(
                "--reports",
                action="store_true",
                help="also remove reports/ — a ~20 minute sweep to regenerate",
            )
            sub.add_argument(
                "--llm-cache",
                action="store_true",
                help="also remove cache/llm/ — committed, and rebuilding it needs API "
                "keys and a daily quota you may not get back today",
            )
            sub.add_argument(
                "--dry-run",
                action="store_true",
                help="list what would be removed, and remove nothing",
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    # Every money figure this prints carries a rupee sign, and on Windows both the
    # console and a redirected pipe default to a legacy code page that has no
    # codepoint for it. Without this, `recoup eval > out.txt` raises
    # UnicodeEncodeError on the first row of the arms table.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
