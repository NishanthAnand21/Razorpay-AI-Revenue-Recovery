#!/usr/bin/env python3
"""Run the recovery agent live.

    ./serve.py                          # simulated gateway, dry run, as fast as it goes
    ./serve.py --rate 4 --limit 40      # paced, so it can be watched
    ./serve.py --gateway razorpay       # real Razorpay test-mode reads
    ./serve.py --execute                # actually act (refuses without test keys)
    cat events.jsonl | ./serve.py --source stdin

Prints one line per decision, including the ones where it decides to do nothing,
because a recovery agent's refusals are the interesting half.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

from reclaim.classify import train as train_classifier
from reclaim.diagnose import ThreeTierDiagnoser
from reclaim.gateway import GatewayError, RazorpayTestGateway, SimulatedGateway
from reclaim.models import Action, FailedPayment, RootCause
from reclaim.service import RecoveryService

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


DIM, BOLD, RED, GREEN, YELLOW, BLUE, GREY = "2", "1", "31", "32", "33", "34", "90"

ACTION_STYLE = {
    Action.RETRY_NOW: GREEN, Action.RETRY_SCHEDULED: GREEN,
    Action.SWITCH_METHOD: GREEN, Action.NUDGE_CUSTOMER: BLUE,
    Action.REQUEST_INSTRUMENT_UPDATE: BLUE, Action.ISSUE_INTEREST_NOTICE: YELLOW,
    Action.REFER_MSEFC: YELLOW, Action.RECONCILE: YELLOW,
    Action.ESCALATE_MANUAL: YELLOW, Action.STOP: GREY,
}


def load_events(source: str, limit: int) -> list[FailedPayment]:
    if source == "stdin":
        rows = []
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("settlement_actually_succeeded", False)
            d["true_root_cause"] = RootCause(d.get("true_root_cause", "unknown"))
            rows.append(FailedPayment(**d))
            if len(rows) >= limit:
                break
        return rows
    from run_eval import load
    return load("test")[:limit]


def open_gateway(kind: str, payments, execute: bool):
    if kind == "simulated":
        return SimulatedGateway(payments), None
    try:
        return RazorpayTestGateway(allow_writes=execute), None
    except GatewayError as exc:
        return SimulatedGateway(payments), str(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("synthetic", "stdin"), default="synthetic")
    ap.add_argument("--gateway", choices=("simulated", "razorpay"), default="simulated")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="events per second; 0 = unpaced")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--execute", action="store_true",
                    help="actually act instead of dry-running")
    args = ap.parse_args()

    events = load_events(args.source, args.limit)
    if not events:
        print("no events", file=sys.stderr)
        sys.exit(1)

    gateway, fallback_reason = open_gateway(
        args.gateway, {p.payment_id: p for p in events}, args.execute)

    print(c("\n  Reclaim — live recovery agent", BOLD))
    mode = c("EXECUTING", RED) if args.execute else c("dry run", GREY)
    live = c("LIVE", GREEN) if getattr(gateway, "live", False) else c("simulated", GREY)
    print(f"  gateway {live}   mode {mode}   {len(events)} events"
          f"{f'   {args.rate:g}/s' if args.rate else ''}")
    if fallback_reason:
        print(c(f"  ! razorpay unavailable, fell back to the simulator: "
                f"{fallback_reason}", YELLOW))
    if args.execute and not getattr(gateway, "live", False):
        print(c("  ! --execute on a simulated gateway moves no real money", YELLOW))
    print()

    # Train the diagnosis tier the same way the evals do: on train, never on
    # what we are about to process.
    from run_eval import load as load_split
    learned = train_classifier(load_split("train"))
    svc = RecoveryService(diagnoser=ThreeTierDiagnoser(learned),
                          gateway=gateway, dry_run=not args.execute)

    hdr = (f"  {'#':>4}  {'payment':<20}{'diagnosis':<20}{'tier':<9}"
           f"{'action':<26}{'outcome'}")
    print(c(hdr, DIM))
    print(c("  " + "-" * 104, DIM))

    interval = 1.0 / args.rate if args.rate else 0.0
    started = time.time()
    for i, p in enumerate(events, 1):
        tick = time.time()
        h = svc.handle(p)
        d = h.decision

        # Pad on the PLAIN text, then colour the padded cell. Padding a string
        # that already contains escape codes counts them as visible characters,
        # which is why the long action names were overflowing their column.
        plain = d.action.value + (f" +{d.delay_hours}h" if d.delay_hours else "")
        action = c(plain.ljust(26), ACTION_STYLE.get(d.action, GREY))

        blocked = c(f"  [{d.blocked_by}]", YELLOW) if d.blocked_by else ""
        outcome = h.result
        if "avoided" in outcome:
            outcome = c(outcome, GREEN)
        elif "dry run" in outcome or "no messaging" in outcome:
            outcome = c(outcome, GREY)

        print(f"  {i:>4}  {p.payment_id:<20}{d.diagnosed_cause.value:<20}"
              f"{d.diagnosis_source:<9}{action}{outcome}{blocked}")

        if interval:
            time.sleep(max(0.0, interval - (time.time() - tick)))

    elapsed = time.time() - started
    s = svc.stats
    print(c("  " + "-" * 104, DIM))
    print(f"\n  {c('handled', DIM)} {s.handled}   {c('acted', DIM)} {s.executed}   "
          f"{c('refused', DIM)} {s.refused}   {c('reconciled', DIM)} {s.reconciled}   "
          f"{c('spend', DIM)} INR {s.spend_inr:.2f}")
    if s.already_settled:
        print(c(f"  {s.already_settled} payment(s) had already settled — "
                f"double charges avoided by reconciling first", GREEN))

    if s.refusal_reasons:
        print(f"\n  {c('the kernel refused:', DIM)}")
        for rule, n in sorted(s.refusal_reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>4}  {rule}")

    health = svc.health()
    ok = c("verified", GREEN) if health["audit_intact"] else c("TAMPERED", RED)
    print(f"\n  {c('audit', DIM)} {health['audit_entries']} entries, {ok}, "
          f"head {health['audit_head']}...")
    print(f"  {c('latency', DIM)} p99 {health['p99_latency_us']:.0f}us   "
          f"{c('throughput', DIM)} {s.handled/max(elapsed,1e-9):,.0f}/s"
          f"{c(' (paced)', DIM) if args.rate else ''}")
    print()


if __name__ == "__main__":
    main()
