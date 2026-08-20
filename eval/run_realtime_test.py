#!/usr/bin/env python3
"""Real-time recovery against Razorpay test mode.

    export RAZORPAY_KEY_ID=rzp_test_...  RAZORPAY_KEY_SECRET=...
    python3 eval/run_realtime_test.py --seed 8        # create real links
    python3 eval/run_realtime_test.py --execute       # actually send reminders

Everything else in this repo runs against synthetic events. This runs against
Razorpay's sandbox: real objects, real HTTP, real latency, and -- with
--execute -- a real reminder leaving the building.

The point of the exercise is not that an agent can call an API. It is that the
compliance kernel decides whether the call happens at all, and refuses on facts
rather than on a model's opinion. Below, the same book of receivables is walked
at several times of day, and the API call genuinely does not occur outside RBI's
collections window.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.compliance import ObservableState, Rail, feasible_actions
from reclaim.gateway import GatewayError, PaymentLink, RazorpayTestGateway
from reclaim.models import Action
from reclaim.security import AuditLog, idempotency_key, pseudonymise

# A small book of receivables, in paise. Contacts deliberately avoid repeated
# digits: the real API rejects them with "Recurring digits in customer contact
# are disallowed", which is the kind of validation rule you want to discover
# before a demo rather than during one.
BOOK = [
    ("Asha Menon", "asha.menon@example.com", "+919845012345", 189900),
    ("Ravi Kumar", "ravi.kumar@example.com", "+919845067312", 42500),
    ("Priya Nair", "priya.nair@example.com", "+919845098176", 725000),
    ("Imran Shaikh", "imran.shaikh@example.com", "+919845043821", 12800),
    ("Neha Gupta", "neha.gupta@example.com", "+919845076231", 96400),
    ("Arjun Rao", "arjun.rao@example.com", "+919845031907", 254300),
    ("Fatima Ali", "fatima.ali@example.com", "+919845052846", 61200),
    ("Vikram Setu", "vikram.setu@example.com", "+919845089143", 318700),
]


def _valid(contact: str) -> bool:
    digits = "".join(ch for ch in contact if ch.isdigit())
    return len(digits) == 12 and digits.startswith("91")


assert all(_valid(c) for _n, _e, c, _a in BOOK), "a contact would be rejected"

# Hours to walk the book at. 22:00 is outside RBI's 08:00-19:00 collections
# window; 11:00 and 15:00 are inside it.
HOURS = (11.0, 15.0, 22.0)


# The sandbox rate-limits creation, so the seed loop paces itself rather than
# leaning on the client's retry path. Retries are for surprises; a limit you
# already know about should be respected on the way in.
SEED_PACING_SECONDS = 1.5


def discover(gw: RazorpayTestGateway) -> list[PaymentLink]:
    """Find receivables already in the sandbox.

    Discovery before creation, for two reasons. It is what a real deployment
    does -- an agent is pointed at a book that already exists, it does not
    manufacture one -- and the sandbox rate-limits creation hard enough that a
    harness which always seeds becomes unrunnable after a few passes.
    """
    d = gw._request("GET", "/payment_links")
    return [gw._to_link(x) for x in d.get("payment_links", [])]


def top_up(gw: RazorpayTestGateway, have: int, target: int) -> int:
    """Create up to `target` receivables, stopping cleanly at a rate limit."""
    made = 0
    for i, (name, email, contact, paise) in enumerate(BOOK[:max(0, target - have)]):
        if i:
            time.sleep(SEED_PACING_SECONDS)
        try:
            link = gw.create_payment_link(
                paise / 100.0, description="Reclaim receivable",
                name=name, email=email, contact=contact,
                notes={"source": "reclaim", "surface": "receivable"})
        except GatewayError as exc:
            # A rate limit while seeding is not a failure of the test. Say so
            # and carry on with the book we have.
            print(f"    stopped seeding after {made}: {str(exc)[:80]}")
            break
        made += 1
        print(f"    {link.link_id}  INR {link.amount_inr:>10,.2f}  {link.short_url}")
    return made


def observe_link(link: PaymentLink, hour: float, contacts_7d: int) -> ObservableState:
    """Facts about a real receivable. Nothing here is a model output."""
    return ObservableState(
        rail=Rail.INVOICE, local_hour=hour, is_working_day=True,
        is_collections=True, contacts_7d=contacts_7d, max_contacts_7d=2,
        is_disputed=False,
        # Not asserted: we have no evidence this supplier is a registered MSME,
        # so the statutory remedies stay unavailable. Conservative by default is
        # the only safe direction for a claim that must be true to make.
        supplier_is_msme=False,
        days_past_appointed_day=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0,
                    help="create this many real payment links first")
    ap.add_argument("--limit", type=int, default=6,
                    help="how many receivables to work")
    ap.add_argument("--execute", action="store_true",
                    help="actually send reminders through the API")
    args = ap.parse_args()

    try:
        gw = RazorpayTestGateway(allow_writes=True)
    except GatewayError as exc:
        print(f"Razorpay test mode not configured: {exc}")
        print("\n  export RAZORPAY_KEY_ID=rzp_test_...")
        print("  export RAZORPAY_KEY_SECRET=...")
        sys.exit(1)

    print(f"\n  Reclaim — real-time test against {gw.name} ({gw.key_id})")
    print(f"  mode: {'EXECUTING real reminders' if args.execute else 'dry run'}\n")

    existing = discover(gw)
    print(f"  found {len(existing)} receivables already in the sandbox")
    if args.seed and len(existing) < args.seed:
        print(f"  topping up to {args.seed} (paced {SEED_PACING_SECONDS}s apart)...")
        top_up(gw, len(existing), args.seed)
        existing = discover(gw)

    if not existing:
        print("  nothing to work — run once with --seed 5")
        sys.exit(1)

    # --- reconcile against the real API ------------------------------------
    ids = [l.link_id for l in existing][:args.limit]
    print(f"\n  RECONCILE — fetching live status for {len(ids)} receivables")
    latencies: list[float] = []
    links: list[PaymentLink] = []
    for link_id in ids:
        t = time.perf_counter()
        link = gw.fetch_payment_link(link_id)
        latencies.append((time.perf_counter() - t) * 1000)
        links.append(link)
        flag = "AT RISK" if link.at_risk else link.status.upper()
        print(f"    {link.link_id}  {flag:<9} INR {link.amount_inr:>10,.2f}"
              f"  paid {link.amount_paid_inr:>10,.2f}   {latencies[-1]:5.0f}ms")

    at_risk = [l for l in links if l.at_risk]
    print(f"\n  revenue at risk: INR {sum(l.amount_inr for l in at_risk):,.2f} "
          f"across {len(at_risk)} receivables")

    # --- the kernel gates real API calls ------------------------------------
    audit = AuditLog()
    contacts: dict[str, int] = {}
    sent = refused = 0
    refusals: dict[str, int] = {}

    print(f"\n  WALKING THE BOOK AT THREE TIMES OF DAY")
    print(f"  {'hour':>6}  {'receivable':<22}{'action':<18}{'result'}")
    print("  " + "-" * 82)

    for hour in HOURS:
        for link in at_risk:
            key = link.link_id
            state = observe_link(link, hour, contacts.get(key, 0))
            allowed, fired = feasible_actions(state)
            want = Action.NUDGE_CUSTOMER
            cleared = want in allowed
            blocked = next((v.rule for v in fired if want in v.forbids), None)

            if not cleared:
                refused += 1
                refusals[blocked or "?"] = refusals.get(blocked or "?", 0) + 1
                result = f"refused — {blocked}"
            elif not args.execute:
                result = "dry run: would notify by email"
            else:
                t = time.perf_counter()
                try:
                    okd = gw.notify_payment_link(key, "email")
                    latencies.append((time.perf_counter() - t) * 1000)
                    result = (f"REMINDER SENT ({latencies[-1]:.0f}ms)" if okd
                              else "api returned success=false")
                    sent += 1
                    contacts[key] = contacts.get(key, 0) + 1
                except GatewayError as exc:
                    result = f"gateway error: {type(exc).__name__}"

            audit.append({
                "receivable": key, "hour": hour,
                "customer": pseudonymise(key),
                "amount_inr": link.amount_inr,
                "action": want.value if cleared else Action.STOP.value,
                "kernel_cleared": cleared, "blocked_by": blocked,
                "executed": cleared and args.execute,
                "idempotency_key": idempotency_key(key, int(hour), want.value),
            })
            print(f"  {hour:>5.0f}h  {key:<22}"
                  f"{('notify' if cleared else 'stop'):<18}{result}")
        print("  " + "-" * 82)

    # --- what happened ------------------------------------------------------
    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 2 else max(latencies)
    ok, _bad = audit.verify()

    print(f"""
  RESULT

    reminders sent      {sent}
    refused by kernel   {refused}
    audit               {len(audit.entries)} entries, {'verified' if ok else 'TAMPERED'}, head {audit.head[:16]}...

  KERNEL REFUSALS (real API calls that did not happen)""")
    for rule, n in sorted(refusals.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>4}  {rule}")

    print(f"""
  REAL API LATENCY   p50 {p50:.0f}ms   p99 {p99:.0f}ms   ({len(latencies)} calls)

  Against the compliance kernel's 2.5 microseconds, one gateway round trip is
  about {p50 * 1000 / 2.5:,.0f}x slower. That ratio is the whole argument for the tiering:
  legality is decided locally on facts before anything touches the network, so
  the expensive call is only ever made for actions that are already permitted.

  Note what the 22:00 rows show. The kernel refused, so no HTTP request was
  issued at all -- the reminder was not sent and then logged as suppressed, it
  was never sent. RBI confines collections contact to 08:00-19:00, and that is a
  fact about a clock, so no amount of model confidence can talk past it.""")


if __name__ == "__main__":
    main()
