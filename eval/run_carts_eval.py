"""Abandoned cart recovery, and the one parameter the whole decision turns on."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim import carts as C
from reclaim.carts import Cart, messages_worth_sending
from reclaim.compliance import ObservableState, Rail, feasible_actions
from reclaim.models import Action

STREAM = Path(__file__).resolve().parents[1] / "data" / "events.json"


def load() -> list[Cart]:
    raw = json.loads(STREAM.read_text())["checkouts"]
    rng = random.Random(19)
    out = []
    for c in raw:
        if c["completed_hour"] is not None and c["completed_hour"] <= c["started_hour"] + 2:
            continue                      # converted before we would ever look
        out.append(Cart(
            cart_id=c["session_id"], customer_id=c["customer_id"],
            amount_inr=c["amount_inr"],
            # A customer's typical order value, which is what an opt-out costs us
            # a share of. Correlated with this cart but not identical to it.
            customer_value_inr=round(c["amount_inr"] * rng.uniform(0.6, 1.5), 2),
            stall_hour=(c["started_hour"] + 2) % 24,
            is_new_customer=c["is_new_customer"],
            reached_payment_page=c["reached_payment_page"],
            would_self_recover=c["would_self_recover"]))
    return out


def simulate(carts: list[Cart], n_messages, rng: random.Random) -> dict:
    gross = incremental = spend = optout_cost = 0.0
    messages = optouts = 0
    for cart in carts:
        n = n_messages(cart)
        opted_out = False
        recovered = cart.would_self_recover      # they were coming back anyway
        caused = False
        for i in range(n):
            if opted_out:
                break
            # Contact windows still apply; a cart nudge is transactional, 09-21.
            state = ObservableState(rail=Rail.UPI_COLLECT,
                                    local_hour=float((cart.stall_hour + i * 6) % 24),
                                    contacts_7d=i, max_contacts_7d=3)
            allowed, _ = feasible_actions(state)
            if Action.NUDGE_CUSTOMER not in allowed:
                continue
            messages += 1
            spend += C.MESSAGE_COST_INR
            if not cart.would_self_recover and not caused:
                if rng.random() < C.UPLIFT_BY_MESSAGE[i]:
                    recovered = True
                    caused = True
            if rng.random() < C.OPT_OUT_BY_MESSAGE[i]:
                opted_out = True
                optouts += 1
                optout_cost += C.opt_out_cost(cart.customer_value_inr)
        if recovered:
            gross += cart.amount_inr
        if caused:
            incremental += cart.amount_inr
    return {"gross": gross, "incremental": incremental, "messages": messages,
            "optouts": optouts, "optout_cost": optout_cost, "spend": spend,
            "net": incremental - spend - optout_cost}


def average(carts, fn, trials: int = 40) -> dict:
    """Average over seeds.

    Opt-outs are rare events on a few hundred carts, so a single run produces a
    visibly jagged curve -- an earlier version of the sweep showed the gated
    policy doing better at a higher opt-out cost, which is impossible and was
    purely sampling noise. Averaging is cheap here and reporting the jagged
    version would have been misleading.
    """
    keys = ("gross", "incremental", "messages", "optouts", "optout_cost", "spend", "net")
    acc = {k: 0.0 for k in keys}
    for t in range(trials):
        r = simulate(carts, fn, random.Random(1000 + t))
        for k in keys:
            acc[k] += r[k]
    return {k: v / trials for k, v in acc.items()}


POLICIES = [
    ("no messaging", lambda c: 0),
    ("blanket, 3 messages", lambda c: 3),
    ("single message", lambda c: 1),
    ("EV-gated", messages_worth_sending),
]


def main() -> None:
    carts = load()
    print(f"Abandoned cart recovery -- {len(carts):,} stalled checkouts, "
          f"INR {sum(c.amount_inr for c in carts):,.0f} in carts")
    print(f"{sum(c.would_self_recover for c in carts):,} of these customers come back "
          f"unaided.\n")

    print(f"{'policy':<22}{'gross':>14}{'incremental':>14}{'messages':>10}"
          f"{'opt-outs':>10}{'LTV lost':>13}{'net':>13}")
    print("-" * 96)
    rows = {}
    for name, fn in POLICIES:
        r = average(carts, fn)
        rows[name] = r
        print(f"{name:<22}{r['gross']:>14,.0f}{r['incremental']:>14,.0f}"
              f"{r['messages']:>10,.0f}{r['optouts']:>10,.1f}{r['optout_cost']:>13,.0f}"
              f"{r['net']:>13,.0f}")
    print("-" * 96)

    blanket = rows["blanket, 3 messages"]
    ev = rows["EV-gated"]
    print(f"""
GROSS VERSUS INCREMENTAL, AGAIN

The blanket sequence would report INR {blanket['gross']:,.0f} of recovered carts. It
actually caused INR {blanket['incremental']:,.0f} of them -- {blanket['incremental']/blanket['gross']:.0%}. The rest were customers
who returned on their own and got a message about it.

This is the same gap the regression discontinuity found on mandates, on a
completely different surface and by a completely different method. It is not an
artefact of either.

THE PARAMETER NOBODY MEASURES

Every policy here is EV-positive, and the EV-gated policy sends nearly as many
messages as the blanket one. On these assumptions, messaging abandoned carts is
simply worth doing, and a prior that it is wasteful does not survive contact
with the arithmetic.

But the whole decision rests on one number: what a customer opting out of
contact costs you. Here that is {C.FUTURE_RECOVERABLE_EVENTS:.0f} future recoverable events at
{C.FUTURE_RECOVERY_UPLIFT:.0%} uplift. Almost nobody measures it, and the sweep below shows the
answer flipping as it moves.""")

    print(f"\n{'future events':>15}{'blanket net':>15}{'EV-gated net':>15}"
          f"{'EV-gated msgs':>15}")
    print("-" * 60)
    original = C.FUTURE_RECOVERABLE_EVENTS
    sweep = []
    for k in (0.0, 2.0, 5.0, 12.0, 25.0, 50.0):
        C.FUTURE_RECOVERABLE_EVENTS = k
        b = average(carts, lambda c: 3)
        e = average(carts, messages_worth_sending)
        sweep.append((k, b["net"], e["net"]))
        print(f"{k:>15.0f}{b['net']:>15,.0f}{e['net']:>15,.0f}{e['messages']:>15,.0f}")
    C.FUTURE_RECOVERABLE_EVENTS = original
    print("-" * 60)

    # Where the blanket sequence stops being worth sending. Read off the sweep
    # rather than written down, so it cannot go stale when the model changes.
    flip = next(((lo[0], hi[0]) for lo, hi in zip(sweep, sweep[1:])
                 if lo[1] >= 0 > hi[1]), None)
    worst_gated = min(e for _k, _b, e in sweep)
    print(f"""Blanket messaging turns value-destroying between {flip[0]:.0f} and {flip[1]:.0f} future
recoverable events per customer. That is not an exotic range for a subscription
or a marketplace, and it is the difference between a growth tactic and an
expensive one.

The EV-gated policy never goes below INR {worst_gated:,.0f} anywhere in the sweep, because
it stops sending as the cost rises. That is the actual argument for it, and it
is not the one I expected to be making: it does not recover more today -- at low
opt-out costs it recovers less -- it just does not have to be right about a
parameter almost nobody measures.""")


if __name__ == "__main__":
    main()
