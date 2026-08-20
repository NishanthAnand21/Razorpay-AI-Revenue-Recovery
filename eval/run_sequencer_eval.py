"""Scheduling three retries across a mandate cycle, and pricing who pays for them.

Run:
    python3 eval/run_sequencer_eval.py
    python3 eval/run_sequencer_eval.py --harm-sweep
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.sequencer import (BOUNCE_PENALTY_INR, GATEWAY_FEE_INR, MandateCycle,
                               immediate_burst, industry_default, next_salary_only,
                               solve, success_probability, validate)

DATA = Path(__file__).resolve().parents[1] / "data" / "cycles.jsonl"


def load() -> list[MandateCycle]:
    out = []
    for line in DATA.read_text().splitlines():
        d = json.loads(line)
        d["funded_days"] = set(d["funded_days"])
        out.append(MandateCycle(**d))
    return out


def simulate(cycle: MandateCycle, days: list[int], rng: random.Random) -> dict:
    """Execute a schedule against the ground truth and see what happened."""
    merchant_gain = 0.0
    spend = 0.0
    penalties = 0.0
    attempts = 0
    for i, d in enumerate(days, start=1):
        attempts += 1
        spend += GATEWAY_FEE_INR
        if rng.random() < success_probability(cycle, d, i):
            merchant_gain = cycle.amount_inr
            break
        if cycle.cause == "funds":
            # Every failed presentation on an underfunded account is a bank
            # charge to the customer.
            penalties += BOUNCE_PENALTY_INR
    return {"recovered": merchant_gain > 0, "gain": merchant_gain, "spend": spend,
            "penalties": penalties, "attempts": attempts}


POLICIES = [
    ("immediate burst x3", lambda c: immediate_burst(c)),
    ("industry 24h/72h/d7", lambda c: industry_default(c)),
    ("next salary, once", lambda c: next_salary_only(c)),
    ("optimal, merchant only", lambda c: solve(c, price_customer_harm=False)),
    ("optimal, prices harm", lambda c: solve(c, price_customer_harm=True)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harm-sweep", action="store_true")
    args = ap.parse_args()

    cycles = load()
    at_risk = sum(c.amount_inr for c in cycles)
    print(f"Mandate retry sequencing -- {len(cycles):,} failed cycles, "
          f"INR {at_risk:,.0f} at risk")
    print(f"NPCI allows 1 execution + 3 retries per cycle; RBI requires 24h notice")
    print(f"before each; Autopay executes only in non-peak windows.\n")

    print(f"{'policy':<24}{'recovered':>11}{'merchant net':>15}"
          f"{'customer penalties':>20}{'attempts':>10}{'rule breaks':>13}")
    print("-" * 93)

    results = {}
    for name, fn in POLICIES:
        rng = random.Random(7)          # common random numbers across policies
        gain = spend = pen = att = 0.0
        rec = 0
        breaks = 0
        for c in cycles:
            plan = fn(c)
            if validate(plan, c):
                breaks += 1
            r = simulate(c, plan.days, rng)
            gain += r["gain"]; spend += r["spend"]; pen += r["penalties"]
            att += r["attempts"]; rec += r["recovered"]
        net = gain - spend
        results[name] = {"net": net, "pen": pen, "rec": rec / len(cycles), "att": att}
        print(f"{name:<24}{rec/len(cycles):>10.1%}{net:>15,.0f}"
              f"{pen:>20,.0f}{att/len(cycles):>10.2f}{breaks:>13,}")
    print("-" * 93)

    ind = results["industry 24h/72h/d7"]
    opt = results["optimal, prices harm"]
    mo = results["optimal, merchant only"]

    print(f"""
Against the published industry default of 24h / 72h / day 7:

  merchant net        INR {ind['net']:>12,.0f}  ->  INR {opt['net']:>12,.0f}   ({opt['net']/ind['net']-1:+.1%})
  customer penalties  INR {ind['pen']:>12,.0f}  ->  INR {opt['pen']:>12,.0f}   ({opt['pen']/ind['pen']-1:+.1%})
  attempts per cycle  {ind['att']/len(cycles):>12.2f}  ->  {opt['att']/len(cycles):>12.2f}

The industry default retries at a fixed offset from the failure, which is the one
variable the outcome does not depend on. What determines whether a mandate clears
is when the customer next gets paid. Waiting for that is better for the merchant
AND cheaper for the customer -- these are not in tension, which is the useful
finding here.

THE EXTERNALITY

  optimising merchant net alone     INR {mo['net']:>12,.0f} net, INR {mo['pen']:>11,.0f} of customer penalties
  optimising the joint objective    INR {opt['net']:>12,.0f} net, INR {opt['pen']:>11,.0f} of customer penalties

  difference                        INR {mo['net']-opt['net']:>12,.0f} extra revenue
                                    INR {mo['pen']-opt['pen']:>12,.0f} extra cost to customers

A merchant chasing its own P&L takes INR {mo['net']-opt['net']:,.0f} more and hands its customers
INR {mo['pen']-opt['pen']:,.0f} in bank charges to do it -- destroying roughly
{(mo['pen']-opt['pen'])/max(1,(mo['net']-opt['net'])):.1f} rupees of value for every rupee it gains.

That cost appears in no merchant's accounts, which is exactly why it gets
ignored. It is also the cost that shows up later as churn.

ONE NUMBER THAT LOOKS WRONG

The joint-objective policy recovers {opt['rec']:.1%} of cycles against the industry
default's {ind['rec']:.1%} -- a LOWER recovery rate -- while earning INR {opt['net']-ind['net']:,.0f} more.

It is not a contradiction. It declines to chase small cycles whose expected
recovery cannot cover the bank charge it would inflict, and spends the freed
attempts on cycles that clear. Fewer payments recovered, more money recovered.

Recovery rate is the metric this industry reports and it is the wrong one, for
the same reason gross recovery is the wrong one: both count events instead of
value, and both reward activity that destroys it.""")

    if args.harm_sweep:
        print(f"\n{'harm weight':>12}{'merchant net':>15}{'penalties':>14}{'attempts':>10}")
        print("-" * 51)
        for w in (0.0, 0.25, 0.5, 1.0, 2.0):
            rng = random.Random(7)
            gain = spend = pen = att = 0.0
            for c in cycles:
                plan = solve(c, price_customer_harm=True, harm_weight=w)
                r = simulate(c, plan.days, rng)
                gain += r["gain"]; spend += r["spend"]
                pen += r["penalties"]; att += r["attempts"]
            print(f"{w:>12.2f}{gain-spend:>15,.0f}{pen:>14,.0f}{att/len(cycles):>10.2f}")
        print("The knob is a policy choice, not a hyperparameter. Publishing the")
        print("curve lets whoever owns that decision make it with the numbers in view.")


if __name__ == "__main__":
    main()
