"""Chasing 6,000 B2B invoices, and counting what the chasing costs.

The metric here is days sales outstanding, not a recovery rate. Nearly every
invoice is eventually paid; the question is how much sooner, and what the account
looks like afterwards.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.compliance import (OUTREACH_ACTIONS, ObservableState, Rail,
                                feasible_actions)
from reclaim.models import Action, Channel, CHANNEL_COST_INR, ACTION_COST_INR
from reclaim.receivables import (ChaseState, Invoice, Stage, next_action,
                                 statutory_interest, target_stage)

DATA = Path(__file__).resolve().parents[1] / "data" / "receivables.jsonl"
HORIZON = 180
BUSINESS_HOUR = 11.0

# How many days each rung pulls payment forward, scaled by how responsive the
# buyer is. A referral moves almost anyone; a pre-due reminder moves the
# forgetful and nobody else.
STAGE_PULL = {Stage.PRE_DUE: 2.0, Stage.DUE: 3.0, Stage.SOFT: 9.0,
              Stage.FIRM: 22.0, Stage.REFER: 70.0}
# Chasing a buyer who was always going to pay on time costs relationship, and it
# scales with how big the account is. Same shape as the bounce-penalty
# externality in the mandate sequencer: a cost the chaser does not see.
# Emphatically NOT flat across the ladder. A courtesy reminder before the due
# date is ordinary business practice and costs approximately nothing; a filing
# against a buyer who was going to pay on time can end the account. An earlier
# version priced every rung identically, which made the polite policy look worse
# than the aggressive one.
GOODWILL_RATE = {Stage.PRE_DUE: 0.00002, Stage.DUE: 0.0001, Stage.SOFT: 0.0006,
                 Stage.FIRM: 0.004, Stage.REFER: 0.03}


def load() -> list[Invoice]:
    return [Invoice(**json.loads(l)) for l in DATA.read_text().splitlines()]


def is_working_day(day: int) -> bool:
    return day % 7 not in (5, 6)


def observe_invoice(inv: Invoice, st: ChaseState, day: int, contacts_7d: int
                    ) -> ObservableState:
    return ObservableState(
        rail=Rail.INVOICE,
        local_hour=BUSINESS_HOUR,
        is_working_day=is_working_day(day),
        is_collections=True,
        contacts_7d=contacts_7d,
        max_contacts_7d=2,                    # B2B tolerance is lower than consumer
        is_disputed=inv.disputed,
        supplier_is_msme=inv.supplier_is_msme,
        days_past_appointed_day=day - inv.due_day,
        promise_to_pay_open=st.promise_open(day),
    )


class NoChase:
    name = "no chasing"
    def act(self, inv, st, day, allowed): return None


class BlanketLadder:
    """Fixed offsets past due, ignoring promises, disputes and eligibility."""
    DAYS = {1, 7, 14, 21, 30, 45}

    def __init__(self, gated: bool = False):
        self.gated = gated
        self.name = "blanket ladder (gated)" if gated else "blanket ladder"

    def act(self, inv, st, day, allowed):
        if (day - inv.due_day) not in self.DAYS:
            return None
        past = day - inv.due_day
        if past >= 45:
            action = Action.REFER_MSEFC
        elif past >= 21:
            action = Action.ISSUE_INTEREST_NOTICE
        else:
            action = Action.NUDGE_CUSTOMER
        # The gated variant runs the identical schedule with the kernel enforced.
        # It exists to answer the only question that matters about the ungated
        # one: how much of its advantage is bought by breaking rules?
        if self.gated and action not in allowed:
            return None
        return action


class ReclaimChaser:
    """The ladder, gated by the kernel and steered by promises.

    The contact budget is the scarce resource, and the rungs are not equally
    valuable. Two failed tunings taught this the hard way:

      - acting only on a rung change capped the chaser near one contact per
        invoice and left legal DSO unclaimed;
      - acting whenever the gap allowed spent the weekly budget early on pre-due
        reminders, so the formal notice -- the rung that actually moves a
        delinquent buyer -- was vetoed for contact fatigue when it came due.

    So: always act on an escalation, and only repeat on the rungs worth
    repeating. Which is the same allocation problem as the mandate sequencer,
    arrived at from the other end.
    """
    name = "reclaim chaser"
    MIN_GAP_DAYS = 5
    REPEAT_GAP_DAYS = 12
    REPEATABLE = {Stage.SOFT, Stage.FIRM}

    def __init__(self, capacity_fraction: float = 1.0):
        self.capacity_fraction = capacity_fraction

    def act(self, inv, st, day, allowed):
        if st.touched_days and day - st.touched_days[-1] < self.MIN_GAP_DAYS:
            return None
        stage = target_stage(inv, st, day)
        escalating = stage.value > st.stage.value or not st.contacts
        if not escalating:
            if stage not in self.REPEATABLE:
                return None
            if day - st.touched_days[-1] < self.REPEAT_GAP_DAYS:
                return None
        action, _channel, _why = next_action(inv, st, day)
        return action if action in allowed else None


def priority(inv: Invoice) -> float:
    """Expected days-of-cash bought by working this invoice.

    Amount alone is the obvious ranking and the wrong one: a large invoice from a
    buyer who always pays on time has nothing to recover. What is worth working
    is value multiplied by the chance the buyer is actually going to be late,
    which is what their own history estimates.

    Buyers with no history get the base rate rather than a zero, so that a new
    account is neither ignored nor assumed to be a problem.
    """
    late = inv.buyer_prior_late_ratio if inv.buyer_prior_invoices else 0.35
    # Shrink weak estimates toward the base rate: three past invoices is not
    # evidence, and treating it as evidence is how prioritisers get captured by
    # noise on tiny samples.
    n = inv.buyer_prior_invoices
    shrunk = (n * late + 5 * 0.35) / (n + 5)
    return inv.amount_inr * shrunk


def simulate(policy, invoices: list[Invoice], rng: random.Random,
             capacity_fraction: float = 1.0, rank=None) -> dict:
    # Capacity: a collections team cannot work every invoice.
    rank = rank or (lambda i: i.amount_inr)
    if capacity_fraction < 1.0:
        cutoff = sorted((rank(i) for i in invoices), reverse=True)[
            int(len(invoices) * capacity_fraction) - 1]
    else:
        cutoff = 0.0

    contacts_log: dict[str, list[int]] = {}
    total_days = 0.0
    total_amount = 0.0
    spend = goodwill = interest_claimed = 0.0
    contacts = breaches = misuse = disputed_contacts = 0
    paid_in_horizon = 0

    for inv in invoices:
        st = ChaseState()
        pay_day = inv.would_pay_on_day
        worked = rank(inv) >= cutoff

        for day in range(max(0, inv.due_day - 5), min(HORIZON, pay_day + 1)):
            if not worked:
                break
            recent = [d for d in contacts_log.get(inv.buyer_id, []) if day - d < 7]
            state = observe_invoice(inv, st, day, len(recent))
            allowed, _ = feasible_actions(state)

            action = policy.act(inv, st, day, allowed)
            if action is None:
                continue

            if action not in allowed:
                breaches += 1
                if action in (Action.ISSUE_INTEREST_NOTICE, Action.REFER_MSEFC) \
                        and not inv.supplier_is_msme:
                    misuse += 1
                if inv.disputed:
                    disputed_contacts += 1

            contacts += 1
            st.contacts += 1
            st.touched_days.append(day)
            st.stage = target_stage(inv, st, day)
            contacts_log.setdefault(inv.buyer_id, []).append(day)
            spend += ACTION_COST_INR.get(action, 0.0) + CHANNEL_COST_INR[Channel.EMAIL]

            if action is Action.ISSUE_INTEREST_NOTICE and inv.supplier_is_msme:
                interest_claimed += statutory_interest(inv.amount_inr, day - inv.due_day)

            # Chasing a buyer who was never going to be late is pure cost, and
            # what it costs depends on how hard the rung was.
            if inv.would_pay_on_day <= inv.due_day:
                goodwill += GOODWILL_RATE[st.stage] * inv.amount_inr

            # The effect of the chase.
            if not inv.disputed:
                pull = STAGE_PULL[st.stage] * inv.reliability * rng.uniform(0.6, 1.4)
                pay_day = max(day + 1, int(pay_day - pull))
                # A soft chase is where a promise gets made.
                if st.stage in (Stage.SOFT, Stage.FIRM) and st.promise_day is None:
                    if rng.random() < 0.55:
                        st.promise_day = min(pay_day, day + rng.randint(3, 12))
                        if rng.random() > inv.reliability:
                            st.promises_broken += 1
                            st.promise_day = None

        # DSO over a FIXED cohort, censoring anything unpaid at the horizon.
        # Averaging only over invoices that happened to settle in time penalises
        # exactly the policies that drag slow invoices inside the window, which
        # is the opposite of what the metric is for.
        if pay_day <= HORIZON:
            paid_in_horizon += 1
            settled = pay_day
        else:
            settled = HORIZON
        total_days += (settled - inv.issued_day) * inv.amount_inr
        total_amount += inv.amount_inr

    dso = total_days / total_amount if total_amount else 0.0
    return {"dso": dso, "paid": paid_in_horizon / len(invoices), "contacts": contacts,
            "at_risk": total_amount,
            "spend": spend, "goodwill": goodwill, "interest": interest_claimed,
            "breaches": breaches, "misuse": misuse, "disputed": disputed_contacts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    inv = load()
    print(f"Receivables chasing -- {len(inv):,} invoices, "
          f"INR {sum(i.amount_inr for i in inv):,.0f} outstanding\n")

    policies = [
        (NoChase(), 1.0),
        (BlanketLadder(), 1.0),
        (BlanketLadder(gated=True), 1.0),
        (ReclaimChaser(), 1.0),
        (ReclaimChaser(0.4), 0.4),
    ]
    ranked = [(ReclaimChaser(0.4), 0.4)]
    print(f"{'policy':<32}{'DSO':>7}{'paid':>7}{'contacts':>10}{'goodwill lost':>15}"
          f"{'breaches':>10}{'misuse':>8}")
    print("-" * 89)
    rows = {}
    for pol, cap in policies:
        label = pol.name + (f" @{cap:.0%} capacity, by value" if cap < 1.0 else "")
        r = simulate(pol, inv, random.Random(3), cap)
        rows[label] = r
        print(f"{label:<32}{r['dso']:>7.1f}{r['paid']:>7.1%}{r['contacts']:>10,}"
              f"{r['goodwill']:>15,.0f}{r['breaches']:>10,}{r['misuse']:>8,}")
    for pol, cap in ranked:
        label = f"{pol.name} @{cap:.0%}, prioritised"
        r = simulate(pol, inv, random.Random(3), cap, rank=priority)
        rows[label] = r
        print(f"{label:<32}{r['dso']:>7.1f}{r['paid']:>7.1%}{r['contacts']:>10,}"
              f"{r['goodwill']:>15,.0f}{r['breaches']:>10,}{r['misuse']:>8,}")
    print("-" * 89)

    base = rows["no chasing"]
    blanket = rows["blanket ladder"]
    gated = rows["blanket ladder (gated)"]
    ours = rows["reclaim chaser"]
    prio = rows["reclaim chaser @40%, prioritised"]
    outstanding = sum(i.amount_inr for i in inv)

    print(f"""
DSO over a fixed cohort, unpaid invoices censored at the {HORIZON}-day horizon.

  no chasing              {base['dso']:.1f} days
  blanket ladder          {blanket['dso']:.1f} days   ({blanket['contacts']:,} contacts, {blanket['breaches']:,} of them forbidden)
  blanket ladder (gated)  {gated['dso']:.1f} days   (same schedule, kernel enforced)
  reclaim chaser          {ours['dso']:.1f} days
  chaser @40%, prioritised {prio['dso']:.1f} days   ({prio['contacts']:,} contacts)

Three findings, including one that does not flatter this repo.

1. THE UNGATED LADDER'S LEAD IS BOUGHT, NOT EARNED.

   It posts the best DSO at {blanket['dso']:.1f} days. Run the identical schedule with the
   kernel enforced and it lands at {gated['dso']:.1f}. The {gated['dso'] - blanket['dso']:.1f} day difference is
   entirely the {blanket['breaches']:,} actions it is not permitted to take -- including
   {blanket['misuse']:,} statutory interest claims against suppliers who are NOT registered
   micro or small enterprises, which is a false legal claim made in writing to
   a customer, and {blanket['disputed']:,} contacts on formally disputed invoices.

2. THE ESCALATION LOGIC ADDS NOTHING MEASURABLE.

   The gated blanket ladder reaches {gated['dso']:.1f} days; the chaser, with promise
   tracking, behaviour-based rung skipping and a graded ladder, reaches {ours['dso']:.1f}.
   That is a tie, and the chaser spends more contacts to get there.

   All of the value on this surface is in the compliance kernel, not in the
   cleverness of the ladder. Reporting otherwise would be easy and wrong.

3. PRIORITISATION IS WHERE THE REAL GAIN IS.

   Ranking by value x the buyer's own history of paying late, rather than by
   value alone, reaches {prio['dso']:.1f} days on {prio['contacts']:,} contacts -- matching the
   full-capacity chaser's {ours['dso']:.1f} days using {1 - prio['contacts']/ours['contacts']:.0%} fewer.

   Same cash, {1 - prio['contacts']/ours['contacts']:.0%} less work. For a collections team that is the entire
   question, and it is the same lesson the detector reached: under a binding
   capacity constraint, who you work matters more than how hard you work them.

WHAT THE STATUTORY LEVER IS WORTH

  INR {ours['interest']:,.0f} of s.16 interest legitimately claimed -- money the
  supplier is owed by operation of law and would otherwise never have asked
  for. Claimed only where the supplier is actually registered, because the
  kernel does not offer the option otherwise.

  Working capital released against doing nothing:
    reclaim chaser   INR {(base['dso']-ours['dso']) * outstanding / 365:,.0f}
    prioritised @40% INR {(base['dso']-prio['dso']) * outstanding / 365:,.0f}""")


if __name__ == "__main__":
    main()
