"""Choosing which three moments in a mandate cycle to spend.

WHY THIS IS A REAL PROBLEM
--------------------------
NPCI allows one execution plus three retries per Autopay cycle. RBI requires a
pre-debit notification at least 24 hours before every debit, and a retry is a
debit -- so a retry can never be immediate, and each one has to be committed to a
day in advance. Autopay may only execute in non-peak windows.

So the sequencer cannot do what dunning systems normally do, which is retry
quickly and often. It gets three shots, each booked 24h ahead, and the only
question is *when*.

WHAT MAKES THE ANSWER NON-OBVIOUS
---------------------------------
In India the answer is dominated by the salary cycle. A mandate that fails for
insufficient funds on the 28th and one that fails on the 2nd are different
problems: the first customer is days away from a credit, the second has just been
paid and is unlikely to be funded again soon. The published industry default --
retry at 24h, 72h, then day 7 -- is a fixed offset from the failure, so it is
blind to exactly the thing that determines the outcome.

THE COST NOBODY PRICES
----------------------
A failed auto-debit does not only cost the merchant a gateway fee. Indian banks
charge the *customer* a bounce penalty of roughly Rs 250-500 per failed
presentation. That cost is real, it is large relative to a small subscription,
and it appears nowhere in the merchant's P&L -- which is precisely why a merchant
optimising its own net revenue will over-retry and push the loss onto the person
it is trying to keep.

This module therefore solves for two objectives and reports both. The gap between
them is the externality, measured in rupees.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CYCLE_DAYS = 30
MAX_RETRIES = 3               # NPCI: 1 execution + 3 retries
NOTICE_DAYS = 1               # RBI pre-debit notice: at least 24h ahead
GATEWAY_FEE_INR = 2.0
# Mean bank bounce penalty charged to the customer per failed presentation.
BOUNCE_PENALTY_INR = 350.0
# A legal execution hour: before 10:00 is outside both NPCI peak windows, and
# salary credits have normally settled overnight by then.
EXECUTION_HOUR = 9.0


@dataclass
class MandateCycle:
    """One subscription charge that failed, and the cycle it has to be won back in."""

    cycle_id: str
    amount_inr: float
    fail_day: int                 # day of the cycle the original execution failed
    salary_day: int               # day of the month this customer is paid
    cause: str                    # funds | transient | instrument
    # ground truth, eval only
    funded_days: set[int] = field(default_factory=set)


def funds_probability(day: int, salary_day: int) -> float:
    """Probability the account can cover the debit on a given day.

    Rises sharply the day money lands and decays through the month. The shape
    matters more than the exact constants: what the sequencer has to learn is
    that waiting for a credit beats retrying quickly, and that the right wait is
    a function of the calendar rather than of how long ago we last failed.
    """
    gap = (day - salary_day) % CYCLE_DAYS
    if gap <= 1:
        return 0.82
    if gap <= 3:
        return 0.74
    if gap <= 7:
        return 0.58
    if gap <= 14:
        return 0.38
    if gap <= 21:
        return 0.24
    return 0.16


def success_probability(cycle: MandateCycle, day: int, attempt: int) -> float:
    """P(this retry works), given when it lands and how many came before."""
    if cycle.cause == "instrument":
        return 0.0                       # a dead mandate cannot be revived by timing
    if cycle.cause == "transient":
        # Issuer blips clear fast; the calendar is irrelevant.
        base = 0.62
    else:
        base = funds_probability(day, cycle.salary_day)
    # Each successive presentation on the same cycle is worth a little less.
    return max(0.0, base * (0.9 ** (attempt - 1)))


# --- the optimal policy ------------------------------------------------------

@dataclass
class Plan:
    """When to present, and what the plan is expected to be worth."""

    days: list[int]
    expected_merchant_inr: float
    expected_customer_penalty_inr: float


def solve(cycle: MandateCycle, *, price_customer_harm: bool,
          harm_weight: float = 1.0) -> Plan:
    """Dynamic programming over (day, retries left).

    The state is small and the horizon is finite, so the optimum is exact rather
    than heuristic -- there is no need to approximate a problem this size, and an
    exact answer is what makes the comparison against the industry default
    meaningful.

    `price_customer_harm` toggles whether the bounce penalty enters the
    objective. Both settings are solved and reported; the difference between them
    is the cost a merchant-only objective silently exports to its customers.
    """
    first_legal = cycle.fail_day + NOTICE_DAYS
    memo: dict[tuple[int, int], tuple[float, list[int]]] = {}

    def value(day: int, retries_left: int) -> tuple[float, list[int]]:
        if retries_left == 0 or day > CYCLE_DAYS:
            return 0.0, []
        key = (day, retries_left)
        if key in memo:
            return memo[key]

        # Option 1: wait a day. Free, but the cycle is finite.
        best, best_plan = value(day + 1, retries_left)

        # Option 2: present today.
        attempt = MAX_RETRIES - retries_left + 1
        p = success_probability(cycle, day, attempt)
        fail_cost = GATEWAY_FEE_INR
        if price_customer_harm and cycle.cause == "funds":
            fail_cost += harm_weight * BOUNCE_PENALTY_INR
        cont, cont_plan = value(day + 1, retries_left - 1)
        act = p * cycle.amount_inr - GATEWAY_FEE_INR - (1 - p) * (fail_cost - GATEWAY_FEE_INR) \
            + (1 - p) * cont
        if act > best:
            best, best_plan = act, [day] + cont_plan

        memo[key] = (best, best_plan)
        return memo[key]

    _v, days = value(first_legal, MAX_RETRIES)

    # Score the chosen plan on both axes, so the externality is always visible
    # even when the objective ignored it.
    merchant = 0.0
    penalty = 0.0
    survive = 1.0
    for i, d in enumerate(days, start=1):
        p = success_probability(cycle, d, i)
        merchant += survive * (p * cycle.amount_inr - GATEWAY_FEE_INR)
        if cycle.cause == "funds":
            penalty += survive * (1 - p) * BOUNCE_PENALTY_INR
        survive *= (1 - p)
    return Plan(days, merchant, penalty)


# --- baselines ---------------------------------------------------------------

def industry_default(cycle: MandateCycle) -> Plan:
    """Retry at 24h, 72h, then day 7 -- the published smart-retry default.

    A fixed offset from the failure, which is the one thing the outcome does not
    depend on.
    """
    days = [cycle.fail_day + 1, cycle.fail_day + 3, cycle.fail_day + 7]
    return _score(cycle, [d for d in days if d <= CYCLE_DAYS])


def immediate_burst(cycle: MandateCycle) -> Plan:
    """Three presentations the same day, hours apart. What a naive retry loop does.

    Every one of them breaks the pre-debit notice rule, and on a funds failure
    each one earns the customer another bounce penalty for money that was never
    going to be there. Included because it is what you get by default.
    """
    return _score(cycle, [cycle.fail_day] * 3)


def validate(plan: Plan, cycle: MandateCycle) -> list[str]:
    """Check a schedule against the rules the kernel enforces per-decision."""
    problems = []
    if len(plan.days) > MAX_RETRIES:
        problems.append(f"exceeds NPCI cycle cap ({len(plan.days)} > {MAX_RETRIES})")
    prev = cycle.fail_day
    for d in plan.days:
        if d - prev < NOTICE_DAYS:
            problems.append(f"day {d}: less than 24h since the previous debit, "
                            "so no valid pre-debit notice can be outstanding")
        prev = d
    if any(d > CYCLE_DAYS for d in plan.days):
        problems.append("presentation falls outside the mandate cycle")
    return problems


def next_salary_only(cycle: MandateCycle) -> Plan:
    """A single presentation the day after the next salary credit.

    The obvious domain heuristic, and a strong baseline -- worth beating rather
    than worth ignoring.
    """
    d = cycle.fail_day + 1
    while d <= CYCLE_DAYS and (d % CYCLE_DAYS) != (cycle.salary_day + 1) % CYCLE_DAYS:
        d += 1
    return _score(cycle, [d] if d <= CYCLE_DAYS else [])


def _score(cycle: MandateCycle, days: list[int]) -> Plan:
    merchant = penalty = 0.0
    survive = 1.0
    for i, d in enumerate(days, start=1):
        p = success_probability(cycle, d, i)
        merchant += survive * (p * cycle.amount_inr - GATEWAY_FEE_INR)
        if cycle.cause == "funds":
            penalty += survive * (1 - p) * BOUNCE_PENALTY_INR
        survive *= (1 - p)
    return Plan(days, merchant, penalty)
