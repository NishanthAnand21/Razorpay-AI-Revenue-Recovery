"""A 90-day panel of mandate debits, for measuring what interventions cause.

Built so that the causal estimator can be graded. The generator plants a known
true treatment effect and two confounders that are specifically designed to fool
the naive comparison:

  1. A smooth time-of-day trend in baseline self-cure. People top up during the
     day, so recovery drifts upward with the clock regardless of what we do. A
     difference of means across the peak boundary would credit that drift to the
     intervention.
  2. Compositional selection. Peak hours carry genuinely different traffic --
     more retail, smaller tickets, more first-time mandates. So the blocked group
     and the treated group are not alike *in aggregate*, which is exactly why the
     global comparison fails and a local one at the boundary does not.

Treatment is not assigned by this file. It is assigned by the compliance kernel,
from the NPCI peak windows, which is the entire point.
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 424242
N_EVENTS = 400_000

# The true causal effect of an immediate retry on recovery within 24h. The
# estimator does not see this; the eval grades against it.
TRUE_LIFT_TRANSIENT = 0.150
TRUE_LIFT_FUNDS = 0.060      # retrying instantly helps far less when the issue
                             # is an empty account, which is the whole reason
                             # timing policy exists


@dataclass
class DebitEvent:
    event_id: str
    fail_hour: float             # time of day the debit failed, continuous
    day: int
    amount_inr: float
    cause: str                   # transient | funds
    first_mandate: bool
    recovered_24h: int           # the outcome
    # ground truth, eval only
    true_lift_applied: float
    self_cured: int              # would have recovered with no intervention


def _baseline_self_cure(hour: float, cause: str, first: bool) -> float:
    """Probability the customer fixes it themselves within 24h.

    Rises smoothly through the day and dips overnight -- a real pattern, and the
    confounder that a naive boundary comparison would misread as our doing.
    """
    trend = 0.18 + 0.055 * math.sin((hour - 4.0) / 24.0 * 2 * math.pi)
    base = trend + (0.10 if cause == "funds" else 0.02)
    base -= 0.05 if first else 0.0
    return max(0.02, min(0.85, base))


def _peak(hour: float) -> bool:
    """NPCI peak windows: Autopay may not execute inside these."""
    return (10.0 <= hour < 13.0) or (17.0 <= hour < 21.5)


def generate(n: int = N_EVENTS, seed: int = SEED) -> list[DebitEvent]:
    rng = random.Random(seed)
    out: list[DebitEvent] = []
    for i in range(n):
        hour = rng.uniform(0.0, 24.0)
        day = rng.randrange(90)

        # Compositional selection: peak traffic really is different traffic.
        if _peak(hour):
            first = rng.random() < 0.46
            amount = round(rng.uniform(99, 2500), 2)
            cause = rng.choices(["transient", "funds"], [0.35, 0.65])[0]
        else:
            first = rng.random() < 0.28
            amount = round(rng.uniform(299, 9000), 2)
            cause = rng.choices(["transient", "funds"], [0.55, 0.45])[0]

        p0 = _baseline_self_cure(hour, cause, first)
        self_cured = 1 if rng.random() < p0 else 0

        # Treatment is whatever the compliance kernel permits: an immediate retry
        # is only legal outside the peak windows.
        treated = not _peak(hour)
        lift = (TRUE_LIFT_TRANSIENT if cause == "transient" else TRUE_LIFT_FUNDS)

        if self_cured:
            recovered = 1                      # would have come back regardless
        elif treated:
            recovered = 1 if rng.random() < lift / max(1e-9, 1 - p0) * (1 - p0) else 0
        else:
            recovered = 0

        out.append(DebitEvent(
            f"ev_{i:06d}", round(hour, 4), day, amount, cause, first,
            recovered, lift if treated else 0.0, self_cured))
    return out


def true_late_at(cutoff: float, events: list[DebitEvent], bandwidth: float) -> float:
    """The true local average treatment effect at a cutoff.

    The RD estimand is local -- it is the effect on units near the boundary, not
    the average effect overall. Grading the estimator against the global average
    would be marking it wrong for being right.
    """
    near = [e for e in events if abs(e.fail_hour - cutoff) <= bandwidth]
    if not near:
        return float("nan")
    # An intervention can only help someone who was not going to self-cure, so
    # the effect on the recovery *rate* is (1 - p_self_cure) * lift, not lift.
    total = 0.0
    for e in near:
        p0 = _baseline_self_cure(e.fail_hour, e.cause, e.first_mandate)
        lift = TRUE_LIFT_TRANSIENT if e.cause == "transient" else TRUE_LIFT_FUNDS
        total += (1.0 - p0) * lift
    return total / len(near)


def main() -> None:
    ev = generate()
    path = Path(__file__).parent / "panel.jsonl"
    with path.open("w") as fh:
        for e in ev:
            fh.write(json.dumps(asdict(e)) + "\n")
    treated = [e for e in ev if not _peak(e.fail_hour)]
    print(f"wrote {len(ev):,} mandate debit failures over 90 days -> {path.name}")
    print(f"  treated (outside NPCI peak): {len(treated):,}")
    print(f"  blocked (inside NPCI peak):  {len(ev) - len(treated):,}")
    print(f"  overall recovery:            {sum(e.recovered_24h for e in ev)/len(ev):.3f}")
    print(f"  true lift, transient/funds:  {TRUE_LIFT_TRANSIENT} / {TRUE_LIFT_FUNDS}")


if __name__ == "__main__":
    main()
