"""Mandate cycles to schedule retries against.

Salary dates are the load-bearing distribution here: most Indian salaried
customers are paid within the first few days of the month, but a meaningful tail
is paid at month end, and the whole point of the sequencer is that those two
groups need opposite schedules after the same failure.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.sequencer import CYCLE_DAYS, MandateCycle, funds_probability

SEED = 90210
N = 20_000

SALARY_DAYS = [1, 2, 3, 5, 7, 25, 28, 30]
SALARY_WEIGHTS = [0.34, 0.16, 0.10, 0.09, 0.08, 0.07, 0.08, 0.08]
CAUSES = ["funds", "transient", "instrument"]
CAUSE_WEIGHTS = [0.71, 0.21, 0.08]   # NACH failures are dominated by balance


def generate(n: int = N, seed: int = SEED) -> list[MandateCycle]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        salary_day = rng.choices(SALARY_DAYS, SALARY_WEIGHTS)[0]
        cause = rng.choices(CAUSES, CAUSE_WEIGHTS)[0]
        amount = round(rng.choice([
            rng.uniform(99, 499), rng.uniform(499, 1999), rng.uniform(1999, 9999),
        ]), 2)
        # Mandates are anchored to a debit day; failures cluster where balances
        # are thin, so the failure day is sampled against the funds curve.
        weights = [1.0 - funds_probability(d, salary_day) for d in range(1, CYCLE_DAYS)]
        fail_day = rng.choices(range(1, CYCLE_DAYS), weights)[0]

        c = MandateCycle(f"cyc_{i:06d}", amount, fail_day, salary_day, cause)
        # Ground truth: which days the account could actually have covered it.
        c.funded_days = {d for d in range(1, CYCLE_DAYS + 1)
                         if rng.random() < funds_probability(d, salary_day)}
        out.append(c)
    return out


def main() -> None:
    cycles = generate()
    path = Path(__file__).parent / "cycles.jsonl"
    with path.open("w") as fh:
        for c in cycles:
            d = asdict(c)
            d["funded_days"] = sorted(c.funded_days)
            fh.write(json.dumps(d) + "\n")
    print(f"wrote {len(cycles):,} mandate cycles -> {path.name}")
    from collections import Counter
    print("  causes:", dict(Counter(c.cause for c in cycles)))
    print(f"  mean days from failure to next salary: "
          f"{sum((c.salary_day - c.fail_day) % CYCLE_DAYS for c in cycles)/len(cycles):.1f}")


if __name__ == "__main__":
    main()
