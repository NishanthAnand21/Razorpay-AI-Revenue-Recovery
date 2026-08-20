"""The whole pipeline on one raw event stream, with one audit trail."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.detect import (LearnedDetector, LogisticModel, PlattCalibrator,
                            candidates_from_stream, featurise, load_stream, split)
from reclaim.orchestrator import run
from reclaim.security import AuditLog
from reclaim.surfaces import Surface


def main() -> None:
    items = candidates_from_stream(load_stream())
    train, _test = split(items)
    model = LogisticModel().fit([featurise(i) for i in train],
                                [1 if i.is_worth_chasing else 0 for i in train])
    # Calibrated: the queue ranks by probability x value, which calibration
    # reorders at the top -- exactly the part a bounded budget consumes.
    detector = LearnedDetector(PlattCalibrator(model).fit(train), 0.1)

    print(f"End to end -- one raw event stream, one queue, one audit trail\n")
    print(f"{len(items):,} stalls detected across {len(set(i.surface for i in items))} surfaces, "
          f"INR {sum(i.amount_inr for i in items):,.0f} at risk\n")

    print(f"{'capacity':>9}{'acted':>7}{'blocked':>8}{'expected recovery':>19}"
          f"{'marginal':>12}{'spend':>9}   {'surfaces served'}")
    print("-" * 104)
    curve = []
    prev = 0.0
    for capacity in (25, 100, 250, 500, 1000):
        log = AuditLog()
        r = run(items, detector=detector, capacity=capacity, log=log)
        mix = collections.Counter(i.surface.value for i in r.interventions)
        mix_s = ", ".join(f"{k.split('_')[0]}:{v}" for k, v in mix.most_common())
        marginal = r.expected_recovery_inr - prev
        curve.append((capacity, r.expected_recovery_inr, marginal, len(r.interventions)))
        prev = r.expected_recovery_inr
        print(f"{capacity:>9}{len(r.interventions):>7}{sum(r.blocked.values()):>8}"
              f"{r.expected_recovery_inr:>19,.0f}{marginal:>12,.0f}"
              f"{r.spend_inr:>9,.2f}   {mix_s}")
    print("-" * 104)

    # Where more capacity stops paying for itself. Comparing each marginal to
    # the one before it finds the bend; comparing it to the first marginal (an
    # earlier attempt) just finds the end of the table.
    knee = next((c for (c, _t, m, _n), (_pc, _pt, pm, _pn) in zip(curve[1:], curve)
                 if pm > 0 and m < 0.2 * pm), curve[-1][0])
    saturated = curve[-1][3] < 1000
    print(f"""
DIMINISHING RETURNS ARE SHARP

Going from 25 to 250 interventions adds INR {curve[2][1] - curve[0][1]:,.0f} of expected recovery.
Going from 250 to 1,000 adds INR {curve[-1][1] - curve[2][1]:,.0f}. The curve bends at {knee}, past which the
queue has run out
of items whose value justifies touching them{" -- it saturates at " + str(curve[-1][3]) + " and stops" if saturated else ""}.

Staffing a collections team to work everything is therefore the wrong shape of
answer. The curve says where to stop, which is a question ops leads are usually
asked to answer from intuition.""")

    log = AuditLog()
    r = run(items, detector=detector, capacity=250, log=log)

    print(f"""
A NOTE ON THE EXPECTED RECOVERY COLUMN

It is an expectation under the detector's own model, not a measurement. When the
detector was calibrated, this column fell by about a fifth at every budget --
nothing got worse, the earlier figure was the model flattering itself with
probabilities that ran high. An expected-value number is only as honest as the
probability inside it, which is the same lesson as gross recovery in a different
costume.

ONE QUEUE, NOT FOUR

At small budgets the queue is entirely receivables, and that is the correct
answer rather than a bug: a single overdue invoice can be worth more than every
abandoned cart in the batch put together. Four surfaces each optimising inside
their own silo would have spent that budget on carts, because a cart team's
budget is measured in carts.

Surfaces only start sharing the queue once capacity exceeds the supply of
valuable receivables. Where that crossover sits is the single most useful number
an ops lead could get from this system, and it falls out of running one queue
instead of four.

WHAT THE KERNEL REFUSED""")
    for rule, n in sorted(r.blocked.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {rule}")
    print(f"""
  Refusals are logged, not silently dropped. A trail that records only what was
  done cannot answer the question an auditor actually asks, which is what else
  was considered and why it was not done.

AUDIT INTEGRITY

  {len(log.entries):,} entries, head {log.audit_head if hasattr(log, 'audit_head') else log.head[:32]}...""")

    ok, bad = log.verify()
    print(f"  chain verifies: {ok}")
    log.entries[len(log.entries) // 2].payload["action"] = "retry_now"
    ok2, bad2 = log.verify()
    print(f"  after editing one entry in the middle: verifies={ok2}, first bad={bad2}")
    print("""
  The head hash commits to the entire history, so publishing it -- to a log
  service, a compliance mailbox, anywhere outside this process -- makes the
  whole trail tamper-evident without needing a second copy of it.""")


if __name__ == "__main__":
    main()
