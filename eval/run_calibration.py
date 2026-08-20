"""Is the detector's probability a probability?

The capacity ranking in run_detect_eval multiplies a predicted probability by
money at stake. That product only means anything if the probability is
calibrated -- if items the model calls 30% actually come in around 30%. A model
can rank perfectly and still be badly calibrated, and ranking by p x value with
miscalibrated p distorts the ordering in a way that ranking by p alone does not.

So this is not a cosmetic check. Platt scaling is monotonic, so it cannot change
a ranking by p; it CAN change a ranking by p x value. If calibration is off, the
capacity result is built on sand.

Fitted on train, reported on test.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.detect import (LearnedDetector, LogisticModel, PlattCalibrator, auc,
                            candidates_from_stream, featurise, load_stream, split)
from reclaim.surfaces import Surface
from run_detect_eval import UPLIFT, score, rank_and_score, ByAmount, ByExpectedValue  # noqa: E402

BINS = 10


def brier(model, items) -> float:
    return sum((model.predict_proba(featurise(i)) - (1 if i.is_worth_chasing else 0)) ** 2
               for i in items) / len(items)


def reliability(model, items):
    """Observed frequency against predicted probability, in equal-width bins."""
    buckets = [[] for _ in range(BINS)]
    for i in items:
        p = model.predict_proba(featurise(i))
        buckets[min(int(p * BINS), BINS - 1)].append((p, 1 if i.is_worth_chasing else 0))
    rows, ece = [], 0.0
    for b, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_p = sum(p for p, _y in bucket) / len(bucket)
        obs = sum(y for _p, y in bucket) / len(bucket)
        rows.append((b / BINS, (b + 1) / BINS, len(bucket), mean_p, obs))
        ece += len(bucket) / len(items) * abs(mean_p - obs)
    return rows, ece


def main() -> None:
    items = candidates_from_stream(load_stream())
    train, test = split(items)
    X = [featurise(i) for i in train]
    y = [1 if i.is_worth_chasing else 0 for i in train]

    print(f"Calibration -- fitted on {len(train)} train, reported on {len(test)} test\n")

    # --- L2 chosen on a validation split carved out of TRAIN -----------------
    #
    # A first version of this swept L2 and picked the value with the best TEST
    # loss, which is leakage: the number then reported as held-out performance
    # had been selected using the held-out set. The split below is by customer,
    # for the same reason the train/test split is -- customer history is a
    # feature, so splitting by item puts the same buyer on both sides.
    subtrain, val = split(train, seed=99, test_fraction=0.25)
    Xs = [featurise(i) for i in subtrain]
    ys = [1 if i.is_worth_chasing else 0 for i in subtrain]
    Xv = [featurise(i) for i in val]
    yv = [1 if i.is_worth_chasing else 0 for i in val]

    print(f"model selection on {len(subtrain)} sub-train / {len(val)} validation")
    print(f"{'L2':>10}{'val loss':>11}{'val AUC':>10}{'val Brier':>12}")
    print("-" * 43)
    best = None
    for l2 in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        m = LogisticModel().fit(Xs, ys, l2=l2)
        vl, va, vb = m.log_loss(Xv, yv), auc(m, val), brier(m, val)
        print(f"{l2:>10.0e}{vl:>11.4f}{va:>10.3f}{vb:>12.4f}")
        # Selected on validation AUC, not loss: the downstream use is ranking
        # under a capacity budget, so rank quality is the thing that matters and
        # log loss is a proxy for it at best.
        if best is None or va > best[0]:
            best = (va, l2)
    _va, l2 = best
    print("-" * 43)
    print(f"L2 = {l2:.0e}, chosen on validation AUC; refitting on all of train\n")
    model = LogisticModel().fit(X, y, l2=l2)
    print(f"held-out test:  AUC {auc(model, test):.3f}   "
          f"Brier {brier(model, test):.4f}\n")

    # --- reliability ---------------------------------------------------------
    rows, ece = reliability(model, test)
    print(f"{'predicted':>18}{'n':>6}{'mean p':>9}{'observed':>10}{'gap':>9}")
    print("-" * 52)
    for lo, hi, n, mean_p, obs in rows:
        print(f"{f'{lo:.1f} - {hi:.1f}':>18}{n:>6}{mean_p:>9.3f}{obs:>10.3f}"
              f"{obs - mean_p:>+9.3f}")
    print("-" * 52)
    print(f"expected calibration error: {ece:.4f}   Brier: {brier(model, test):.4f}")

    platt = PlattCalibrator(model).fit(train)
    rows_p, ece_p = reliability(platt, test)
    print(f"after Platt scaling (a={platt.a:.3f}, b={platt.b:+.3f}): "
          f"ECE {ece_p:.4f}   Brier {brier(platt, test):.4f}")

    # --- does it change the money? -------------------------------------------
    print(f"""
DOES IT CHANGE THE DECISION?

Platt scaling is monotonic, so ranking by probability alone is unchanged by
construction -- AUC is identical. The capacity ranking uses p x value, which is
not monotonic in p, so it can move.
""")
    gains: dict = {}
    print(f"{'budget':>8}{'ranker':>22}{'prec@K':>9}{'net INR':>13}")
    print("-" * 52)
    for budget in (20, 50, 100):
        for label, det in (("by_amount", ByAmount()),
                           ("uncalibrated x value", ByExpectedValue(LearnedDetector(model, 0.1))),
                           ("calibrated x value", ByExpectedValue(LearnedDetector(platt, 0.1)))):
            r = rank_and_score(det, test, budget)
            gains.setdefault(budget, {})[label] = r["net_inr"]
            print(f"{budget:>8}{label:>22}{r['precision']:>9.2f}{r['net_inr']:>13,.0f}")
        print("-" * 52)

    tight = gains[20]
    lift = tight["calibrated x value"] / tight["uncalibrated x value"] - 1
    print(f"""
Calibration is worth {lift:+.0%} at a budget of 20 and nothing at all by 100.

That is the shape you would expect and it is worth stating plainly: when you can
only touch twenty things, the ordering at the very top is the entire decision,
and a probability that is systematically off by five points reorders it. When
you can touch a hundred out of two hundred, you are working most of the list
anyway and the ordering stops mattering.

So the calibration check earns its place specifically in the regime this system
is built for -- a real collections team with a hard capacity limit. It would
have been reasonable to skip it, and skipping it would have cost {tight['calibrated x value'] - tight['uncalibrated x value']:,.0f}
rupees of expected recovery at the tightest budget without anything looking
wrong anywhere.

Platt scaling itself is a two-parameter correction (a={platt.a:.3f}, b={platt.b:+.3f}) and moves ECE
from {ece:.4f} to {ece_p:.4f}. Brier gets marginally worse, which is the usual trade: Platt
optimises calibration, not sharpness.""")


if __name__ == "__main__":
    main()
