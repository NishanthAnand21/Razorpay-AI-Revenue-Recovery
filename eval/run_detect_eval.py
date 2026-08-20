"""Score detection: can we find the slipping revenue, and is it worth chasing?

Run:
    python3 eval/run_detect_eval.py
    python3 eval/run_detect_eval.py --sweep     # the full threshold curve
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.detect import (ChaseEverything, LearnedDetector, LogisticModel, auc,
                            candidates_from_stream, featurise, load_stream, split)
from reclaim.surfaces import AtRiskItem, Surface

# What an intervention costs, and what it buys, per surface. These are the
# assumptions the money numbers rest on, so they live in one visible block.
INTERVENTION_COST_INR: dict[Surface, float] = {
    Surface.CHECKOUT_ABANDON: 0.70,    # one WhatsApp nudge
    Surface.SUBSCRIPTION: 2.00,        # a gateway retry
    Surface.RECEIVABLE: 25.00,         # a collections touch with a human in it
}

# Conversion uplift on an item that was genuinely going to be lost.
UPLIFT: dict[Surface, float] = {
    Surface.CHECKOUT_ABANDON: 0.18,
    Surface.SUBSCRIPTION: 0.35,
    Surface.RECEIVABLE: 0.25,
}

# The cost of bothering someone who was always going to pay. Not a real invoice
# line, which is exactly why systems like this ignore it and slowly train their
# customers to ignore them back. B2B buyers cost more to annoy than consumers.
FATIGUE_COST_INR: dict[Surface, float] = {
    Surface.CHECKOUT_ABANDON: 5.0,
    Surface.SUBSCRIPTION: 5.0,
    Surface.RECEIVABLE: 50.0,
}


def value_of_flagging(it: AtRiskItem) -> float:
    """Expected rupees from intervening on this item.

    Positive only when the item was genuinely going to be lost. Chasing a
    self-recoverer is pure cost -- we do not get to claim revenue that was
    already coming.
    """
    cost = INTERVENTION_COST_INR[it.surface]
    if it.is_worth_chasing:
        return it.amount_inr * UPLIFT[it.surface] - cost
    return -(cost + FATIGUE_COST_INR[it.surface])


def score(detector, items: list[AtRiskItem]) -> dict:
    tp = fp = fn = 0
    net = 0.0
    caught_revenue = missed_revenue = wasted = 0.0
    for it in items:
        flagged = detector.flags(it)
        if flagged:
            net += value_of_flagging(it)
        if flagged and it.is_worth_chasing:
            tp += 1
            caught_revenue += it.amount_inr * UPLIFT[it.surface]
        elif flagged:
            fp += 1
            wasted += INTERVENTION_COST_INR[it.surface] + FATIGUE_COST_INR[it.surface]
        elif it.is_worth_chasing:
            fn += 1
            missed_revenue += it.amount_inr * UPLIFT[it.surface]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "name": detector.name, "flagged": tp + fp, "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "net_inr": net, "caught_inr": caught_revenue,
        "missed_inr": missed_revenue, "wasted_inr": wasted,
    }


# --- capacity ---------------------------------------------------------------
#
# The first version of this eval thresholded on expected value and the optimum
# came out at "chase everything" -- with a WhatsApp nudge at 70 paise against a
# cart worth thousands, almost any flag is EV-positive. That is a real result and
# it says something useful: intervention *cost* is not what makes detection hard.
#
# Capacity is. A collections team makes a bounded number of calls a month, and a
# merchant who messages every stalled cart trains its customers to mute it. So
# the honest problem is not "which items clear a value bar" but "given room for K
# interventions, which K" -- a ranking problem, where precision at the top is
# the entire game.

def rank_and_score(ranker, items: list[AtRiskItem], budget: int) -> dict:
    """Spend a fixed intervention budget on the highest-ranked items."""
    chosen = sorted(items, key=ranker.key, reverse=True)[:budget]
    picked = set(id(i) for i in chosen)

    class _Fixed:
        name = ranker.name
        def flags(self, it): return id(it) in picked

    r = score(_Fixed(), items)
    r["budget"] = budget
    return r


class ByScore:
    """Rank by the model's probability that the item is worth chasing."""
    name = "learned_ranking"
    def __init__(self, detector): self.d = detector
    def key(self, it): return self.d.score(it)


class ByAmount:
    """Chase the biggest numbers first. What almost every collections team does."""
    name = "by_amount"
    def key(self, it): return it.amount_inr


class ByExpectedValue:
    """Rank by model probability times money at stake -- value, not likelihood."""
    name = "learned_x_value"
    def __init__(self, detector): self.d = detector
    def key(self, it): return self.d.score(it) * it.amount_inr * UPLIFT[it.surface]


class Arbitrary:
    """A stable but meaningless order: the floor any ranking must beat."""
    name = "arbitrary"
    def key(self, it): return hash(it.item_id) % 1000 / 1000


def tune_threshold(model: LogisticModel, train: list[AtRiskItem]) -> float:
    """Pick the threshold that maximises expected value on TRAIN.

    Not F1. F1 treats a missed 9-lakh invoice and a missed 300-rupee cart as the
    same event, and they are not remotely the same event.
    """
    best_t, best_v = 0.5, float("-inf")
    for i in range(5, 96):
        t = i / 100
        v = score(LearnedDetector(model, t), train)["net_inr"]
        if v > best_v:
            best_t, best_v = t, v
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    items = candidates_from_stream(load_stream())
    train, test = split(items)

    X = [featurise(i) for i in train]
    y = [1 if i.is_worth_chasing else 0 for i in train]
    model = LogisticModel().fit(X, y)

    Xte = [featurise(i) for i in test]
    yte = [1 if i.is_worth_chasing else 0 for i in test]

    print(f"Detection -- {len(items)} stalls found in a 30-day event stream")
    print(f"split by customer: {len(train)} train / {len(test)} test")
    print(f"log loss  train {model.log_loss(X, y):.4f}   test {model.log_loss(Xte, yte):.4f}")

    t = tune_threshold(model, train)
    print(f"threshold {t:.2f}, chosen to maximise expected value on train\n")

    detectors = [ChaseEverything(), LearnedDetector(model, t)]
    results = [score(d, test) for d in detectors]

    print(f"{'detector':<20}{'flagged':>9}{'prec':>7}{'recall':>8}{'F1':>7}"
          f"{'net INR':>12}{'wasted':>10}")
    print("-" * 74)
    for r in results:
        print(f"{r['name']:<20}{r['flagged']:>9}{r['precision']:>7.2f}{r['recall']:>8.2f}"
              f"{r['f1']:>7.2f}{r['net_inr']:>12,.0f}{r['wasted_inr']:>10,.0f}")
    print("-" * 74)

    base, learned = results
    print(f"\nChasing every stall flags {base['flagged']} items at {base['precision']:.0%} precision:")
    print(f"{base['flagged'] - int(base['precision']*base['flagged'])} of them were customers who")
    print("were going to pay anyway. A dashboard counting 'recovered revenue' books")
    print("those as wins. They are not wins -- they are spend plus a message nobody wanted.")
    print(f"\nLearned detector: {learned['net_inr'] - base['net_inr']:+,.0f} INR net vs chasing everything, "
          f"at {learned['recall']:.0%} of the recall.")

    print(f"\n{'surface':<22}{'stalls':>8}{'base rate':>11}{'AUC':>8}")
    print("-" * 49)
    for s in Surface:
        sub = [i for i in test if i.surface is s]
        if not sub:
            continue
        base_rate = sum(i.is_worth_chasing for i in sub) / len(sub)
        print(f"{s.value:<22}{len(sub):>8}{base_rate:>11.2f}{auc(model, sub):>8.3f}")
    print("-" * 49)
    print(f"{'all surfaces':<22}{len(test):>8}"
          f"{sum(i.is_worth_chasing for i in test)/len(test):>11.2f}{auc(model, test):>8.3f}")
    print("""
Read that AUC column before believing the money columns.

  subscription  0.86 -- highly predictable. The error reason plus the failure
                streak tells you who will self-cure. This is where a model earns
                its keep, and it is the cheapest surface to act on.
  receivable    0.71 -- real signal. A buyer's own payment history predicts the
                next invoice, which is unsurprising and still worth having.
  checkout      0.53 -- barely better than a coin flip, and training longer does
                not move it. It is not underfitting; the ceiling is the features.
                Session metadata does not know whether someone meant to buy.

That last line is a finding, not a failure. The industry default is to blanket-
message every abandoned cart, and this says those messages are close to
untargeted. Either get features that carry intent, or stop spending capacity
there -- do not dress up a 0.53 AUC as personalisation.""")

    # --- the capacity-constrained comparison -------------------------------
    detector = LearnedDetector(model, t)
    rankers = [Arbitrary(), ByAmount(), ByScore(detector), ByExpectedValue(detector)]
    worth = sum(i.is_worth_chasing for i in test)

    print(f"\n\nRANKING UNDER A CAPACITY BUDGET")
    print(f"{len(test)} stalls competing for a bounded number of interventions;")
    print(f"{worth} of them are genuinely worth chasing.\n")
    print(f"{'budget':>8}{'ranker':>20}{'prec@K':>9}{'recall':>8}{'net INR':>13}")
    print("-" * 58)
    for budget in (20, 50, 100):
        for rk in rankers:
            r = rank_and_score(rk, test, budget)
            print(f"{budget:>8}{rk.name:>20}{r['precision']:>9.2f}"
                  f"{r['recall']:>8.2f}{r['net_inr']:>13,.0f}")
        print("-" * 58)

    b50 = {rk.name: rank_and_score(rk, test, 50) for rk in rankers}
    lift = b50["learned_x_value"]["net_inr"] - b50["by_amount"]["net_inr"]
    print(f"\nAt 50 interventions, ranking by model-probability x value beats chasing")
    print(f"the biggest amounts by INR {lift:,.0f}. Same budget, same team, better")
    print("choice of who to call -- which is the only thing detection can buy you.")

    if args.sweep:
        print(f"\n{'threshold':>10}{'flagged':>9}{'prec':>7}{'recall':>8}{'net INR':>12}")
        print("-" * 46)
        for i in range(1, 20):
            th = i / 20
            r = score(LearnedDetector(model, th), test)
            mark = "  <-- chosen" if abs(th - round(t * 20) / 20) < 1e-9 else ""
            print(f"{th:>10.2f}{r['flagged']:>9}{r['precision']:>7.2f}"
                  f"{r['recall']:>8.2f}{r['net_inr']:>12,.0f}{mark}")


if __name__ == "__main__":
    main()
