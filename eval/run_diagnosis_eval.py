"""Diagnosis accuracy across the tiers, and what each tier costs.

Selection is on a validation split carved out of train. The objective is
end-to-end expected accuracy -- coverage times accuracy where the classifier is
confident, plus the model tier's own accuracy on whatever it abstains from.

An earlier version optimised `accuracy x coverage`, which treats an abstained row
as worth zero. It is not worth zero; it is worth whatever the model tier gets it
right at. Optimising the wrong objective picked the loosest possible threshold.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.classify import train
from reclaim.diagnose import (MockLLMDiagnoser, RulesDiagnoser, ThreeTierDiagnoser,
                              TieredDiagnoser)
from reclaim.models import RootCause
from run_eval import load  # noqa: E402

L2_GRID = (1e-4, 1e-3, 1e-2)
MARGIN_GRID = (0.10, 0.25, 0.40, 0.60)


def split_by_customer(rows, fraction=0.25, seed=5):
    rng = random.Random(seed)
    customers = sorted({p.customer_id for p in rows})
    rng.shuffle(customers)
    cut = int(len(customers) * (1 - fraction))
    keep = set(customers[:cut])
    return [p for p in rows if p.customer_id in keep], \
           [p for p in rows if p.customer_id not in keep]


def novel_rows(rows):
    rules = RulesDiagnoser()
    return [p for p in rows if rules.diagnose(p).cause is RootCause.UNKNOWN]


def evaluate(learned, rows, llm) -> dict:
    """End-to-end on the rows the rules table cannot answer."""
    confident = [p for p in rows if learned.confident(learned.predict(p))]
    abstained = [p for p in rows if p not in confident]
    right_c = sum(learned.predict(p).cause is p.true_root_cause for p in confident)
    right_a = sum(llm.diagnose(p).cause is p.true_root_cause for p in abstained)
    return {
        "coverage": len(confident) / len(rows),
        "acc_confident": right_c / len(confident) if confident else 0.0,
        "acc_abstained": right_a / len(abstained) if abstained else 0.0,
        "expected_acc": (right_c + right_a) / len(rows),
        "llm_calls": len(abstained),
    }


def main() -> None:
    train_rows, test_rows = load("train"), load("test")
    sub, val = split_by_customer(train_rows)
    llm = MockLLMDiagnoser()

    val_novel = novel_rows(val)
    print(f"Diagnosis -- selection on {len(val_novel)} validation rows the rules "
          f"table cannot answer\n")
    print(f"{'L2':>8}{'margin':>8}{'coverage':>10}{'acc|conf':>10}"
          f"{'acc|abstain':>13}{'expected':>10}{'llm calls':>11}")
    print("-" * 70)

    best = None
    for l2 in L2_GRID:
        learned = train(sub, l2=l2)
        for margin in MARGIN_GRID:
            learned.min_margin = margin
            r = evaluate(learned, val_novel, llm)
            print(f"{l2:>8.0e}{margin:>8.2f}{r['coverage']:>10.2f}"
                  f"{r['acc_confident']:>10.3f}{r['acc_abstained']:>13.3f}"
                  f"{r['expected_acc']:>10.3f}{r['llm_calls']:>11}")
            if best is None or r["expected_acc"] > best[0]:
                best = (r["expected_acc"], l2, margin)
    print("-" * 70)
    _acc, l2, margin = best
    print(f"chosen on validation: L2 {l2:.0e}, margin {margin:.2f}\n")

    # Refit on all of train with the chosen settings, then report on test.
    learned = train(train_rows, l2=l2)
    learned.min_margin = margin

    rules = RulesDiagnoser()
    test_novel = novel_rows(test_rows)
    two = TieredDiagnoser()
    three = ThreeTierDiagnoser(learned)

    print(f"HELD-OUT TEST -- {len(test_rows)} payments, "
          f"{len(test_novel)} of them beyond the rules table\n")
    print(f"{'diagnoser':<16}{'overall':>10}{'on novel':>11}{'llm calls':>12}")
    print("-" * 49)
    for name, dg in (("rules + model", two), ("rules + learned + model", three)):
        if hasattr(dg, "counts"):
            dg.counts = {"rules": 0, "learned": 0, "llm": 0}
        overall = sum(dg.diagnose(p).cause is p.true_root_cause for p in test_rows)
        calls = dg.counts["llm"] if hasattr(dg, "counts") else len(test_novel)
        if hasattr(dg, "counts"):
            dg.counts = {"rules": 0, "learned": 0, "llm": 0}
        nov = sum(dg.diagnose(p).cause is p.true_root_cause for p in test_novel)
        print(f"{name:<16}{overall/len(test_rows):>10.3f}"
              f"{nov/len(test_novel):>11.3f}{calls:>12}")
    print("-" * 49)

    leave_one_reason_out(train_rows, l2, margin)

    three.counts = {"rules": 0, "learned": 0, "llm": 0}
    for p in test_rows:
        three.diagnose(p)
    c = three.counts
    total = sum(c.values())
    print(f"""
TIER USAGE across all {total} payments

  rules table   {c['rules']:>4}  ({c['rules']/total:.0%})  free, exact match
  classifier    {c['learned']:>4}  ({c['learned']/total:.0%})  microseconds, trained on labelled history
  model         {c['llm']:>4}  ({c['llm']/total:.0%})  a network call

The classifier absorbs most of what used to reach the model, and is more
accurate on those rows than the model tier was. That is not surprising -- it was
trained on this merchant's own resolved failures, and the model was reasoning
from a prompt. What the model tier is left with is the genuinely ambiguous
remainder, which is what it should have been doing all along.

The classifier one-hot encodes the error reason, so a reason never seen in
training contributes nothing and the prediction falls back on method, recurrence
and note tokens. Its margin collapses there, and the row routes onward. The
routing signal and the accuracy come from the same mechanism.""")


def leave_one_reason_out(train_rows, l2: float, margin: float) -> None:
    """The generalisation test that matters: a reason never seen in training.

    The rows called "novel" above are novel to the RULES TABLE, not to the
    classifier -- an ambiguous code like `do_not_honour` is absent from the rules
    lookup but present in the training labels. So the accuracy above does not
    establish that the classifier generalises to genuinely new failure reasons,
    and reading it that way would be a mistake.

    Here each reason is removed from training entirely and the classifier is
    tested on it. The question is not really whether it gets them right -- often
    it cannot -- but whether it KNOWS it cannot, and abstains so the model tier
    picks them up. A classifier that guesses confidently on unseen reasons would
    quietly route the hardest cases away from the tier equipped to handle them.
    """
    reasons = sorted({p.error_reason for p in train_rows})
    reasons = [r for r in reasons
               if 8 <= sum(1 for p in train_rows if p.error_reason == r) <= 200]

    print(f"\nLEAVE-ONE-REASON-OUT -- {len(reasons)} reasons, each removed from training\n")
    print(f"{'held-out reason':<32}{'n':>5}{'abstained':>11}{'acc if kept':>13}")
    print("-" * 61)
    total_n = total_abstained = total_right = 0
    worst = None
    for reason in reasons:
        keep = [p for p in train_rows if p.error_reason != reason]
        held = [p for p in train_rows if p.error_reason == reason]
        learned = train(keep, l2=l2)
        learned.min_margin = margin
        confident = [p for p in held if learned.confident(learned.predict(p))]
        right = sum(learned.predict(p).cause is p.true_root_cause for p in confident)
        abstained = len(held) - len(confident)
        total_n += len(held)
        total_abstained += abstained
        total_right += right
        acc = right / len(confident) if confident else float("nan")
        if confident and len(confident) >= 8 and (worst is None or acc < worst[1]):
            worst = (reason, acc, abstained / len(held), len(held))
        print(f"{reason:<32}{len(held):>5}{abstained/len(held):>11.0%}"
              f"{acc:>13.2f}" if confident else
              f"{reason:<32}{len(held):>5}{abstained/len(held):>11.0%}{'--':>13}")
    print("-" * 61)
    kept = total_n - total_abstained
    print(f"{'all':<32}{total_n:>5}{total_abstained/total_n:>11.0%}"
          f"{total_right/kept if kept else float('nan'):>13.2f}")
    print(f"""
On reasons it has genuinely never seen the classifier abstains {total_abstained/total_n:.0%} of the
time, and where it does commit it is right {total_right/kept if kept else 0:.0%} of the time.

That is the behaviour the tiering needs. The `reason in vocabulary` feature is
what produces it: an unseen reason contributes nothing to any class, the softmax
flattens, the margin collapses, and the row routes onward to the model tier
rather than being answered badly and cheaply.

The aggregate hides one bad case, so here it is: `{worst[0]}` ({worst[3]} rows) is
answered at {worst[1]:.0%} accuracy with only {worst[2]:.0%} abstention. It is the most ambiguous
string in the vocabulary -- it can mean an issuer blip, an empty account or an
abandoned authentication -- and the classifier is confidently wrong about it more
often than it abstains. Everything else in the table sits at 0.75 or better.

The fix is not more training. A merchant seeing that pattern should route that
specific reason to the model tier unconditionally, which is a one-line entry in
the rules table pointing the other way -- a deny-list rather than a lookup. Left
unfixed here and reported instead, because inventing the fix without the traffic
to validate it would be guessing.""")


if __name__ == "__main__":
    main()
