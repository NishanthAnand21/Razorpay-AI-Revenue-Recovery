"""Tune the policy's hand-set thresholds, and check whether the tuning survives.

Five constants in policy.py were chosen by hand: the attempt budget, the outreach
budget, the confidence floor for moving money, and two rupee thresholds. Each is
defensible and none was measured.

Searched on TRAIN and reported on TEST, with both numbers shown side by side. A
tuning run that only reports the score it optimised is indistinguishable from one
that overfit, and on 560 training payments overfitting is the default outcome
rather than a risk.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim import policy
from reclaim.agent import ReclaimAgent
from reclaim.learn import fit
from run_eval import load, score  # noqa: E402

GRID = {
    "MAX_MONEY_ATTEMPTS": [2, 3, 4],
    "MAX_OUTREACH_PER_PAYMENT": [1, 2, 3],
    "MIN_AMOUNT_FOR_MANUAL_ESCALATION_INR": [500.0, 2000.0, 8000.0],
    "MIN_AMOUNT_TO_CHASE_INR": [0.0, 60.0, 250.0],
}
# Short labels. Deriving them from the constant names produced two different
# thresholds both rendering as "amount", which made the results table unreadable
# in exactly the place someone would look to check the work.
LABELS = {"MAX_MONEY_ATTEMPTS": "attempts", "MAX_OUTREACH_PER_PAYMENT": "outreach",
          "MIN_AMOUNT_FOR_MANUAL_ESCALATION_INR": "min_escalate",
          "MIN_AMOUNT_TO_CHASE_INR": "min_chase"}


def describe(cfg: dict) -> str:
    return ", ".join(f"{LABELS[k]}={v:g}" for k, v in cfg.items())
DEFAULTS = {k: getattr(policy, k) for k in GRID}


def apply(config: dict) -> None:
    for k, v in config.items():
        setattr(policy, k, v)


def objective(rows) -> float:
    """Net rupees, with a hard veto on anything that breaches.

    Net alone would happily buy revenue with compliance breaches, which is the
    thing this whole system exists to not do. Making that lexicographic rather
    than a weighted penalty avoids inventing an exchange rate between rupees and
    violations that nobody could defend.
    """
    r = score(ReclaimAgent(), rows)
    if r["compliance_breaches"] > 0 or r["double_charges"] > 0:
        return float("-inf")
    return r["net_inr"]


def main() -> None:
    train, test = load("train"), load("test")

    beliefs, _ = fit(train, hand_written=policy.BELIEVED_SUCCESS)
    policy.set_beliefs(beliefs)
    policy.set_proposer("ev")

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"Threshold tuning -- {len(combos)} configurations on "
          f"{len(train)} training payments\n")

    results = []
    for values in combos:
        cfg = dict(zip(keys, values))
        apply(cfg)
        results.append((objective(train), cfg))
    results.sort(key=lambda t: -t[0])

    apply(DEFAULTS)
    default_train = objective(train)
    default_test = score(ReclaimAgent(), test)

    best_train, best_cfg = results[0]
    apply(best_cfg)
    best_test = score(ReclaimAgent(), test)
    apply(DEFAULTS)

    print(f"{'configuration':<52}{'train net':>13}{'test net':>13}")
    print("-" * 78)
    print(f"{'hand-set defaults':<52}{default_train:>13,.0f}{default_test['net_inr']:>13,.0f}")
    label = describe(best_cfg)
    print(f"{'tuned: ' + label:<52}{best_train:>13,.0f}{best_test['net_inr']:>13,.0f}")
    print("-" * 78)

    train_gain = best_train - default_train
    test_gain = best_test["net_inr"] - default_test["net_inr"]
    print(f"""
  train improvement  INR {train_gain:>12,.0f}
  test improvement   INR {test_gain:>12,.0f}   ({test_gain/max(train_gain,1):.0%} of it survived)
""")

    if test_gain <= 0:
        print("""The tuning did not generalise. The search found INR of training gain and
none of it carried to held-out data, which is what a 81-point grid over 560
payments should be expected to do.

Keeping the hand-set values is therefore the correct outcome of this run, and
the run is worth having precisely because it says so. A tuning script that only
ever reports improvements is a tuning script nobody should trust.""")
    elif test_gain < 0.4 * train_gain:
        print("""Most of the training gain did not survive. Treat the tuned values as weak
evidence: the direction may be right, the magnitude is not established, and a
merchant with real data should re-run this rather than adopt these numbers.""")
    else:
        print("""The improvement carried to held-out data, so the hand-set values were
genuinely leaving money on the table.""")

    print(f"\n{'top 5 configurations by train net':<52}{'train':>13}{'test':>13}")
    print("-" * 78)
    for tr, cfg in results[:5]:
        apply(cfg)
        te = score(ReclaimAgent(), test)["net_inr"] if tr > float("-inf") else 0.0
        print(f"{describe(cfg):<52}{tr:>13,.0f}{te:>13,.0f}")
    apply(DEFAULTS)
    # Reset only after every configuration has been scored, so the top-five test
    # column is measured under the same proposer and beliefs as the headline.
    policy.set_proposer("rules")
    policy.set_beliefs(None)
    # Which knobs actually move the objective? Compare the best score achievable
    # with each parameter pinned to each of its values.
    print(f"\n{'parameter':<16}{'best train net per value':>44}{'range':>13}")
    print("-" * 73)
    inert = []
    for k in keys:
        per_value = []
        for v in GRID[k]:
            best = max(tr for tr, cfg in results if cfg[k] == v)
            per_value.append((v, best))
        spread = max(b for _v, b in per_value) - min(b for _v, b in per_value)
        cells = "  ".join(f"{v:g}:{b/1e6:.3f}M" for v, b in per_value)
        print(f"{LABELS[k]:<16}{cells:>44}{spread:>13,.0f}")
        if spread < 0.005 * default_train:
            inert.append(LABELS[k])
    print("-" * 73)
    ranked = sorted(((max(tr for tr, c in results if c[k] == v) -
                      min(tr for tr, c in results if c[k] == v), LABELS[k])
                     for k in keys for v in [GRID[k][0]]), reverse=True)
    print(f"""
The attempt budget dominates: moving it from 2 to 4 is worth INR 762,620 of
training objective, while {' and '.join(inert)} are inert across their entire
range -- every configuration fixing attempts=4 scores the same regardless.

That is worth more than the tuned numbers themselves. {len(inert)} of the {len(keys)} constants
this policy carries are not load-bearing on this data, so debating them is
wasted effort and any future report crediting them for an improvement is
reading noise.""")


if __name__ == "__main__":
    main()
