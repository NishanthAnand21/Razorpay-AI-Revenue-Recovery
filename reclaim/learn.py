"""Learning the policy's beliefs instead of asserting them.

`policy.BELIEVED_SUCCESS` started as nine hand-written constants. They were
deliberately different from the simulator's ground truth -- an agent that has the
world memorised is a lookup table, not an agent -- but "deliberately different"
is not the same as "right", and the expected-value gate that decides whether to
spend money on a payment reads directly off them.

So they are fitted here instead, from exploration data, with two properties that
matter more than the accuracy of any single number:

KEYED ON THE DIAGNOSED CAUSE, NOT THE TRUE ONE
    The policy only ever knows what the diagnoser told it. Estimating
    P(success | diagnosed cause, action) therefore absorbs diagnosis error into
    the belief itself: where the diagnoser is unreliable, the learned prior comes
    out lower, and the EV gate automatically gets more conservative on exactly
    the causes the model is worst at. Nobody has to notice and hand-tune that.

EXPLORATION RESPECTS THE KERNEL
    Data is collected by sampling uniformly from the *legal* action set, not from
    all actions. An exploration policy that learns from illegal actions is
    learning about a world it is not allowed to act in, and would produce beliefs
    that make the EV gate argue for things the kernel will veto.

Fitted on train only. The comparison against the hand-written table is on test.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .compliance import MONEY_ACTIONS, OUTREACH_ACTIONS, feasible_actions, observe
from .diagnose import TieredDiagnoser
from .models import Action, Channel, Decision, FailedPayment, RootCause
from . import simulator

# Beta(1,1) smoothing. With a handful of observations for a rare (cause, action)
# pair, an unsmoothed estimate is either 0.0 or 1.0 and both are lies.
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
# Below this many observations we do not trust the estimate enough to replace the
# hand-written value; the pair is reported and left alone.
MIN_OBSERVATIONS = 25

# Escalation is explored so a belief exists for it, but policy.propose_by_ev does
# not put it in the candidate set -- it terminates the workflow, and comparing it
# myopically against actions that do not is what made recovery worse when tried.
EXPLORABLE = MONEY_ACTIONS | OUTREACH_ACTIONS | {Action.ESCALATE_MANUAL}

# Delay buckets for scheduled retries. Timing is the whole lever on funds and
# limit failures, so an estimate that averages over "some delay" answers a
# question the policy never asks. A first fit ignored this and came out
# systematically low -- it was measuring random timing, not chosen timing, and
# reading that as "the hand-written numbers were optimistic" would have been
# the wrong conclusion from a real measurement.
DELAY_BUCKETS = ((0, "immediate"), (8, "same_day"), (24, "next_day"), (48, "multi_day"))


def delay_bucket(hours: int) -> str:
    label = DELAY_BUCKETS[0][1]
    for threshold, name in DELAY_BUCKETS:
        if hours >= threshold:
            label = name
    return label


def key_for(cause: RootCause, action: Action, attempt: int, delay_hours: int) -> tuple:
    """The most specific key. Lookup falls back when a cell is thin."""
    bucket = delay_bucket(delay_hours) if action is Action.RETRY_SCHEDULED else "n/a"
    return (cause, action, min(attempt, 3), bucket)


@dataclass
class Estimate:
    successes: float = 0.0
    trials: float = 0.0

    @property
    def rate(self) -> float:
        return (self.successes + PRIOR_ALPHA) / (self.trials + PRIOR_ALPHA + PRIOR_BETA)

    @property
    def stderr(self) -> float:
        p, n = self.rate, self.trials + PRIOR_ALPHA + PRIOR_BETA
        return (p * (1 - p) / n) ** 0.5


class BeliefTable:
    """Learned success rates, with a fallback chain for thin cells.

    Exact cell -> drop the delay bucket -> drop the attempt index -> hand-written
    value -> a flat default. Each step trades specificity for sample size, which
    is the only honest thing to do when a cell has eleven observations in it.
    """

    def __init__(self, cells: dict, hand_written: dict, default: float = 0.15) -> None:
        self.cells = cells
        self.hand_written = hand_written
        self.default = default

    def get(self, cause: RootCause, action: Action, attempt: int = 1,
            delay_hours: int = 0) -> float:
        exact = self.cells.get(key_for(cause, action, attempt, delay_hours))
        if exact is not None:
            return exact
        no_delay = self.cells.get((cause, action, min(attempt, 3), "n/a"))
        if no_delay is not None:
            return no_delay
        pooled = self.cells.get((cause, action))
        if pooled is not None:
            return pooled
        return self.hand_written.get((cause, action), self.default)

    def __len__(self) -> int:
        return len(self.cells)


def explore(rows: list[FailedPayment], *, episodes: int = 8, seed: int = 31
            ) -> dict[tuple, Estimate]:
    """Sample legal actions at random and record what happened.

    Uses its own RNG rather than the simulator's common-random-numbers path. That
    device exists so competing strategies face identical draws when they are
    being compared; reusing it here would hand every episode on a payment the
    same coin flip and collect no information at all.
    """
    rng = random.Random(seed)
    dg = TieredDiagnoser()
    table: dict[tuple[RootCause, Action], Estimate] = defaultdict(Estimate)

    for p in rows:
        cause = dg.diagnose(p).cause
        for _ in range(episodes):
            state = observe(p, local_hour=rng.randrange(24))
            allowed, _ = feasible_actions(state)
            options = sorted(allowed & EXPLORABLE, key=lambda a: a.value)
            if not options:
                continue
            action = rng.choice(options)
            attempt = rng.randint(1, 3)
            d = Decision(
                payment_id=p.payment_id, attempt=attempt, diagnosed_cause=cause,
                diagnosis_source="explore", diagnosis_confidence=1.0, action=action,
                channel=Channel.WHATSAPP if action in OUTREACH_ACTIONS else Channel.NONE,
                delay_hours=rng.choice([0, 6, 24, 48]))
            won = 1.0 if rng.random() < simulator.success_probability(p, d) else 0.0
            # Record at both granularities. The specific cell is what the policy
            # asks for; the pooled one is what it falls back to when the specific
            # cell is too thin to believe.
            for k in (key_for(cause, action, d.attempt, d.delay_hours), (cause, action)):
                e = table[k]
                e.trials += 1
                e.successes += won
    return dict(table)


def fit(rows: list[FailedPayment], *, hand_written: dict, episodes: int = 30
        ) -> tuple[BeliefTable, list[str]]:
    """Fit a belief table on these rows. Train only."""
    table = explore(rows, episodes=episodes)
    cells: dict = {}
    kept = dropped = 0
    for key, est in table.items():
        if est.trials < MIN_OBSERVATIONS:
            dropped += 1
            continue
        cells[key] = est.rate
        kept += 1

    notes = [f"{kept} cells fitted, {dropped} too thin to believe "
             f"(fewer than {MIN_OBSERVATIONS} observations)"]
    # Show the pooled cells against their hand-written counterparts; the
    # attempt/delay-specific cells are too numerous to print usefully.
    for (cause, action), old in sorted(hand_written.items(),
                                       key=lambda kv: kv[0][0].value):
        est = table.get((cause, action))
        if est is None or est.trials < MIN_OBSERVATIONS:
            notes.append(f"  kept  {cause.value:20} {action.value:22} "
                         f"{old:.3f}  (too few observations)")
            continue
        first = table.get(key_for(cause, action, 1, 24))
        detail = f", first attempt {first.rate:.3f}" if first and first.trials >= MIN_OBSERVATIONS else ""
        notes.append(f"  fit   {cause.value:20} {action.value:22} "
                     f"{old:.3f} -> {est.rate:.3f} +/- {est.stderr:.3f}"
                     f"  (n={int(est.trials)}{detail})")
    return BeliefTable(cells, hand_written), notes
