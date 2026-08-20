"""A world model to score against.

This is the honest weak point of the project and it is labelled as such: no real
merchant is going to hand a student live retry outcomes. So outcomes are drawn
from a hand-specified table of success probabilities.

Two design choices keep the resulting numbers meaningful rather than circular:

  1. This table is NOT the table the policy believes (see policy.BELIEVED_SUCCESS).
     The agent is scored against a world it does not have memorised.
  2. Common random numbers: the coin flip for a given (payment, attempt) is seeded
     from the payment id, so every strategy faces the identical draw. Differences
     between strategies are differences in decisions, not luck.

`eval/run_eval.py --sensitivity` re-runs the whole comparison under perturbed
tables to show the ranking is not an artefact of these exact constants.
"""
from __future__ import annotations

import hashlib
import random

from .compliance import MONEY_ACTIONS
from .models import Action, Decision, FailedPayment, RootCause

# Ground-truth P(success) for a first attempt.
TRUE_SUCCESS: dict[tuple[RootCause, Action], float] = {
    (RootCause.TRANSIENT_ISSUER, Action.RETRY_NOW): 0.50,
    (RootCause.TRANSIENT_ISSUER, Action.RETRY_SCHEDULED): 0.68,
    (RootCause.TRANSIENT_ISSUER, Action.SWITCH_METHOD): 0.45,
    (RootCause.TRANSIENT_ISSUER, Action.NUDGE_CUSTOMER): 0.05,

    (RootCause.INSUFFICIENT_FUNDS, Action.RETRY_NOW): 0.08,
    (RootCause.INSUFFICIENT_FUNDS, Action.RETRY_SCHEDULED): 0.18,  # upgraded by delay below
    (RootCause.INSUFFICIENT_FUNDS, Action.SWITCH_METHOD): 0.10,
    (RootCause.INSUFFICIENT_FUNDS, Action.NUDGE_CUSTOMER): 0.28,

    (RootCause.AUTH_FRICTION, Action.RETRY_NOW): 0.20,
    (RootCause.AUTH_FRICTION, Action.RETRY_SCHEDULED): 0.24,
    (RootCause.AUTH_FRICTION, Action.SWITCH_METHOD): 0.18,
    (RootCause.AUTH_FRICTION, Action.NUDGE_CUSTOMER): 0.42,

    # A dead instrument is dead. No amount of retrying changes that -- this zero
    # is the single biggest source of wasted spend in the naive baselines.
    (RootCause.INSTRUMENT_INVALID, Action.RETRY_NOW): 0.00,
    (RootCause.INSTRUMENT_INVALID, Action.RETRY_SCHEDULED): 0.00,
    (RootCause.INSTRUMENT_INVALID, Action.SWITCH_METHOD): 0.12,
    (RootCause.INSTRUMENT_INVALID, Action.REQUEST_INSTRUMENT_UPDATE): 0.26,

    (RootCause.LIMIT_EXCEEDED, Action.RETRY_NOW): 0.05,
    (RootCause.LIMIT_EXCEEDED, Action.RETRY_SCHEDULED): 0.30,       # upgraded by delay below
    (RootCause.LIMIT_EXCEEDED, Action.SWITCH_METHOD): 0.40,
    (RootCause.LIMIT_EXCEEDED, Action.NUDGE_CUSTOMER): 0.12,

    (RootCause.RISK_DECLINED, Action.RETRY_NOW): 0.02,
    (RootCause.RISK_DECLINED, Action.RETRY_SCHEDULED): 0.02,
    (RootCause.RISK_DECLINED, Action.SWITCH_METHOD): 0.02,
    (RootCause.RISK_DECLINED, Action.NUDGE_CUSTOMER): 0.02,
}

# Manual review recovers some money, slowly, at analyst cost.
MANUAL_REVIEW_SUCCESS = 0.20
# Each successive attempt on the same payment is worth less.
ATTEMPT_DECAY = 0.80


def _coin(payment_id: str, attempt: int) -> float:
    """Deterministic uniform draw, shared across strategies."""
    h = hashlib.sha256(f"{payment_id}:{attempt}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big")).random()


def success_probability(p: FailedPayment, d: Decision, noise: float = 0.0) -> float:
    cause = p.true_root_cause
    if d.action is Action.ESCALATE_MANUAL:
        return MANUAL_REVIEW_SUCCESS
    if d.action is Action.STOP:
        return 0.0

    base = TRUE_SUCCESS.get((cause, d.action), 0.05)

    # Timing is the real lever for these two causes: waiting for a balance to be
    # topped up or a daily cap to reset is what actually converts.
    if cause is RootCause.INSUFFICIENT_FUNDS and d.action is Action.RETRY_SCHEDULED:
        base = 0.45 if d.delay_hours >= 24 else (0.30 if d.delay_hours >= 8 else 0.12)
    if cause is RootCause.LIMIT_EXCEEDED and d.action is Action.RETRY_SCHEDULED:
        base = 0.55 if d.delay_hours >= 8 else 0.10

    base *= ATTEMPT_DECAY ** (d.attempt - 1)
    # A customer who keeps failing is a worse bet.
    base *= 1.0 - min(0.30, 0.10 * p.customer_prior_failures)
    return max(0.0, min(1.0, base * (1.0 + noise)))


def attempt_succeeds(p: FailedPayment, d: Decision, noise: float = 0.0) -> bool:
    return _coin(p.payment_id, d.attempt) < success_probability(p, d, noise)


def is_compliance_violation(p: FailedPayment, d: Decision) -> bool:
    """Automatically re-charging a risk-declined payment. Should never happen."""
    return p.true_root_cause is RootCause.RISK_DECLINED and d.action in {
        Action.RETRY_NOW, Action.RETRY_SCHEDULED, Action.SWITCH_METHOD,
    }


def is_wasted_retry(p: FailedPayment, d: Decision) -> bool:
    """A gateway fee spent on an instrument with a literal zero success rate."""
    return p.true_root_cause is RootCause.INSTRUMENT_INVALID and d.action in {
        Action.RETRY_NOW, Action.RETRY_SCHEDULED,
    }
