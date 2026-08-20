"""The four ways revenue slips away, and what they have in common.

The brief's framing is that loss "rarely happens in one clean step" -- a payment
degrades, a checkout is abandoned, a subscription fails, an invoice goes overdue.
Those look like four different problems and are usually built as four different
systems. They are not four problems. Each one is:

    something of known value stalled, and there is a window in which an
    intervention can still convert it.

What actually differs between them is only how loud the stall is:

    payment_failure   an explicit error code arrives. Detection is free.
    subscription      a mandate debit fails. Detection is free, but the retry
                      window is days long, not minutes.
    checkout_abandon  NOTHING arrives. The signal is an absence -- a session that
                      started and never finished. Detection is inference.
    receivable        an invoice quietly passes its due date. Detection is a
                      clock, and the valuable version is predicting it early.

So the recovery spine (diagnose -> decide -> guardrails -> act -> audit) is
shared, and only detection is written per surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Surface(str, Enum):
    PAYMENT_FAILURE = "payment_failure"
    CHECKOUT_ABANDON = "checkout_abandon"
    SUBSCRIPTION = "subscription"
    RECEIVABLE = "receivable"


@dataclass
class AtRiskItem:
    """A unit of revenue the detector believes is slipping away.

    This is the interface between detection and recovery. Everything downstream
    -- policy, guardrails, audit -- consumes this and does not care which
    surface produced it.
    """

    item_id: str
    surface: Surface
    customer_id: str
    amount_inr: float
    detected_at_hour: int          # hour of day we raised the flag
    hours_since_stall: float       # how long it had been stalled when we caught it
    evidence: dict[str, Any] = field(default_factory=dict)
    detector_score: float = 0.0    # confidence that this is genuinely at risk

    # --- ground truth, eval only. Never read by detector or policy. ---
    truly_at_risk: bool = True     # would this actually have been lost?
    would_self_recover: bool = False  # would the customer have converted anyway?

    @property
    def is_worth_chasing(self) -> bool:
        """The honest target: at risk AND not going to fix itself.

        Chasing someone who was always going to pay is not a win. It costs a
        message, it costs goodwill, and on a big enough base it costs more than
        the revenue it 'saves'. This is the label the detector is scored on.
        """
        return self.truly_at_risk and not self.would_self_recover


# How long after a stall an intervention is still worth making, per surface.
# Past this the money is usually gone regardless of what we do.
RECOVERY_WINDOW_HOURS: dict[Surface, float] = {
    Surface.PAYMENT_FAILURE: 72.0,
    Surface.CHECKOUT_ABANDON: 24.0,    # abandoned carts go cold fast
    Surface.SUBSCRIPTION: 168.0,       # a week of mandate retries is normal
    Surface.RECEIVABLE: 720.0,         # 30 days; B2B collections is a slow game
}
