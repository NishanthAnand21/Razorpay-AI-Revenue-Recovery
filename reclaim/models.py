"""Core domain types for Reclaim.

Everything is a plain dataclass so the whole pipeline stays inspectable and
serialisable without pulling in a dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class RootCause(str, Enum):
    """Why a payment failed, at the level of abstraction that changes what we do.

    The taxonomy is deliberately action-oriented rather than error-code-shaped:
    two different gateway codes that call for the same intervention belong in
    the same class.
    """

    TRANSIENT_ISSUER = "transient_issuer"      # issuer/gateway blip; retrying works
    INSUFFICIENT_FUNDS = "insufficient_funds"  # money isn't there *yet*; timing matters
    AUTH_FRICTION = "auth_friction"            # OTP failed, collect request expired
    INSTRUMENT_INVALID = "instrument_invalid"  # expired card, dead VPA, revoked mandate
    LIMIT_EXCEEDED = "limit_exceeded"          # per-txn or daily cap hit
    RISK_DECLINED = "risk_declined"            # fraud/compliance block — never retry
    UNKNOWN = "unknown"                        # unmapped; routed to the LLM diagnoser


class Action(str, Enum):
    """The bounded set of things the agent is allowed to do."""

    RETRY_NOW = "retry_now"
    RETRY_SCHEDULED = "retry_scheduled"        # retry at a chosen future hour
    SWITCH_METHOD = "switch_method"            # re-attempt on a different instrument
    NUDGE_CUSTOMER = "nudge_customer"          # outreach asking the customer to act
    REQUEST_INSTRUMENT_UPDATE = "request_instrument_update"
    ESCALATE_MANUAL = "escalate_manual"        # hand to a human, take no money action
    STOP = "stop"                              # give up, on purpose, with a reason


class Channel(str, Enum):
    NONE = "none"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"


# Rupee cost of taking an action once. Retry cost is the gateway fee we eat on a
# failed attempt; outreach costs are per-message vendor rates.
ACTION_COST_INR: dict[Action, float] = {
    Action.RETRY_NOW: 2.0,
    Action.RETRY_SCHEDULED: 2.0,
    Action.SWITCH_METHOD: 2.0,
    Action.NUDGE_CUSTOMER: 0.0,               # priced by channel instead
    Action.REQUEST_INSTRUMENT_UPDATE: 0.0,
    Action.ESCALATE_MANUAL: 25.0,             # ~2 min of an ops analyst's time
    Action.STOP: 0.0,
}

CHANNEL_COST_INR: dict[Channel, float] = {
    Channel.NONE: 0.0,
    Channel.SMS: 0.15,
    Channel.WHATSAPP: 0.70,
    Channel.EMAIL: 0.02,
}


@dataclass
class FailedPayment:
    """A single payment that did not go through."""

    payment_id: str
    customer_id: str
    amount_inr: float
    method: str                    # card | upi | netbanking | wallet
    error_code: str                # gateway-level code
    error_reason: str              # gateway-level reason slug
    merchant_note: str             # free text, sometimes the only real signal
    failed_at_hour: int            # 0-23, local time of the failure
    is_recurring: bool             # subscription / mandate debit
    customer_prior_failures: int   # how many times this customer failed recently
    # Ground truth, used only for evaluation — never read by the agent.
    true_root_cause: RootCause

    def to_public_dict(self) -> dict[str, Any]:
        """The view the agent is allowed to see (no ground-truth leakage)."""
        d = asdict(self)
        d.pop("true_root_cause")
        return d


@dataclass
class Decision:
    """One bounded step the agent chose to take, with its reasoning."""

    payment_id: str
    attempt: int
    diagnosed_cause: RootCause
    diagnosis_source: str          # "rules" | "llm" | "llm_fallback"
    diagnosis_confidence: float
    action: Action
    channel: Channel = Channel.NONE
    delay_hours: int = 0
    rationale: str = ""
    blocked_by: str | None = None  # which guardrail vetoed a stronger action

    @property
    def cost_inr(self) -> float:
        return ACTION_COST_INR[self.action] + CHANNEL_COST_INR[self.channel]


@dataclass
class RecoveryOutcome:
    """What happened when we ran the whole recovery workflow for one payment."""

    payment_id: str
    amount_inr: float
    recovered: bool
    decisions: list[Decision] = field(default_factory=list)
    recovered_on_attempt: int | None = None

    @property
    def spend_inr(self) -> float:
        return sum(d.cost_inr for d in self.decisions)

    @property
    def net_inr(self) -> float:
        """Money actually gained: recovered amount less what we spent chasing it."""
        return (self.amount_inr if self.recovered else 0.0) - self.spend_inr
