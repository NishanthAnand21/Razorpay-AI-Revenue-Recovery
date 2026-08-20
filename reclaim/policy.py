"""What to do about a diagnosed failure, and -- more importantly -- what not to.

The value in a recovery agent is mostly in restraint. Retrying everything three
times is trivial to build and actively loses money: it burns gateway fees on
instruments that can never succeed, and it re-triggers fraud rules on payments a
risk engine already declined.

Every action here passes through `apply_guardrails`, which can only ever weaken
an action, never strengthen one. That gives a single place to audit and a single
place for a compliance reviewer to read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Action, Channel, Decision, FailedPayment, RootCause
from .diagnose import Diagnosis

# --- tunables (fitted on train, frozen before touching test) ------------------

MAX_MONEY_ATTEMPTS = 3        # retries/switches per payment
MAX_OUTREACH_PER_PAYMENT = 2  # nudges per payment
QUIET_HOURS = range(21, 24)   # plus 0..8, see _in_quiet_hours
QUIET_HOURS_MORNING = range(0, 9)
# Below this confidence we will not move money. This one constant is the whole
# safety/recovery tradeoff: the eval reports both settings side by side, because
# picking a point on that curve is a business decision, not an engineering one.
MIN_CONFIDENCE_TO_ACT = 0.50
STRICT_MIN_CONFIDENCE_TO_ACT = 0.60

CONFIG = {"min_confidence": MIN_CONFIDENCE_TO_ACT}


def set_strict(on: bool) -> None:
    """Strict mode refuses to charge on any diagnosis the model was unsure of."""
    CONFIG["min_confidence"] = STRICT_MIN_CONFIDENCE_TO_ACT if on else MIN_CONFIDENCE_TO_ACT
# Chasing a small payment with a costly action is negative-value by construction.
MIN_AMOUNT_FOR_MANUAL_ESCALATION_INR = 2000.0
MIN_AMOUNT_TO_CHASE_INR = 60.0

# The agent's *beliefs* about how likely each action is to work. Deliberately a
# separate table from the simulator's ground truth -- an agent that knows the
# world model exactly is not an agent, it is a lookup.
BELIEVED_SUCCESS: dict[tuple[RootCause, Action], float] = {
    (RootCause.TRANSIENT_ISSUER, Action.RETRY_NOW): 0.55,
    (RootCause.TRANSIENT_ISSUER, Action.RETRY_SCHEDULED): 0.62,
    (RootCause.INSUFFICIENT_FUNDS, Action.RETRY_SCHEDULED): 0.42,
    (RootCause.INSUFFICIENT_FUNDS, Action.NUDGE_CUSTOMER): 0.30,
    (RootCause.AUTH_FRICTION, Action.NUDGE_CUSTOMER): 0.46,
    (RootCause.AUTH_FRICTION, Action.RETRY_NOW): 0.22,
    (RootCause.LIMIT_EXCEEDED, Action.RETRY_SCHEDULED): 0.50,
    (RootCause.LIMIT_EXCEEDED, Action.SWITCH_METHOD): 0.44,
    (RootCause.INSTRUMENT_INVALID, Action.REQUEST_INSTRUMENT_UPDATE): 0.28,
}


@dataclass
class RecoveryState:
    """Everything the policy is allowed to remember about a payment in flight."""

    money_attempts: int = 0
    outreach_count: int = 0
    clock_hour: int = 0
    escalated: bool = False
    tried: list[Action] = field(default_factory=list)


def _in_quiet_hours(hour: int) -> bool:
    """No outreach late at night. TRAI-style courtesy, and it converts worse."""
    return hour in QUIET_HOURS or hour in QUIET_HOURS_MORNING


def _hours_until(target_hour: int, now_hour: int) -> int:
    return (target_hour - now_hour) % 24 or 24


# --- the plan ----------------------------------------------------------------

def propose(p: FailedPayment, dx: Diagnosis, st: RecoveryState) -> Decision:
    """Pick the best next action, ignoring guardrails. Guardrails come after."""
    cause = dx.cause
    d = Decision(
        payment_id=p.payment_id, attempt=st.money_attempts + st.outreach_count + 1,
        diagnosed_cause=cause, diagnosis_source=dx.source,
        diagnosis_confidence=dx.confidence, action=Action.STOP,
        rationale="no plan matched",
    )

    if cause is RootCause.TRANSIENT_ISSUER:
        # First attempt goes out immediately -- issuer blips clear in minutes.
        # A second attempt waits, because an instant repeat hits the same dead switch.
        if st.money_attempts == 0:
            d.action, d.rationale = Action.RETRY_NOW, "issuer blip; immediate retry usually clears"
        else:
            d.action, d.delay_hours = Action.RETRY_SCHEDULED, 6
            d.rationale = "issuer still degraded; backing off 6h before retrying"

    elif cause is RootCause.INSUFFICIENT_FUNDS:
        # Money problems are timing problems. Retrying now is near-worthless;
        # retrying when the balance is likely topped up is the whole trick.
        note = p.merchant_note.lower()
        if "salary" in note or "payday" in note:
            d.action, d.delay_hours = Action.RETRY_SCHEDULED, 48
            d.rationale = "customer flagged an incoming credit; retrying after payday"
        elif st.money_attempts == 0:
            d.action, d.delay_hours = Action.RETRY_SCHEDULED, _hours_until(10, st.clock_hour)
            d.rationale = "retrying at 10:00, after overnight credits settle"
        else:
            d.action, d.channel = Action.NUDGE_CUSTOMER, Channel.WHATSAPP
            d.rationale = "two silent attempts failed; asking the customer to fund the account"

    elif cause is RootCause.AUTH_FRICTION:
        # The customer has to do something. A silent retry cannot fix it.
        if st.outreach_count == 0:
            d.action, d.channel = Action.NUDGE_CUSTOMER, Channel.WHATSAPP
            d.rationale = "authentication was abandoned; sending a one-tap link to complete it"
        else:
            d.action, d.rationale = Action.RETRY_NOW, "re-issuing the collect request"

    elif cause is RootCause.LIMIT_EXCEEDED:
        if Action.RETRY_SCHEDULED not in st.tried:
            d.action, d.delay_hours = Action.RETRY_SCHEDULED, _hours_until(9, st.clock_hour)
            d.rationale = "daily cap resets at midnight; retrying next morning"
        else:
            d.action, d.rationale = Action.SWITCH_METHOD, "cap still binding; moving to another instrument"

    elif cause is RootCause.INSTRUMENT_INVALID:
        # Retrying a dead card is a pure loss, every single time.
        d.action, d.channel = Action.REQUEST_INSTRUMENT_UPDATE, Channel.SMS
        d.rationale = "instrument is permanently dead; retrying cannot succeed, asking for a new one"

    elif cause is RootCause.RISK_DECLINED:
        d.action = Action.ESCALATE_MANUAL
        d.rationale = "declined by risk; automated retry is not permitted"

    else:  # UNKNOWN
        d.action = Action.ESCALATE_MANUAL
        d.rationale = "cause not established; handing to a human instead of guessing with money"

    return d


# --- the brakes --------------------------------------------------------------

def apply_guardrails(d: Decision, p: FailedPayment, st: RecoveryState) -> Decision:
    """Weaken the proposed action wherever a rule says we must.

    Ordered most-severe first. Each veto records itself in `blocked_by`, so the
    audit trail shows not just what we did but what we were stopped from doing.
    """
    money_actions = {Action.RETRY_NOW, Action.RETRY_SCHEDULED, Action.SWITCH_METHOD}

    # 1. Hard compliance stop. Nothing overrides this.
    if d.diagnosed_cause is RootCause.RISK_DECLINED and d.action in money_actions:
        d.blocked_by = "risk_declined_never_retried"
        d.action, d.channel = Action.ESCALATE_MANUAL, Channel.NONE

    # 2. Never spend a gateway fee on an instrument that cannot work.
    if d.diagnosed_cause is RootCause.INSTRUMENT_INVALID and d.action in money_actions:
        d.blocked_by = "dead_instrument_not_retried"
        d.action, d.channel = Action.REQUEST_INSTRUMENT_UPDATE, Channel.SMS

    # 3. Don't move money on a diagnosis we don't believe.
    if d.diagnosis_confidence < CONFIG["min_confidence"] and d.action in money_actions:
        d.blocked_by = "low_confidence_diagnosis"
        d.action, d.channel = Action.ESCALATE_MANUAL, Channel.NONE

    # 4. Attempt budget.
    if d.action in money_actions and st.money_attempts >= MAX_MONEY_ATTEMPTS:
        d.blocked_by = "attempt_budget_exhausted"
        d.action, d.channel = Action.STOP, Channel.NONE

    # 5. Contact budget -- we are not going to harass anyone.
    outreach = {Action.NUDGE_CUSTOMER, Action.REQUEST_INSTRUMENT_UPDATE}
    if d.action in outreach and st.outreach_count >= MAX_OUTREACH_PER_PAYMENT:
        d.blocked_by = "outreach_budget_exhausted"
        d.action, d.channel = Action.STOP, Channel.NONE

    # 6. Quiet hours: defer the message rather than cancel it.
    if d.action in outreach and _in_quiet_hours(st.clock_hour):
        d.blocked_by = "quiet_hours_deferred"
        d.delay_hours = _hours_until(9, st.clock_hour)

    # 7. Don't spend an analyst on a payment worth less than the analyst.
    if d.action is Action.ESCALATE_MANUAL and p.amount_inr < MIN_AMOUNT_FOR_MANUAL_ESCALATION_INR:
        d.blocked_by = "below_manual_review_threshold"
        d.action, d.channel = Action.STOP, Channel.NONE
        d.rationale += " (too small to justify a human review)"

    # 8. Expected value. If believed recovery can't cover the cost, don't bother.
    if d.action in money_actions:
        ev = BELIEVED_SUCCESS.get((d.diagnosed_cause, d.action), 0.15) * p.amount_inr
        if ev < d.cost_inr or p.amount_inr < MIN_AMOUNT_TO_CHASE_INR:
            d.blocked_by = "negative_expected_value"
            d.action, d.channel = Action.STOP, Channel.NONE

    return d


def decide(p: FailedPayment, dx: Diagnosis, st: RecoveryState) -> Decision:
    return apply_guardrails(propose(p, dx, st), p, st)
