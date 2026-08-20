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

from .compliance import (MONEY_ACTIONS, OUTREACH_ACTIONS, Rail, SettlementState,
                         earliest_legal_hour, feasible_actions, observe)
from .models import (ACTION_COST_INR, CHANNEL_COST_INR, Action, Channel, Decision,
                     FailedPayment, RootCause)
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

CONFIG = {"min_confidence": MIN_CONFIDENCE_TO_ACT, "proposer": "rules"}


def set_proposer(mode: str) -> None:
    """'rules' for the hand-written ladder, 'ev' for belief-driven selection."""
    CONFIG["proposer"] = mode


def set_strict(on: bool) -> None:
    """Strict mode refuses to charge on any diagnosis the model was unsure of."""
    CONFIG["min_confidence"] = STRICT_MIN_CONFIDENCE_TO_ACT if on else MIN_CONFIDENCE_TO_ACT
# Chasing a small payment with a costly action is negative-value by construction.
MIN_AMOUNT_FOR_MANUAL_ESCALATION_INR = 2000.0
MIN_AMOUNT_TO_CHASE_INR = 60.0

# The agent's *beliefs* about how likely each action is to work. Deliberately a
# separate table from the simulator's ground truth -- an agent that knows the
# world model exactly is not an agent, it is a lookup.
# Hand-written fallbacks. `reclaim/learn.py` fits a richer table -- conditioned
# on the attempt index and, for scheduled retries, on the delay bucket -- and
# these remain as the fallback for cells too thin to estimate. They are keyed on
# the *diagnosed* cause, which is all the policy ever knows.
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


class _FlatBeliefs:
    """The hand-written table behind the same interface as a fitted one."""

    def get(self, cause: RootCause, action: Action, attempt: int = 1,
            delay_hours: int = 0) -> float:
        return BELIEVED_SUCCESS.get((cause, action), 0.15)


# Swapped for a fitted BeliefTable by set_beliefs(). Defaults to the
# hand-written values so the policy works with no training step at all.
BELIEFS = _FlatBeliefs()


def set_beliefs(table) -> None:
    """Install a fitted belief table. Fit on train; never on the evaluation set."""
    global BELIEFS
    BELIEFS = table if table is not None else _FlatBeliefs()


@dataclass
class Ledger:
    """Counters that span workflows.

    The network caps are per card per 30 days and the contact budget is per
    customer per week -- neither is a property of any single recovery workflow. A
    customer with six subscriptions can breach the Visa cap without any one
    workflow exceeding its own budget, which is precisely the bug that made the
    earlier per-payment budgets insufficient.
    """

    reattempts_30d: dict[str, int] = field(default_factory=dict)
    contacts_7d: dict[str, int] = field(default_factory=dict)
    mandate_cycle_attempts: dict[str, int] = field(default_factory=dict)

    def note_money_action(self, p: FailedPayment) -> None:
        self.reattempts_30d[p.customer_id] = self.reattempts_30d.get(p.customer_id, 0) + 1
        if p.is_recurring:
            self.mandate_cycle_attempts[p.payment_id] = \
                self.mandate_cycle_attempts.get(p.payment_id, 0) + 1

    def note_contact(self, p: FailedPayment) -> None:
        self.contacts_7d[p.customer_id] = self.contacts_7d.get(p.customer_id, 0) + 1


@dataclass
class RecoveryState:
    """Everything the policy is allowed to remember about a payment in flight."""

    money_attempts: int = 0
    outreach_count: int = 0
    clock_hour: int = 0
    escalated: bool = False
    reconciled: bool = False
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


# --- choosing by expected value instead of by rule ---------------------------
#
# `propose` above is a hand-written ladder: if the cause is X on attempt N, do Y.
# It encodes real domain knowledge and it is readable, which is worth a lot. But
# it cannot use a fitted belief table, and fitting one had no effect on outcomes
# precisely because nothing downstream consulted it -- the expected-value gate
# only ever fires on payments too small to chase, so it was inert.
#
# This proposer picks the legal action, and for a scheduled retry the delay, that
# maximises believed expected value. With the hand-written flat beliefs it should
# do roughly nothing, since flat beliefs cannot distinguish a 48-hour retry from
# an immediate one. With fitted beliefs it can.

# Delays worth considering for a scheduled retry. Deliberately coarse: these
# match the buckets the belief table is fitted on, and proposing a delay finer
# than the evidence supports would be false precision.
CANDIDATE_DELAYS = (6, 24, 48)


def propose_by_ev(p: FailedPayment, dx: Diagnosis, st: RecoveryState,
                  allowed: set[Action]) -> Decision:
    """Pick the highest-expected-value legal action, and say why."""
    attempt = st.money_attempts + st.outreach_count + 1
    options: list[tuple[float, Action, Channel, int]] = []

    for action in sorted(allowed & (MONEY_ACTIONS | OUTREACH_ACTIONS),
                         key=lambda a: a.value):
        if action in st.tried and action is not Action.RETRY_SCHEDULED:
            continue                      # repeating the same lever rarely pays
        channel = Channel.WHATSAPP if action in OUTREACH_ACTIONS else Channel.NONE
        delays = CANDIDATE_DELAYS if action is Action.RETRY_SCHEDULED else (0,)
        for delay in delays:
            rate = BELIEFS.get(dx.cause, action, attempt, delay)
            cost = ACTION_COST_INR[action] + CHANNEL_COST_INR[channel]
            options.append((rate * p.amount_inr - cost, action, channel, delay))

    if not options:
        return Decision(
            payment_id=p.payment_id, attempt=attempt, diagnosed_cause=dx.cause,
            diagnosis_source=dx.source, diagnosis_confidence=dx.confidence,
            action=Action.STOP, rationale="no legal action has positive value")

    ev, action, channel, delay = max(options, key=lambda t: t[0])
    if ev <= 0:
        return Decision(
            payment_id=p.payment_id, attempt=attempt, diagnosed_cause=dx.cause,
            diagnosis_source=dx.source, diagnosis_confidence=dx.confidence,
            action=Action.STOP,
            rationale=f"best legal option is worth INR {ev:,.2f}; stopping")

    rate = BELIEFS.get(dx.cause, action, attempt, delay)
    return Decision(
        payment_id=p.payment_id, attempt=attempt, diagnosed_cause=dx.cause,
        diagnosis_source=dx.source, diagnosis_confidence=dx.confidence,
        action=action, channel=channel, delay_hours=delay,
        rationale=f"highest expected value of {len(options)} legal options: "
                  f"believed {rate:.0%} success, INR {ev:,.0f} expected")


# --- the brakes --------------------------------------------------------------

def apply_guardrails(d: Decision, p: FailedPayment, st: RecoveryState) -> Decision:
    """Weaken the proposed action wherever a rule says we must.

    Ordered most-severe first. Each veto records itself in `blocked_by`, so the
    audit trail shows not just what we did but what we were stopped from doing.
    """
    money_actions = MONEY_ACTIONS

    # Legality is no longer decided here. Rules 1 and 2 used to re-check risk
    # declines and dead instruments off the *diagnosed* cause, which is exactly
    # how a wrong diagnosis used to reach the money. The kernel now decides both
    # from the raw decline reason, before this function is ever called, so what
    # remains below is business policy: budgets, courtesy, and value.

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
        ev = BELIEFS.get(d.diagnosed_cause, d.action, d.attempt, d.delay_hours) * p.amount_inr
        if ev < d.cost_inr or p.amount_inr < MIN_AMOUNT_TO_CHASE_INR:
            d.blocked_by = "negative_expected_value"
            d.action, d.channel = Action.STOP, Channel.NONE

    return d


# --- the kernel comes first --------------------------------------------------

def _state_for(p: FailedPayment, st: RecoveryState, ledger: Ledger):
    """Assemble the observable facts. Nothing here is a model output."""
    # A pre-debit notification covers the scheduled debit. A *retry* is another
    # debit, so it needs its own notice -- which is why a mandate retry can never
    # be immediate, and why the sequencer's job is choosing which moments to spend.
    notice = 25.0 if st.money_attempts == 0 else None
    s = observe(
        p,
        local_hour=st.clock_hour,
        reattempts_30d=ledger.reattempts_30d.get(p.customer_id, 0),
        mandate_cycle_attempts=ledger.mandate_cycle_attempts.get(p.payment_id, 0),
        contacts_7d=ledger.contacts_7d.get(p.customer_id, 0),
        hours_since_pre_debit_notice=notice,
    )
    if st.reconciled:
        # We asked the gateway and it told us. The outcome is no longer unknown.
        s.settlement_state = SettlementState.FAILED
    return s


def _blocking_rule(action: Action, fired) -> str | None:
    for v in fired:
        if action in v.forbids:
            return v.rule
    return None


def project(d: Decision, allowed: set[Action], fired, state, st: RecoveryState) -> Decision:
    """Move a proposal into the legal set, preferring to defer over to abandon.

    A rule that says 'not now' is not a rule that says 'never'. Most of the
    recoverable money on the timing-constrained rails is in that distinction: an
    Autopay retry blocked at 11:00 is worth scheduling for 13:00, not dropping.
    """
    if d.action in allowed:
        return d

    d.blocked_by = _blocking_rule(d.action, fired)

    if d.action in MONEY_ACTIONS and Action.RETRY_SCHEDULED in allowed:
        hour = earliest_legal_hour(state, Action.RETRY_NOW)
        delay = int((hour - state.local_hour) % 24) if hour is not None else 6
        # A retry on a mandate needs its own 24h pre-debit notice.
        if state.rail is Rail.UPI_AUTOPAY and (
                state.hours_since_pre_debit_notice is None
                or state.hours_since_pre_debit_notice < 24.0):
            delay = max(delay, 24)
        d.action, d.channel = Action.RETRY_SCHEDULED, Channel.NONE
        d.delay_hours = max(1, delay)
        d.rationale += f" (deferred {d.delay_hours}h: {d.blocked_by})"
        return d

    # A message blocked only by the clock is a message to send later, not a
    # reason to wake an analyst. An earlier version tested `d.action in allowed`
    # here, which is false by construction at this point in the function, so this
    # branch never fired and every after-hours nudge became an escalation.
    TIME_ONLY = {"outside_contact_window", "non_working_day"}
    if d.action in OUTREACH_ACTIONS and d.blocked_by in TIME_ONLY:
        hour = earliest_legal_hour(state, d.action)
        if hour is not None:
            d.delay_hours = max(1, int((hour - state.local_hour) % 24))
            d.rationale += f" (held to {hour:.0f}:00: {d.blocked_by})"
            return d

    d.action = Action.ESCALATE_MANUAL if Action.ESCALATE_MANUAL in allowed else Action.STOP
    d.channel = Channel.NONE
    d.rationale += f" (no legal alternative: {d.blocked_by})"
    return d


def decide(p: FailedPayment, dx: Diagnosis, st: RecoveryState,
           ledger: Ledger | None = None) -> Decision:
    """Legality first, then intent, then business policy.

    The ordering is the whole design. `feasible_actions` reads only facts, so the
    set it returns is unaffected by anything the diagnoser got wrong. Only inside
    that set does the model's opinion get to matter.
    """
    ledger = ledger if ledger is not None else Ledger()
    state = _state_for(p, st, ledger)
    allowed, fired = feasible_actions(state)

    # An unconfirmed outcome has exactly one correct next move, and it is not a
    # guess about the cause. Find out what actually happened first.
    if state.settlement_state is SettlementState.UNKNOWN:
        return Decision(
            payment_id=p.payment_id,
            attempt=st.money_attempts + st.outreach_count + 1,
            diagnosed_cause=dx.cause, diagnosis_source=dx.source,
            diagnosis_confidence=dx.confidence, action=Action.RECONCILE,
            rationale="the gateway never confirmed an outcome; checking before "
                      "doing anything that could debit the customer twice",
            blocked_by="unknown_settlement",
        )

    raw = (propose_by_ev(p, dx, st, allowed) if CONFIG["proposer"] == "ev"
           else propose(p, dx, st))
    d = project(raw, allowed, fired, state, st)
    d = apply_guardrails(d, p, st)
    d.kernel_cleared = d.action in allowed
    return d
