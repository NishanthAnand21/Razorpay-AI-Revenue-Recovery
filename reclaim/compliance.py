"""The compliance kernel: what is *legal* to do, decided without any model.

WHY THIS MODULE EXISTS
----------------------
The first version of this system put guardrails after the model: the diagnoser
said "this is a transient issuer error", the policy proposed a retry, and a
guardrail checked the *diagnosed cause* before allowing it. Evaluation showed the
obvious consequence -- when the model misread an ambiguous decline, the guardrail
was reading a wrong label and waved the action through. Three violations on 240
payments, caused entirely by the safety layer trusting the thing it was meant to
be protecting against.

That is a design error, not a tuning problem, and researching the actual rules
showed why. Every hard constraint in Indian payments is decidable from facts a
system of record can assert, with no inference at all:

    Visa Category 1 response codes may never be reattempted -- the first retry is
    a violation, and it is a fee. (04, 07, 12, 14, 15, 41, 43, 46, 57, R0, R1, R3)
    Visa allows at most 15 reattempts per 30 days; Mastercard at most 10.
    A UPI Autopay mandate gets 1 execution plus 3 retries per cycle. Then stop.
    UPI Autopay may only execute in non-peak windows: before 10:00, 13:00-17:00,
    and after 21:30.
    RBI's Digital Payments E-mandate Framework (2026) requires a pre-debit
    notification at least 24 hours before every debit, with an opt-out.
    RBI's recovery-agent rules confine collections contact to 08:00-19:00 on
    working days -- and that covers SMS, WhatsApp and email, not just calls.

Not one of those needs a model. They are functions of a response code, a counter,
a clock and a calendar.

So the control flow is inverted. Instead of

    model proposes  ->  guardrail vetoes

we do

    kernel computes the legal set  ->  model chooses inside it

The model never sees an illegal option, so it cannot pick one. That turns "we
tuned the confidence threshold until violations went to zero" into a property
that holds no matter how wrong the model is -- which is testable, and is tested
in eval/run_adversarial.py by swapping in a diagnoser that is deliberately
malicious and confirming the violation count stays at zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .models import Action, Channel


class Rail(str, Enum):
    CARD_VISA = "card_visa"
    CARD_MASTERCARD = "card_mastercard"
    UPI_AUTOPAY = "upi_autopay"        # standing mandate
    UPI_COLLECT = "upi_collect"        # one-off pull
    NETBANKING = "netbanking"
    INVOICE = "invoice"                # no rail; a receivable being chased


class SettlementState(str, Enum):
    """Whether we actually know the outcome of the last attempt.

    UNKNOWN is the dangerous one and the reason this field exists. A gateway
    timeout does not mean the payment failed -- it means nobody told us. Retrying
    on a timeout is how customers get charged twice, and it is the single most
    expensive bug a recovery system can ship.
    """

    FAILED = "failed"
    UNKNOWN = "unknown"
    SUCCEEDED = "succeeded"


# Visa Category 1, "never retry". A reattempt is a violation on the first try.
VISA_NEVER_RETRY = {"04", "07", "12", "14", "15", "41", "43", "46", "57",
                    "R0", "R1", "R3"}
# Mastercard's do-not-retry merchant advice codes.
MASTERCARD_NEVER_RETRY = {"MAC03", "MAC21"}

VISA_REATTEMPT_CAP_30D = 15
MASTERCARD_REATTEMPT_CAP_30D = 10
UPI_AUTOPAY_ATTEMPTS_PER_CYCLE = 4          # 1 original + 3 retries
EMANDATE_PRE_DEBIT_NOTICE_HOURS = 24.0

# NPCI peak windows, during which Autopay must not execute.
UPI_PEAK_WINDOWS = ((10.0, 13.0), (17.0, 21.5))

# RBI recovery-agent contact window, working days only.
COLLECTIONS_CONTACT_WINDOW = (8.0, 19.0)
# Ordinary transactional messaging is less restricted than collections.
TRANSACTIONAL_CONTACT_WINDOW = (9.0, 21.0)

# Raw gateway reasons that must never be re-charged, on ANY rail.
#
# The card networks publish their own never-retry codes, but they only bind card
# traffic -- and a fraud decline on UPI is exactly as wrong to re-charge as one on
# Visa. An earlier version of this kernel only checked the card lists, so a
# security evaluation walked straight through it on UPI payments. These are raw
# reason strings straight off the gateway, so keying on them still involves no
# inference.
RISK_DECLINE_REASONS = {
    "suspected_fraud", "payment_declined_by_risk", "velocity_rule_triggered",
}
# Instruments that are permanently dead. Retrying is not a compliance breach here,
# it is a guaranteed-zero spend, so the kernel refuses it on value grounds.
DEAD_INSTRUMENT_REASONS = {
    "mandate_revoked", "card_reported_lost", "token_deprovisioned",
    "invalid_vpa", "card_expired", "debit_not_registered",
}

MONEY_ACTIONS = {Action.RETRY_NOW, Action.RETRY_SCHEDULED, Action.SWITCH_METHOD}
OUTREACH_ACTIONS = {Action.NUDGE_CUSTOMER, Action.REQUEST_INSTRUMENT_UPDATE}


@dataclass
class ObservableState:
    """Facts only. Nothing on this record is a model output.

    That restriction is the whole point: if a field here could be wrong because a
    model was wrong, the guarantee this module provides would be worthless.
    """

    rail: Rail
    network_response_code: str | None = None   # raw code from the network
    decline_reason: str | None = None          # raw reason slug from the gateway
    settlement_state: SettlementState = SettlementState.FAILED

    # counters, straight from the ledger
    reattempts_30d: int = 0                    # this card, rolling 30 days
    attempts_this_mandate_cycle: int = 0
    hours_since_pre_debit_notice: float | None = None

    # clock and calendar
    local_hour: float = 12.0
    is_working_day: bool = True

    # customer state
    contacts_7d: int = 0
    max_contacts_7d: int = 3
    consent_transactional: bool = True
    consent_marketing: bool = True
    mandate_active: bool = True
    is_disputed: bool = False
    is_collections: bool = False               # receivables chasing


@dataclass
class Veto:
    """One rule refusing one class of action, with the reason it exists."""

    rule: str
    forbids: set[Action]
    because: str
    citation: str = ""


def _in_window(hour: float, window: tuple[float, float]) -> bool:
    lo, hi = window
    return lo <= hour < hi


def _in_upi_peak(hour: float) -> bool:
    return any(lo <= hour < hi for lo, hi in UPI_PEAK_WINDOWS)


def vetoes(s: ObservableState) -> list[Veto]:
    """Every rule that fires against this state. Pure, ordered, and auditable."""
    out: list[Veto] = []

    # --- settlement ambiguity beats everything -------------------------------
    if s.settlement_state is SettlementState.UNKNOWN:
        out.append(Veto(
            "unknown_settlement", MONEY_ACTIONS | OUTREACH_ACTIONS,
            "last attempt's outcome is unconfirmed; charging again risks a double "
            "debit, and telling the customer they failed may be false",
            "idempotency / reconcile-before-retry",
        ))
    if s.settlement_state is SettlementState.SUCCEEDED:
        out.append(Veto(
            "already_paid", MONEY_ACTIONS | OUTREACH_ACTIONS,
            "the money arrived, most likely after we flagged it; there is nothing "
            "to recover and a nudge now is actively damaging",
            "self-recovery race",
        ))

    # --- rail-agnostic decline rules -----------------------------------------
    reason = (s.decline_reason or "").lower()
    if reason in RISK_DECLINE_REASONS:
        out.append(Veto(
            "risk_decline_never_recharged", MONEY_ACTIONS,
            f"the gateway declined this for risk ({reason}); re-charging it "
            "automatically is not permitted on any rail",
            "risk/compliance decline",
        ))
    if reason in DEAD_INSTRUMENT_REASONS:
        out.append(Veto(
            "dead_instrument", MONEY_ACTIONS,
            f"the instrument is permanently dead ({reason}); a retry has a zero "
            "success rate and costs a gateway fee",
            "value, not law",
        ))

    # --- network rules -------------------------------------------------------
    code = (s.network_response_code or "").upper()
    if s.rail is Rail.CARD_VISA and code in VISA_NEVER_RETRY:
        out.append(Veto(
            "visa_category_1", MONEY_ACTIONS,
            f"Visa response {code} is Category 1; the first reattempt is a "
            "violation and carries a per-transaction fee",
            "Visa rules for declined transaction resubmission",
        ))
    if s.rail is Rail.CARD_MASTERCARD and code in MASTERCARD_NEVER_RETRY:
        out.append(Veto(
            "mastercard_do_not_retry", MONEY_ACTIONS,
            f"Mastercard advice {code} means stop; further attempts burn the "
            "allowance and trigger fines",
            "Mastercard merchant advice codes",
        ))
    if s.rail is Rail.CARD_VISA and s.reattempts_30d >= VISA_REATTEMPT_CAP_30D:
        out.append(Veto(
            "visa_30d_cap", MONEY_ACTIONS,
            f"{s.reattempts_30d} reattempts in the last 30 days; Visa allows "
            f"{VISA_REATTEMPT_CAP_30D}",
            "Visa excessive reattempts fee",
        ))
    if s.rail is Rail.CARD_MASTERCARD and s.reattempts_30d >= MASTERCARD_REATTEMPT_CAP_30D:
        out.append(Veto(
            "mastercard_30d_cap", MONEY_ACTIONS,
            f"{s.reattempts_30d} reattempts in the last 30 days; Mastercard "
            f"allows {MASTERCARD_REATTEMPT_CAP_30D}",
            "Mastercard retry limits",
        ))

    # --- mandate rules -------------------------------------------------------
    if s.rail is Rail.UPI_AUTOPAY:
        if s.attempts_this_mandate_cycle >= UPI_AUTOPAY_ATTEMPTS_PER_CYCLE:
            out.append(Veto(
                "upi_autopay_cycle_cap", MONEY_ACTIONS,
                f"{s.attempts_this_mandate_cycle} attempts this cycle; NPCI allows "
                f"{UPI_AUTOPAY_ATTEMPTS_PER_CYCLE} (1 original + 3 retries)",
                "NPCI UPI Autopay rules, Aug 2025",
            ))
        if _in_upi_peak(s.local_hour):
            # Not a ban on retrying, a ban on retrying *now*. Scheduling survives.
            out.append(Veto(
                "upi_peak_hours", {Action.RETRY_NOW},
                f"{s.local_hour:04.1f} falls in an NPCI peak window; Autopay "
                "executes only before 10:00, 13:00-17:00, or after 21:30",
                "NPCI non-peak execution windows",
            ))
        notice = s.hours_since_pre_debit_notice
        if notice is None or notice < EMANDATE_PRE_DEBIT_NOTICE_HOURS:
            out.append(Veto(
                "emandate_pre_debit_notice", {Action.RETRY_NOW},
                "RBI requires a pre-debit notification at least 24h before every "
                "debit, with an opt-out; no valid notice is outstanding",
                "RBI Digital Payments E-mandate Framework, 2026",
            ))
    if not s.mandate_active:
        out.append(Veto(
            "mandate_revoked", MONEY_ACTIONS,
            "the customer revoked the mandate; debiting anyway is unauthorised",
            "e-mandate revocation",
        ))

    # --- contact rules -------------------------------------------------------
    window = COLLECTIONS_CONTACT_WINDOW if s.is_collections else TRANSACTIONAL_CONTACT_WINDOW
    if not _in_window(s.local_hour, window):
        out.append(Veto(
            "outside_contact_window", OUTREACH_ACTIONS,
            f"{s.local_hour:04.1f} is outside the permitted "
            f"{window[0]:.0f}:00-{window[1]:.0f}:00 window",
            "RBI recovery-agent guidelines" if s.is_collections else "messaging policy",
        ))
    if s.is_collections and not s.is_working_day:
        out.append(Veto(
            "non_working_day", OUTREACH_ACTIONS,
            "collections contact is confined to working days",
            "RBI recovery-agent guidelines",
        ))
    if s.contacts_7d >= s.max_contacts_7d:
        out.append(Veto(
            "contact_fatigue_cap", OUTREACH_ACTIONS,
            f"{s.contacts_7d} contacts in 7 days across all surfaces; the cap is "
            f"{s.max_contacts_7d}",
            "per-customer global budget",
        ))
    if not s.consent_transactional:
        out.append(Veto(
            "consent_withdrawn", OUTREACH_ACTIONS,
            "the customer withdrew consent to be contacted",
            "DPDP Act 2023",
        ))
    if s.is_disputed:
        out.append(Veto(
            "under_dispute", MONEY_ACTIONS | OUTREACH_ACTIONS,
            "the amount is formally disputed; chasing it can prejudice the "
            "dispute and reads as harassment",
            "dispute interlock",
        ))

    return out


ALL_ACTIONS = set(Action)


def feasible_actions(s: ObservableState) -> tuple[set[Action], list[Veto]]:
    """The legal set. Everything downstream chooses from this and only this."""
    fired = vetoes(s)
    forbidden: set[Action] = set()
    for v in fired:
        forbidden |= v.forbids
    allowed = ALL_ACTIONS - forbidden
    # STOP, ESCALATE_MANUAL and RECONCILE are always available: doing nothing,
    # asking a human, and asking the gateway what actually happened are never
    # themselves violations. RECONCILE in particular has to survive the
    # unknown-settlement veto, because it is the way out of that state.
    allowed |= {Action.STOP, Action.ESCALATE_MANUAL, Action.RECONCILE}
    return allowed, fired


def earliest_legal_hour(s: ObservableState, action: Action) -> float | None:
    """When this action first becomes legal, if it is only blocked by the clock.

    Lets the policy convert "not now" into "at 13:00" instead of giving up, which
    is where most of the recoverable money in the timing-constrained rails lives.
    """
    if action is Action.RETRY_NOW and s.rail is Rail.UPI_AUTOPAY and _in_upi_peak(s.local_hour):
        for lo, hi in UPI_PEAK_WINDOWS:
            if lo <= s.local_hour < hi:
                return hi
    if action in OUTREACH_ACTIONS:
        window = COLLECTIONS_CONTACT_WINDOW if s.is_collections else TRANSACTIONAL_CONTACT_WINDOW
        if s.local_hour < window[0]:
            return window[0]
        if s.local_hour >= window[1]:
            return window[0]  # next day
    return None


# --- adapter: our payment records -> observable facts -------------------------

# Raw gateway reason -> the network response code it corresponds to. This is a
# lookup over strings we literally received, so it involves no inference. Codes
# marked (C1) are Visa Category 1 and can never be reattempted.
REASON_TO_NETWORK_CODE: dict[str, str] = {
    "card_reported_lost": "41",           # (C1) lost card
    "mandate_revoked": "R1",              # (C1) revocation of authorisation
    "token_deprovisioned": "14",          # (C1) invalid account number
    "suspected_fraud": "07",              # (C1) pickup card, fraud
    "payment_declined_by_risk": "04",     # (C1) pickup card
    "velocity_rule_triggered": "57",      # (C1) transaction not permitted
    "card_expired": "54",                 # retryable in theory, pointless in fact
    "insufficient_funds": "51",           # retryable, and worth retrying
    "do_not_honour": "05",                # retryable -- see the note below
    "debit_declined": "05",
    "transaction_not_permitted": "57",    # (C1)
}

# Reasons where the outcome is genuinely unknown rather than known-failed.
UNCONFIRMED_REASONS = {"gateway_timeout", "acquirer_switch_error", "npci_downtime",
                       "payment_failed_at_bank"}


def observe(payment, *, local_hour: float, reattempts_30d: int = 0,
            mandate_cycle_attempts: int = 0, contacts_7d: int = 0,
            hours_since_pre_debit_notice: float | None = None,
            is_working_day: bool = True) -> ObservableState:
    """Build the fact record for a FailedPayment. No model output enters here."""
    reason = payment.error_reason
    if payment.is_recurring and payment.method == "upi":
        rail = Rail.UPI_AUTOPAY
    elif payment.method == "upi":
        rail = Rail.UPI_COLLECT
    elif payment.method == "card":
        # Real systems read the BIN. Ours splits on a digest of the id: Python's
        # str hash is salted per process, so `hash()` here would silently assign
        # a payment to a different network on every run and make the whole
        # evaluation unreproducible.
        import hashlib
        h = int.from_bytes(hashlib.sha256(payment.payment_id.encode()).digest()[:4], "big")
        rail = Rail.CARD_VISA if h % 3 else Rail.CARD_MASTERCARD
    elif payment.method == "netbanking":
        rail = Rail.NETBANKING
    else:
        rail = Rail.UPI_COLLECT

    settlement = (SettlementState.UNKNOWN if reason in UNCONFIRMED_REASONS
                  else SettlementState.FAILED)

    return ObservableState(
        rail=rail,
        network_response_code=REASON_TO_NETWORK_CODE.get(reason),
        decline_reason=reason,
        settlement_state=settlement,
        reattempts_30d=reattempts_30d,
        attempts_this_mandate_cycle=mandate_cycle_attempts,
        hours_since_pre_debit_notice=hours_since_pre_debit_notice,
        local_hour=float(local_hour),
        is_working_day=is_working_day,
        contacts_7d=contacts_7d,
        mandate_active=reason != "mandate_revoked",
        is_collections=False,
    )
