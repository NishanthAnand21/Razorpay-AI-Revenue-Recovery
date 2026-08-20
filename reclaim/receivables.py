"""Chasing B2B invoices without burning the customer you are chasing.

Receivables are the surface where the compliance kernel does the most work, and
where the naive approach does the most damage.

WHAT MAKES THIS DIFFERENT FROM CONSUMER RECOVERY
------------------------------------------------
A failed card payment is a transaction. An overdue invoice is a *relationship*
that happens to have money outstanding, and the buyer will still be a customer
next quarter. So the objective is not "extract this invoice" but "get paid sooner
without spending the account", and the metric is days sales outstanding rather
than a recovery rate.

THE LEVER NOBODY PULLS
----------------------
Under s.15 of the MSMED Act a buyer must pay a registered micro or small supplier
within 45 days of acceptance, and under s.16 anything later accrues compound
interest at three times the RBI bank rate with monthly rests. A supplier who is
eligible is owed that money by operation of law, and most never mention it.

It is also a claim that must be true. Asserting a statutory remedy a supplier is
not entitled to is a false legal claim, so eligibility is a fact the compliance
kernel checks -- `not_an_msme_supplier` vetoes the interest notice and the MSEFC
referral outright, and no amount of model confidence unlocks them.

PROMISES
--------
The single most useful signal in collections is a promise to pay, and the single
most common way to waste one is to keep chasing after receiving it. A promise is
leverage precisely because it was voluntary; chasing through it converts a
cooperative buyer into an uncooperative one and costs the promise as well. So an
open promise vetoes outreach in the kernel, and a *broken* promise is what earns
escalation -- not elapsed time alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .models import Action, Channel

# RBI bank rate, and the statutory multiple under s.16.
RBI_BANK_RATE = 0.0625
STATUTORY_MULTIPLE = 3.0
STATUTORY_ANNUAL_RATE = RBI_BANK_RATE * STATUTORY_MULTIPLE     # 18.75%

MAX_TERMS_DAYS = 45          # s.15 ceiling, whatever the contract says
PROMISE_GRACE_DAYS = 2       # a day or two late on a promise is not a broken one


class Stage(IntEnum):
    """The escalation ladder. Each rung is harder to walk back than the last."""

    PRE_DUE = 0        # a reminder before the money is even late
    DUE = 1            # it is due today
    SOFT = 2           # a polite chase
    FIRM = 3           # a formal notice, with statutory interest if it applies
    REFER = 4          # a filing with the facilitation council


# How far past the appointed day each rung becomes appropriate.
STAGE_THRESHOLD_DAYS = {Stage.PRE_DUE: -3, Stage.DUE: 0, Stage.SOFT: 7,
                        Stage.FIRM: 21, Stage.REFER: 45}


def statutory_interest(principal: float, days_late: int) -> float:
    """Compound interest under s.16, at monthly rests.

    Monthly rests, not daily: the Act says monthly, and quoting a number the Act
    does not support is the fastest way to lose the credibility the notice is
    supposed to buy.
    """
    if days_late <= 0:
        return 0.0
    months = days_late / 30.0
    monthly = STATUTORY_ANNUAL_RATE / 12.0
    return principal * ((1.0 + monthly) ** months - 1.0)


def appointed_day(accepted_day: int, agreed_terms_days: int) -> int:
    """The day payment falls due, capped by statute regardless of contract."""
    return accepted_day + min(agreed_terms_days, MAX_TERMS_DAYS)


@dataclass
class Invoice:
    invoice_id: str
    buyer_id: str
    amount_inr: float
    issued_day: int
    accepted_day: int
    agreed_terms_days: int
    supplier_is_msme: bool
    disputed: bool
    # Observable: how often this buyer has paid late before. The only signal we
    # actually have about whether chasing them will move anything.
    buyer_prior_late_ratio: float = 0.0
    buyer_prior_invoices: int = 0
    # ground truth, eval only
    would_pay_on_day: int = 0        # when this buyer pays with no chasing at all
    reliability: float = 0.5         # how much a chase actually moves them

    @property
    def due_day(self) -> int:
        return appointed_day(self.accepted_day, self.agreed_terms_days)


@dataclass
class ChaseState:
    """What we have already done, and what the buyer said."""

    stage: Stage = Stage.PRE_DUE
    contacts: int = 0
    promise_day: int | None = None
    promises_broken: int = 0
    referred: bool = False
    touched_days: list[int] = field(default_factory=list)

    def promise_open(self, today: int) -> bool:
        return self.promise_day is not None and today <= self.promise_day + PROMISE_GRACE_DAYS

    def promise_broken(self, today: int) -> bool:
        return self.promise_day is not None and today > self.promise_day + PROMISE_GRACE_DAYS


def target_stage(inv: Invoice, st: ChaseState, today: int) -> Stage:
    """Which rung this invoice has earned.

    Elapsed time proposes; behaviour disposes. A buyer who has broken a promise
    skips a rung, because the thing that predicts non-payment is not lateness --
    most late payers pay -- it is having said they would and not done so.
    """
    days_past = today - inv.due_day
    stage = Stage.PRE_DUE
    for s, threshold in STAGE_THRESHOLD_DAYS.items():
        if days_past >= threshold:
            stage = s
    if st.promise_broken(today) and stage < Stage.REFER:
        stage = Stage(stage + 1)
    return stage


def next_action(inv: Invoice, st: ChaseState, today: int) -> tuple[Action, Channel, str]:
    """The chase we would like to make. The kernel decides whether we may."""
    stage = target_stage(inv, st, today)
    days_past = today - inv.due_day

    if stage is Stage.REFER:
        interest = statutory_interest(inv.amount_inr, days_past)
        # The rationale goes into an audit trail that a buyer may eventually
        # read, so it states what is actually on the record and nothing else.
        because = ("after a broken commitment to pay" if st.promises_broken
                   else "with no payment commitment obtained")
        return (Action.REFER_MSEFC, Channel.EMAIL,
                f"{days_past}d past the appointed day {because}; referring, "
                f"claiming INR {interest:,.0f} of statutory interest accrued")

    if stage is Stage.FIRM:
        interest = statutory_interest(inv.amount_inr, days_past)
        return (Action.ISSUE_INTEREST_NOTICE, Channel.EMAIL,
                f"{days_past}d past the appointed day; formal notice quoting "
                f"s.16 interest of INR {interest:,.0f} accrued to date")

    if stage is Stage.SOFT:
        return (Action.NUDGE_CUSTOMER, Channel.WHATSAPP,
                f"{days_past}d past due; asking for a payment date")

    if stage is Stage.DUE:
        return (Action.NUDGE_CUSTOMER, Channel.EMAIL, "due today; a courtesy note")

    return (Action.NUDGE_CUSTOMER, Channel.EMAIL,
            f"due in {-days_past}d; reminder while it is still nobody's fault")
