"""The whole pipeline, in one place.

Raw merchant events in; audited, compliance-cleared actions out.

    detect ──▶ prioritise under capacity ──▶ route by surface ──▶ kernel gate
           ──▶ execute ──▶ hash-chained audit

Four things are deliberately shared across every surface rather than
reimplemented per surface, because each one is a place where per-surface
implementations silently disagree in production:

  the compliance kernel      one definition of what is legal
  the contact ledger         one budget per customer, across all surfaces, so a
                             customer with a failed subscription, an abandoned
                             cart and an overdue invoice does not receive three
                             messages from three subsystems that cannot see each
                             other
  the capacity budget        one queue, so surfaces compete on value rather than
                             each spending freely inside its own silo
  the audit log              one chain, so the trail for a customer is a single
                             ordered record rather than four partial ones
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .compliance import ObservableState, Rail, feasible_actions
from .models import ACTION_COST_INR, CHANNEL_COST_INR, Action, Channel
from .security import AuditLog, idempotency_key
from .surfaces import AtRiskItem, Surface

# What a single intervention costs on each surface, and what fraction of the
# money it recovers when it lands. Kept together so the routing decision is
# legible in one place.
SURFACE_UPLIFT = {Surface.CHECKOUT_ABANDON: 0.11, Surface.SUBSCRIPTION: 0.35,
                  Surface.RECEIVABLE: 0.25, Surface.PAYMENT_FAILURE: 0.30}


@dataclass
class Intervention:
    item_id: str
    surface: Surface
    customer_id: str
    amount_inr: float
    action: Action
    channel: Channel
    expected_value_inr: float
    idempotency_key: str
    cleared: bool
    blocked_by: str | None = None

    @property
    def cost_inr(self) -> float:
        return ACTION_COST_INR.get(self.action, 0.0) + CHANNEL_COST_INR.get(self.channel, 0.0)


@dataclass
class RunReport:
    considered: int = 0
    interventions: list[Intervention] = field(default_factory=list)
    blocked: dict[str, int] = field(default_factory=dict)
    skipped_capacity: int = 0
    skipped_value: int = 0
    audit_head: str = ""

    @property
    def expected_recovery_inr(self) -> float:
        return sum(i.expected_value_inr for i in self.interventions)

    @property
    def spend_inr(self) -> float:
        return sum(i.cost_inr for i in self.interventions)


# Which action each surface wants. The kernel decides whether it may have it.
SURFACE_ACTION = {
    Surface.PAYMENT_FAILURE: (Action.RETRY_SCHEDULED, Channel.NONE),
    Surface.SUBSCRIPTION: (Action.RETRY_SCHEDULED, Channel.NONE),
    Surface.CHECKOUT_ABANDON: (Action.NUDGE_CUSTOMER, Channel.WHATSAPP),
    Surface.RECEIVABLE: (Action.NUDGE_CUSTOMER, Channel.EMAIL),
}


def _observe(item: AtRiskItem, contacts_7d: int, hour: float) -> ObservableState:
    rail = {Surface.SUBSCRIPTION: Rail.UPI_AUTOPAY,
            Surface.RECEIVABLE: Rail.INVOICE}.get(item.surface, Rail.UPI_COLLECT)
    ev = item.evidence
    return ObservableState(
        rail=rail,
        decline_reason=ev.get("error_reason"),
        local_hour=hour,
        is_working_day=True,
        contacts_7d=contacts_7d,
        is_collections=item.surface is Surface.RECEIVABLE,
        is_disputed=bool(ev.get("disputed")),
        # Scheduled mandate debits carry a standing pre-debit notice.
        hours_since_pre_debit_notice=25.0,
        # Receivables facts. Conservative defaults: a supplier is not assumed
        # eligible for a statutory remedy, and nothing is assumed overdue.
        supplier_is_msme=bool(ev.get("supplier_is_msme")),
        days_past_appointed_day=int(ev.get("days_past_due", 0)),
    )


def run(items: list[AtRiskItem], *, detector, capacity: int,
        hour: float = 11.0, log: AuditLog | None = None) -> RunReport:
    """Score, rank, gate and act -- once, across every surface together."""
    log = log if log is not None else AuditLog()
    report = RunReport(considered=len(items))

    # Rank every surface in one queue, by expected rupees rather than by count.
    # Surfaces competing against each other is the point: a single overdue
    # invoice can be worth more than every abandoned cart in the batch.
    scored = []
    for it in items:
        p = detector.score(it)
        ev = p * it.amount_inr * SURFACE_UPLIFT[it.surface]
        scored.append((ev, it))
    scored.sort(key=lambda t: t[0], reverse=True)

    contacts: dict[str, int] = {}
    for ev, it in scored:
        if len(report.interventions) >= capacity:
            report.skipped_capacity += 1
            continue

        action, channel = SURFACE_ACTION[it.surface]
        cost = ACTION_COST_INR.get(action, 0.0) + CHANNEL_COST_INR.get(channel, 0.0)
        if ev <= cost:
            report.skipped_value += 1
            continue

        state = _observe(it, contacts.get(it.customer_id, 0), hour)
        allowed, fired = feasible_actions(state)
        cleared = action in allowed
        blocked = next((v.rule for v in fired if action in v.forbids), None)

        iv = Intervention(
            item_id=it.item_id, surface=it.surface, customer_id=it.customer_id,
            amount_inr=it.amount_inr, action=action if cleared else Action.STOP,
            channel=channel if cleared else Channel.NONE,
            expected_value_inr=ev if cleared else 0.0,
            idempotency_key=idempotency_key(it.item_id, 1, action.value),
            cleared=cleared, blocked_by=blocked)

        if not cleared:
            report.blocked[blocked or "unknown"] = report.blocked.get(blocked or "unknown", 0) + 1
        else:
            report.interventions.append(iv)
            if channel is not Channel.NONE:
                contacts[it.customer_id] = contacts.get(it.customer_id, 0) + 1

        # Everything is logged, including what we refused to do and why. A trail
        # that records only the actions taken cannot answer the question an
        # auditor actually asks, which is what else was considered.
        log.append({
            "item": it.item_id, "surface": it.surface.value,
            "customer": it.customer_id, "amount_inr": round(it.amount_inr, 2),
            "action": iv.action.value, "expected_value_inr": round(ev, 2),
            "kernel_cleared": cleared, "blocked_by": blocked,
            "idempotency_key": iv.idempotency_key,
        })

    report.audit_head = log.head
    return report
