"""The agent as a running process.

Everything else in this repo is a batch evaluation. This is the thing that would
actually be deployed: it takes one event at a time, decides, acts through a
gateway, and records what it did and what it refused to do.

The split matters. An evaluation can afford to know the future -- it scores
against ground truth it holds in memory. A service cannot. So this class is
written to depend on nothing an evaluation would have and a production process
would not: no true root cause, no counterfactual, no batch to look ahead in.
If it compiles against that constraint, it can be deployed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .compliance import MONEY_ACTIONS, OUTREACH_ACTIONS
from .diagnose import Diagnosis, TieredDiagnoser
from .gateway import GatewayError, PaymentStatus, SimulatedGateway
from .models import Action, Decision, FailedPayment, RecoveryOutcome
from .policy import Ledger, RecoveryState, decide
from .security import AuditLog, idempotency_key


@dataclass
class Handled:
    """One event, start to finish, as the service saw it."""

    payment_id: str
    decision: Decision
    diagnosis: Diagnosis
    executed: bool
    result: str
    latency_us: float
    gateway_calls: int = 0
    audit_seq: int | None = None


@dataclass
class ServiceStats:
    handled: int = 0
    executed: int = 0
    refused: int = 0
    reconciled: int = 0
    already_settled: int = 0
    money_actions: int = 0
    outreach: int = 0
    spend_inr: float = 0.0
    refusal_reasons: dict[str, int] = field(default_factory=dict)
    latencies_us: list[float] = field(default_factory=list)

    @property
    def p99_us(self) -> float:
        if not self.latencies_us:
            return 0.0
        return sorted(self.latencies_us)[int(len(self.latencies_us) * 0.99)]


class RecoveryService:
    """Stateful across events, because the budgets that matter are.

    The ledger has to persist between events -- the 30-day network caps and the
    weekly contact budget are properties of a card and a customer, not of a
    request -- so this object is the unit of deployment, not the function call.
    """

    def __init__(self, *, diagnoser=None, gateway=None, ledger: Ledger | None = None,
                 audit: AuditLog | None = None, dry_run: bool = True) -> None:
        self.diagnoser = diagnoser or TieredDiagnoser()
        self.gateway = gateway or SimulatedGateway()
        self.ledger = ledger or Ledger()
        self.audit = audit or AuditLog()
        # Default is dry run. A recovery agent that executes by default is one
        # config mistake away from charging people, and the cost of the opposite
        # mistake is a demo that does nothing.
        self.dry_run = dry_run
        self.stats = ServiceStats()
        self._states: dict[str, RecoveryState] = {}

    # --- the loop ------------------------------------------------------------

    def handle(self, payment: FailedPayment, *, now_hour: int | None = None) -> Handled:
        started = time.perf_counter_ns()
        calls_before = getattr(self.gateway, "calls", 0)

        state = self._states.setdefault(
            payment.payment_id,
            RecoveryState(clock_hour=now_hour if now_hour is not None
                          else payment.failed_at_hour))

        dx = self.diagnoser.diagnose(payment)
        d = decide(payment, dx, state, self.ledger)
        executed, result = self._execute(payment, d, state)

        latency = (time.perf_counter_ns() - started) / 1000.0
        entry = self.audit.append({
            "payment_id": payment.payment_id,
            "action": d.action.value,
            "diagnosed": d.diagnosed_cause.value,
            "tier": d.diagnosis_source,
            "confidence": round(d.diagnosis_confidence, 3),
            "kernel_cleared": d.kernel_cleared,
            "blocked_by": d.blocked_by,
            "executed": executed,
            "result": result,
            "idempotency_key": idempotency_key(
                payment.payment_id, d.attempt, d.action.value),
            "cost_inr": round(d.cost_inr, 2),
        })

        self._record(d, executed, latency, result)
        return Handled(payment.payment_id, d, dx, executed, result, latency,
                       getattr(self.gateway, "calls", 0) - calls_before, entry.seq)

    # --- doing the thing -----------------------------------------------------

    def _execute(self, p: FailedPayment, d: Decision,
                 state: RecoveryState) -> tuple[bool, str]:
        if d.action is Action.STOP:
            return False, "stopped"
        if d.action is Action.ESCALATE_MANUAL:
            return False, "queued for a human"

        if d.action is Action.RECONCILE:
            # The one action that is always safe to take for real, even in a dry
            # run: it is a read, and skipping it is how a dry run learns the
            # wrong thing about whether a payment settled.
            try:
                status = self.gateway.fetch_payment(p.payment_id)
            except GatewayError as exc:
                return False, f"reconcile failed: {type(exc).__name__}"
            state.reconciled = True
            if status.settled:
                state.settled = True
                return True, "already settled -- double charge avoided"
            return True, f"confirmed {status.status}"

        if self.dry_run:
            return False, f"dry run: would {d.action.value}"

        if d.action in MONEY_ACTIONS:
            try:
                key = idempotency_key(p.payment_id, d.attempt, d.action.value)
                status = self.gateway.capture_payment(
                    p.payment_id, p.amount_inr, idempotency_key=key)
            except GatewayError as exc:
                return False, f"gateway refused: {type(exc).__name__}"
            self.ledger.note_money_action(p)
            state.money_attempts += 1
            return True, f"charged: {status.status}"

        if d.action in OUTREACH_ACTIONS:
            # No messaging vendor is wired up, and pretending otherwise in a
            # demo would be the exact dishonesty this project keeps arguing
            # against. The decision, the channel and the cost are all real; the
            # send is not.
            self.ledger.note_contact(p)
            state.outreach_count += 1
            return False, f"no messaging vendor configured ({d.channel.value})"

        return False, "no executor for this action"

    # --- bookkeeping ---------------------------------------------------------

    def _record(self, d: Decision, executed: bool, latency_us: float,
                result: str) -> None:
        s = self.stats
        s.handled += 1
        s.latencies_us.append(latency_us)
        s.spend_inr += d.cost_inr
        if d.action is Action.RECONCILE:
            s.reconciled += 1
            if "already settled" in result:
                s.already_settled += 1
        if d.action in MONEY_ACTIONS:
            s.money_actions += 1
        if d.action in OUTREACH_ACTIONS:
            s.outreach += 1
        if executed:
            s.executed += 1
        if d.blocked_by:
            s.refused += 1
            s.refusal_reasons[d.blocked_by] = s.refusal_reasons.get(d.blocked_by, 0) + 1

    def health(self) -> dict[str, Any]:
        """What a readiness probe would ask."""
        ok, bad = self.audit.verify()
        return {
            "gateway": self.gateway.name,
            "gateway_live": getattr(self.gateway, "live", False),
            "dry_run": self.dry_run,
            "handled": self.stats.handled,
            "audit_entries": len(self.audit.entries),
            "audit_intact": ok,
            "audit_first_bad": bad,
            "audit_head": self.audit.head[:16],
            "p99_latency_us": round(self.stats.p99_us, 1),
        }
