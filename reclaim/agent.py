"""The recovery loop: diagnose, decide, act, record -- until we win or stop."""
from __future__ import annotations

from .diagnose import TieredDiagnoser
from .models import Action, Channel, Decision, FailedPayment, RecoveryOutcome, RootCause
from .policy import Ledger, RecoveryState, decide
from . import simulator

MAX_STEPS = 6  # hard ceiling; guardrails normally stop us well before this


class ReclaimAgent:
    """Diagnose-then-act, with every step passed through the guardrail layer."""

    name = "reclaim_agent"

    def __init__(self, diagnoser=None, ledger: Ledger | None = None) -> None:
        self.diagnoser = diagnoser or TieredDiagnoser()
        # One ledger across the whole batch, not one per payment. The 30-day
        # network caps and the weekly contact budget are properties of a card and
        # a customer, and a per-workflow counter cannot see either.
        self.ledger = ledger if ledger is not None else Ledger()

    def run(self, p: FailedPayment, noise: float = 0.0) -> RecoveryOutcome:
        dx = self.diagnoser.diagnose(p)
        st = RecoveryState(clock_hour=p.failed_at_hour)
        out = RecoveryOutcome(payment_id=p.payment_id, amount_inr=p.amount_inr, recovered=False)

        for _ in range(MAX_STEPS):
            d = decide(p, dx, st, self.ledger)
            out.decisions.append(d)

            if d.action is Action.STOP:
                break

            if d.action is Action.RECONCILE:
                st.reconciled = True
                if p.settlement_actually_succeeded:
                    # The money was already there. Not a recovery -- there was
                    # nothing to recover -- but a double debit that did not happen,
                    # which is worth considerably more than the 10 paise it cost
                    # to ask.
                    out.resolved_already_paid = True
                    break
                st.clock_hour = (st.clock_hour + 1) % 24
                continue

            if simulator.attempt_succeeds(p, d, noise):
                out.recovered = True
                out.recovered_on_attempt = d.attempt
                break

            # Book-keeping so the next decision knows what we already spent.
            st.tried.append(d.action)
            if d.action in {Action.RETRY_NOW, Action.RETRY_SCHEDULED, Action.SWITCH_METHOD}:
                st.money_attempts += 1
                self.ledger.note_money_action(p)
            if d.action in {Action.NUDGE_CUSTOMER, Action.REQUEST_INSTRUMENT_UPDATE}:
                st.outreach_count += 1
                self.ledger.note_contact(p)
            if d.action is Action.ESCALATE_MANUAL:
                st.escalated = True
                break  # a human owns it now; the agent takes no further action
            st.clock_hour = (st.clock_hour + max(1, d.delay_hours)) % 24

        return out


# --- baselines ---------------------------------------------------------------

class DoNothing:
    """The floor. Whatever we do has to beat abandoning the money."""

    name = "do_nothing"

    def run(self, p: FailedPayment, noise: float = 0.0) -> RecoveryOutcome:
        return RecoveryOutcome(payment_id=p.payment_id, amount_inr=p.amount_inr, recovered=False)


class RetryAll:
    """What most teams ship: hammer every failure three times, immediately."""

    name = "retry_all_x3"

    def run(self, p: FailedPayment, noise: float = 0.0) -> RecoveryOutcome:
        out = RecoveryOutcome(payment_id=p.payment_id, amount_inr=p.amount_inr, recovered=False)
        for attempt in range(1, 4):
            d = Decision(
                payment_id=p.payment_id, attempt=attempt, diagnosed_cause=RootCause.UNKNOWN,
                diagnosis_source="none", diagnosis_confidence=0.0, action=Action.RETRY_NOW,
                rationale="blanket retry",
            )
            out.decisions.append(d)
            if simulator.attempt_succeeds(p, d, noise):
                out.recovered, out.recovered_on_attempt = True, attempt
                break
        return out


class RetryBackoff:
    """A more careful blanket policy: same idea, but spaced 24h apart."""

    name = "retry_backoff_x3"

    def run(self, p: FailedPayment, noise: float = 0.0) -> RecoveryOutcome:
        out = RecoveryOutcome(payment_id=p.payment_id, amount_inr=p.amount_inr, recovered=False)
        for attempt in range(1, 4):
            d = Decision(
                payment_id=p.payment_id, attempt=attempt, diagnosed_cause=RootCause.UNKNOWN,
                diagnosis_source="none", diagnosis_confidence=0.0,
                action=Action.RETRY_SCHEDULED, delay_hours=24, rationale="blanket retry, 24h apart",
            )
            out.decisions.append(d)
            if simulator.attempt_succeeds(p, d, noise):
                out.recovered, out.recovered_on_attempt = True, attempt
                break
        return out
