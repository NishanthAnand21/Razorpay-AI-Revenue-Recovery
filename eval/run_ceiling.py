"""How much of this money is recoverable at all?

65.8% invites an obvious question -- why not 100? -- and the answer is not "the
agent needs tuning". Recovery rate is not an accuracy metric that a better model
pushes toward 1.0. It is the share of at-risk money that comes back, and most of
what limits it is a property of the payments, not of the policy.

So this measures the ceiling directly, with an oracle: an agent that KNOWS the
true root cause of every payment (no diagnosis error at all), picks the single
best-performing legal action every time, and spends every attempt the rules
allow. Nothing real can beat it. Whatever it leaves on the table is money that
was never available.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim import simulator
from reclaim.compliance import (MONEY_ACTIONS, OUTREACH_ACTIONS, SettlementState,
                                feasible_actions, observe)
from reclaim.models import Action, Channel, Decision, RecoveryOutcome, RootCause
from reclaim.policy import (MAX_MONEY_ATTEMPTS, MAX_OUTREACH_PER_PAYMENT, Ledger,
                            RecoveryState, _state_for)
from run_eval import load, score  # noqa: E402

CANDIDATE_DELAYS = (0, 6, 24, 48)


class Oracle:
    """Perfect diagnosis, best legal action, same budgets as the real agent.

    A first version of this scored BELOW Reclaim, which is impossible for an
    upper bound and so was a bug rather than a result. Two causes, both from
    writing a separate loop instead of reusing the real one:

      - it computed the legal action set once at the hour of failure and never
        advanced the clock, so it could never take an action that only becomes
        legal later -- and deferring into a legal window is most of what the
        timing policy is for;
      - it capped itself at three actions total, while the real agent has a
        budget of three money attempts AND two outreach attempts.

    So this now mirrors `ReclaimAgent.run` exactly -- same state machine, same
    budgets, same clock, same kernel -- and changes one thing: instead of
    consulting a belief table it asks the simulator for the TRUE probability of
    every legal action and takes the best. That is the only advantage it has,
    and it is an advantage nothing real can have.
    """

    name = "oracle (perfect knowledge)"
    MAX_STEPS = 6
    kernel_aware = True

    def __init__(self, obey_kernel: bool = True) -> None:
        self.obey_kernel = obey_kernel

    def run(self, p, noise: float = 0.0) -> RecoveryOutcome:
        st = RecoveryState(clock_hour=p.failed_at_hour)
        ledger = Ledger()
        out = RecoveryOutcome(payment_id=p.payment_id, amount_inr=p.amount_inr,
                              recovered=False)

        for _ in range(self.MAX_STEPS):
            state = _state_for(p, st, ledger)

            # Reconcile first on an unconfirmed outcome, exactly as the real
            # agent does. Without this the kernel vetoes every money and
            # outreach action -- correctly, since the outcome is unknown -- and
            # the oracle escalated immediately on every gateway timeout while
            # the real agent reconciled and then acted. That alone put the
            # "upper bound" below the thing it was supposed to bound.
            if state.settlement_state is SettlementState.UNKNOWN and not st.reconciled:
                st.reconciled = True
                if p.settlement_actually_succeeded:
                    out.resolved_already_paid = True
                    break
                st.clock_hour = (st.clock_hour + 1) % 24
                continue

            allowed, _fired = feasible_actions(state)
            allowed_before_override = allowed
            if not self.obey_kernel:
                allowed = set(Action)

            # Budgets, exactly as the real policy applies them.
            #
            # ESCALATE_MANUAL belongs in this set and was missing from the first
            # version. Handing a payment to an analyst recovers it 20% of the
            # time in the simulator, so an oracle that ignores it scores below
            # the real agent on exactly the payments where every money action is
            # vetoed -- dead instruments and risk declines -- which is how a
            # supposed upper bound ended up beneath the thing it bounds.
            usable = set()
            for a in allowed:
                if a in MONEY_ACTIONS and st.money_attempts >= MAX_MONEY_ATTEMPTS:
                    continue
                if a in OUTREACH_ACTIONS and st.outreach_count >= MAX_OUTREACH_PER_PAYMENT:
                    continue
                if a in (MONEY_ACTIONS | OUTREACH_ACTIONS | {Action.ESCALATE_MANUAL}):
                    usable.add(a)

            attempt = st.money_attempts + st.outreach_count + 1
            best = None
            for action in sorted(usable, key=lambda a: a.value):
                delays = CANDIDATE_DELAYS if action is Action.RETRY_SCHEDULED else (0,)
                for delay in delays:
                    d = Decision(
                        payment_id=p.payment_id, attempt=attempt,
                        diagnosed_cause=p.true_root_cause,      # the oracle bit
                        diagnosis_source="oracle", diagnosis_confidence=1.0,
                        action=action, delay_hours=delay,
                        channel=(Channel.WHATSAPP if action in OUTREACH_ACTIONS
                                 else Channel.NONE))
                    prob = simulator.success_probability(p, d)
                    if best is None or prob > best[0]:
                        best = (prob, d)

            if best is None or best[0] <= 0.0:
                break
            _prob, d = best
            # Recorded at the moment of choice, like the real agent. The lawless
            # variant marks its own actions as uncleared, so its breaches count.
            d.kernel_cleared = d.action in allowed_before_override
            out.decisions.append(d)

            if simulator.attempt_succeeds(p, d, noise):
                out.recovered, out.recovered_on_attempt = True, attempt
                break

            st.tried.append(d.action)
            if d.action is Action.ESCALATE_MANUAL:
                break                      # a human owns it; the agent stops
            if d.action in MONEY_ACTIONS:
                st.money_attempts += 1
                ledger.note_money_action(p)
            if d.action in OUTREACH_ACTIONS:
                st.outreach_count += 1
                ledger.note_contact(p)
            st.clock_hour = (st.clock_hour + max(1, d.delay_hours)) % 24

        return out


def main() -> None:
    test = load("test")
    at_risk = sum(p.amount_inr for p in test if not p.settlement_actually_succeeded)

    from reclaim import policy
    from reclaim.agent import ReclaimAgent
    from reclaim.classify import train as train_classifier
    from reclaim.diagnose import ThreeTierDiagnoser
    from reclaim.learn import fit

    train_rows = load("train")
    beliefs, _ = fit(train_rows, hand_written=policy.BELIEVED_SUCCESS)
    learned = train_classifier(train_rows)
    policy.set_beliefs(beliefs)
    policy.set_proposer("ev")
    tuned = score(ReclaimAgent(diagnoser=ThreeTierDiagnoser(learned)), test)
    policy.set_proposer("rules")
    policy.set_beliefs(None)

    lawful = score(Oracle(obey_kernel=True), test)
    lawless = score(Oracle(obey_kernel=False), test)

    print(f"Recovery ceiling -- {len(test)} held-out payments, "
          f"INR {at_risk:,.0f} genuinely at risk\n")
    print(f"{'agent':<34}{'recovered':>11}{'net INR':>14}{'breaches':>10}"
          f"{'double chg':>12}")
    print("-" * 81)
    print(f"{'Reclaim (tuned)':<34}{tuned['recovery_rate']:>10.1%}"
          f"{tuned['net_inr']:>14,.0f}{tuned['compliance_breaches']:>10}"
          f"{tuned['double_charges']:>12}")
    print(f"{'oracle, obeying the rules':<34}{lawful['recovery_rate']:>10.1%}"
          f"{lawful['net_inr']:>14,.0f}{lawful['compliance_breaches']:>10}"
          f"{lawful['double_charges']:>12}")
    print(f"{'oracle, ignoring the rules':<34}{lawless['recovery_rate']:>10.1%}"
          f"{lawless['net_inr']:>14,.0f}{lawless['compliance_breaches']:>10}"
          f"{lawless['double_charges']:>12}")
    print("-" * 81)

    gap = lawful["recovery_rate"] - tuned["recovery_rate"]
    print(f"""
  The oracle knows the true cause of every payment, picks the best-performing
  legal action every time, and spends every attempt the rules allow. Nothing
  real can beat it.

  It reaches {lawful['recovery_rate']:.1%}. Reclaim reaches {tuned['recovery_rate']:.1%} -- which is
  {tuned['recovery_rate']/lawful['recovery_rate']:.1%} of everything that was available.

  The remaining {gap:.1%} is the entire prize for perfect diagnosis. It is worth
  INR {lawful['net_inr'] - tuned['net_inr']:,.0f}, and no model can capture all of it, because a model
  that knew every true cause with certainty is what the oracle already is.
""")

    # --- where the rest of the money is ------------------------------------
    #
    # Split by the best legal CHARGE, not by the best legal action overall.
    # Escalating to a human recovers 20% of anything, so folding it in makes
    # every bucket look reachable and erases the distinction that matters: an
    # instrument that cannot be charged by anyone, at any price, with any model.
    buckets: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    escalation_only = 0.0

    for p in test:
        if p.settlement_actually_succeeded:
            continue
        state = observe(p, local_hour=p.failed_at_hour)
        allowed, _ = feasible_actions(state)
        best_charge = 0.0
        for action in sorted(MONEY_ACTIONS, key=lambda a: a.value):
            if action not in allowed:
                continue
            for delay in CANDIDATE_DELAYS:
                d = Decision(p.payment_id, 1, p.true_root_cause, "oracle", 1.0,
                             action, delay_hours=delay)
                best_charge = max(best_charge, simulator.success_probability(p, d))

        if best_charge <= 0.0:
            bucket = "cannot be charged at all (best legal charge = 0%)"
            escalation_only += p.amount_inr
        elif best_charge < 0.25:
            bucket = "best legal charge under 25%"
        elif best_charge < 0.6:
            bucket = "best legal charge 25-60%"
        else:
            bucket = "best legal charge over 60%"
        buckets[bucket] += p.amount_inr
        counts[bucket] += 1

    print(f"  {'WHY THE MONEY IS NOT ALL RECOVERABLE':<48}{'count':>7}{'INR':>14}{'share':>8}")
    print("  " + "-" * 77)
    for bucket in ("cannot be charged at all (best legal charge = 0%)",
                   "best legal charge under 25%",
                   "best legal charge 25-60%",
                   "best legal charge over 60%"):
        if bucket in buckets:
            print(f"  {bucket:<48}{counts[bucket]:>7}{buckets[bucket]:>14,.0f}"
                  f"{buckets[bucket]/at_risk:>8.1%}")
    print("  " + "-" * 77)

    print(f"""
  INR {escalation_only:,.0f} ({escalation_only/at_risk:.0%}) cannot be charged by anyone. A revoked mandate
  or a cancelled card has a success probability of exactly zero, and a risk
  decline must not be retried at all. The only route left is a human, which
  recovers about one in five -- so most of that money is simply gone, and no
  amount of tuning changes it.

  Even the oracle that IGNORES every rule -- retrying fraud declines, breaching
  network caps, contacting people at 3am -- reaches only {lawless['recovery_rate']:.1%}. That is the
  hard ceiling with omniscience and no law. 100% is roughly {100 - lawless['recovery_rate']*100:.0f} points out of
  reach of a system that cheats, let alone one that does not.

  So 100% is not a target that better tuning approaches. It is a number you can
  only report by counting things that did not happen -- which is exactly what
  `retry_all_x3` does when it books 11 double charges as recovered revenue, and
  what the industry does when it reports gross recovery that overstates real
  lift by 3.9x.

  The honest ceiling is {lawful['recovery_rate']:.1%}. Reclaim is at {tuned['recovery_rate']/lawful['recovery_rate']:.0%} of it, and the whole
  remaining prize -- perfect diagnosis, forever, on every payment -- is worth
  INR {lawful['net_inr'] - tuned['net_inr']:,.0f}.""")


if __name__ == "__main__":
    main()
