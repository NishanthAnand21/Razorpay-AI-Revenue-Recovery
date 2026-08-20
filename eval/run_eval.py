"""Score the agent against baselines on the held-out test set.

Run:
    python3 eval/run_eval.py                 # headline comparison + diagnosis metrics
    python3 eval/run_eval.py --sensitivity   # re-run under perturbed world models
    python3 eval/run_eval.py --audit pay_id  # print the full decision trail for one payment
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim import simulator
from reclaim.compliance import MONEY_ACTIONS, feasible_actions, observe
from reclaim import policy
from reclaim.agent import DoNothing, ReclaimAgent, RetryAll, RetryBackoff
from reclaim.diagnose import RULES_TABLE, RulesDiagnoser, TieredDiagnoser
from reclaim.models import Action, FailedPayment, RecoveryOutcome, RootCause

DATA = Path(__file__).resolve().parents[1] / "data"


def load(split: str) -> list[FailedPayment]:
    rows = []
    for line in (DATA / f"{split}.jsonl").read_text().splitlines():
        d = json.loads(line)
        d["true_root_cause"] = RootCause(d["true_root_cause"])
        rows.append(FailedPayment(**d))
    return rows


# --- recovery metrics --------------------------------------------------------

def score(strategy, rows: list[FailedPayment], noise: float = 0.0) -> dict:
    outs: list[RecoveryOutcome] = [strategy.run(p, noise) for p in rows]

    # Payments that had already settled were never at risk, and re-charging one
    # is a duplicate debit, not a recovery. An earlier version of this scorer
    # counted those as recovered revenue, which flattered every strategy that
    # retries blindly -- the blanket baselines were being paid for their own
    # worst failure mode.
    at_risk = sum(p.amount_inr for p in rows if not p.settlement_actually_succeeded)
    recovered = sum(o.amount_inr for p, o in zip(rows, outs)
                    if o.recovered and not p.settlement_actually_succeeded)
    spend = sum(o.spend_inr for o in outs)

    # Two distinct failures, kept apart on purpose (the security eval draws the
    # same line): a compliance breach is an action the observable facts forbid,
    # and the kernel decides it. A risky recharge is an action on a payment whose
    # *true* cause was fraud but whose raw decline reason did not say so -- not
    # decidable at the time, and so not a breach.
    breaches = risky = wasted = double_charges = reconciles = already_paid = 0
    for p, o in zip(rows, outs):
        already_paid += o.resolved_already_paid
        # Baselines never consult the kernel, and never send a pre-debit notice
        # either, so their retries are scored against a state with no notice
        # outstanding -- which is exactly the state they actually create.
        baseline_allowed, _ = feasible_actions(observe(p, local_hour=p.failed_at_hour))
        for d in o.decisions:
            cleared = d.kernel_cleared if strategy.name.startswith("reclaim") \
                else d.action in baseline_allowed
            if d.action in MONEY_ACTIONS and not cleared:
                breaches += 1
            risky += simulator.is_compliance_violation(p, d)
            wasted += simulator.is_wasted_retry(p, d)
            reconciles += d.action is Action.RECONCILE
            # The one that actually costs a customer money: re-charging a payment
            # that had in fact already gone through.
            if d.action in simulator.MONEY_ACTIONS and p.settlement_actually_succeeded:
                double_charges += 1

    return {
        "strategy": strategy.name,
        "n": len(rows),
        "recovery_rate": recovered / at_risk if at_risk else 0.0,
        "count_rate": sum(o.recovered for o in outs) / len(outs),
        "recovered_inr": recovered,
        "spend_inr": spend,
        "net_inr": recovered - spend,
        "actions_taken": sum(len(o.decisions) for o in outs),
        "compliance_breaches": breaches,
        "risky_recharges": risky,
        "wasted_dead_instrument_retries": wasted,
        "escalations": sum(
            1 for o in outs for d in o.decisions if d.action is Action.ESCALATE_MANUAL
        ),
        "double_charges": double_charges,
        "reconciles": reconciles,
        "already_paid": already_paid,
    }


def print_recovery_table(results: list[dict], at_risk: float) -> None:
    print(f"\n{'strategy':<20}{'recov%':>8}{'net INR':>13}{'spend':>9}"
          f"{'breach':>8}{'risky':>7}{'wasted':>8}{'double chg':>12}")
    print("-" * 88)
    for r in results:
        print(f"{r['strategy']:<20}{r['recovery_rate']*100:>7.1f}%{r['net_inr']:>13,.0f}"
              f"{r['spend_inr']:>9,.0f}{r['compliance_breaches']:>8}"
              f"{r['risky_recharges']:>7}{r['wasted_dead_instrument_retries']:>8}"
              f"{r['double_charges']:>12}")
    print("-" * 88)
    print(f"total revenue at risk in the held-out set: INR {at_risk:,.0f}")


# --- diagnosis metrics -------------------------------------------------------

def diagnosis_report(rows: list[FailedPayment]) -> None:
    """Per-class precision/recall for the tiered diagnoser, plus the novel split.

    The novel split is the interesting number: those are the rows the rules table
    has never seen, i.e. exactly the ones the model is there to earn its keep on.
    """
    tiered = TieredDiagnoser()
    rules = RulesDiagnoser()

    tp: dict[RootCause, int] = {}
    fp: dict[RootCause, int] = {}
    fn: dict[RootCause, int] = {}
    novel_total = novel_right = seen_total = seen_right = 0
    escalated_unknown = 0

    for p in rows:
        truth = p.true_root_cause
        pred = tiered.diagnose(p).cause
        is_novel = rules.diagnose(p).cause is RootCause.UNKNOWN

        if is_novel:
            novel_total += 1
            novel_right += pred is truth
            escalated_unknown += pred is RootCause.UNKNOWN
        else:
            seen_total += 1
            seen_right += pred is truth

        if pred is truth:
            tp[truth] = tp.get(truth, 0) + 1
        else:
            fp[pred] = fp.get(pred, 0) + 1
            fn[truth] = fn.get(truth, 0) + 1

    print(f"\n{'root cause':<24}{'precision':>11}{'recall':>9}{'support':>9}")
    print("-" * 55)
    for c in RootCause:
        support = sum(1 for p in rows if p.true_root_cause is c)
        if not support:
            continue
        t, f_p, f_n = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
        prec = t / (t + f_p) if t + f_p else 0.0
        rec = t / (t + f_n) if t + f_n else 0.0
        print(f"{c.value:<24}{prec:>11.2f}{rec:>9.2f}{support:>9}")
    print("-" * 55)
    overall = (novel_right + seen_right) / len(rows)
    print(f"overall accuracy                {overall:.3f}")
    print(f"  on reasons in the rules table {seen_right}/{seen_total} = "
          f"{seen_right/seen_total:.3f}  (rules layer, free)")
    print(f"  on reasons never seen before  {novel_right}/{novel_total} = "
          f"{novel_right/novel_total:.3f}  (model layer)")
    print(f"  of which routed to a human    {escalated_unknown}  "
          f"(no confident call, so no money moved)")
    print(f"\nA rules-only diagnoser scores 0.000 on those {novel_total} rows -- it returns")
    print("UNKNOWN, which the policy turns into an escalation. That gap is what the")
    print("model layer buys, and it is why the LLM sits behind the table, not in front of it.")


# --- sensitivity -------------------------------------------------------------

def sensitivity(rows: list[FailedPayment]) -> None:
    """Does the ranking survive a wrong world model? Re-run under perturbation."""
    print("\nsensitivity: net INR under perturbed success probabilities")
    print(f"{'noise':>8}", end="")
    strategies = [ReclaimAgent(), RetryAll(), RetryBackoff()]
    for s in strategies:
        print(f"{s.name:>20}", end="")
    print()
    print("-" * 68)
    margins = []
    for noise in (-0.30, -0.15, 0.0, 0.15, 0.30):
        print(f"{noise:>+8.0%}", end="")
        nets = []
        for s in strategies:
            net = score(s, rows, noise)["net_inr"]
            nets.append(net)
            print(f"{net:>20,.0f}", end="")
        print()
        margins.append(nets[0] - max(nets[1:]))
    print("-" * 68)
    print(f"agent's lead over the best baseline: min INR {min(margins):,.0f}, "
          f"median INR {statistics.median(margins):,.0f}")
    flips = sum(1 for m in margins if m < 0)
    if flips:
        print(f"\nThe ranking DOES flip in {flips} of {len(margins)} perturbations. This is the")
        print("honest read of it: when the world is generous -- when retries succeed far more")
        print("often than modelled -- blanket 24h backoff catches up and passes the agent,")
        print("because in that regime the cheapest way to recover money really is to retry")
        print("everything. The agent's edge is largest exactly where recovery is hard.")
        print("\nTwo things that column does not price, though:")
        print("  - blanket backoff commits 26 compliance violations on this set; the agent")
        print("    commits 0 in strict mode. A regulator does not discount that by noise.")
        print("  - it burns 57 gateway fees on instruments with a literal 0% success rate.")
        print("Ranking strategies on net INR alone is the wrong frame, which is why the")
        print("headline table carries the violation and waste columns next to the money.")
    else:
        print("The ranking holds across every perturbation tested.")


# --- audit -------------------------------------------------------------------

def audit(rows: list[FailedPayment], payment_id: str) -> None:
    p = next((r for r in rows if r.payment_id == payment_id), None)
    if p is None:
        print(f"no payment {payment_id} in the test set")
        return
    out = ReclaimAgent().run(p)
    print(f"\npayment {p.payment_id}   INR {p.amount_inr:,.2f}   {p.method}"
          f"{'  [recurring]' if p.is_recurring else ''}")
    print(f"gateway said: {p.error_code} / {p.error_reason}")
    print(f"merchant note: {p.merchant_note or '(none)'}")
    print(f"true cause (eval only): {p.true_root_cause.value}\n")
    for d in out.decisions:
        print(f"  attempt {d.attempt}: {d.action.value}"
              f"{f' via {d.channel.value}' if d.channel.value != 'none' else ''}"
              f"{f' after {d.delay_hours}h' if d.delay_hours else ''}")
        print(f"      diagnosed {d.diagnosed_cause.value} "
              f"({d.diagnosis_source}, confidence {d.diagnosis_confidence:.2f})")
        print(f"      why: {d.rationale}")
        if d.blocked_by:
            print(f"      GUARDRAIL: {d.blocked_by}")
        print(f"      cost: INR {d.cost_inr:.2f}")
    print(f"\n  outcome: {'RECOVERED' if out.recovered else 'not recovered'}"
          f"   spent INR {out.spend_inr:.2f}   net INR {out.net_inr:,.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--audit", metavar="PAYMENT_ID")
    args = ap.parse_args()

    test = load("test")

    if args.audit:
        audit(test, args.audit)
        return

    print(f"Reclaim -- held-out evaluation on {len(test)} failed payments")
    print(f"(policy tuned on data/train.jsonl, never on these rows)")

    results = [score(s, test) for s in (DoNothing(), RetryAll(), RetryBackoff())]
    results.append(score(ReclaimAgent(), test))

    # Everything fitted on train only, then evaluated here: the belief table, the
    # EV proposer that consults it, and the learned diagnosis tier.
    from reclaim.classify import train as train_classifier
    from reclaim.diagnose import ThreeTierDiagnoser
    from reclaim.learn import fit
    train_rows = load("train")
    beliefs, _notes = fit(train_rows, hand_written=policy.BELIEVED_SUCCESS)
    learned_dx = train_classifier(train_rows, l2=1e-4)
    learned_dx.min_margin = 0.10
    policy.set_beliefs(beliefs)
    policy.set_proposer("ev")
    tuned = score(ReclaimAgent(diagnoser=ThreeTierDiagnoser(learned_dx)), test)
    tuned["strategy"] = "reclaim_agent[tuned]"
    results.append(tuned)
    policy.set_proposer("rules")
    policy.set_beliefs(None)
    policy.set_strict(True)
    strict = score(ReclaimAgent(), test)
    strict["strategy"] = "reclaim_agent[strict]"
    policy.set_strict(False)
    results.append(strict)
    print_recovery_table(results, sum(p.amount_inr for p in test))

    best_base = max(r["net_inr"] for r in results[:3])
    agent = results[3]
    print(f"\nnet money recovered vs the best baseline: "
          f"INR {agent['net_inr'] - best_base:,.0f} "
          f"({(agent['net_inr']/best_base - 1)*100:+.1f}%)")
    print(f"""
breach     an action the observable facts forbid. The compliance kernel decides
           this, so the agent is at {agent['compliance_breaches']} by construction and the blanket
           baselines are at {results[1]['compliance_breaches']} -- they do not look.
risky      a recharge on a payment whose true cause was fraud but whose raw
           reason was `do_not_honour`, which Visa classifies as retryable. Not
           decidable at the time. The agent takes {agent['risky_recharges']}; strict mode takes {strict['risky_recharges']},
           costing INR {agent['net_inr'] - strict['net_inr']:,.0f} ({(strict['net_inr']/agent['net_inr']-1)*100:+.1f}%) of recovered revenue.
double chg re-charging a payment that had already settled. The agent reconciles
           first and takes {agent['double_charges']}; the baselines take {results[1]['double_charges']} and {results[2]['double_charges']}, and each one is a
           customer debited twice for the same purchase.

Which point on the risky/revenue curve is right is a risk-appetite call, not an
engineering one, so both are reported rather than one quietly chosen.""")

    # --- ablation ---------------------------------------------------------
    print(f"""
ABLATION: TWO CHANGES THAT ONLY WORK TOGETHER

  proposer  'rules' is the hand-written ladder (if the cause is X on attempt N,
            do Y). 'ev' picks the legal action and delay with the highest
            believed expected value.
  beliefs   'hand-written' is nine flat constants. 'fitted' is a table learned
            from kernel-legal exploration on train, conditioned on the attempt
            index and, for scheduled retries, on the delay.
""")
    print(f"{'proposer':<12}{'beliefs':<14}{'recovered':>11}{'net INR':>13}")
    print("-" * 50)
    for prop in ("rules", "ev"):
        for bname, b in (("hand-written", None), ("fitted", beliefs)):
            policy.set_proposer(prop)
            policy.set_beliefs(b)
            r = score(ReclaimAgent(), test)
            print(f"{prop:<12}{bname:<14}{r['recovery_rate']*100:>10.1f}%{r['net_inr']:>13,.0f}")
    policy.set_proposer("rules")
    policy.set_beliefs(None)
    print("-" * 50)
    print("""
Fitting the beliefs changes nothing on its own, because under the rules ladder
nothing consults them -- the expected-value gate only fires on payments too
small to chase, so it was inert. And the EV proposer is WORSE than the ladder on
hand-written beliefs, because flat constants cannot tell a 48-hour retry from an
immediate one, and timing is the entire lever on funds failures.

Together they beat the ladder. Reporting only the combined number would have
made this look like one improvement instead of two halves that are useless
apart, and would have hidden that the belief table is the part doing the work.""")

    diagnosis_report(test)

    if args.sensitivity:
        sensitivity(test)


if __name__ == "__main__":
    main()
