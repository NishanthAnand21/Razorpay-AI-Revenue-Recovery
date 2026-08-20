"""Can a bad model cause a compliance violation? It should be impossible.

The point of moving constraints into a kernel that reads only facts is that the
guarantee stops depending on the model being right. That is a claim, and claims
should be attacked rather than asserted -- so this harness runs the pipeline with
diagnosers that are progressively more hostile and counts violations.

    honest      the real tiered diagnoser
    confused    uniformly random causes, at full confidence
    adversarial deliberately picks the label that unlocks the most dangerous
                action available, and claims certainty

If the kernel holds, all three produce zero violations of the network and mandate
rules. The recovered revenue collapses, of course -- a broken model makes bad
*business* decisions. It just cannot make illegal ones.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.compliance import (MONEY_ACTIONS, Rail, SettlementState, feasible_actions,
                                observe)
from reclaim.diagnose import Diagnosis, TieredDiagnoser
from reclaim.models import Action, RootCause
from reclaim.policy import RecoveryState, propose
from run_eval import load  # noqa: E402


class Honest:
    name = "honest"
    def __init__(self): self.d = TieredDiagnoser()
    def diagnose(self, p): return self.d.diagnose(p)


class Confused:
    """Uniformly random, and completely sure of itself."""
    name = "confused"
    def __init__(self, seed=1): self.rng = random.Random(seed)
    def diagnose(self, p):
        c = self.rng.choice([c for c in RootCause if c is not RootCause.UNKNOWN])
        return Diagnosis(c, 1.0, "adversarial", "random label at full confidence")


class Adversarial:
    """Actively tries to get money moved on payments that must not be touched."""
    name = "adversarial"
    def diagnose(self, p):
        # TRANSIENT_ISSUER is the label that unlocks an immediate retry, so claim
        # it for everything -- including revoked mandates and fraud declines.
        return Diagnosis(RootCause.TRANSIENT_ISSUER, 1.0, "adversarial",
                         "claiming a retryable blip regardless of the evidence")


def run(diagnoser, rows, *, kernel: bool) -> dict:
    """Run one decision step per payment, with and without the kernel."""
    violations = unknown_settlement_charges = money_actions = 0

    for p in rows:
        dx = diagnoser.diagnose(p)
        st = RecoveryState(clock_hour=p.failed_at_hour)
        d = propose(p, dx, st)          # raw proposal, before any safety layer

        s = observe(p, local_hour=p.failed_at_hour)
        allowed, _ = feasible_actions(s)

        if kernel and d.action not in allowed:
            # The kernel does not argue with the model. It removes the option.
            d.action = Action.ESCALATE_MANUAL

        if d.action in MONEY_ACTIONS:
            money_actions += 1
            code = (s.network_response_code or "").upper()
            from reclaim.compliance import MASTERCARD_NEVER_RETRY, VISA_NEVER_RETRY
            if (s.rail is Rail.CARD_VISA and code in VISA_NEVER_RETRY) or \
               (s.rail is Rail.CARD_MASTERCARD and code in MASTERCARD_NEVER_RETRY) or \
               not s.mandate_active:
                violations += 1
            if s.settlement_state is SettlementState.UNKNOWN:
                unknown_settlement_charges += 1

    return {"money_actions": money_actions, "violations": violations,
            "double_charge_risk": unknown_settlement_charges}


def main() -> None:
    rows = load("test")
    print(f"Adversarial safety check -- {len(rows)} payments, one decision each\n")
    print(f"{'diagnoser':<14}{'kernel':>8}{'money actions':>15}"
          f"{'network violations':>20}{'double-charge risk':>20}")
    print("-" * 77)
    for dg in (Honest(), Confused(), Adversarial()):
        for kernel in (False, True):
            r = run(dg, rows, kernel=kernel)
            print(f"{dg.name:<14}{('on' if kernel else 'off'):>8}"
                  f"{r['money_actions']:>15}{r['violations']:>20}"
                  f"{r['double_charge_risk']:>20}")
        print("-" * 77)

    print("""
Read the 'off' rows first. With constraints living behind the model, an
adversarial diagnoser walks straight through them: it asserts a retryable blip on
revoked mandates and fraud declines, and the policy obliges.

With the kernel on, every one of those columns is zero, for every diagnoser --
including the one built to break it. Nothing was tuned to achieve that. The
kernel reads a response code, a counter and a clock; the model's opinion is not
an input, so the model's error cannot be a cause.

The double-charge column is the one a payments engineer will care about most. A
gateway timeout does not mean the payment failed, it means nobody told us yet.
The original policy treated it as a transient blip and retried immediately, which
is precisely how a customer gets debited twice. The kernel refuses any money
action while settlement is unconfirmed -- reconcile first, then decide.""")


if __name__ == "__main__":
    main()
