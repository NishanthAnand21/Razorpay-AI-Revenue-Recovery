"""Generate the synthetic failed-payment dataset.

Deterministic: the same seed always produces the same file, so every number in
the eval report is reproducible by anyone who clones the repo.

The generator deliberately plants two hard things:

  1. ~18% of rows carry a failure reason that is NOT in the rules lookup table.
     A rules-only diagnoser must fall through to UNKNOWN on these. This is the
     gap the LLM diagnoser is measured against.
  2. Some reasons are genuinely ambiguous from the code alone and are only
     resolvable from the free-text merchant note.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.models import FailedPayment, RootCause

SEED = 20260820
N = 800
TEST_FRACTION = 0.30

# (error_code, error_reason, is_novel) grouped by the ground-truth root cause.
# "novel" reasons are held out of the rules table on purpose.
REASON_BANK: dict[RootCause, list[tuple[str, str, bool]]] = {
    RootCause.TRANSIENT_ISSUER: [
        ("GATEWAY_ERROR", "payment_failed_at_bank", False),
        ("GATEWAY_ERROR", "gateway_timeout", False),
        ("GATEWAY_ERROR", "issuer_unavailable", False),
        ("GATEWAY_ERROR", "npci_downtime", True),
        ("GATEWAY_ERROR", "acquirer_switch_error", True),
    ],
    RootCause.INSUFFICIENT_FUNDS: [
        ("BAD_REQUEST_ERROR", "insufficient_funds", False),
        ("BAD_REQUEST_ERROR", "account_balance_low", False),
        ("BAD_REQUEST_ERROR", "debit_failed_low_balance", True),
    ],
    RootCause.AUTH_FRICTION: [
        ("BAD_REQUEST_ERROR", "incorrect_otp", False),
        ("BAD_REQUEST_ERROR", "otp_attempts_exceeded", False),
        ("BAD_REQUEST_ERROR", "upi_collect_expired", False),
        ("BAD_REQUEST_ERROR", "payment_cancelled_by_user", False),
        ("BAD_REQUEST_ERROR", "3ds_challenge_abandoned", True),
    ],
    RootCause.INSTRUMENT_INVALID: [
        ("BAD_REQUEST_ERROR", "card_expired", False),
        ("BAD_REQUEST_ERROR", "invalid_vpa", False),
        ("BAD_REQUEST_ERROR", "mandate_revoked", False),
        ("BAD_REQUEST_ERROR", "card_reported_lost", True),
        ("BAD_REQUEST_ERROR", "token_deprovisioned", True),
    ],
    RootCause.LIMIT_EXCEEDED: [
        ("BAD_REQUEST_ERROR", "per_transaction_limit_exceeded", False),
        ("BAD_REQUEST_ERROR", "daily_limit_exceeded", False),
        ("BAD_REQUEST_ERROR", "upi_txn_cap_breached", True),
    ],
    RootCause.RISK_DECLINED: [
        ("BAD_REQUEST_ERROR", "payment_declined_by_risk", False),
        ("BAD_REQUEST_ERROR", "suspected_fraud", False),
        ("BAD_REQUEST_ERROR", "velocity_rule_triggered", True),
    ],
}

# How often each root cause shows up. Roughly mirrors the shape of a real
# failure mix: transient and funds dominate, risk is rare but expensive.
CAUSE_WEIGHTS: dict[RootCause, float] = {
    RootCause.TRANSIENT_ISSUER: 0.30,
    RootCause.INSUFFICIENT_FUNDS: 0.26,
    RootCause.AUTH_FRICTION: 0.20,
    RootCause.INSTRUMENT_INVALID: 0.13,
    RootCause.LIMIT_EXCEEDED: 0.07,
    RootCause.RISK_DECLINED: 0.04,
}

NOTE_TEMPLATES: dict[RootCause, list[str]] = {
    RootCause.TRANSIENT_ISSUER: [
        "bank side error, customer says app showed 'try again later'",
        "spike of these in the last 20 min, looks like a bank outage",
        "",
    ],
    RootCause.INSUFFICIENT_FUNDS: [
        "customer messaged saying salary credits on the 1st",
        "customer asked us to try again after payday",
        "",
    ],
    RootCause.AUTH_FRICTION: [
        "customer did not approve the collect request in time",
        "customer says he never got the OTP sms",
        "",
    ],
    RootCause.INSTRUMENT_INVALID: [
        "customer got a new card, old one is dead",
        "customer closed that bank account last month",
        "",
    ],
    RootCause.LIMIT_EXCEEDED: [
        "customer hit his UPI daily cap, wants to pay tomorrow",
        "",
    ],
    RootCause.RISK_DECLINED: [
        "flagged by risk, do not auto retry",
        "",
    ],
}

# Reasons that genuinely arise from more than one root cause. These are the ones
# that make the problem real: `do_not_honour` is the classic example -- the issuer
# declines without saying why, and it can mean no money, a dead card, or a fraud
# rule. The surface string cannot resolve them. Only the merchant note sometimes
# can, and often nothing can, in which case the honest answer is to escalate
# rather than to guess with somebody's money.
AMBIGUOUS_REASONS: dict[str, tuple[RootCause, ...]] = {
    "do_not_honour": (RootCause.INSUFFICIENT_FUNDS, RootCause.INSTRUMENT_INVALID,
                      RootCause.RISK_DECLINED),
    "transaction_not_permitted": (RootCause.LIMIT_EXCEEDED, RootCause.INSTRUMENT_INVALID,
                                  RootCause.RISK_DECLINED),
    "authentication_failed": (RootCause.AUTH_FRICTION, RootCause.INSTRUMENT_INVALID),
    "payment_failed": (RootCause.TRANSIENT_ISSUER, RootCause.INSUFFICIENT_FUNDS,
                       RootCause.AUTH_FRICTION),
    "debit_declined": (RootCause.INSUFFICIENT_FUNDS, RootCause.RISK_DECLINED,
                       RootCause.INSTRUMENT_INVALID),
}
# Share of rows whose reason string is overwritten with an ambiguous one.
AMBIGUOUS_FRACTION = 0.24

METHODS = ["upi", "card", "netbanking", "wallet"]
METHOD_WEIGHTS = [0.52, 0.30, 0.13, 0.05]


def _amount(rng: random.Random) -> float:
    """Long-tailed amounts, the way real payment volume actually looks."""
    bucket = rng.random()
    if bucket < 0.55:
        return round(rng.uniform(99, 1500), 2)
    if bucket < 0.88:
        return round(rng.uniform(1500, 12000), 2)
    return round(rng.uniform(12000, 95000), 2)


def generate(n: int = N, seed: int = SEED) -> list[FailedPayment]:
    rng = random.Random(seed)
    causes = list(CAUSE_WEIGHTS)
    weights = [CAUSE_WEIGHTS[c] for c in causes]

    rows: list[FailedPayment] = []
    for i in range(n):
        cause = rng.choices(causes, weights=weights, k=1)[0]
        code, reason, _novel = rng.choice(REASON_BANK[cause])
        is_recurring = rng.random() < 0.34
        # Mandate debits skew to instrument/funds problems in the real world;
        # keep the method consistent with that.
        method = "upi" if is_recurring and rng.random() < 0.6 else rng.choices(METHODS, METHOD_WEIGHTS, k=1)[0]

        # Overwrite some reasons with an ambiguous string that this cause could
        # plausibly have produced. The ground-truth label stays the same, so the
        # diagnoser is being asked to do something genuinely hard.
        ambiguous_for_cause = [r for r, causes in AMBIGUOUS_REASONS.items() if cause in causes]
        if ambiguous_for_cause and rng.random() < AMBIGUOUS_FRACTION:
            reason = rng.choice(ambiguous_for_cause)
            code = "BAD_REQUEST_ERROR"

        rows.append(
            FailedPayment(
                payment_id=f"pay_{seed}{i:05d}",
                customer_id=f"cust_{rng.randint(1, 420):04d}",
                amount_inr=_amount(rng),
                method=method,
                error_code=code,
                error_reason=reason,
                merchant_note=rng.choice(NOTE_TEMPLATES[cause]),
                failed_at_hour=rng.randint(0, 23),
                is_recurring=is_recurring,
                customer_prior_failures=rng.choices([0, 1, 2, 3], [0.62, 0.24, 0.10, 0.04], k=1)[0],
                true_root_cause=cause,
            )
        )
    return rows


def split(rows: list[FailedPayment], seed: int = SEED) -> tuple[list[FailedPayment], list[FailedPayment]]:
    """Hold out a test set. The policy is tuned on train and reported on test."""
    rng = random.Random(seed + 1)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - TEST_FRACTION))
    return shuffled[:cut], shuffled[cut:]


def main() -> None:
    rows = generate()
    train, test = split(rows)
    out = Path(__file__).parent
    for name, subset in (("train", train), ("test", test)):
        path = out / f"{name}.jsonl"
        with path.open("w") as fh:
            for r in subset:
                d = r.to_public_dict()
                d["true_root_cause"] = r.true_root_cause.value
                fh.write(json.dumps(d) + "\n")
        print(f"wrote {len(subset):4d} rows -> {path.relative_to(out.parent)}")


if __name__ == "__main__":
    main()
