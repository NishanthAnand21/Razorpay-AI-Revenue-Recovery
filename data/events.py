"""A 30-day raw event stream for one merchant.

This is deliberately not a list of problems. It is what a merchant's systems
actually emit -- checkout sessions, mandate debits, invoices -- most of which are
completely fine. Finding the slipping revenue in here is the detector's job, and
the false-positive cost of getting it wrong is real money.

Ground truth on every record answers the counterfactual the detector cannot see:
would this have been lost, and would the customer have fixed it themselves?
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 20260820
DAYS = 30
HOURS = DAYS * 24


@dataclass
class CheckoutSession:
    """A customer reached checkout. Most convert. Some stall. Some come back."""

    session_id: str
    customer_id: str
    amount_inr: float
    started_hour: int
    method_selected: str | None      # None = never got as far as picking one
    reached_payment_page: bool
    completed_hour: int | None       # when they actually paid, if they did
    customer_prior_orders: int
    is_new_customer: bool
    # ground truth
    would_self_recover: bool         # came back unaided within the window


@dataclass
class MandateDebit:
    """A subscription charge against a standing mandate."""

    debit_id: str
    customer_id: str
    amount_inr: float
    attempted_hour: int
    succeeded: bool
    error_reason: str | None
    mandate_age_days: int
    consecutive_failures: int
    would_self_recover: bool         # customer tops up / re-authorises unprompted


@dataclass
class Invoice:
    """A B2B invoice. The valuable detection here happens BEFORE the due date."""

    invoice_id: str
    customer_id: str
    amount_inr: float
    issued_hour: int
    due_hour: int
    paid_hour: int | None
    customer_avg_days_late: float    # from this buyer's own history
    customer_prior_invoices: int
    disputed: bool
    would_self_recover: bool         # pays late but pays, with no chasing


def _amount(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 2)


def generate_checkouts(rng: random.Random, n: int = 900) -> list[CheckoutSession]:
    out = []
    for i in range(n):
        started = rng.randrange(HOURS)
        new_cust = rng.random() < 0.42
        prior = 0 if new_cust else rng.choices([1, 2, 3, 8], [0.4, 0.3, 0.2, 0.1])[0]
        amount = _amount(rng, 199, 1200) if rng.random() < 0.6 else _amount(rng, 1200, 24000)
        reached_payment = rng.random() < 0.72
        method = rng.choice(["upi", "card", "netbanking", "wallet"]) if reached_payment else None

        # Conversion odds: returning customers and cheaper carts convert more.
        p_convert = 0.58 + (0.14 if not new_cust else 0.0) - (0.10 if amount > 8000 else 0.0)
        p_convert += 0.10 if reached_payment else -0.25
        completed = rng.random() < max(0.05, min(0.95, p_convert))

        if completed:
            out.append(CheckoutSession(
                f"cs_{i:05d}", f"cust_{rng.randint(1, 500):04d}", amount, started,
                method, reached_payment, started + rng.randint(0, 1), prior, new_cust, False))
            continue

        # Stalled. Some of these customers come back on their own -- and those are
        # precisely the ones it is a mistake to spend a nudge on.
        p_self = 0.30 + (0.15 if not new_cust else 0.0) + (0.10 if reached_payment else 0.0)
        self_rec = rng.random() < p_self
        out.append(CheckoutSession(
            f"cs_{i:05d}", f"cust_{rng.randint(1, 500):04d}", amount, started,
            method, reached_payment,
            started + rng.randint(2, 20) if self_rec else None,
            prior, new_cust, self_rec))
    return out


def generate_mandates(rng: random.Random, n: int = 420) -> list[MandateDebit]:
    reasons = ["insufficient_funds", "mandate_revoked", "debit_not_registered",
               "gateway_timeout", "do_not_honour"]
    out = []
    for i in range(n):
        ok = rng.random() < 0.79
        reason = None if ok else rng.choices(reasons, [0.46, 0.12, 0.10, 0.20, 0.12])[0]
        consec = 0 if ok else rng.choices([1, 2, 3], [0.68, 0.22, 0.10])[0]
        # A customer whose balance was simply low often tops up unprompted.
        self_rec = (not ok) and reason == "insufficient_funds" and rng.random() < 0.34
        out.append(MandateDebit(
            f"md_{i:05d}", f"cust_{rng.randint(1, 500):04d}",
            _amount(rng, 149, 4999), rng.randrange(HOURS), ok, reason,
            rng.randint(30, 900), consec, self_rec))
    return out


def generate_invoices(rng: random.Random, n: int = 260) -> list[Invoice]:
    out = []
    for i in range(n):
        issued = rng.randrange(0, HOURS - 24)
        terms_days = rng.choice([15, 30, 45])
        due = issued + terms_days * 24
        avg_late = rng.choices([0.0, 3.0, 9.0, 22.0], [0.44, 0.28, 0.19, 0.09])[0]
        prior = rng.randint(0, 14)
        disputed = rng.random() < 0.06

        # Buyers behave like their own history, mostly.
        if disputed:
            paid, self_rec = None, False
        elif avg_late <= 1.0 and rng.random() < 0.88:
            paid, self_rec = due - rng.randint(0, 48), False       # pays on time
        elif rng.random() < 0.62:
            late_h = int(rng.gauss(avg_late, 4) * 24)
            paid, self_rec = due + max(24, late_h), True            # late, but pays unaided
        else:
            paid, self_rec = None, False                           # goes bad
        out.append(Invoice(
            f"inv_{i:05d}", f"cust_{rng.randint(1, 500):04d}",
            _amount(rng, 12000, 900000), issued, due, paid,
            avg_late, prior, disputed, self_rec))
    return out


def main() -> None:
    rng = random.Random(SEED)
    stream = {
        "checkouts": [asdict(x) for x in generate_checkouts(rng)],
        "mandates": [asdict(x) for x in generate_mandates(rng)],
        "invoices": [asdict(x) for x in generate_invoices(rng)],
    }
    path = Path(__file__).parent / "events.json"
    path.write_text(json.dumps(stream, indent=1))
    tot = sum(len(v) for v in stream.values())
    print(f"wrote {tot} raw events over {DAYS} days -> {path.name}")
    for k, v in stream.items():
        print(f"  {k:<12} {len(v):>5}")


if __name__ == "__main__":
    main()
