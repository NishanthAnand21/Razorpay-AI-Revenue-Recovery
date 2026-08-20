"""A book of B2B invoices to chase.

Buyers differ along two axes that matter and are not the same thing: when they
would pay if left alone, and how much chasing actually moves them. Conflating
those is why blanket dunning ladders waste effort -- most late payers are simply
slow, and chasing a slow-but-certain payer buys days at the cost of the account.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.receivables import Invoice, appointed_day

SEED = 771177
N = 6_000
HORIZON_DAYS = 180


def generate(n: int = N, seed: int = SEED) -> list[Invoice]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        issued = rng.randrange(0, 60)
        accepted = issued + rng.randint(0, 5)
        terms = rng.choices([15, 30, 45, 60, 90], [0.12, 0.42, 0.26, 0.14, 0.06])[0]
        amount = round(rng.choice([
            rng.uniform(8_000, 60_000), rng.uniform(60_000, 400_000),
            rng.uniform(400_000, 2_500_000)]), 2)
        # Most suppliers on a payments platform are small; MSME registration is
        # common but far from universal, and it gates the statutory remedies.
        is_msme = rng.random() < 0.58
        disputed = rng.random() < 0.05

        due = appointed_day(accepted, terms)
        archetype = rng.choices(["prompt", "slow", "delinquent"], [0.46, 0.38, 0.16])[0]
        if archetype == "prompt":
            would_pay, reliability = due - rng.randint(0, 4), 0.15
        elif archetype == "slow":
            would_pay, reliability = due + rng.randint(5, 40), 0.55
        else:
            # Pays very late or not within the horizon at all.
            would_pay = due + rng.randint(60, 200)
            reliability = 0.80
        if disputed:
            would_pay = due + 999          # nothing moves until the dispute clears

        # Buyer history is observable and informative but noisy -- a handful of
        # past invoices is a weak estimate of behaviour, and the generator makes
        # it weak on purpose so the prioritiser has to work with what a real
        # ledger would actually give it.
        prior_n = rng.randint(0, 12)
        true_rate = {"prompt": 0.12, "slow": 0.62, "delinquent": 0.86}[archetype]
        observed = (sum(1 for _ in range(prior_n) if rng.random() < true_rate) / prior_n
                    if prior_n else 0.35)

        out.append(Invoice(
            invoice_id=f"inv_{i:06d}", buyer_id=f"buy_{rng.randint(1, 900):04d}",
            amount_inr=amount, issued_day=issued, accepted_day=accepted,
            agreed_terms_days=terms, supplier_is_msme=is_msme, disputed=disputed,
            buyer_prior_late_ratio=round(observed, 3), buyer_prior_invoices=prior_n,
            would_pay_on_day=would_pay, reliability=reliability))
    return out


def main() -> None:
    inv = generate()
    path = Path(__file__).parent / "receivables.jsonl"
    with path.open("w") as fh:
        for x in inv:
            fh.write(json.dumps(asdict(x)) + "\n")
    late = [x for x in inv if x.would_pay_on_day > x.due_day]
    print(f"wrote {len(inv):,} invoices -> {path.name}")
    print(f"  would go late unaided: {len(late):,} ({len(late)/len(inv):.0%})")
    print(f"  MSME suppliers:        {sum(x.supplier_is_msme for x in inv):,}")
    print(f"  disputed:              {sum(x.disputed for x in inv):,}")
    print(f"  value at risk:         INR {sum(x.amount_inr for x in inv):,.0f}")


if __name__ == "__main__":
    main()
