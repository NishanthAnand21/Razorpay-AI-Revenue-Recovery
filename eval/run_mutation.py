"""Does the verifier actually catch anything?

A suite that always reports success is indistinguishable from a suite that
checks nothing, and the verification output is the strongest claim in this repo.
So each rule in the kernel is deliberately broken, one at a time, and the
verifier must notice. A mutation that survives is a rule nobody is checking.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim import compliance
from reclaim.verify import PROPERTIES, check

# Each mutation is a plausible edit -- an off-by-one in a cap, a window trimmed,
# a code dropped from a list -- paired with the property that should die.
MUTATIONS = [
    ("drop code 41 from Visa never-retry",
     "visa_category_1_never_retried",
     lambda: setattr(compliance, "VISA_NEVER_RETRY",
                     compliance.VISA_NEVER_RETRY - {"41"})),

    ("drop MAC21 from Mastercard never-retry",
     "mastercard_do_not_retry_honoured",
     lambda: setattr(compliance, "MASTERCARD_NEVER_RETRY",
                     compliance.MASTERCARD_NEVER_RETRY - {"MAC21"})),

    ("Visa 30-day cap off by one (15 -> 16)",
     "visa_30d_cap_respected",
     lambda: setattr(compliance, "VISA_REATTEMPT_CAP_30D", 16)),

    ("Mastercard 30-day cap off by one (10 -> 11)",
     "mastercard_30d_cap_respected",
     lambda: setattr(compliance, "MASTERCARD_REATTEMPT_CAP_30D", 11)),

    ("Autopay cycle cap off by one (4 -> 5)",
     "autopay_cycle_cap_respected",
     lambda: setattr(compliance, "UPI_AUTOPAY_ATTEMPTS_PER_CYCLE", 5)),

    ("evening peak window ends early (21.5 -> 21.0)",
     "autopay_never_executes_in_peak",
     lambda: setattr(compliance, "UPI_PEAK_WINDOWS", ((10.0, 13.0), (17.0, 21.0)))),

    ("pre-debit notice requirement weakened to 12h",
     "autopay_requires_pre_debit_notice",
     lambda: setattr(compliance, "EMANDATE_PRE_DEBIT_NOTICE_HOURS", 12.0)),

    ("collections contact window widened to 08:00-21:00",
     "contact_window_respected",
     lambda: setattr(compliance, "COLLECTIONS_CONTACT_WINDOW", (8.0, 21.0))),
]


def snapshot() -> dict:
    return {k: getattr(compliance, k) for k in (
        "VISA_NEVER_RETRY", "MASTERCARD_NEVER_RETRY", "VISA_REATTEMPT_CAP_30D",
        "MASTERCARD_REATTEMPT_CAP_30D", "UPI_AUTOPAY_ATTEMPTS_PER_CYCLE",
        "UPI_PEAK_WINDOWS", "EMANDATE_PRE_DEBIT_NOTICE_HOURS",
        "COLLECTIONS_CONTACT_WINDOW")}


def restore(snap: dict) -> None:
    for k, v in snap.items():
        setattr(compliance, k, v)


def main() -> None:
    by_name = {p.name: p for p in PROPERTIES}
    original = snapshot()

    print("Mutation testing the compliance kernel\n")
    print(f"{'injected bug':<46}{'property':<36}{'':>8}")
    print("-" * 92)

    escaped = []
    for label, prop_name, mutate in MUTATIONS:
        restore(original)
        mutate()
        r = check(by_name[prop_name])
        caught = not r.holds
        if not caught:
            escaped.append(label)
        print(f"{label:<46}{prop_name:<36}"
              f"{('CAUGHT' if caught else 'ESCAPED'):>8}")
    restore(original)

    print("-" * 92)
    print(f"{len(MUTATIONS) - len(escaped)}/{len(MUTATIONS)} mutations caught")

    # And confirm the suite is clean again once the bugs are reverted.
    clean = all(check(p).holds for p in PROPERTIES)
    print(f"kernel restored and re-verified: {'clean' if clean else 'STILL FAILING'}")

    if escaped:
        print("\nESCAPED MUTATIONS -- these rules are not actually being checked:")
        for e in escaped:
            print(f"  - {e}")
        sys.exit(1)

    print("""
Every injected bug was caught, and the suite goes green again once they are
reverted. The verifier is therefore doing work: each rule has at least one
property that fails when the rule is wrong, which is the thing a passing test
run cannot tell you on its own.""")


if __name__ == "__main__":
    main()
