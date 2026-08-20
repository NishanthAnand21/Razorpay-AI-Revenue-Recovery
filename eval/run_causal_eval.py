"""What did the agent actually cause? Graded against a known answer.

Run:
    python3 eval/run_causal_eval.py
    python3 eval/run_causal_eval.py --bandwidth-sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.causal import (density_continuity, gross_recovery_rate,
                            naive_difference, sharp_rd)
from data.panel import _peak, true_late_at, DebitEvent  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "panel.jsonl"

# Boundaries of the NPCI peak windows. At each one, the legality of an immediate
# retry flips -- and nothing about the customer does.
REAL_CUTOFFS = [
    (10.0, "peak opens: retry legal below, blocked above", False),
    (13.0, "peak closes: retry blocked below, legal above", True),
    (17.0, "peak opens: retry legal below, blocked above", False),
    (21.5, "peak closes: retry blocked below, legal above", True),
]
# Hours where no rule changes. A design that finds effects here is finding noise.
PLACEBO_CUTOFFS = [8.0, 11.5, 14.5, 15.5, 19.5]

BANDWIDTH = 1.0


def load() -> list[DebitEvent]:
    return [DebitEvent(**json.loads(l)) for l in DATA.read_text().splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth-sweep", action="store_true")
    ap.add_argument("--power", action="store_true")
    args = ap.parse_args()

    ev = load()
    treated = [float(e.recovered_24h) for e in ev if not _peak(e.fail_hour)]
    blocked = [float(e.recovered_24h) for e in ev if _peak(e.fail_hour)]
    running = [e.fail_hour for e in ev]
    outcome = [float(e.recovered_24h) for e in ev]

    print(f"Causal evaluation -- {len(ev):,} mandate debit failures over 90 days")
    print("Treatment (an immediate retry) is assigned by the compliance kernel:")
    print("legal outside the NPCI peak windows, illegal inside them.\n")

    gross = gross_recovery_rate(treated)
    naive = naive_difference(treated, blocked)

    print("WHAT THE INDUSTRY WOULD REPORT")
    print(f"  gross recovery on chased payments   {gross:>8.4f}   "
          f"<- 'we recover {gross:.0%} of failed payments'")
    print(f"  naive treated-minus-blocked          {naive:>8.4f}")

    print("\nWHAT ACTUALLY HAPPENED, at each regulatory boundary")
    print(f"{'cutoff':>8}{'RD estimate':>34}{'true LATE':>12}{'error':>9}")
    print("-" * 65)
    ests = []
    for cut, _desc, treat_above in REAL_CUTOFFS:
        r = sharp_rd(running, outcome, cutoff=cut, bandwidth=BANDWIDTH,
                     treat_above=treat_above)
        truth = true_late_at(cut, ev, BANDWIDTH)
        ests.append((r, truth))
        covered = "yes" if r.ci95[0] <= truth <= r.ci95[1] else "NO"
        print(f"{cut:>8.1f}{str(r):>34}{truth:>12.4f}{r.estimate - truth:>+9.4f}"
              f"   CI covers truth: {covered}")
    print("-" * 65)

    pooled = sum(r.estimate / (r.std_error ** 2) for r, _ in ests) / \
             sum(1 / (r.std_error ** 2) for r, _ in ests)
    pooled_se = (1 / sum(1 / (r.std_error ** 2) for r, _ in ests)) ** 0.5
    truth_avg = sum(t for _, t in ests) / len(ests)
    print(f"{'pooled':>8}{f'{pooled:+.4f} +/- {pooled_se:.4f}':>34}"
          f"{truth_avg:>12.4f}{pooled - truth_avg:>+9.4f}")

    print(f"""
The gross number overstates the true effect by {gross / truth_avg:.1f}x.

Of every 100 mandate debits we chase and see recover, roughly
{100 * (1 - truth_avg / gross):.0f} would have come back on their own. Reporting the
gross figure is not a rounding error -- it is claiming credit for other
people's behaviour, and it is what almost every recovery product does.

The naive difference is not the fix either: at {naive:+.4f} it is
{'over' if naive > truth_avg else 'under'}stated, because peak hours carry different traffic than
off-peak hours. Comparing the two groups compares two populations.""")

    print("\nVALIDITY CHECKS")
    print("  placebo cutoffs, where no rule changes and the answer should be 0:")
    worst = 0.0
    for cut in PLACEBO_CUTOFFS:
        r = sharp_rd(running, outcome, cutoff=cut, bandwidth=BANDWIDTH)
        flag = "  <- SIGNIFICANT, investigate" if r.significant else ""
        worst = max(worst, abs(r.estimate))
        print(f"    {cut:>5.1f}   {r}{flag}")
    print(f"  largest placebo effect: {worst:.4f} "
          f"({worst / truth_avg:.0%} of the real effect)")

    print("\n  density continuity at each real cutoff (manipulation check):")
    for cut, _d, _t in REAL_CUTOFFS:
        ratio = density_continuity(running, cut, BANDWIDTH)
        print(f"    {cut:>5.1f}   above/below density ratio {ratio:.3f}")
    print("  Ratios near 1.0: nobody is choosing which side to fail on, which is")
    print("  the assumption the whole design rests on. Here it holds by construction.")

    if args.power:
        power_analysis(ev)

    if args.bandwidth_sweep:
        print(f"\n{'bandwidth':>10}{'estimate at 13:00':>22}{'true LATE':>12}")
        print("-" * 44)
        for h in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            r = sharp_rd(running, outcome, cutoff=13.0, bandwidth=h, treat_above=True)
            print(f"{h:>10.2f}{str(r):>22}{true_late_at(13.0, ev, h):>12.4f}")
        print("Wide bandwidths trade variance for bias: they borrow strength from")
        print("hours where the populations genuinely differ. The estimate should be")
        print("stable across the narrow end and drift as the window widens.")


def power_analysis(ev: list[DebitEvent]) -> None:
    """How much volume does this design need before it can be believed?

    A first run at 60k events produced a placebo effect worth 72% of the real
    one -- the design was not wrong, it was underpowered, and an underpowered RD
    is indistinguishable from a broken one. So the honest deliverable is not just
    an estimate but the volume at which the estimate becomes trustworthy.

    Minimum detectable effect is quoted at 80% power and 5% significance, which
    is the 2.8 * SE convention.
    """
    import random
    rng = random.Random(11)
    print(f"\n{'events':>12}{'n at cutoff':>13}{'SE':>9}{'MDE (80% power)':>18}"
          f"{'usable?':>10}")
    print("-" * 62)
    true_effect = true_late_at(13.0, ev, BANDWIDTH)
    for n in (25_000, 50_000, 100_000, 200_000, 400_000):
        sub = rng.sample(ev, min(n, len(ev)))
        r = sharp_rd([e.fail_hour for e in sub], [float(e.recovered_24h) for e in sub],
                     cutoff=13.0, bandwidth=BANDWIDTH, treat_above=True)
        mde = 2.8 * r.std_error
        verdict = "yes" if mde < true_effect else "no"
        print(f"{n:>12,}{r.n_left + r.n_right:>13,}{r.std_error:>9.4f}"
              f"{mde:>18.4f}{verdict:>10}")
    print("-" * 62)
    print(f"The effect we are trying to see is {true_effect:.4f}. A merchant needs")
    print("roughly 100k failed mandate debits in the window before this design can")
    print("resolve it -- about 1.1k/day over 90 days. Below that, report the")
    print("confidence interval and refuse to report a point estimate.")


if __name__ == "__main__":
    main()
