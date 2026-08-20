"""Machine-check every safety property of the compliance kernel."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.verify import (PROPERTIES, assert_permissive_is_permissive, check,
                            check_liveness, check_monotonicity)


def main() -> None:
    print("Compliance kernel verification\n")
    assert_permissive_is_permissive()
    print("baseline check: PERMISSIVE fires no vetoes on any rail\n")
    print(f"{'property':<38}{'states':>10}{'result':>10}{'ms':>8}")
    print("-" * 66)

    total_states = 0
    failures = []

    mono = check_monotonicity()
    total_states += mono.states_checked
    print(f"{mono.name:<38}{mono.states_checked:>10,}"
          f"{('HOLDS' if mono.holds else 'FAILED'):>10}{mono.seconds*1000:>8.0f}")
    if not mono.holds:
        failures.append(mono)

    for prop in PROPERTIES:
        r = check(prop)
        total_states += r.states_checked
        print(f"{r.name:<38}{r.states_checked:>10,}"
              f"{('HOLDS' if r.holds else 'FAILED'):>10}{r.seconds*1000:>8.0f}")
        if not r.holds:
            failures.append((r, prop))

    live = check_liveness()
    total_states += live.states_checked
    print(f"{live.name:<38}{live.states_checked:>10,}"
          f"{('HOLDS' if live.holds else 'FAILED'):>10}{live.seconds*1000:>8.0f}")
    if not live.holds:
        failures.append(live)

    print("-" * 66)
    print(f"{'total':<38}{total_states:>10,}")

    if failures:
        print(f"\n{len(failures)} PROPERTIES FAILED")
        for f in failures:
            r = f[0] if isinstance(f, tuple) else f
            print(f"\n  {r.name}: {len(r.counterexamples)} counterexample(s)")
            s, leaked = r.counterexamples[0]
            print(f"    leaked actions: {sorted(a.value for a in leaked)}")
            print(f"    state: rail={s.rail.value} code={s.network_response_code} "
                  f"settlement={s.settlement_state.value} hour={s.local_hour} "
                  f"reattempts={s.reattempts_30d} cycle={s.attempts_this_mandate_cycle}")
        sys.exit(1)

    print("""
All properties hold.

The projected properties are exhaustive, not sampled: each enumerates every
combination of the fields its trigger reads, with all other fields set to their
most-permissive values. Because the kernel only ever subtracts actions, a
property that holds in the most-permissive completion holds in every completion.
That argument depends on monotonicity, which is checked above rather than
assumed.

Liveness is the exception and is marked as sampled, not proved: it reads every
field, the full product is ~1.5 x 10^8 states, and it holds by construction
anyway -- STOP and ESCALATE_MANUAL are unioned back in after the veto loop. The
sample exists to catch a future edit that moves that union, not to establish
something the code already says plainly.""")


if __name__ == "__main__":
    main()
