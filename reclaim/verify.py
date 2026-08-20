"""Proving the compliance kernel, rather than testing it.

Tests sample a state space. For something that decides whether money moves, that
is the wrong strength of evidence: a test suite that passes tells you the states
you thought of are fine.

The state space here is finite once the continuous fields are discretised at
their decision boundaries, but the naive product is around 10^8 states, which is
not enumerable in pure Python in a reasonable time. Two observations make it
tractable, and both are properties of how the kernel is written rather than
tricks:

MONOTONICITY
    `feasible_actions` computes `ALL_ACTIONS - union(vetoes)`. Vetoes only ever
    subtract. So adding any condition to a state can only shrink the allowed set,
    never grow it.

MOST-PERMISSIVE COMPLETION
    Therefore, to prove "in every state satisfying T, action a is forbidden", it
    suffices to enumerate the variables T actually reads, and set every *other*
    variable to the value that fires the fewest vetoes. If a is already forbidden
    in that most-permissive completion, it is forbidden in every completion,
    because every other completion has at least as many vetoes.

That reduces each property to a few thousand states, and the result is a proof
over the whole discretised space rather than a sample of it. The discretisation
is itself sound because every constraint in the kernel is a threshold comparison:
between two adjacent boundary values nothing changes, so checking the boundaries
and one point either side covers the continuum.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterator

from .compliance import (DEAD_INSTRUMENT_REASONS, MASTERCARD_NEVER_RETRY,
                         MONEY_ACTIONS, OUTREACH_ACTIONS, RISK_DECLINE_REASONS,
                         Rail, SettlementState, VISA_NEVER_RETRY, ObservableState,
                         feasible_actions)
from .models import Action

# The most permissive state: every optional constraint switched off. Any veto
# that fires from here fires because of the variables under test, not because of
# background conditions.
PERMISSIVE = ObservableState(
    rail=Rail.NETBANKING,
    network_response_code=None,
    decline_reason=None,
    settlement_state=SettlementState.FAILED,
    reattempts_30d=0,
    attempts_this_mandate_cycle=0,
    hours_since_pre_debit_notice=48.0,
    # 14:00 is the only kind of hour that is permissive on every axis at once:
    # outside both NPCI peak windows, and inside both the collections (08-19) and
    # transactional (09-21) contact windows. An earlier version used 12:00, which
    # sits inside the morning peak -- so the peak veto silently blocked RETRY_NOW
    # in the baseline and masked a mutation to the pre-debit notice rule. That is
    # exactly the failure mode `assert_permissive_is_permissive` now prevents.
    local_hour=14.0,
    is_working_day=True,
    contacts_7d=0,
    max_contacts_7d=3,
    consent_transactional=True,
    consent_marketing=True,
    mandate_active=True,
    is_disputed=False,
    is_collections=False,
    # The receivables rules veto by default -- statutory remedies are unavailable
    # unless the supplier qualifies and the money is actually late -- so the
    # permissive baseline has to satisfy them, or every other property would be
    # checked against a baseline that is already blocking things.
    supplier_is_msme=True,
    days_past_appointed_day=1,
    promise_to_pay_open=False,
)

# Values to enumerate per field: every threshold, plus one point either side.
DOMAINS: dict[str, list] = {
    "rail": list(Rail),
    "network_response_code": sorted(VISA_NEVER_RETRY | MASTERCARD_NEVER_RETRY
                                    | {"05", "51", "54", "00"}) + [None],
    "decline_reason": sorted(RISK_DECLINE_REASONS | DEAD_INSTRUMENT_REASONS
                             | {"insufficient_funds", "do_not_honour", "gateway_timeout"}) + [None],
    "settlement_state": list(SettlementState),
    "reattempts_30d": [0, 9, 10, 11, 14, 15, 16, 40],
    "attempts_this_mandate_cycle": [0, 1, 3, 4, 5],
    "hours_since_pre_debit_notice": [None, 0.0, 23.9, 24.0, 25.0],
    "local_hour": [0.0, 7.9, 8.0, 9.0, 9.9, 10.0, 12.9, 13.0, 16.9, 17.0,
                   18.9, 19.0, 20.9, 21.0, 21.4, 21.5, 23.9],
    "is_working_day": [True, False],
    "contacts_7d": [0, 2, 3, 4],
    "consent_transactional": [True, False],
    "mandate_active": [True, False],
    "is_disputed": [True, False],
    "is_collections": [True, False],
    "supplier_is_msme": [True, False],
    "days_past_appointed_day": [-5, -1, 0, 1, 20],
    "promise_to_pay_open": [True, False],
}


def assert_permissive_is_permissive() -> None:
    """The baseline must fire no vetoes at all, on any rail.

    The whole projection argument assumes PERMISSIVE is the weakest state. If any
    veto fires there, properties are being checked against a baseline that is
    already blocking things, and a broken rule can hide behind an unrelated one.
    """
    from .compliance import vetoes
    for rail in Rail:
        s = replace(PERMISSIVE, rail=rail)
        fired = vetoes(s)
        if fired:
            raise AssertionError(
                f"PERMISSIVE baseline is not permissive on {rail.value}: "
                f"{[v.rule for v in fired]}"
            )


@dataclass
class Property:
    """A safety claim: whenever `trigger` holds, `forbidden` must be unavailable."""

    name: str
    reads: tuple[str, ...]                       # fields the trigger depends on
    trigger: Callable[[ObservableState], bool]
    forbidden: set[Action]
    why: str


def _states_over(fields: tuple[str, ...]) -> Iterator[ObservableState]:
    """Every combination of `fields`, with all other fields most-permissive."""
    domains = [DOMAINS[f] for f in fields]
    for combo in itertools.product(*domains):
        yield replace(PERMISSIVE, **dict(zip(fields, combo)))


PROPERTIES: list[Property] = [
    Property("risk_declines_never_recharged_any_rail",
             ("rail", "decline_reason"),
             lambda s: (s.decline_reason or "") in RISK_DECLINE_REASONS,
             MONEY_ACTIONS,
             "a fraud decline is as wrong to re-charge on UPI as on a card"),

    Property("dead_instruments_never_recharged",
             ("rail", "decline_reason"),
             lambda s: (s.decline_reason or "") in DEAD_INSTRUMENT_REASONS,
             MONEY_ACTIONS,
             "a dead instrument has a zero success rate on every rail"),

    Property("visa_category_1_never_retried",
             ("rail", "network_response_code"),
             lambda s: s.rail is Rail.CARD_VISA
                       and (s.network_response_code or "") in VISA_NEVER_RETRY,
             MONEY_ACTIONS,
             "Visa Category 1: the first reattempt is already a violation"),

    Property("mastercard_do_not_retry_honoured",
             ("rail", "network_response_code"),
             lambda s: s.rail is Rail.CARD_MASTERCARD
                       and (s.network_response_code or "") in MASTERCARD_NEVER_RETRY,
             MONEY_ACTIONS,
             "Mastercard advice codes MAC03/MAC21 mean stop"),

    Property("visa_30d_cap_respected",
             ("rail", "reattempts_30d"),
             lambda s: s.rail is Rail.CARD_VISA and s.reattempts_30d >= 15,
             MONEY_ACTIONS,
             "Visa allows at most 15 reattempts in 30 days"),

    Property("mastercard_30d_cap_respected",
             ("rail", "reattempts_30d"),
             lambda s: s.rail is Rail.CARD_MASTERCARD and s.reattempts_30d >= 10,
             MONEY_ACTIONS,
             "Mastercard allows at most 10 retries in 30 days"),

    Property("autopay_cycle_cap_respected",
             ("rail", "attempts_this_mandate_cycle"),
             lambda s: s.rail is Rail.UPI_AUTOPAY and s.attempts_this_mandate_cycle >= 4,
             MONEY_ACTIONS,
             "NPCI allows 1 execution plus 3 retries per mandate cycle"),

    Property("autopay_never_executes_in_peak",
             ("rail", "local_hour"),
             lambda s: s.rail is Rail.UPI_AUTOPAY
                       and ((10.0 <= s.local_hour < 13.0) or (17.0 <= s.local_hour < 21.5)),
             {Action.RETRY_NOW},
             "NPCI confines Autopay execution to non-peak windows"),

    Property("autopay_requires_pre_debit_notice",
             ("rail", "hours_since_pre_debit_notice"),
             lambda s: s.rail is Rail.UPI_AUTOPAY
                       and (s.hours_since_pre_debit_notice is None
                            or s.hours_since_pre_debit_notice < 24.0),
             {Action.RETRY_NOW},
             "RBI requires a pre-debit notification at least 24h before every debit"),

    Property("no_charge_while_settlement_unknown",
             ("settlement_state",),
             lambda s: s.settlement_state is SettlementState.UNKNOWN,
             MONEY_ACTIONS | OUTREACH_ACTIONS,
             "a timeout means unknown, not failed; charging again risks a double debit"),

    Property("no_action_once_already_paid",
             ("settlement_state",),
             lambda s: s.settlement_state is SettlementState.SUCCEEDED,
             MONEY_ACTIONS | OUTREACH_ACTIONS,
             "the money arrived; there is nothing to recover"),

    Property("revoked_mandate_never_debited",
             ("mandate_active",),
             lambda s: not s.mandate_active,
             MONEY_ACTIONS,
             "debiting a revoked mandate is unauthorised"),

    Property("contact_window_respected",
             ("local_hour", "is_collections"),
             lambda s: not ((8.0 if s.is_collections else 9.0) <= s.local_hour
                            < (19.0 if s.is_collections else 21.0)),
             OUTREACH_ACTIONS,
             "RBI confines collections contact to 08:00-19:00"),

    Property("contact_fatigue_cap_respected",
             ("contacts_7d",),
             lambda s: s.contacts_7d >= 3,
             OUTREACH_ACTIONS,
             "per-customer contact budget across all surfaces"),

    Property("withdrawn_consent_respected",
             ("consent_transactional",),
             lambda s: not s.consent_transactional,
             OUTREACH_ACTIONS,
             "DPDP: withdrawal of consent must be honoured"),

    Property("statutory_remedies_require_msme_status",
             ("supplier_is_msme",),
             lambda s: not s.supplier_is_msme,
             {Action.ISSUE_INTEREST_NOTICE, Action.REFER_MSEFC},
             "asserting a remedy a supplier is not entitled to is a false legal claim"),

    Property("statutory_remedies_require_overdue",
             ("days_past_appointed_day",),
             lambda s: s.days_past_appointed_day <= 0,
             {Action.ISSUE_INTEREST_NOTICE, Action.REFER_MSEFC},
             "s.16 interest accrues only from the appointed day"),

    Property("open_promise_suppresses_chasing",
             ("promise_to_pay_open",),
             lambda s: s.promise_to_pay_open,
             OUTREACH_ACTIONS | {Action.REFER_MSEFC},
             "chasing through a promise we accepted destroys the leverage it gave"),

    Property("disputed_amounts_untouched",
             ("is_disputed",),
             lambda s: s.is_disputed,
             MONEY_ACTIONS | OUTREACH_ACTIONS,
             "chasing a disputed amount can prejudice the dispute"),
]

# Liveness: the kernel must never leave the agent with nothing legal to do.
# A safety layer that can deadlock is its own kind of outage.
LIVENESS = (Action.STOP, Action.ESCALATE_MANUAL)


@dataclass
class Result:
    name: str
    states_checked: int
    counterexamples: list[tuple[ObservableState, set[Action]]]
    seconds: float

    @property
    def holds(self) -> bool:
        return not self.counterexamples


def check(prop: Property) -> Result:
    """Exhaustive over the fields the property reads; sound for the rest."""
    t0 = time.perf_counter()
    bad: list[tuple[ObservableState, set[Action]]] = []
    n = 0
    for s in _states_over(prop.reads):
        n += 1
        if not prop.trigger(s):
            continue
        allowed, _ = feasible_actions(s)
        leaked = allowed & prop.forbidden
        if leaked:
            bad.append((s, leaked))
    return Result(prop.name, n, bad, time.perf_counter() - t0)


def check_liveness(samples: int = 400_000, seed: int = 5) -> Result:
    """The agent must never be left with nothing legal to do.

    Unlike the properties above this one reads every field, and the full
    discretised product is ~1.5 x 10^8 states -- not enumerable here. It is also
    the one property that holds by construction rather than by argument:
    `feasible_actions` unions STOP and ESCALATE_MANUAL back in unconditionally,
    after all vetoes have been applied.

    So this is a random sample, and it is labelled as a sample rather than a
    proof. Its job is to catch a future edit that moves that union above the veto
    loop, not to establish something the code does not already say plainly.
    """
    import random
    rng = random.Random(seed)
    t0 = time.perf_counter()
    fields = list(DOMAINS)
    bad: list[tuple[ObservableState, set[Action]]] = []
    for _ in range(samples):
        s = replace(PERMISSIVE, **{f: rng.choice(DOMAINS[f]) for f in fields})
        allowed, _ = feasible_actions(s)
        if not all(a in allowed for a in LIVENESS):
            bad.append((s, allowed))
            if len(bad) > 5:
                break
    return Result("always_has_a_legal_action [sampled]", samples, bad,
                  time.perf_counter() - t0)


def check_monotonicity(samples: int = 200_000, seed: int = 3) -> Result:
    """The assumption the projection argument rests on, tested directly.

    Take a random state, tighten one field toward its stricter value, and confirm
    the allowed set never grows. If this ever failed, every proof above would be
    void -- so it is checked rather than asserted.
    """
    import random
    rng = random.Random(seed)
    t0 = time.perf_counter()
    bad = []
    fields = list(DOMAINS)
    for _ in range(samples):
        base = replace(PERMISSIVE, **{f: rng.choice(DOMAINS[f]) for f in fields})
        allowed_before, _ = feasible_actions(base)
        # Tighten: turn on a constraint that can only add vetoes.
        tighter = replace(
            base,
            is_disputed=True if rng.random() < 0.5 else base.is_disputed,
            consent_transactional=False if rng.random() < 0.5 else base.consent_transactional,
            contacts_7d=max(base.contacts_7d, 3) if rng.random() < 0.5 else base.contacts_7d,
        )
        allowed_after, _ = feasible_actions(tighter)
        if not allowed_after <= allowed_before:
            bad.append((tighter, allowed_after - allowed_before))
    return Result("monotone_under_tightening", samples, bad, time.perf_counter() - t0)
