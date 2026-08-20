"""Turning a gateway error into an actionable root cause.

Two diagnosers, deliberately layered:

  RulesDiagnoser  — a lookup table over failure reasons we have already seen.
                    Free, instant, perfectly auditable, and blind to anything new.
  LLMDiagnoser    — handles what the table has never seen, plus the cases where
                    the free-text merchant note contradicts the error code.

The layering matters: we do not pay a model to classify `card_expired`. The LLM
is spent only where the cheap path genuinely cannot answer, and its accuracy on
exactly those rows is measured in eval/run_eval.py.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .models import FailedPayment, RootCause

# Reasons we have seen enough times to hard-code. Anything outside this table is
# an unknown to the rules layer -- by design.
RULES_TABLE: dict[str, RootCause] = {
    "payment_failed_at_bank": RootCause.TRANSIENT_ISSUER,
    "gateway_timeout": RootCause.TRANSIENT_ISSUER,
    "issuer_unavailable": RootCause.TRANSIENT_ISSUER,
    "insufficient_funds": RootCause.INSUFFICIENT_FUNDS,
    "account_balance_low": RootCause.INSUFFICIENT_FUNDS,
    "incorrect_otp": RootCause.AUTH_FRICTION,
    "otp_attempts_exceeded": RootCause.AUTH_FRICTION,
    "upi_collect_expired": RootCause.AUTH_FRICTION,
    "payment_cancelled_by_user": RootCause.AUTH_FRICTION,
    "card_expired": RootCause.INSTRUMENT_INVALID,
    "invalid_vpa": RootCause.INSTRUMENT_INVALID,
    "mandate_revoked": RootCause.INSTRUMENT_INVALID,
    "per_transaction_limit_exceeded": RootCause.LIMIT_EXCEEDED,
    "daily_limit_exceeded": RootCause.LIMIT_EXCEEDED,
    "payment_declined_by_risk": RootCause.RISK_DECLINED,
    "suspected_fraud": RootCause.RISK_DECLINED,
}


@dataclass
class Diagnosis:
    cause: RootCause
    confidence: float
    source: str        # "rules" | "llm" | "llm_fallback"
    rationale: str


class RulesDiagnoser:
    """Exact-match lookup. Returns UNKNOWN rather than guessing."""

    name = "rules"

    def diagnose(self, p: FailedPayment) -> Diagnosis:
        cause = RULES_TABLE.get(p.error_reason)
        if cause is None:
            return Diagnosis(
                RootCause.UNKNOWN, 0.0, "rules",
                f"reason '{p.error_reason}' is not in the rules table",
            )
        return Diagnosis(cause, 1.0, "rules", f"exact match on '{p.error_reason}'")


# --- LLM layer ---------------------------------------------------------------

SYSTEM_PROMPT = """You classify failed payment attempts for an Indian payments \
company into exactly one root cause. Answer with one label and nothing else.

Labels:
- transient_issuer: bank or gateway was temporarily down. Retrying the same way works.
- insufficient_funds: the customer does not have the money right now.
- auth_friction: the customer failed or abandoned authentication (OTP, collect request).
- instrument_invalid: the card/VPA/mandate is permanently dead. Retrying can never work.
- limit_exceeded: a per-transaction or daily cap was hit.
- risk_declined: blocked by fraud or compliance. Must never be retried automatically.

The merchant's free-text note, when present, is written by a human who spoke to \
the customer. Trust it over the raw error code when they disagree."""


def _build_user_prompt(p: FailedPayment) -> str:
    note = p.merchant_note or "(none)"
    return (
        f"error_code: {p.error_code}\n"
        f"error_reason: {p.error_reason}\n"
        f"method: {p.method}\n"
        f"recurring: {p.is_recurring}\n"
        f"merchant_note: {note}\n"
    )


# Keyword signatures used by the offline stand-in. These generalise over unseen
# reason strings the way an LLM would -- imperfectly, which is the point.
_SIGNATURES: list[tuple[RootCause, tuple[str, ...]]] = [
    (RootCause.RISK_DECLINED, ("risk", "fraud", "velocity", "blocked", "do not auto retry")),
    (RootCause.INSTRUMENT_INVALID, ("expired", "lost", "stolen", "deprovision", "revoked",
                                    "invalid", "closed that bank account", "new card", "dead")),
    (RootCause.LIMIT_EXCEEDED, ("limit", "cap", "exceed", "breach")),
    (RootCause.INSUFFICIENT_FUNDS, ("insufficient", "balance", "low_balance", "low balance",
                                    "funds", "salary", "payday")),
    (RootCause.AUTH_FRICTION, ("otp", "3ds", "challenge", "abandon", "collect", "cancel",
                               "did not approve", "never got")),
    (RootCause.TRANSIENT_ISSUER, ("timeout", "downtime", "unavailable", "switch_error",
                                  "npci", "acquirer", "gateway", "outage", "try again later")),
]


class MockLLMDiagnoser:
    """Offline stand-in for the real model, so the eval runs with no API key.

    It reads only what the real diagnoser is given -- error code, reason, method
    and the merchant note -- and never touches ground truth. It is wrong on a
    real fraction of rows, which keeps the reported accuracy honest.
    """

    name = "llm(mock)"

    def diagnose(self, p: FailedPayment) -> Diagnosis:
        haystack = f"{p.error_reason} {p.merchant_note}".lower()
        haystack = re.sub(r"[^a-z0-9 _]", " ", haystack)
        for cause, keys in _SIGNATURES:
            hit = next((k for k in keys if k in haystack), None)
            if hit:
                return Diagnosis(cause, 0.78, "llm", f"matched signal '{hit}'")
        # No keyword fired -- this is an ambiguous decline like `do_not_honour`,
        # where the string genuinely does not identify the cause. A real model
        # does not shrug here, it commits on context and is sometimes wrong, so
        # the stand-in does the same. The confidence it reports is what keeps
        # those errors cheap: see MIN_CONFIDENCE_TO_ACT in policy.py.
        if p.customer_prior_failures >= 2:
            # A repeatedly failing customer is the one case where guessing is
            # worse than asking. Abstain and let the policy escalate.
            return Diagnosis(
                RootCause.UNKNOWN, 0.30, "llm_fallback",
                "ambiguous decline on a repeatedly failing customer; not guessing with money",
            )
        if p.is_recurring:
            guess = RootCause.INSUFFICIENT_FUNDS   # mandate debits mostly fail on balance
            why = "ambiguous decline on a mandate debit; balance is the usual cause"
        elif p.method == "card":
            guess = RootCause.INSTRUMENT_INVALID
            why = "ambiguous card decline; issuer most often means the card is bad"
        else:
            guess = RootCause.TRANSIENT_ISSUER
            why = "ambiguous decline on a one-off collect; treating as an issuer blip"
        return Diagnosis(guess, 0.55, "llm", why)


class ClaudeDiagnoser:
    """The real thing. Used when ANTHROPIC_API_KEY is set and --llm is passed.

    Falls back to the mock on any error so a flaky network can never corrupt an
    eval run -- the fallback is recorded in the decision's diagnosis_source.
    """

    name = "llm(claude)"

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self._fallback = MockLLMDiagnoser()
        try:
            from anthropic import Anthropic  # imported lazily: optional dependency
            self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        except Exception:
            self._client = None

    def diagnose(self, p: FailedPayment) -> Diagnosis:
        if self._client is None:
            d = self._fallback.diagnose(p)
            d.source = "llm_fallback"
            return d
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=16,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_prompt(p)}],
            )
            label = resp.content[0].text.strip().lower()
            return Diagnosis(RootCause(label), 0.85, "llm", f"model returned '{label}'")
        except Exception as exc:  # network, rate limit, or an unparseable label
            d = self._fallback.diagnose(p)
            d.source = "llm_fallback"
            d.rationale = f"{d.rationale} (after model error: {type(exc).__name__})"
            return d


class TieredDiagnoser:
    """Rules first, model only where rules came up empty."""

    name = "tiered"

    def __init__(self, llm=None) -> None:
        self.rules = RulesDiagnoser()
        self.llm = llm or MockLLMDiagnoser()

    def diagnose(self, p: FailedPayment) -> Diagnosis:
        d = self.rules.diagnose(p)
        if d.cause is not RootCause.UNKNOWN:
            return d
        return self.llm.diagnose(p)
