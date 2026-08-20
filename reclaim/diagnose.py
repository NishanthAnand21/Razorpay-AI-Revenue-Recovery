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


class HardenedDiagnoser:
    """A diagnoser that treats the merchant note as hostile input.

    Three changes from the plain tiered path, in order of how much they matter:

    1. If the note looks like an injection attempt, we do not try to clean it and
       carry on. We abstain -- return UNKNOWN at low confidence, which the policy
       turns into a human escalation. A note arguing with the classifier is not a
       note worth trusting once it has been tidied up.
    2. PII is stripped before any text reaches the model, independently of
       whether the note is suspicious.
    3. Confidence derived from note text is capped. Free text supplied by a third
       party should never be able to push a diagnosis over the threshold at which
       money moves; the structured fields have to carry that weight.
    """

    name = "hardened"
    NOTE_DERIVED_CONFIDENCE_CAP = 0.60

    def __init__(self, inner=None) -> None:
        self.inner = inner or TieredDiagnoser()

    def diagnose(self, p: FailedPayment) -> Diagnosis:
        from .security import prepare_for_model

        safe_note, scan, _pii = prepare_for_model(p.merchant_note)

        if scan.suspicious:
            return Diagnosis(
                RootCause.UNKNOWN, 0.0, "quarantined",
                f"merchant note rejected ({', '.join(scan.findings)}); "
                "escalating rather than classifying on adversarial input",
            )

        # Re-diagnose against the sanitised note rather than the original, so the
        # model never sees the raw bytes even when they looked harmless.
        import dataclasses
        cleaned = dataclasses.replace(p, merchant_note=safe_note)
        d = self.inner.diagnose(cleaned)

        if safe_note and d.source in ("llm", "llm_fallback"):
            d.confidence = min(d.confidence, self.NOTE_DERIVED_CONFIDENCE_CAP)
        return d


class CachedDiagnoser:
    """Memoise diagnoses by signature, and count what that saves.

    The scalability argument in the README is that diagnosis is a low-cardinality
    function: the space of `(error_reason, method, recurring)` signatures is
    bounded by the gateway's vocabulary, not by traffic. This class makes that
    claim executable rather than rhetorical -- `calls` and `hits` are the
    measured numbers, not an estimate.

    Two keying strategies, because the first one measured far worse than the
    projection suggested:

      "strict"    key on (reason, method, recurring) and let anything with a
                  merchant note bypass the cache. Safe, and on data where most
                  payments carry a note it produced a 23% hit rate against a
                  projected 99.98% -- the projection was a bound on note-free
                  traffic and was being quoted as though it were the rate.

      "note"      fold a normalised note into the key. Merchant notes are mostly
                  templated -- picked from a dropdown, or typed from habit -- so
                  identical text recurs constantly and caching it is both safe
                  (the input really is identical) and effective.

    Normalisation is case and whitespace only. Nothing semantic: two notes that
    mean the same thing but read differently are two different inputs, and
    deciding otherwise is exactly the kind of cleverness that makes a cache
    return the wrong answer.
    """

    def __init__(self, inner=None, *, key_strategy: str = "note") -> None:
        self.key_strategy = key_strategy
        self.inner = inner or TieredDiagnoser()
        self.name = f"cached({getattr(self.inner, 'name', 'inner')})"
        self._cache: dict[tuple, Diagnosis] = {}
        self.calls = 0
        self.hits = 0
        self.uncacheable = 0

    def signature(self, p: FailedPayment) -> tuple | None:
        base = (p.error_reason, p.method, p.is_recurring)
        if not p.merchant_note:
            return base
        if self.key_strategy == "strict":
            return None
        return base + (" ".join(p.merchant_note.lower().split()),)

    def diagnose(self, p: FailedPayment) -> Diagnosis:
        key = self.signature(p)
        if key is None:
            self.uncacheable += 1
            self.calls += 1
            return self.inner.diagnose(p)
        if key in self._cache:
            self.hits += 1
            import dataclasses
            return dataclasses.replace(self._cache[key])
        self.calls += 1
        d = self.inner.diagnose(p)
        self._cache[key] = d
        return d

    @property
    def hit_rate(self) -> float:
        total = self.calls + self.hits
        return self.hits / total if total else 0.0
