"""Security controls for an agent that moves money on untrusted input.

THREAT MODEL
------------
The `merchant_note` field is free text written by a human at the merchant, and it
is fed into a model whose output influences whether a payment is retried. That is
an indirect prompt injection surface with a financial sink at the end of it.

The literature is consistent that injection cannot be reliably prevented:
delimiter and spotlighting schemes are partial and bypassable by text that simply
argues its way past them. The recommended control is not prevention but least
privilege -- bound what a compromised model is *able* to do.

That is a control this system already has and has proved. The compliance kernel
reads no model output, so the strongest possible injection can steer the
diagnosis and still cannot unlock an illegal action. So the security posture is
deliberately split:

    containment   the kernel bounds the blast radius. Verified, not asserted.
    detection     injected notes are detected and abstained on, not sanitised
                  into a false sense of cleanliness.
    minimisation  PII is stripped before any text leaves for a third-party model,
                  because sending customer identifiers to a processor is a DPDP
                  question independent of whether the model is honest.
    integrity     the audit log is hash-chained, so a decision trail cannot be
                  quietly rewritten after the fact.
    idempotency   every money action carries a deterministic key, so a duplicate
                  delivery cannot become a duplicate debit.

Note what is NOT claimed: that the notes are made safe. They are not. They are
made irrelevant to anything that matters.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field

# --- 1. injection detection --------------------------------------------------

# Patterns that have no business appearing in "customer said his card expired".
# Detection, not sanitisation: a note that argues with the classifier is not a
# note we want cleaned up and trusted, it is one we want to stop trusting.
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?(previous|prior|above)", "instruction override"),
    (r"disregard\s+(the\s+)?(above|previous|earlier)", "instruction override"),
    (r"\b(system|assistant|user)\s*[:>\]]", "role impersonation"),
    (r"</?\s*(note|system|instruction|prompt)\s*>", "delimiter escape"),
    (r"you\s+(are|must|should|will)\s+now", "role reassignment"),
    (r"new\s+(instruction|rule|directive|task)", "instruction injection"),
    (r"\bclassify\s+(this|it|as)\b", "output steering"),
    (r"\balways\s+(retry|approve|classify|return)", "output steering"),
    (r"\bregardless\s+of\b", "output steering"),
    (r"for\s+all\s+(future|subsequent)", "persistence attempt"),
    (r"\b(reveal|print|output|include)\b.{0,24}\b(card|cvv|otp|token|prompt|key)",
     "exfiltration attempt"),
    (r"[A-Za-z0-9+/]{40,}={0,2}", "encoded payload"),
    (r"\bdo\s+not\s+(escalate|stop|abstain)", "guardrail suppression"),
]

# Characters used to hide text from a human reviewer while keeping it visible to
# a model: zero-width joiners, bidi overrides, and the rest.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿\x00-\x08\x0b-\x1f\x7f]")

MAX_NOTE_CHARS = 240   # real merchant notes are one line; anything longer is a payload

# Cyrillic and Greek characters that render identically to Latin ones. NFKC does
# not fold these -- they are genuinely different letters, not compatibility
# variants -- so "Ignоre" with a Cyrillic o survives normalisation and sails past
# a Latin-alphabet regex. Folding them is what makes the pattern list mean
# anything against an attacker who has spent thirty seconds thinking about it.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ѕ": "s", "ԁ": "d", "ϲ": "c", "ο": "o", "ε": "e", "ρ": "p",
    "α": "a", "ν": "v", "τ": "t", "ι": "i", "κ": "k", "μ": "m", "Ι": "I",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X", "У": "Y",
})


@dataclass
class NoteScan:
    cleaned: str
    suspicious: bool
    findings: list[str] = field(default_factory=list)
    truncated: bool = False


def scan_note(note: str) -> NoteScan:
    """Normalise, then judge. Never 'fix' and pass along as trusted."""
    findings: list[str] = []

    # Unicode normalisation first: homoglyph and compatibility tricks are meant to
    # survive a naive regex pass, and NFKC collapses most of them.
    text = unicodedata.normalize("NFKC", note or "")
    if _INVISIBLE.search(text):
        findings.append("hidden characters")
    text = _INVISIBLE.sub("", text)

    # Fold lookalike scripts before matching, but keep the folded form only for
    # matching -- the cleaned text handed onward stays as the merchant wrote it,
    # minus the things that were actively hiding.
    folded = text.translate(_CONFUSABLES)
    if folded != text:
        findings.append("mixed-script lookalikes")

    truncated = len(text) > MAX_NOTE_CHARS
    if truncated:
        findings.append("over-length note")
        text = text[:MAX_NOTE_CHARS]

    lowered = folded.lower()
    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            findings.append(label)

    return NoteScan(cleaned=text, suspicious=bool(findings),
                    findings=sorted(set(findings)), truncated=truncated)


# --- 2. PII minimisation -----------------------------------------------------

def _luhn(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Order matters and is load-bearing. An earlier version listed the 12-digit
# Aadhaar rule first, and it ate the national part of "+919876543210" -- a phone
# number redacted under the wrong label is still a bug, because the label is what
# a reviewer uses to decide how serious a leak was.
_PII_RULES: list[tuple[str, str]] = [
    (r"(?:\+?91[\-\s]?|0)?[6-9]\d{9}(?!\d)", "[PHONE]"),
    # Aadhaar never starts below 2, and must not be preceded by a digit or plus,
    # so it cannot re-consume the tail of something already handled above.
    (r"(?<![\d+])[2-9]\d{11}(?!\d)", "[AADHAAR]"),
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN]"),                    # income-tax PAN
    # UPI VPAs look like ravi@okhdfc -- no dot, no TLD. A pattern that insists on
    # a TLD misses every VPA in the country, which is most of the identifiers
    # that actually turn up in these notes.
    (r"\b[\w.\-]{2,}@[A-Za-z][\w\-]*(?:\.[A-Za-z]{2,})?\b", "[EMAIL_OR_VPA]"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Strip identifiers before any text is sent to a third-party model.

    This is a data-minimisation control, not an anti-injection one. It applies
    whether or not the model is trustworthy, because the question 'may we send
    this customer's identifiers to a processor' is answered by the DPDP Act and
    not by our confidence in the vendor.
    """
    found: list[str] = []
    out = text

    # Card numbers first, and only when they actually pass a Luhn check, so that
    # order ids and amounts are not mangled into [CARD].
    def _card(m: re.Match) -> str:
        digits = re.sub(r"[^\d]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.append("card")
            return "[CARD]"
        return m.group(0)

    out = re.sub(r"\b(?:\d[ \-]?){13,19}\b", _card, out)

    for pattern, tag in _PII_RULES:
        if re.search(pattern, out):
            found.append(tag.strip("[]").lower())
            out = re.sub(pattern, tag, out)
    return out, sorted(set(found))


def prepare_for_model(note: str) -> tuple[str, NoteScan, list[str]]:
    """The only path by which merchant text may reach a model."""
    scan = scan_note(note)
    redacted, pii = redact_pii(scan.cleaned)
    return redacted, scan, pii


# --- 3. tamper-evident audit log ---------------------------------------------

@dataclass
class LogEntry:
    seq: int
    timestamp: float
    payload: dict
    prev_hash: str
    hash: str


class AuditLog:
    """Append-only, hash-chained. Each entry commits to the entire history.

    A decision trail that can be edited after an incident is not evidence. Chaining
    means altering any past record invalidates every hash after it, so tampering
    is detectable without needing a second copy of the log.
    """

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []

    @staticmethod
    def _digest(seq: int, ts: float, payload: dict, prev: str) -> str:
        blob = json.dumps({"seq": seq, "ts": ts, "payload": payload, "prev": prev},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def append(self, payload: dict, *, timestamp: float | None = None) -> LogEntry:
        seq = len(self.entries)
        prev = self.entries[-1].hash if self.entries else self.GENESIS
        ts = time.time() if timestamp is None else timestamp
        h = self._digest(seq, ts, payload, prev)
        entry = LogEntry(seq, ts, payload, prev, h)
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, int | None]:
        """Return (intact, first_bad_index)."""
        prev = self.GENESIS
        for e in self.entries:
            if e.prev_hash != prev:
                return False, e.seq
            if self._digest(e.seq, e.timestamp, e.payload, e.prev_hash) != e.hash:
                return False, e.seq
            prev = e.hash
        return True, None

    @property
    def head(self) -> str:
        """The single value that commits to the whole log. Publish it anywhere."""
        return self.entries[-1].hash if self.entries else self.GENESIS


# --- 4. idempotency ----------------------------------------------------------

def idempotency_key(payment_id: str, attempt: int, action: str,
                    *, cycle: int = 0) -> str:
    """A deterministic key for one logical money action.

    Recovery pipelines retry themselves: a queue redelivers, a worker restarts
    mid-flight, an operator replays a batch. Without a stable key derived from
    the *logical* action rather than the physical call, each of those becomes a
    second debit. The key is deliberately not time-based for that reason.
    """
    raw = f"{payment_id}|{cycle}|{attempt}|{action}"
    return "idm_" + hashlib.sha256(raw.encode()).hexdigest()[:32]
