"""Talking to a real payment gateway, or a faithful stand-in.

Everything in this repo up to here decided what to do. Nothing did it. This is
the boundary where a decision becomes an API call, and it is the boundary where
being careless costs real money -- so the design is deliberately conservative:

  READS ARE REAL, WRITES ARE GATED
      Fetching a payment's status against Razorpay test mode is safe and is
      implemented for real. Charging is not the same kind of operation, and a
      recovery agent that can charge by default is a bug waiting for a bad
      config. Writes require an explicit opt-in and refuse to run against live
      keys at all.

  EVERY WRITE CARRIES AN IDEMPOTENCY KEY
      Derived from the logical action, not the physical call, so a redelivered
      queue message or a restarted worker replays the same key rather than
      creating a second debit.

  NO CREDENTIALS EVER REACH A LOG
      Auth is built per request and never stored on the instance in a form that
      repr() or a traceback would surface.

Written against the standard library. `urllib.request` verifies TLS by default
and the whole repo stays installable by cloning it.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

API_BASE = "https://api.razorpay.com/v1"
# Point the client at a mock server instead. Useful for exercising the real HTTP
# path -- auth headers, retries, timeouts, JSON parsing, latency -- without live
# credentials, which is most of what can actually break in an integration.
#
# A mock is NOT test mode. Razorpay test mode is Razorpay's own sandbox with
# their semantics; a mock returns whatever you told it to. Both are useful and
# they are not the same claim, so the client reports which one it is talking to.
API_BASE_ENV = "RECLAIM_API_BASE"
# Set to surface gateway error bodies. Off by default so error payloads do not
# reach logs as a side effect of a failure; on when a human is debugging.
DEBUG_ENV = "RECLAIM_GATEWAY_DEBUG"
TIMEOUT_SECONDS = 10.0
MAX_RETRIES = 4
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GatewayError(RuntimeError):
    """A gateway call failed in a way the caller has to decide about."""


@dataclass
class PaymentLink:
    """An issued, unpaid payment link: revenue at risk with an id you can act on.

    The recovery surfaces elsewhere in this repo are reconstructed from synthetic
    events. This one is real and API-addressable end to end -- a link can be
    created, chased with an actual reminder, and reconciled -- so it is the
    surface where the agent can be demonstrated against a live gateway rather
    than described.
    """

    link_id: str
    status: str                 # created | partially_paid | paid | cancelled | expired
    amount_inr: float
    amount_paid_inr: float = 0.0
    short_url: str = ""
    reminders_sent: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def settled(self) -> bool:
        return self.status == "paid"

    @property
    def at_risk(self) -> bool:
        return self.status in ("created", "partially_paid")


@dataclass
class PaymentStatus:
    """What the gateway says about a payment, right now."""

    payment_id: str
    status: str                 # created | authorized | captured | refunded | failed
    amount_inr: float
    error_code: str | None = None
    error_reason: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def settled(self) -> bool:
        """Money actually moved. The question RECONCILE exists to answer."""
        return self.status in ("captured", "refunded")

    @property
    def confirmed_failed(self) -> bool:
        return self.status == "failed"


class SimulatedGateway:
    """A local stand-in, used when no credentials are configured.

    Not a mock in the testing sense -- it is the same interface backed by the
    repo's own simulator, so the service runs end to end with nothing configured
    and a reader can watch it work before deciding whether to point it at an API.
    It reports `live = False` and every caller surfaces that, because a demo that
    looks live and is not is worse than no demo.
    """

    live = False
    name = "simulated"

    def __init__(self, payments: dict[str, Any] | None = None) -> None:
        self._payments = payments or {}
        self.calls = 0

    def fetch_payment(self, payment_id: str) -> PaymentStatus:
        self.calls += 1
        p = self._payments.get(payment_id)
        if p is None:
            return PaymentStatus(payment_id, "failed", 0.0, "NOT_FOUND", "unknown payment")
        settled = getattr(p, "settlement_actually_succeeded", False)
        return PaymentStatus(
            payment_id=payment_id,
            status="captured" if settled else "failed",
            amount_inr=p.amount_inr,
            error_code=None if settled else p.error_code,
            error_reason=None if settled else p.error_reason,
        )

    def capture_payment(self, payment_id: str, amount_inr: float, *,
                        idempotency_key: str) -> PaymentStatus:
        self.calls += 1
        return PaymentStatus(payment_id, "captured", amount_inr)


class RazorpayTestGateway:
    """Razorpay REST client, test mode only unless explicitly overridden.

    Reads the key pair from RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET. Test keys
    begin `rzp_test_`; anything else is refused unless RECLAIM_ALLOW_LIVE_KEYS is
    set, because the difference between a test key and a live key is the
    difference between a demo and charging somebody.
    """

    def __init__(self, key_id: str | None = None, key_secret: str | None = None,
                 *, allow_writes: bool = False, base: str | None = None) -> None:
        self.base = (base or os.environ.get(API_BASE_ENV) or API_BASE).rstrip("/")
        self.is_mock = "api.razorpay.com" not in self.base

        key_id = key_id or os.environ.get("RAZORPAY_KEY_ID", "")
        key_secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET", "")

        if self.is_mock:
            # A mock does not authenticate anything, so requiring real keys to
            # talk to one would only push people toward putting real keys where
            # they are not needed. Placeholders, and the object says it is a mock.
            key_id = key_id or "rzp_test_mock"
            key_secret = key_secret or "mock"
        else:
            if not key_id or not key_secret:
                raise GatewayError(
                    "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET are not set")
            if not key_id.startswith("rzp_test_") and \
                    not os.environ.get("RECLAIM_ALLOW_LIVE_KEYS"):
                raise GatewayError(
                    f"refusing to run against a non-test key ({key_id[:12]}...). "
                    "Set RECLAIM_ALLOW_LIVE_KEYS=1 only if you mean it.")
        # Held as an opaque header value rather than as the key pair, so a
        # traceback or a repr of this object cannot print the secret.
        self._auth = "Basic " + base64.b64encode(
            f"{key_id}:{key_secret}".encode()).decode()
        self.key_id = key_id                       # public half; safe to show
        self.allow_writes = allow_writes
        self.calls = 0

    @property
    def live(self) -> bool:
        """True only against Razorpay itself. A mock is a real round trip to
        fabricated data, which is a different claim and is reported as one."""
        return not self.is_mock

    @property
    def name(self) -> str:
        return "razorpay-mock" if self.is_mock else "razorpay"

    @staticmethod
    def _backoff(exc, attempt: int) -> float:
        """How long to wait before retrying.

        A rate limit is not a transient error and must not be treated as one.
        The first version backed off 0.5s then 1.0s for everything, which is
        sensible for a 503 and useless against a 429 -- the sandbox rate-limits
        payment-link creation and three attempts inside 1.5 seconds simply
        produced three 429s. Rate limits get seconds, and the server's own
        Retry-After wins over any guess we make.
        """
        if getattr(exc, "code", None) == 429:
            retry_after = (exc.headers or {}).get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), 30.0)
                except ValueError:
                    pass
            return min(2.0 * (2 ** attempt), 15.0)
        return 0.5 * (2 ** attempt)

    def _request(self, method: str, path: str, body: dict | None = None,
                 idempotency_key: str | None = None) -> dict:
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        last: Exception | None = None

        for attempt in range(MAX_RETRIES):
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Authorization", self._auth)
            req.add_header("Content-Type", "application/json")
            if idempotency_key:
                req.add_header("X-Razorpay-Idempotency-Key", idempotency_key)
            try:
                self.calls += 1
                with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                    raw = resp.read().decode(errors="replace").strip()
                # A mock with no rule for this path answers 200 with an empty
                # body or a friendly greeting. Letting json.loads raise here
                # would surface as a generic decode error three frames away, so
                # it is named at the boundary instead.
                if not raw:
                    raise GatewayError(
                        f"{method} {path}: empty response from {self.base} "
                        "(no mock rule configured for this path?)")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    raise GatewayError(
                        f"{method} {path}: response was not JSON "
                        f"({raw[:60]!r}...)") from None
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                    time.sleep(self._backoff(exc, attempt))
                    last = exc
                    continue
                # The body is withheld by default: gateway error payloads echo
                # request context and should not land in a log as a side effect
                # of something failing. But withholding it makes a 400 during
                # development almost undebuggable, so it is opt-in rather than
                # unavailable -- a control nobody can turn on when they
                # legitimately need it just gets worked around.
                detail = ""
                if os.environ.get(DEBUG_ENV):
                    detail = f" -- {exc.read(400).decode(errors='replace')}"
                raise GatewayError(
                    f"{method} {path} failed: HTTP {exc.code}{detail}") from None
            except urllib.error.URLError as exc:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    last = exc
                    continue
                raise GatewayError(f"{method} {path} unreachable: "
                                   f"{type(exc).__name__}") from None
        raise GatewayError(f"{method} {path} failed after {MAX_RETRIES} attempts") \
            from last

    def fetch_payment(self, payment_id: str) -> PaymentStatus:
        """The read behind RECONCILE. Safe, idempotent, and the whole point."""
        d = self._request("GET", f"/payments/{payment_id}")
        return PaymentStatus(
            payment_id=d.get("id", payment_id),
            status=d.get("status", "unknown"),
            amount_inr=float(d.get("amount", 0)) / 100.0,
            error_code=d.get("error_code"),
            error_reason=d.get("error_reason") or d.get("error_description"),
            raw=d,
        )

    def list_payments(self, count: int = 10) -> list[PaymentStatus]:
        d = self._request("GET", f"/payments?count={count}")
        return [PaymentStatus(
            payment_id=i.get("id", ""), status=i.get("status", "unknown"),
            amount_inr=float(i.get("amount", 0)) / 100.0,
            error_code=i.get("error_code"),
            error_reason=i.get("error_reason") or i.get("error_description"),
            raw=i) for i in d.get("items", [])]

    # --- payment links: the one surface that is real end to end --------------

    def create_payment_link(self, amount_inr: float, *, description: str,
                            name: str, email: str, contact: str,
                            notes: dict | None = None) -> PaymentLink:
        if not self.allow_writes:
            raise GatewayError("writes are disabled; pass allow_writes=True")
        d = self._request("POST", "/payment_links", {
            "amount": int(round(amount_inr * 100)), "currency": "INR",
            "description": description,
            "customer": {"name": name, "email": email, "contact": contact},
            # Suppressed at creation on purpose: whether to contact this customer
            # is the compliance kernel's decision, not a flag set at issue time.
            "notify": {"sms": False, "email": False},
            "reminder_enable": True,
            "notes": notes or {},
        })
        return self._to_link(d)

    def fetch_payment_link(self, link_id: str) -> PaymentLink:
        return self._to_link(self._request("GET", f"/payment_links/{link_id}"))

    def notify_payment_link(self, link_id: str, medium: str = "email") -> bool:
        """Send a real reminder. The recovery action, actually executed.

        Gated behind allow_writes like every other write: this one leaves the
        building and reaches a person.
        """
        if not self.allow_writes:
            raise GatewayError("writes are disabled; pass allow_writes=True")
        if medium not in ("email", "sms"):
            raise GatewayError(f"unsupported medium {medium!r}")
        d = self._request("POST", f"/payment_links/{link_id}/notify_by/{medium}")
        return bool(d.get("success"))

    @staticmethod
    def _to_link(d: dict) -> PaymentLink:
        return PaymentLink(
            link_id=d.get("id", ""), status=d.get("status", "unknown"),
            amount_inr=float(d.get("amount", 0)) / 100.0,
            amount_paid_inr=float(d.get("amount_paid", 0)) / 100.0,
            short_url=d.get("short_url", ""),
            reminders_sent=int((d.get("reminders") or {}).get("count", 0) or 0),
            raw=d)

    def capture_payment(self, payment_id: str, amount_inr: float, *,
                        idempotency_key: str) -> PaymentStatus:
        if not self.allow_writes:
            raise GatewayError(
                "writes are disabled. Pass allow_writes=True deliberately; the "
                "default is read-only so a misconfiguration cannot move money.")
        d = self._request("POST", f"/payments/{payment_id}/capture",
                          {"amount": int(round(amount_inr * 100)), "currency": "INR"},
                          idempotency_key=idempotency_key)
        return PaymentStatus(payment_id, d.get("status", "unknown"),
                             float(d.get("amount", 0)) / 100.0, raw=d)


def open_gateway(*, allow_writes: bool = False, payments=None):
    """The real gateway when credentials exist, the stand-in otherwise.

    Never raises for a missing key pair: the service must run out of the box, and
    the caller reports which one it got rather than pretending they are the same.
    """
    try:
        return RazorpayTestGateway(allow_writes=allow_writes)
    except GatewayError:
        return SimulatedGateway(payments)
