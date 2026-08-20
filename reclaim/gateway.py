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
TIMEOUT_SECONDS = 10.0
MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GatewayError(RuntimeError):
    """A gateway call failed in a way the caller has to decide about."""


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

    live = True
    name = "razorpay"

    def __init__(self, key_id: str | None = None, key_secret: str | None = None,
                 *, allow_writes: bool = False) -> None:
        key_id = key_id or os.environ.get("RAZORPAY_KEY_ID", "")
        key_secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET", "")
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

    def _request(self, method: str, path: str, body: dict | None = None,
                 idempotency_key: str | None = None) -> dict:
        url = f"{API_BASE}{path}"
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
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                    # Exponential backoff. A 429 answered by an immediate retry
                    # is just a second 429.
                    time.sleep(0.5 * (2 ** attempt))
                    last = exc
                    continue
                # Deliberately does not include the response body: gateway error
                # payloads echo request context and should not land in a log by
                # default.
                raise GatewayError(f"{method} {path} failed: HTTP {exc.code}") from None
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
