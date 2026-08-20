#!/usr/bin/env python3
"""A local stand-in for the Razorpay API, over real HTTP.

    python3 tools/mock_razorpay.py &
    RECLAIM_API_BASE=http://127.0.0.1:8787/v1 ./serve.py --gateway razorpay

Why this exists alongside a hosted mock: the hosted one depends on somebody's
account still being configured, and it cannot run in CI or on a plane. This
serves the same shapes from the standard library, so the full HTTP path -- auth
header, retries, timeouts, JSON parsing, idempotency -- is exercised by
`run_all.sh` on any machine with no setup at all.

It is a mock. It answers from the payment id, not from anything that happened:

    ...settled... or ...captured...   -> captured   (forced, for tests)
    ...auth...                        -> authorized
    otherwise                         -> captured for ~18% of ids, else failed

That 18% is the interesting part. A gateway that reports "failed" for everything
never exercises the case the agent exists to catch: a payment that actually went
through while the notification was lost. Retrying one of those debits a customer
twice, so the mock has to be able to produce them.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8787

DECLINES = [
    ("BAD_REQUEST_ERROR", "insufficient_funds"),
    ("GATEWAY_ERROR", "gateway_timeout"),
    ("BAD_REQUEST_ERROR", "do_not_honour"),
    ("BAD_REQUEST_ERROR", "card_expired"),
]


def _bucket(pid: str, mod: int = 100) -> int:
    """Stable 0..mod-1 bucket for an id.

    A digest rather than sum(ord(...)): real payment ids share a long prefix and
    differ in a few trailing digits, so a character sum clusters them all into
    the same few buckets. The first version did that and produced zero
    already-settled payments across 120 events, which silently removed the case
    this mock exists to exercise.
    """
    return int.from_bytes(hashlib.sha256(pid.encode()).digest()[:4], "big") % mod


def payment_body(pid: str, amount_paise: int = 250000) -> dict:
    low = pid.lower()
    if "settled" in low or "captured" in low:
        status, code, reason = "captured", None, None
    elif "auth" in low:
        status, code, reason = "authorized", None, None
    elif _bucket(pid) < 18:
        status, code, reason = "captured", None, None
    else:
        status = "failed"
        # Deterministic in the id, so a replay of the same request gives the
        # same answer. A mock that returns something different each call makes
        # every failure look like a flake.
        code, reason = DECLINES[_bucket(pid, 4)]
    body = {
        "id": pid, "entity": "payment", "amount": amount_paise, "currency": "INR",
        "status": status, "method": "upi", "captured": status == "captured",
        "description": "reclaim mock",
    }
    if code:
        body["error_code"] = code
        body["error_reason"] = reason
        body["error_description"] = reason.replace("_", " ")
    return body


class Handler(BaseHTTPRequestHandler):
    server_version = "reclaim-mock"

    def log_message(self, fmt, *args):
        if self.server.verbose:                       # type: ignore[attr-defined]
            sys.stderr.write("  mock  " + fmt % args + "\n")

    def _send(self, code: int, body: dict) -> None:
        blob = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self) -> None:
        m = re.match(r"^/v1/payments/(pay_[A-Za-z0-9_]+)$", self.path)
        if m:
            return self._send(200, payment_body(m.group(1)))
        if self.path.startswith("/v1/payments"):
            return self._send(200, {
                "entity": "collection", "count": 3,
                "items": [payment_body(f"pay_mock{i}") for i in range(3)]})
        self._send(404, {"error": {"code": "NOT_FOUND", "description": self.path}})

    def do_POST(self) -> None:
        m = re.match(r"^/v1/payments/(pay_[A-Za-z0-9_]+)/capture$", self.path)
        if not m:
            return self._send(404, {"error": {"code": "NOT_FOUND"}})
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        key = self.headers.get("X-Razorpay-Idempotency-Key", "")
        seen = self.server.captured                   # type: ignore[attr-defined]
        if key and key in seen:
            # The behaviour the idempotency key exists for. A mock that happily
            # captures twice would let a double-charge bug pass unnoticed.
            return self._send(200, dict(seen[key], reclaim_replayed=True))
        out = dict(payment_body(m.group(1), body.get("amount", 250000)),
                   status="captured", captured=True)
        if key:
            seen[key] = out
        self._send(200, out)


def serve(port: int = PORT, verbose: bool = False) -> HTTPServer:
    httpd = HTTPServer(("127.0.0.1", port), Handler)
    httpd.verbose = verbose                           # type: ignore[attr-defined]
    httpd.captured = {}                               # type: ignore[attr-defined]
    return httpd


def serve_in_thread(port: int = PORT) -> tuple[HTTPServer, threading.Thread]:
    httpd = serve(port)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    httpd = serve(port, verbose=True)
    print(f"  Razorpay mock on http://127.0.0.1:{port}/v1   (ctrl-c to stop)")
    print(f"  RECLAIM_API_BASE=http://127.0.0.1:{port}/v1 ./serve.py --gateway razorpay")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
