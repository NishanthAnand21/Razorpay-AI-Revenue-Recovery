# Pointing the agent at a mock API

The agent's gateway client is a real HTTP client. Where it points is one
environment variable, so the same code path — auth header, retries, timeouts,
JSON parsing, idempotency keys — is exercised whether it is talking to Razorpay,
to a hosted mock, or to a local one.

```bash
RECLAIM_API_BASE=<base-url> ./bin/reclaim run --gateway razorpay
```

**A mock is not test mode, and the agent will not pretend otherwise.** Razorpay
test mode is Razorpay's own sandbox with their semantics; a mock returns whatever
you configured. The client reports itself as `razorpay-mock` and `live = False`
when the base URL is not `api.razorpay.com`, so a demo can never quietly look
more real than it is.

---

## Option 1 — the local mock (no setup, works offline)

```bash
./bin/reclaim run --gateway mock --limit 40
```

Starts `tools/mock_razorpay.py` in-process and talks to it over real HTTP on
`127.0.0.1:8787`. This is what `run_all.sh` uses, so the network path is covered
by CI on any machine.

To run it standalone:

```bash
python3 tools/mock_razorpay.py
RECLAIM_API_BASE=http://127.0.0.1:8787/v1 ./bin/reclaim run --gateway razorpay
```

It answers deterministically from the payment id: ids containing `settled` or
`captured` come back captured, `auth` comes back authorized, and roughly 18% of
everything else comes back **captured despite being reported as failed** — the
case the agent exists to catch, where retrying would debit a customer twice.

## Option 2 — Beeceptor

Sandbox: `https://razorpay-mock-api.proxy.beeceptor.com`

```bash
RECLAIM_API_BASE=https://razorpay-mock-api.proxy.beeceptor.com/v1 \
  ./bin/reclaim run --gateway razorpay --limit 20
```

No credentials needed — a mock authenticates nothing, and requiring real keys to
reach one would only encourage putting real keys where they are not wanted. The
client substitutes placeholders and labels itself a mock.

Beeceptor answers `200` with an empty body for any path without a rule, and the
client names that case rather than failing with an opaque JSON decode error:

```
GatewayError: GET /payments/pay_TEST123: empty response from
https://razorpay-mock-api.proxy.beeceptor.com/v1 (no mock rule configured?)
```

### Rules to create

In the dashboard, add a **Mocking Rule** for each. Match on path, respond `200`
with `Content-Type: application/json`.

**1. Fetch a failed payment** — `GET /v1/payments/pay_fail*`

```json
{
  "id": "pay_fail001", "entity": "payment", "amount": 250000,
  "currency": "INR", "status": "failed", "method": "upi", "captured": false,
  "error_code": "BAD_REQUEST_ERROR", "error_reason": "insufficient_funds",
  "error_description": "Your account does not have enough balance"
}
```

**2. Fetch a payment that actually settled** — `GET /v1/payments/pay_settled*`

This is the important one. It is the case where the gateway reported a failure,
the notification was lost, and the money had already moved — so a retry would be
a second debit.

```json
{
  "id": "pay_settled001", "entity": "payment", "amount": 250000,
  "currency": "INR", "status": "captured", "method": "upi", "captured": true
}
```

**3. Catch-all fetch** — `GET /v1/payments/*`

```json
{
  "id": "pay_generic", "entity": "payment", "amount": 250000,
  "currency": "INR", "status": "failed", "method": "card", "captured": false,
  "error_code": "GATEWAY_ERROR", "error_reason": "gateway_timeout",
  "error_description": "Payment processing failed due to a timeout"
}
```

**4. Capture** — `POST /v1/payments/*/capture`

```json
{
  "id": "pay_generic", "entity": "payment", "amount": 250000,
  "currency": "INR", "status": "captured", "captured": true
}
```

### What Beeceptor cannot show

It cannot honour an idempotency key — every `POST` gets the same canned reply, so
a replayed capture looks like a fresh one. The local mock *does* track the key and
collapses replays, and `tests/test_all.py` asserts it. If you are demonstrating
the double-charge defence, use the local mock.

## Option 3 — real Razorpay test mode

```bash
export RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
export RAZORPAY_KEY_SECRET=xxxxxxxx
./bin/reclaim run --gateway razorpay --limit 20
```

Reads are real. `fetch_payment` — which is what `RECONCILE` calls, and the whole
reason the double-charge defence works — hits the live sandbox.

Writes are gated twice over: `--execute` must be passed, and a key not beginning
`rzp_test_` is refused unless `RECLAIM_ALLOW_LIVE_KEYS=1` is set. A recovery
agent that executes by default is one config mistake away from charging people.

`./bin/reclaim live` (`eval/run_realtime_test.py`) exercises the full loop against the sandbox:
discovers receivables, reconciles their live status, and — with `--execute` —
sends real reminders through `POST /payment_links/:id/notify_by/:medium`, with
the compliance kernel deciding whether each call happens at all.

A genuine card *retry* still needs a customer-present flow or a saved token,
which a key pair alone cannot produce. Payment links are the surface that is
real end to end, and that distinction is stated rather than blurred.
