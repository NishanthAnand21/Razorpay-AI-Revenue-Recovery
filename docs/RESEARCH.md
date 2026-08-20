# Research notes: what the rules actually say, and what that changes

Everything below was looked up rather than recalled, because the details are the
kind that sound plausible when wrong. Sources at the bottom.

---

## 1. The constraint map

Nine hard constraints govern payment recovery in India. **None of them requires a
model.** Every one is a function of a response code, a counter, a clock or a
calendar.

| constraint | rule | consequence of breaking it |
|---|---|---|
| Visa Category 1 codes | `04 07 12 14 15 41 43 46 57 R0 R1 R3` may **never** be reattempted | violation on the *first* retry, plus a per-transaction fee |
| Visa reattempt cap | ≤ 15 reattempts per 30 days | ~$0.10/txn domestic, +$0.05 cross-border |
| Mastercard cap | ≤ 10 retries per 30 days | fines $0.10–$2.00/txn |
| Mastercard advice | `MAC03`, `MAC21` mean stop immediately | burns allowance, triggers fines |
| UPI Autopay attempts | 1 execution + 3 retries per cycle, then the cycle is failed | mandate marked failed |
| UPI Autopay windows | non-peak only: before 10:00, 13:00–17:00, after 21:30 | attempt rejected |
| RBI e-mandate (2026) | pre-debit notification ≥ 24h before **every** debit, with opt-out | unauthorised debit |
| RBI recovery agents | contact only 08:00–19:00, **working days** — covers SMS, WhatsApp, email, not just calls | harassment finding |
| DPDP Act 2023 | contact requires a lawful basis; withdrawal must be honoured | statutory penalty |

Two of these directly contradicted what v1 of this system did:

- The policy retried `insufficient_funds` **at 10:00**, reasoning that overnight
  credits had settled. 10:00 is the opening minute of an NPCI peak window. On a
  UPI Autopay mandate that retry is not merely suboptimal, it is rejected.
- Quiet hours were set to 21:00–09:00. For anything that counts as collections
  the real window closes at **19:00**, on working days only.

## 2. Ten edge cases v1 did not handle

1. **Unconfirmed settlement.** A gateway timeout does not mean the payment
   failed. It means nobody told us. v1 classified timeouts as `transient_issuer`
   and retried *immediately* — measured at **45 double-charge exposures out of
   240 payments**. This is the most serious defect the research turned up, and no
   amount of model accuracy would have found it, because the model was right.
2. **Category 1 is a first-strike rule.** An attempt budget of 3 is irrelevant
   when attempt 1 is already the violation.
3. **Network caps are per card per 30 days, not per payment.** A customer with
   six subscriptions can breach the Visa cap without any single recovery
   workflow exceeding its own budget. The counter has to be global.
4. **Mandate cycle caps are per cycle, not per failure.**
5. **Peak-hour windows make "retry now" undefined** on Autopay for 6.5 hours of
   every day. The right response is to *schedule*, not to abandon — which is
   revenue v1 left on the table in the other direction.
6. **Pre-debit notice is per debit.** A retry is a debit. So a mandate retry is
   gated on a notification sent ≥24h earlier, which makes fast mandate retries
   structurally impossible and the sequencing problem much more interesting.
7. **Contact fatigue is per customer, not per item.** One customer with a failed
   subscription, an abandoned cart and an overdue invoice receives three messages
   from three subsystems that cannot see each other. Budgets must be global.
8. **Self-recovery races.** The customer pays between detection and intervention.
   Sending the nudge anyway is worse than never having detected it.
9. **Dispute interlock.** Chasing a formally disputed amount can prejudice the
   dispute and reads as harassment. The data had a `disputed` flag; the policy
   ignored it.
10. **Consent withdrawal** is not the same as mandate revocation, and revoking
    one does not revoke the other.

## 3. The invention: constraints in front of the model, not behind it

v1 had the standard shape — **model proposes, guardrail vetoes** — and the
evaluation showed exactly what that shape costs. The guardrail checked the
*diagnosed* cause, so when the diagnosis was wrong the guardrail was reading a
wrong label and waved the action through. Three violations, caused by the safety
layer trusting the thing it existed to protect against. The instinct is to raise
the confidence threshold until the number reaches zero. That buys a number, not a
property: it is still true that a sufficiently confident wrong answer gets through.

Since every hard constraint turned out to be decidable from raw facts, the
control flow can simply be inverted:

```
v1    diagnose ──▶ propose ──▶ veto?      safety depends on the diagnosis
v2    observe  ──▶ legal set ──▶ model chooses inside it
```

`reclaim/compliance.py` computes the feasible action set from an
`ObservableState` on which **no field is a model output** — that restriction is
the whole guarantee. The model then picks within the set. It never sees an
illegal option, so it cannot choose one.

The difference is testable, and `eval/run_adversarial.py` tests it by attacking
it. A diagnoser that claims `transient_issuer` at confidence 1.0 for *every*
payment — revoked mandates, fraud declines, timeouts alike:

| diagnoser | kernel | money actions | network violations | double-charge risk |
|---|---|---|---|---|
| honest | off | 169 | 0 | **45** |
| honest | **on** | 113 | 0 | **0** |
| confused | off | 122 | 3 | 15 |
| confused | **on** | 88 | **0** | **0** |
| adversarial | off | 240 | **7** | 45 |
| adversarial | **on** | 126 | **0** | **0** |

Nothing was tuned to produce those zeros. Recovered revenue does collapse under a
broken model — a bad model makes bad *business* decisions, and it should. It just
cannot make illegal ones.

This also sharpens a distinction v1 blurred. `do_not_honour` (code `05`) is
**retryable** under Visa's rules even though it often means fraud. So retrying it
is a *business* risk, not a compliance breach. v1 counted both as "violations"
and so measured a number that no regulator would recognise. Separating the two
makes the compliance figure provable and leaves the business risk where it
belongs — as an expected-value question the model is allowed to be wrong about.

## 4. Efficiency and scalability

**Diagnosis is a low-cardinality function, so its cost does not scale with
volume.** Measured on our own data:

| events | distinct `(reason, method, recurring)` signatures | model calls needed | cache hit rate |
|---|---|---|---|
| 800 | 159 | 159 | 80.1% |
| 10,000 | ≤ 159 | ≤ 159 | ≥ 98.4% |
| 1,000,000 | ≤ 159 | ≤ 159 | ≥ 99.98% |
| 50,000,000 | ≤ 159 | ≤ 159 | ≥ 99.9997% |

Cost is `O(distinct signatures)`, not `O(events)` — the signature space is
bounded by the gateway's own vocabulary and grows only when a network adds a code.
Combined with the rules table already absorbing **50%** of rows with no model at
all, a merchant doing 50M failed payments a year needs a model budget in the low
hundreds of calls. Per-event LLM inference here would be a straightforward
engineering mistake, and most "AI agent" submissions will make it.

The honest caveat: free-text merchant notes are high-cardinality and cannot be
cached on that key. The design routes noted cases to the model and caches only
the note-free majority, so real hit rates land below the table's ceiling.

Two further scaling properties fall out of the constraint work:

- **Legal windows force batching, which is also what makes it cheap.** Autopay
  can only fire in three windows a day, so the natural architecture is a
  scheduler draining a queue into those windows, not a per-event reactor. The
  compliance constraint and the efficient architecture happen to be the same
  design.
- **Capacity allocation is a knapsack, not a filter.** Ranking by
  `p(worth chasing) × value` is the greedy ratio heuristic, which is near-optimal
  for the unconstrained case but not once per-customer contact caps couple the
  items together. That coupling is the honest next problem.

## 5. What I would build next, in order

1. **Reconcile-before-retry.** Any `UNKNOWN` settlement must hit a status check
   before a money action. This is the double-charge fix and it is worth more than
   any modelling improvement in this repo.
2. **Global counters keyed on card and customer**, not on workflow — the only way
   the 30-day network caps and cross-surface contact fatigue can actually hold.
3. **Notification-aware mandate sequencing.** Given the 24h pre-debit rule and 4
   attempts per cycle, the sequencer's real job is choosing *which three moments
   in the cycle to spend*, which is a small scheduling problem with a clean
   optimal solution.
4. **A holdout for the counterfactual.** `would_self_recover` is knowable in a
   simulator and never in production. Withhold intervention from a random slice
   and measure the difference; everything else is inference on a label that does
   not exist.

---

## Sources

- [Visa: updates to rules for declined transaction resubmission](https://usa.visa.com/dam/VCOM/global/support-legal/documents/updates-to-rules-for-declined-transaction-resubmission-and-use-of-authorization-response-codes.pdf)
- [Visa excessive reattempts rule and fees](https://www.payway.com/visa-excessive-reattempts-rule-fees)
- [Visa decline code grouping (Category 1 list)](https://help.qualpay.com/help/visas-decline-code-grouping)
- [Visa and Mastercard payment retry rules for subscription businesses](https://www.slickerhq.com/resources/blog/visa-mastercard-payment-retry-rules)
- [Merchant advice and response codes](https://gethelp.segpay.com/docs/Content/GatewayDocs/Merchant%20Advice%20&%20Response%20Codes.htm)
- [RBI tightens auto-debit rules: 24-hour prior alert mandatory](https://www.aninews.in/news/business/rbi-tightens-auto-debit-rules-24-hour-prior-alert-now-mandatory-for-recurring-payments20260421202816/)
- [RBI e-mandate framework 2026: AFA rules for recurring payments](https://www.indiaaipulse.com/en/news/rbi-updates-e-mandate-rules-for-recurring-payments)
- [RBI guidelines on e-mandates for recurring transactions](https://www.novojuris.com/thought-leadership/rbi-guidelines-on-e-mandates-for-recurring-transaction.html)
- [NPCI UPI rules from August 1: autopay timing and limits](https://paytm.com/blog/payments/upi/upi-rules-update-august-1-npci-new-guidelines/)
- [NPCI caps UPI balance checks, restricts autopay to non-peak hours](https://www.newsbytesapp.com/news/business/npci-caps-upi-balance-checks-restricts-autopay-to-non-peak-hours/story)
- [UPI autopay revocations hit 20mn per month on low balances](https://www.business-standard.com/amp/finance/news/upi-autopay-revocations-hit-20-mn-monthly-over-low-customer-balances-125090700500_1.html)
- [RBI guidelines: recovery agent calling hours](https://www.credsettle.com/rbi-guidelines-calling-after-7pm)
- [RBI guidelines on recovery agents and borrower rights](https://freed.care/blog/rbi-guidelines-recovery-agents)
- [India outbound call regulations: DPDPA and TRAI compliance](https://talk-q.com/outbound-call-regulations-in-india)
