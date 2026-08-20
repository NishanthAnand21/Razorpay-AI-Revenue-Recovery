# Reclaim

**A revenue-recovery agent that cannot break the rules, and that measures what it
actually caused rather than what it took credit for.**

Submission for the Razorpay AI Buildathon, *AI Revenue Recovery* track.

```bash
./run_all.sh        # 17 suites, ~100 seconds, no dependencies, no network
```

Python 3.11+. Nothing to install, nothing to configure, no network. Every number
below is produced by that command.

Most of the runtime is one suite: leave-one-reason-out trains 25 classifiers from
scratch to test how the diagnoser behaves on failure reasons it has never seen.
That is the check that keeps the accuracy claim honest, so it stays.

---

## The one-paragraph version

Four surfaces lose revenue — failed payments, failed mandates, abandoned carts,
overdue invoices — and they are usually built as four systems. They are one
problem: something of known value stalled, and there is a bounded window to act.
So Reclaim shares one compliance kernel, one contact ledger, one capacity queue
and one audit chain across all four. The kernel reads **no model output**, which
makes its guarantees provable rather than tuned, and its refusals turn out to be
a free randomised experiment for measuring what recovery actually causes.

## What is new, and what is not

Two of the three big pieces here are **not** novel, and saying so is what makes
the third claim worth anything.

- **ML-timed retries already ship** — Stripe Smart Retries, Butter, Churnkey,
  plus granted patents on machine-learned dunning. Competent engineering, not
  invention.
- **The compliance kernel is a safety shield**, an established pattern from safe
  RL and LLM agent safety. Applying it to Indian payment regulation and
  machine-checking it is careful work, not a new idea.

The new part is what those two do together:

> A shield refuses to act for reasons that have nothing to do with the customer.
> That is **exogenous non-treatment** — precisely the randomised holdout causal
> measurement requires, except regulation pays for it instead of you.

A UPI mandate failing at 12:59 cannot be retried; one failing at 13:01 can,
because an NPCI peak window closes at 13:00. Nobody picks their failure minute.
In the shielding literature a shield is pure cost. Here it also **produces the
identification strategy**. §5 grades that against a known answer.

## The pipeline

```
raw events ─▶ detect ─▶ observe ─▶ LEGAL SET ─▶ model picks ─▶ act ─▶ audit
                        (facts)    (kernel)     (inside it)          (hash-chained)
                                       │
                                       └─▶ refusals ─▶ causal measurement
```

---

## 1. The compliance kernel

Nine hard constraints govern Indian payment recovery and **not one needs a
model**: Visa Category 1 codes, the 15/30-day Visa cap and 10 for Mastercard,
NPCI's 4-attempts-per-cycle and non-peak windows, RBI's 2026 e-mandate 24-hour
pre-debit notice, RBI's 08:00–19:00 working-day collections window, MSMED s.15
and s.16, DPDP consent. All are functions of a code, a counter, a clock, a
calendar. `ObservableState` carries **no field that is a model output** — that
restriction is the entire guarantee.

**17 safety properties, verified exhaustively.** Full enumeration is ~10⁸ states,
so the checker uses a soundness argument: the kernel only ever *subtracts*
actions, so a property holding in the most-permissive completion holds in every
completion. Monotonicity underpins that and is checked, not assumed.

**Mutation testing: 12/12 caught.** Rules are broken one at a time — an
off-by-one in a cap, a window trimmed, a code dropped, the MSME gate removed —
and the verifier must notice. On the first run one mutation *escaped*, because
the permissive baseline used `local_hour=12.0`, inside the morning peak window,
so an unrelated veto masked the bug. An assertion now enforces that the baseline
fires no vetoes — and it caught the same class of error again when the
receivables rules landed.

**Adversarial:** a diagnoser claiming `transient_issuer` at confidence 1.0 for
every payment, revoked mandates and fraud declines alike, produces **zero**
network violations and **zero** double-charge exposure. Nothing was tuned to get
those zeros.

## 2. Recovery — payment failures

240 held-out payments, ₹21.8 lakh at risk:

| strategy | recovered | net ₹ | breach | risky | wasted | double charges |
|---|---|---|---|---|---|---|
| retry everything ×3 | 39.2% | 7,93,742 | 279 | 30 | 87 | **11** |
| retry, 24h backoff | 62.3% | 12,62,960 | 144 | 30 | 87 | **8** |
| **Reclaim** | 64.6% | 13,10,158 | **0** | 9 | 7 | **0** |
| **Reclaim (tuned)** | **65.7%** | **13,32,307** | **0** | 9 | 18 | **0** |
| **Reclaim (strict)** | 64.1% | 13,00,454 | **0** | **0** | **0** | **0** |

Three columns kept apart on purpose. A **breach** is an action the observable
facts forbid — the kernel decides it, so the agent is at zero by construction.
**Risky** is a recharge on a payment whose true cause was fraud but whose raw
reason was `do_not_honour`, which Visa classifies as *retryable* — not decidable
at the time, so not a breach. **Double charges** are re-charges of payments that
had already settled.

That last column is what a payments engineer will care about. A gateway timeout
does not mean the payment failed; it means nobody told us. v1 classified
timeouts as transient and retried immediately. The kernel now refuses any money
action while settlement is unconfirmed and issues `RECONCILE` — a 10-paise
status lookup.

## 3. Mandate retry sequencing

NPCI allows 4 presentations per cycle, RBI requires 24h notice before each, and
Autopay executes only in non-peak windows. So a retry can never be immediate and
the only question is **which three moments to spend**. Exact DP over (day,
retries left), on 20,000 cycles:

| policy | recovered | merchant net ₹ | customer penalties ₹ | attempts | rule breaks |
|---|---|---|---|---|---|
| immediate burst ×3 | 59.8% | 3,04,22,332 | 82,13,100 | 2.15 | **20,000** |
| industry 24h/72h/d7 | 56.8% | 2,88,71,366 | 71,65,200 | 1.93 | 0 |
| **optimal, prices harm** | 54.1% | **3,25,92,791** | **29,47,000** | 1.09 | 0 |

**+12.9% merchant revenue and −58.9% customer penalties**, on half the attempts.
The industry default retries at a fixed offset from the *failure*, which is the
one variable the outcome does not depend on. What decides it is the salary cycle.

**The cost nobody prices:** Indian banks charge the *customer* ₹250–500 per
failed presentation. Optimising merchant net alone earns ₹11.2 lakh more while
inflicting ₹30.7 lakh more in bank charges — **2.7 rupees destroyed per rupee
gained.** Both objectives are solved and a harm-weight curve published instead of
a buried constant.

**A number that looks wrong:** the joint policy recovers 54.1% against the
default's 56.8% — *lower* — while earning ₹37 lakh more. It declines cycles whose
expected recovery cannot cover the bank charge it would inflict. Fewer payments
recovered, more money recovered.

## 4. Receivables, and abandoned carts

**Receivables** (6,000 invoices, ₹338 crore outstanding). Uses MSMED s.15's
45-day appointed day and s.16 compound interest at 3× the RBI bank rate — a lever
most chasers never pull, and one that must be *true* to pull, so eligibility is
kernel-checked rather than inferred.

| policy | DSO | contacts | breaches | false statutory claims |
|---|---|---|---|---|
| no chasing | 66.2 | 0 | 0 | 0 |
| blanket ladder | **56.6** | 15,104 | **9,946** | **2,377** |
| blanket ladder (gated) | 61.2 | 5,779 | 0 | 0 |
| reclaim chaser | 61.1 | 7,612 | 0 | 0 |
| **chaser @40%, prioritised** | **60.9** | **4,533** | 0 | 0 |

Three findings, one unflattering. The ungated ladder's 4.6-day lead is **bought
entirely with actions it is not allowed to take**. But gated blanket (61.2) ties
our chaser (61.1) — so **the escalation logic adds nothing measurable**, and all
the value on this surface is the kernel. The real gain is prioritisation: ranking
by value × the buyer's own late-payment history matches full-capacity DSO on 40%
fewer contacts.

**Carts.** The detector measured AUC 0.531 — a coin flip at convergence, a feature
ceiling not underfitting. Session metadata does not know whether someone meant to
buy. The blanket sequence would report ₹9,95,893 recovered having *caused*
₹81,975 — **8%**. The same gross-versus-incremental gap the RD found on mandates,
on a different surface by a different method.

The result contradicted my prior: every policy is EV-positive, so "cart messaging
is wasteful" does not survive the arithmetic. What holds is that the decision
rests entirely on the lifetime cost of an opt-out, which almost nobody measures.
Blanket messaging turns value-destroying between 5 and 12 future recoverable
events per customer; the gated policy stays positive across the whole sweep.

## 5. Fitting the parts that were guesswork

Three things in the policy were hand-set constants. Each was fitted or searched
on **train** and reported on **test**, and two of the three results are negative.

**Beliefs.** `BELIEVED_SUCCESS` was nine flat constants the expected-value gate
read directly. `learn.py` now fits a table from kernel-legal exploration, keyed
on the **diagnosed** cause — so diagnosis error is absorbed into the belief, and
the gate automatically gets more conservative on causes the model is worst at —
and conditioned on attempt index and delay bucket. Funds retries come out at
**0.538 at 48h against 0.129 immediate**, which flat constants cannot express.

A first fit landed systematically below the hand-written values and looked like a
correction. It wasn't: exploration sampled random attempts and delays, so it was
measuring "this action at *some* time" while the policy asks "at *my* chosen
time". Conditioning fixed it.

**The ablation is the result worth keeping:**

| proposer | beliefs | recovered | net ₹ |
|---|---|---|---|
| rules ladder | hand-written | 64.6% | 13,10,158 |
| rules ladder | fitted | 64.6% | 13,10,158 |
| EV | hand-written | 54.2% | 10,99,529 |
| **EV** | **fitted** | **65.7%** | **13,32,307** |

Fitting alone changes **nothing**, because under the rules ladder nothing
consults the beliefs — the EV gate only fires on payments too small to chase, so
it was inert. And an EV proposer on flat beliefs is **worse** than the ladder,
because flat constants cannot tell a 48-hour retry from an immediate one.
Reporting only the combined number would have made two useless halves look like
one improvement.

**Thresholds.** 81 configurations searched on train. **32% of the training gain
survived to test** — and only the attempt budget moves the objective at all.
Two of the four constants are inert across their entire range, which is more
useful than the tuned values: they are not worth arguing about, and any future
report crediting them is reading noise.

**Calibration.** The capacity ranking multiplies probability by value, and that
product is not monotonic in `p` — so miscalibration reorders the queue even
though it cannot change a ranking by `p` alone. I had leaned on this without
checking it. ECE is 0.066, and calibrating is worth **+32% net recovery at a
budget of 20 and nothing at all by 100**. That shape is the point: when you can
only touch twenty things, the ordering at the top *is* the decision.

Calibrating also **lowered** the orchestrator's reported expected recovery by
about a fifth. Nothing got worse — the earlier figure was the model flattering
itself, which is gross recovery's lesson in a different costume.

## 6. Causal measurement — the new part

Sharp regression discontinuity at the four NPCI peak boundaries, 400k mandate
debits, graded against a planted truth the estimator never sees:

| | value |
|---|---|
| gross recovery ("we recover 31% of failed payments") | 0.3092 |
| naive treated − blocked | +0.0832 |
| **RD estimate, pooled** | **+0.0831 ± 0.0053** |
| **true effect (hidden)** | **+0.0789** |

Within 5% of truth; at the 13:00 boundary it lands exactly.

> **Gross recovery overstates the real effect by 3.9×. About 74 of every 100
> "recovered" payments were coming back on their own.**

Validity is checked, not claimed: placebo cutoffs where no rule changes (largest
14% of the real effect, none significant), a McCrary density check (ratios
0.98–1.02), bandwidth sweeps stable across 12×. The first run at 60k events
produced a placebo worth 72% of the real effect — underpowered, not broken — so
`--power` reports the volume needed: **~100k events**, below which the honest
output is an interval, not a point estimate.

## 7. Security

The merchant note is untrusted text feeding a model that influences whether money
moves. Injection cannot be reliably prevented, so the posture is containment.

### The review found three real defects

**High — merchant text reached the model unredacted.** Redaction and injection
screening lived in `HardenedDiagnoser`, a *wrapper*. But `ClaudeDiagnoser` is the
only path that actually leaves the process, and it could be constructed directly,
bypassing all of it. A control you can skip by choosing a different constructor
is not a control. Moved into `_build_user_prompt` — the single place a prompt is
assembled — so no caller can route around it.

**High — the orchestrator fabricated a compliance fact.** `_observe` hardcoded
`hours_since_pre_debit_notice=25.0`, asserting an RBI pre-debit notice existed
without checking. That defeats the premise the entire kernel rests on: every
field it reads is *observed*, not believed. A fabricated fact defeats a proof
about facts, and this one would have authorised mandate retries RBI does not
permit. Now read from the event, defaulting to "no notice".

**Medium — audit truncation was invisible.** Chaining catches edits and
reordering, but deleting the tail leaves a chain that verifies perfectly. Added
`checkpoint()` / `verify(expected)`, and documented that detection needs an
anchor from outside the log rather than pretending chaining suffices.

Also: customer identifiers in the audit log are now HMAC-pseudonymised (stable,
so a reviewer can still follow one customer through it), and the system reports
whether real keying is in force rather than implying it.

**Checked and clean:** no ReDoS in the PII or injection patterns (linear scaling
on adversarial input), exception *messages* never logged — only types, since
client errors can carry URLs and tokens — no secrets stored or serialised, no
`eval`/`exec`, no unsafe deserialisation.

| diagnoser | kernel | steered | compliance breach | business risk |
|---|---|---|---|---|
| plain | off | 43/406 | 11 | 51 |
| plain | on | 43/406 | **0** | 40 |
| hardened | on | **5/406** | **0** | 4 |

**Injection works** — hardening cuts steering from 43 to 5, not to 0. What the
kernel changes is what that buys: breaches go to zero, because no sentence in a
note edits a decline reason or a clock.

The security eval **falsified the containment claim as first written**: the kernel
only enforced never-retry on card rails, so a fraud decline on UPI had no veto.
Also: unicode confusable folding (NFKC does not fold Cyrillic `о`), PII redaction
sized for Indian identifiers — UPI VPAs have no TLD, so the first pattern missed
every one — a hash-chained audit log, and deterministic idempotency keys.

## 8. Efficiency and scalability

Diagnosis is a low-cardinality function: **50% of payments never reach a model at
all** (rules table), and the rest collapse to 337 distinct signatures across 800
payments.

| cache key | distinct | calls | hit rate |
|---|---|---|---|
| strict (note bypasses cache) | 116 | 617 | 22.9% |
| **note folded into key** | 337 | 337 | **57.9%** |

Since the learned tier landed, this matters much less: the classifier absorbs
53% of all payments and model calls fall to **zero on this test set**. The caching
analysis is kept because it is still the right answer for the residual, and
because the correction below is the more instructive part.

The README previously projected ≥99.98%. Measuring it gave **22.9%** — 63% of
payments carry a note and every one bypassed the cache. The projection was a
bound on note-free traffic quoted as if it were the rate. Folding a normalised
note into the key (case and whitespace only, nothing semantic) fixes it honestly,
because merchant notes are templated. Projected forward, cost is `O(distinct
signatures)`, not `O(events)` — 337 calls at 50M payments.

**End to end**, one queue across all surfaces:

| capacity | expected recovery ₹ | marginal ₹ | surfaces served |
|---|---|---|---|
| 25 | 15,05,518 | 15,05,518 | receivable:25 |
| 250 | 51,45,291 | 15,59,348 | receivable:235, checkout:11, sub:4 |
| 500 | 53,09,320 | 1,64,029 | receivable:244, checkout:187, sub:69 |
| 1000 | 53,14,338 | 5,018 | saturates at 636 |

At small budgets the queue is **entirely receivables** — correct, not a bug: one
overdue invoice can outweigh every cart in the batch. Four teams each optimising
inside their own silo would never see that. The curve bends at 500 and saturates
at 636, which answers "how big should the collections team be" with a number.

## 9. What is wrong with this

- **The world models are synthetic.** No student has live retry outcomes. The
  simulator's success table is deliberately *not* the table the policy believes,
  and all strategies face identical random draws.
- **Every accuracy figure is the offline stand-in's, not Claude's.**
  `ClaudeDiagnoser` is wired and `eval/run_llm_eval.py` runs it, but it needs an
  API key. The stand-in is a keyword matcher tuned against the same vocabulary
  the generator uses — a lower bound on plumbing, not evidence about the model.
- **Detection ground truth is a counterfactual we invented.**
  `would_self_recover` is knowable in a simulator and never in production. §5 is
  the start of the real answer; production still needs a deliberate holdout.
- **The RD estimand is local** — the effect near a peak boundary, not the average
  effect. Treating it as global would be a misreading.
- **The sequencer's funds curve and the cart uplift curve are assumed, not
  fitted.** The shapes are right and supported by the reporting; the constants
  are chosen.
- **The receivables escalation logic adds nothing measurable** over a gated fixed
  ladder, as §4 says.
- **The `risky` column cannot reach zero** without strict mode, which costs
  revenue. A risk-appetite decision, not a bug.

## Layout

```
reclaim/compliance.py   the kernel: facts in, legal action set out
reclaim/verify.py       17 machine-checked safety properties
reclaim/causal.py       sharp RD, robust SEs, placebo and density checks
reclaim/security.py     injection detection, PII redaction, hash-chained log
reclaim/sequencer.py    exact DP for mandate scheduling, both objectives
reclaim/receivables.py  escalation ladder, promises, MSMED statutory interest
reclaim/carts.py        cart EV against the lifetime cost of an opt-out
reclaim/detect.py       features, logistic regression, Platt calibration
reclaim/learn.py        fits the policy's beliefs from kernel-legal exploration
reclaim/diagnose.py     rules table, model tier, hardened tier, cache
reclaim/policy.py       legality first, then intent, then business policy
reclaim/orchestrator.py one queue, one ledger, one audit chain
tests/test_all.py       16 regression tests, no test framework required
docs/RESEARCH.md        the regulation, with sources, and ten edge cases
```

Every constraint cited here is sourced in [docs/RESEARCH.md](docs/RESEARCH.md).
