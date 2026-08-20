# Reclaim

**A revenue-recovery agent that cannot break the rules, and that measures what it
actually caused rather than what it took credit for.**

Submission for the Razorpay AI Buildathon, *AI Revenue Recovery* track.

```bash
./run_all.sh        # every suite, ~10 seconds, no dependencies, no network
```

Python 3.11+. Nothing to install. Every number below is reproduced by that command.

---

## What is actually new here

Two of the three big pieces in this repo are **not** novel, and saying so is the
only way the third claim is worth anything.

- **ML-timed retries already ship.** Stripe Smart Retries, Butter and Churnkey do
  this, and there are granted patents on machine-learned dunning. Our learned
  detector is competent engineering, not invention.
- **The compliance kernel is a safety shield**, a well-established pattern from
  safe RL and, more recently, LLM agent safety. Applying it to Indian payment
  regulation and machine-checking it is careful work, not a new idea.

The new part is what those two do *together*:

> A shield refuses to act for reasons that have nothing to do with the customer.
> That is **exogenous non-treatment** — which is exactly the randomised holdout
> that causal measurement requires, except regulation pays for it instead of you.

A UPI mandate failing at 12:59 cannot be retried; one failing at 13:01 can,
because an NPCI peak window closes at 13:00. Nobody picks their failure minute.
The regulation has run a randomised experiment on our behalf and the shield is
where the assignment is recorded.

In the shielding literature a shield is pure cost — it only ever removes options.
Here it also **produces the identification strategy**. That inversion is the
contribution, and §4 grades it against a known answer.

---

## The pipeline

```
raw events ──▶ detect ──▶ observe ──▶ LEGAL SET ──▶ model picks ──▶ act ──▶ audit
                            (facts)   (kernel)      (inside it)            (hash-chained)
                                          │
                                          └──▶ refusals ──▶ causal measurement
```

The one structural decision everything else follows from: **the kernel reads no
model output.** Legality is computed from a response code, a counter and a clock,
before the model's opinion is consulted. So the model chooses *within* the legal
set and never sees an illegal option.

---

## 1. Detection — finding what is slipping away

Revenue rarely announces itself as it leaves. A failed payment does; a stalled
checkout emits **nothing at all**. So the pipeline starts from a raw 30-day event
stream — 1,580 events, most of them fine — and finds the 687 stalls across four
surfaces.

The target label is not "did this stall" but **`is_worth_chasing`: at risk AND not
going to fix itself.** Around half of all stalls self-recover. Chasing them is not
a win; it is a message nobody wanted.

| surface | stalls | base rate | **AUC** |
|---|---|---|---|
| subscription | 31 | 0.84 | **0.908** |
| receivable | 76 | 0.21 | **0.764** |
| checkout abandonment | 95 | 0.57 | **0.528** |
| all | 202 | 0.48 | 0.781 |

Read that column before believing anything below it. **Abandoned carts are a coin
flip**, and training to convergence does not move it — a feature ceiling, not
underfitting. Session metadata does not know whether someone meant to buy. The
industry default is to blanket-message every abandoned cart; this says those
messages are close to untargeted.

### Cost is not the constraint. Capacity is.

Thresholding on expected value produced *chase everything* — a 70-paise nudge
against a cart worth thousands is EV-positive at almost any precision. That dead
end is kept in the repo because it is informative. The real constraint is that a
collections team makes a bounded number of calls, so the question is **given room
for K interventions, which K**.

| budget | ranker | prec@K | net ₹ |
|---|---|---|---|
| 20 | biggest amounts first | 0.05 | 2,08,176 |
| 20 | **model probability × value** | 0.30 | **7,78,414** |
| 100 | biggest amounts first | 0.32 | **14,83,585** |
| 100 | model probability × value | 0.33 | 14,69,552 |

**3.7× at a budget of 20**, and slightly *behind* at 100 — which is what should
happen, since 100 of 202 candidates is drifting back toward chasing everything.
The tighter the capacity, the more the model is worth. Reporting only the budget
where it wins would be the easiest lie in this repo to tell.

Ranking on probability *alone* has the best precision at every budget (1.00 at
K=20) and the **worst** rupees: it selects certainties worth ₹300.

## 2. The compliance kernel — verified, then attacked

Nine hard constraints govern Indian payment recovery, and **not one needs a
model**: Visa Category 1 codes, the 15-per-30-days Visa cap and 10 for Mastercard,
NPCI's 4-attempts-per-mandate-cycle and non-peak execution windows, RBI's 2026
e-mandate 24-hour pre-debit notice, RBI's 08:00–19:00 working-day collections
window, DPDP consent. All are functions of a code, a counter, a clock, a calendar.

So `compliance.py` computes the feasible action set from an `ObservableState` on
which **no field is a model output**, and the policy projects its proposal into
that set — preferring to defer over to abandon, because a rule that says *not now*
is not a rule that says *never*.

**14 safety properties, verified exhaustively.** Full enumeration is ~10⁸ states,
so the checker uses a soundness argument instead: the kernel only ever *subtracts*
actions, so a property holding in the most-permissive completion holds in every
completion. Monotonicity underpins that and is checked, not assumed.

**Mutation testing.** Eight rules are deliberately broken one at a time — an
off-by-one in a cap, a window trimmed, a code dropped — and the verifier must
notice. **10/10 caught**, and the suite goes green again on revert. On the first
run one mutation *escaped*, because the permissive baseline used `local_hour=12.0`
— inside the morning peak window — so an unrelated veto masked the bug. That is
what mutation testing is for.

**Adversarial.** A diagnoser that claims `transient_issuer` at confidence 1.0 for
every payment, revoked mandates and fraud declines alike:

| diagnoser | kernel | network violations | double-charge exposure |
|---|---|---|---|
| honest | off | 0 | **45** |
| adversarial | off | **7** | 45 |
| **all diagnosers** | **on** | **0** | **0** |

Nothing was tuned to get those zeros. A broken model makes bad *business*
decisions, and should. It cannot make illegal ones.

## 3. Recovery — held out, against three baselines

240 failed payments, ₹21.8 lakh at risk, policy tuned only on train:

| strategy | recovered | net ₹ | breach | risky | wasted | double charges |
|---|---|---|---|---|---|---|
| do nothing | 0.0% | 0 | 0 | 0 | 0 | 0 |
| retry everything ×3 | 39.2% | 7,93,742 | 279 | 30 | 87 | **11** |
| retry everything, 24h backoff | 62.3% | 12,62,960 | 144 | 30 | 87 | **8** |
| **Reclaim** | **64.6%** | **13,10,158** | **0** | 9 | 7 | **0** |
| **Reclaim (strict)** | 64.1% | 13,00,454 | **0** | **0** | **0** | **0** |

Three columns are counted apart on purpose, because collapsing them is how
recovery products flatter themselves:

- **breach** — an action the observable facts forbid. The kernel decides it, so
  the agent is at zero *by construction*. The baselines do not look.
- **risky** — a recharge on a payment whose true cause was fraud but whose raw
  reason was `do_not_honour`, which Visa classifies as **retryable**. Not
  decidable at the time, so not a breach. Strict mode takes it to zero for 0.7%
  of revenue.
- **double charges** — re-charging a payment that had already settled. Each one
  is a customer debited twice for the same purchase.

That last column is the one a payments engineer will care about. A gateway
timeout does not mean the payment failed; it means nobody told us. The first
version of this system classified timeouts as transient and retried immediately.
The kernel now refuses any money action while settlement is unconfirmed and
issues `RECONCILE` instead — a 10-paise status lookup.

Diagnosis is tiered: a rules table answers **113/240 rows at 1.000 accuracy for
free**, and the model handles the 127 rows it has never seen at **0.858**, where a
rules-only system scores 0.000 by construction.

## 4. Mandate retry sequencing — three shots, chosen well

NPCI allows one execution plus three retries per cycle; RBI requires a 24-hour
pre-debit notice before each; Autopay executes only in non-peak windows. So a
mandate retry can never be immediate, and the only question is **which three
moments to spend**.

In India that answer is dominated by the salary cycle. The published smart-retry
default — 24h, 72h, then day 7 — is a fixed offset from the *failure*, which is
the one variable the outcome does not depend on. What determines whether a
mandate clears is when the customer next gets paid.

Exact dynamic programming over (day, retries left), on 20,000 failed cycles:

| policy | recovered | merchant net ₹ | customer penalties ₹ | attempts | rule breaks |
|---|---|---|---|---|---|
| immediate burst ×3 | 59.8% | 3,04,22,332 | 82,13,100 | 2.15 | **20,000** |
| industry 24h/72h/d7 | 56.8% | 2,88,71,366 | 71,65,200 | 1.93 | 0 |
| next salary, once | 14.9% | 79,09,334 | 1,84,800 | 0.21 | 0 |
| optimal, merchant only | 65.5% | 3,37,11,000 | 60,15,450 | 1.64 | 0 |
| **optimal, prices harm** | 54.1% | **3,25,92,791** | **29,47,000** | 1.09 | 0 |

**+12.9% merchant revenue and −58.9% customer penalties**, against the industry
default, using half the attempts. Those are not in tension — waiting for the
salary credit is better for both sides — which is the useful finding.

### The cost nobody prices

A failed auto-debit does not only cost the merchant a ₹2 gateway fee. Indian
banks charge the **customer** a bounce penalty of roughly ₹250–500 per failed
presentation. That never appears in the merchant's P&L, which is exactly why
merchants over-retry.

> Optimising merchant net alone earns **₹11.2 lakh more** while inflicting
> **₹30.7 lakh more** in bank charges on customers — destroying **2.7 rupees of
> value for every rupee gained.**

So the sequencer solves both objectives and reports both, with a harm-weight
curve (`--harm-sweep`) instead of a hidden constant. Going from weight 0 to 0.5
cuts customer penalties by a third for 2.4% of merchant net. That knob is a
policy decision, and publishing the curve lets whoever owns it decide with the
numbers in view.

### One number that looks wrong

The joint policy recovers **54.1%** of cycles against the default's 56.8% — a
*lower* recovery rate — while earning ₹37 lakh more. It declines small cycles
whose expected recovery cannot cover the bank charge it would inflict, and spends
the freed attempts on cycles that clear. **Fewer payments recovered, more money
recovered.**

Recovery rate is what this industry reports, and it is the wrong metric for the
same reason gross recovery is: both count events instead of value.

## 5. Causal measurement — the part that is new

The industry reports **gross recovery**. In advertising, where holdouts are
standard, measured lift runs far below platform-reported numbers. Dunning has not
had that reckoning. So: use the shield's refusals as the holdout.

Sharp regression discontinuity at the four NPCI peak-window boundaries, on 400k
mandate debits, graded against a planted ground truth the estimator never sees:

| | value |
|---|---|
| gross recovery ("we recover 31% of failed payments") | 0.3092 |
| naive treated − blocked | +0.0832 |
| **RD estimate, pooled over 4 boundaries** | **+0.0831 ± 0.0053** |
| **true effect (hidden from the estimator)** | **+0.0789** |

**Within 5% of truth.** At the 13:00 boundary it lands exactly. And:

> **Gross recovery overstates the real effect by 3.9×. Roughly 74 of every 100
> "recovered" payments were coming back on their own.**

Validity is checked rather than claimed: placebo cutoffs at hours where no rule
changes (largest 14% of the real effect, none significant), a McCrary-style
density check for manipulation (ratios 0.98–1.02), and bandwidth sweeps stable
across a 12× range.

The first run at 60k events produced a placebo worth **72%** of the real effect.
That was not a broken design, it was an underpowered one — so `--power` reports
the volume needed: **~100k events**, below which the honest output is a
confidence interval and not a point estimate.

## 6. Security — assume injection succeeds

The merchant note is untrusted text feeding a model that influences whether money
moves. The literature is consistent that injection cannot be reliably prevented,
so the posture is **containment, not prevention**.

| diagnoser | kernel | steered | compliance breach | business risk |
|---|---|---|---|---|
| plain | off | 43/406 | 11 | 51 |
| plain | on | 43/406 | **0** | 40 |
| hardened | on | **5/406** | **0** | 4 |

**Injection works** — hardening cuts steering from 43 to 5, not to 0, and keyword
stuffing still gets through. What the kernel changes is what that buys: breaches
go to zero, because no sentence in a merchant note edits a decline reason or a
clock.

The security evaluation also **falsified the containment claim as first written**:
the kernel only enforced never-retry on card rails, so a fraud decline on UPI had
no veto at all. Fixed with rail-agnostic rules keyed on raw gateway reasons, plus
matching verification properties and mutations.

Also: unicode confusable folding (NFKC does not fold Cyrillic `о`), PII redaction
sized for Indian identifiers — UPI VPAs have no TLD, so the first pattern missed
every one of them — a hash-chained tamper-evident audit log, and deterministic
idempotency keys so a redelivered queue message cannot become a second debit.

## 7. Efficiency and scalability

**Diagnosis is a low-cardinality function, so its cost does not scale with
volume.** 800 events contain 159 distinct `(reason, method, recurring)`
signatures, and that space is bounded by the gateway's vocabulary:

| events | model calls needed | cache hit rate |
|---|---|---|
| 1,000,000 | ≤ 159 | ≥ 99.98% |
| 50,000,000 | ≤ 159 | ≥ 99.9997% |

Cost is `O(distinct signatures)`, not `O(events)`. With the rules table absorbing
50% of rows unaided, 50M failed payments a year needs a model budget in the low
hundreds of calls. Per-event LLM inference is the obvious mistake here.

*Caveat:* free-text notes are high-cardinality and bypass the cache, so real hit
rates land below that ceiling.

Two more properties fall out of the constraint work. Autopay can only fire in
three windows a day, so the natural architecture is a **scheduler draining a queue
into legal windows** — the compliance constraint and the efficient architecture
turn out to be the same design. And capacity allocation is a knapsack, not a
filter; `p × value` is the greedy ratio heuristic, near-optimal until per-customer
contact caps couple the items together. That coupling is the honest next problem.

## 8. What is wrong with this

- **The world model is synthetic.** No student has live retry outcomes. Two things
  keep it from being circular: the simulator's success table is *not* the table
  the policy believes, and every strategy faces identical random draws.
- **Detection ground truth is a counterfactual we invented.** `would_self_recover`
  is knowable in a simulator and never in production. §4 is the beginning of the
  real answer, but the production version still needs a deliberate holdout to
  validate against.
- **The RD estimand is local.** It measures the effect on payments near a peak
  boundary, which is not the average effect across all payments. Treating it as
  global would be a misreading.
- **Recovery executes on payment failures and mandate cycles.** Detection covers
  four surfaces; cart recovery and receivables chasing still have no policy of
  their own.
- **The sequencer's funds curve is assumed, not fitted.** Its shape — a jump on
  payday decaying through the month — is the right qualitative story and is
  supported by the reporting on NACH failures, but the exact probabilities are
  chosen, not measured. A merchant with real data should fit it per segment.
- **Reported diagnosis accuracy is the offline stand-in's, not Claude's.**
  `ClaudeDiagnoser` calls the real API; that number is not in this README because
  it has not been measured, and an unmeasured number in a metrics table is how you
  lose an interview.
- **The residual `risky` column cannot reach zero** without strict mode, and
  strict mode costs revenue. That is a risk-appetite decision, not a bug.

## Layout

```
reclaim/compliance.py   the kernel: facts in, legal action set out
reclaim/verify.py       14 machine-checked safety properties
reclaim/causal.py       sharp RD, robust SEs, placebo and density checks
reclaim/security.py     injection detection, PII redaction, hash-chained log
reclaim/sequencer.py    exact DP for mandate retry scheduling, both objectives
reclaim/detect.py       candidate extraction, features, logistic regression
reclaim/diagnose.py     rules table, LLM tier, hardened tier
reclaim/policy.py       legality first, then intent, then business policy
reclaim/agent.py        the recovery loop and three baselines
reclaim/simulator.py    ground-truth world model, common random numbers
data/                   deterministic generators for all three datasets
eval/                   seven suites; run_all.sh runs them in order
docs/RESEARCH.md        the regulation, with sources, and ten edge cases
```

Sources for every constraint cited above are in [docs/RESEARCH.md](docs/RESEARCH.md).
