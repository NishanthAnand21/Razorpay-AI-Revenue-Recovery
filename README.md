# Reclaim

**An agent that recovers failed payments — and knows which ones not to touch.**

Submission for the Razorpay AI Buildathon, *AI Revenue Recovery* track.

```bash
python3 data/generate.py      # regenerate the dataset (deterministic)
python3 eval/run_eval.py      # reproduce every number below
```

No dependencies. Python 3.11+. The whole evaluation runs in under a second.

---

## The claim

On a held-out set of 240 failed payments carrying ₹24.7 lakh of at-risk revenue:

| strategy | recovered | net ₹ | actions | compliance violations | wasted retries |
|---|---|---|---|---|---|
| do nothing | 0.0% | 0 | 0 | 0 | 0 |
| retry everything ×3 | 43.3% | 10,69,350 | 577 | 26 | 57 |
| retry everything, 24h backoff | 61.7% | 15,24,425 | 475 | 26 | 57 |
| **Reclaim** | **66.1%** | **16,32,902** | 545 | 3 | 1 |
| **Reclaim (strict)** | 65.9% | 16,28,968 | 522 | **0** | **0** |

**+7.1% net revenue over the best baseline, with compliance violations down from 26 to 0.**

The policy was tuned on `data/train.jsonl` and never on these rows.

## The idea

Most recovery systems are a retry loop with a cron schedule. That is easy to build and it
quietly loses money in two ways:

1. **It retries the dead.** An expired card or a revoked mandate has a *literal zero* success
   rate. Every retry is a gateway fee spent on an outcome that cannot happen. The blanket
   baselines burn 57 of them on this set.
2. **It retries the forbidden.** Re-charging a payment a risk engine already declined isn't
   just ineffective, it's a compliance event. The blanket baselines do it 26 times.

So the work isn't in the retrying. It's in the **diagnosis** — deciding which of six things
actually went wrong — and then in the **restraint**.

The six causes are chosen to be action-distinguishing rather than error-code-shaped:

| cause | what it means | right move |
|---|---|---|
| `transient_issuer` | bank blipped | retry, immediately |
| `insufficient_funds` | no money *yet* | retry **later** — timing is the entire lever |
| `auth_friction` | customer abandoned OTP | a silent retry can't fix this; ask them |
| `instrument_invalid` | card/VPA/mandate is dead | never retry; request a new instrument |
| `limit_exceeded` | daily cap hit | wait for the reset, or switch method |
| `risk_declined` | fraud block | **never** auto-retry; escalate to a human |

## Where the AI actually sits

Behind a lookup table, not in front of it. A model doesn't need to classify `card_expired`.

- **Rules layer** — exact match on failure reasons we've seen. Free, instant, perfectly
  auditable. Scores **1.000 on 119/240 rows** and returns `UNKNOWN` rather than guessing.
- **Model layer** — handles what the table has never seen, and the ambiguous declines where
  the merchant's free-text note contradicts the error code. Scores **0.876 on the 121 rows
  the rules layer can't touch**, where a rules-only system scores 0.000 by construction.

That split is the point: the model is spent only where the cheap path genuinely fails, and
its accuracy is measured *on exactly those rows* rather than diluted into an overall number.

`ClaudeDiagnoser` calls the real API; `MockLLMDiagnoser` is a keyword-and-context stand-in so
that the eval runs offline with no key. **The reported numbers are the mock's** — see
Honesty below.

### The hard rows

24% of the dataset carries a deliberately ambiguous reason — `do_not_honour`,
`transaction_not_permitted`, `debit_declined`. These are real codes, and they're ambiguous in
production too: an issuer declines without saying why, and it can mean no money, a dead card,
or a fraud rule. The surface string cannot resolve them. Sometimes the merchant note can.
Often nothing can — and then the honest answer is to escalate rather than guess with
somebody's money.

## Every action is bounded and gated

`policy.py` separates *what we'd like to do* (`propose`) from *what we're allowed to do*
(`apply_guardrails`). Guardrails can only ever **weaken** an action, never strengthen one, so
there is a single place to audit and a single place for a compliance reviewer to read.

Eight gates, most-severe first: risk declines are never auto-retried · dead instruments are
never retried · money never moves on a low-confidence diagnosis · a 3-attempt budget · a
2-outreach budget · quiet hours 21:00–09:00 defer outreach rather than cancel it · payments
too small for a human review stop instead · and a final expected-value gate that refuses any
action whose believed recovery can't cover its own cost.

Every decision records *why*, and every veto records itself in `blocked_by` — so the trail
shows not just what the agent did, but what it was **stopped** from doing:

```
$ python3 eval/run_eval.py --audit pay_2026082000754

payment pay_2026082000754   INR 669.55   upi  [recurring]
gateway said: BAD_REQUEST_ERROR / token_deprovisioned
merchant note: customer closed that bank account last month

  attempt 1: request_instrument_update via sms
      diagnosed instrument_invalid (llm, confidence 0.78)
      why: instrument is permanently dead; retrying cannot succeed, asking for a new one
  attempt 3: stop
      GUARDRAIL: outreach_budget_exhausted

  outcome: not recovered   spent INR 0.30
```

That payment is a **loss**, on purpose. We spent 30 paise establishing that ₹669 was
unrecoverable and then stopped. A blanket retry loop spends 6 rupees discovering the same
thing three times over.

## Honesty

Things that are wrong with this, stated before you find them:

**The world model is synthetic.** No student has live retry outcomes, so success is drawn
from a hand-specified table in `simulator.py`. Two things keep the comparison from being
circular: that table is **not** the table the policy believes (`policy.BELIEVED_SUCCESS` is
deliberately different — an agent that has the world memorised isn't an agent), and every
strategy faces the **identical** random draw for a given payment and attempt, so differences
are differences in decisions, not luck.

**The ranking is not robust everywhere.** `--sensitivity` re-runs under ±30% error in the
world model, and the agent **loses to blanket 24h backoff at +15% and +30%**. That's the
honest read: when retries succeed far more often than modelled, the cheapest way to recover
money really is to retry everything. The agent's edge is largest exactly where recovery is
hard. What that column doesn't price is the 26 compliance violations and 57 dead-instrument
retries the baseline commits to get there — which is why the headline table carries those
columns next to the money.

**The guardrail has a hole, and it's measured.** The "never retry a risk decline" gate keys
off the *diagnosed* cause, so a misdiagnosed ambiguous decline still slips through — 3 times
on this set. Strict mode refuses to move money below 0.60 confidence and takes that to 0, for
₹3,935 (−0.2%) of recovered revenue. Which point on that curve is right is a risk-appetite
decision, not an engineering one, so both are reported rather than one being quietly chosen.

**The reported diagnosis accuracy is the mock's, not Claude's.** Swapping in
`ClaudeDiagnoser` and re-running is one flag; that number is not yet in this README because
it hasn't been measured, and putting an unmeasured number in a metrics table is how you lose
an interview.

## Layout

```
reclaim/models.py       domain types; costs in rupees
reclaim/diagnose.py     rules table, LLM diagnosers, tiered fallback
reclaim/policy.py       propose() -> apply_guardrails(); all tunables in one block
reclaim/simulator.py    ground-truth world model + common random numbers
reclaim/agent.py        the recovery loop, and the three baselines
data/generate.py        deterministic synthetic dataset with planted hard cases
eval/run_eval.py        held-out scoring, per-class P/R, sensitivity, audit trail
```
