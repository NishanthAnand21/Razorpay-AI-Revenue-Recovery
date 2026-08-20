"""Finding revenue that is slipping away, before it is gone.

Detection is the half of this problem that has no error code to key off. A
stalled checkout emits nothing at all; an invoice going bad looks exactly like an
invoice that has not come due yet. So the detector's job is to score candidates,
and the hard part is not finding stalls -- it is deciding which stalls are worth
spending money on.

The target label is `is_worth_chasing`: at risk AND not going to fix itself.
Flagging a customer who was always going to pay is not a win. It costs a message,
it costs goodwill, and it is the single easiest way to build a recovery system
that reports a big number while destroying value.

Two detectors are compared in eval/run_detect_eval.py:
  ChaseEverything  -- flag every stall. Perfect recall, and the precision a naive
                      "we recovered X!" dashboard is quietly built on.
  LearnedDetector  -- logistic regression, fitted on a train split, thresholded
                      on expected value rather than on F1.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .surfaces import AtRiskItem, Surface

# Hours after a stall at which we make the call. Early enough to still act,
# late enough that the obvious self-recoveries have already happened.
DECISION_DELAY_HOURS: dict[Surface, float] = {
    Surface.CHECKOUT_ABANDON: 2.0,
    Surface.SUBSCRIPTION: 1.0,
    Surface.RECEIVABLE: -72.0,   # negative: we decide 3 days BEFORE the due date
}


# --- turning raw events into scoreable candidates ----------------------------

def candidates_from_stream(stream: dict) -> list[AtRiskItem]:
    """Every stall in the stream, unfiltered and unscored.

    Only fields observable at the decision time are copied into `evidence`.
    Anything the detector must not see stays on the ground-truth attributes.
    """
    items: list[AtRiskItem] = []

    for c in stream["checkouts"]:
        decide_at = c["started_hour"] + DECISION_DELAY_HOURS[Surface.CHECKOUT_ABANDON]
        # Converted before we would even look? Then it was never a candidate.
        if c["completed_hour"] is not None and c["completed_hour"] <= decide_at:
            continue
        items.append(AtRiskItem(
            item_id=c["session_id"], surface=Surface.CHECKOUT_ABANDON,
            customer_id=c["customer_id"], amount_inr=c["amount_inr"],
            detected_at_hour=int(decide_at) % 24,
            hours_since_stall=DECISION_DELAY_HOURS[Surface.CHECKOUT_ABANDON],
            evidence={
                "reached_payment_page": c["reached_payment_page"],
                "method_selected": c["method_selected"],
                "prior_orders": c["customer_prior_orders"],
                "is_new_customer": c["is_new_customer"],
            },
            truly_at_risk=True,
            would_self_recover=c["would_self_recover"],
        ))

    for m in stream["mandates"]:
        if m["succeeded"]:
            continue
        items.append(AtRiskItem(
            item_id=m["debit_id"], surface=Surface.SUBSCRIPTION,
            customer_id=m["customer_id"], amount_inr=m["amount_inr"],
            detected_at_hour=m["attempted_hour"] % 24, hours_since_stall=1.0,
            evidence={
                "error_reason": m["error_reason"],
                "mandate_age_days": m["mandate_age_days"],
                "consecutive_failures": m["consecutive_failures"],
            },
            truly_at_risk=True,
            would_self_recover=m["would_self_recover"],
        ))

    for v in stream["invoices"]:
        decide_at = v["due_hour"] + DECISION_DELAY_HOURS[Surface.RECEIVABLE]
        # Already settled before we would look. Nothing to detect.
        if v["paid_hour"] is not None and v["paid_hour"] <= decide_at:
            continue
        items.append(AtRiskItem(
            item_id=v["invoice_id"], surface=Surface.RECEIVABLE,
            customer_id=v["customer_id"], amount_inr=v["amount_inr"],
            detected_at_hour=int(decide_at) % 24,
            hours_since_stall=DECISION_DELAY_HOURS[Surface.RECEIVABLE],
            evidence={
                "avg_days_late": v["customer_avg_days_late"],
                "prior_invoices": v["customer_prior_invoices"],
                "disputed": v["disputed"],
                "terms_days": round((v["due_hour"] - v["issued_hour"]) / 24),
            },
            # An invoice not yet due is only 'at risk' if it does in fact go late.
            truly_at_risk=v["paid_hour"] is None or v["paid_hour"] > v["due_hour"],
            would_self_recover=v["would_self_recover"],
        ))

    return items


# --- features ----------------------------------------------------------------

def featurise(it: AtRiskItem) -> list[float]:
    """One shared feature vector across all three surfaces.

    Surface identity is one-hot rather than implicit, so a single model can learn
    that (say) a returning customer means something different for a cart than it
    does for an invoice.
    """
    e = it.evidence
    return [
        1.0,                                                   # bias
        1.0 if it.surface is Surface.CHECKOUT_ABANDON else 0.0,
        1.0 if it.surface is Surface.SUBSCRIPTION else 0.0,
        1.0 if it.surface is Surface.RECEIVABLE else 0.0,
        math.log10(max(it.amount_inr, 1.0)) / 6.0,             # value, compressed
        1.0 if e.get("reached_payment_page") else 0.0,
        1.0 if e.get("is_new_customer") else 0.0,
        min(e.get("prior_orders", 0), 10) / 10.0,
        1.0 if e.get("error_reason") == "insufficient_funds" else 0.0,
        1.0 if e.get("error_reason") in ("mandate_revoked", "debit_not_registered") else 0.0,
        min(e.get("consecutive_failures", 0), 3) / 3.0,
        min(e.get("avg_days_late", 0.0), 30.0) / 30.0,
        1.0 if e.get("disputed") else 0.0,
        min(e.get("prior_invoices", 0), 15) / 15.0,
    ]


N_FEATURES = 14


# --- a logistic regression, by hand ------------------------------------------

class LogisticModel:
    """Plain batch gradient descent with L2. Deliberately small and readable.

    Written out rather than imported so that every number in the report is
    traceable to code in this repo, and so it runs with no dependencies.
    """

    def __init__(self, n: int = N_FEATURES) -> None:
        self.w = [0.0] * n

    def _z(self, x: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, x))

    def predict_proba(self, x: list[float]) -> float:
        z = max(-30.0, min(30.0, self._z(x)))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X: list[list[float]], y: list[int], *,
            epochs: int = 2000, lr: float = 1.0, l2: float = 1e-3) -> "LogisticModel":
        n = len(X)
        for _ in range(epochs):
            grad = [0.0] * len(self.w)
            for xi, yi in zip(X, y):
                err = self.predict_proba(xi) - yi
                for j, v in enumerate(xi):
                    grad[j] += err * v
            for j in range(len(self.w)):
                # No weight decay on the bias term.
                reg = 0.0 if j == 0 else l2 * self.w[j]
                self.w[j] -= lr * (grad[j] / n + reg)
        return self

    def log_loss(self, X: list[list[float]], y: list[int]) -> float:
        eps = 1e-12
        return -sum(
            yi * math.log(max(self.predict_proba(xi), eps))
            + (1 - yi) * math.log(max(1 - self.predict_proba(xi), eps))
            for xi, yi in zip(X, y)
        ) / len(X)


# --- detectors ---------------------------------------------------------------

class ChaseEverything:
    """Flag every stall. This is what a recovery dashboard does by default."""

    name = "chase_everything"

    def score(self, it: AtRiskItem) -> float:
        return 1.0

    def flags(self, it: AtRiskItem) -> bool:
        return True


class LearnedDetector:
    """Score with the fitted model, flag above a threshold chosen on value."""

    name = "learned"

    def __init__(self, model: LogisticModel, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    def score(self, it: AtRiskItem) -> float:
        return self.model.predict_proba(featurise(it))

    def flags(self, it: AtRiskItem) -> bool:
        return self.score(it) >= self.threshold


def auc(model: "LogisticModel", items: list[AtRiskItem]) -> float:
    """Rank-ordering quality, which is what matters when capacity is the binding
    constraint. Accuracy at a threshold would hide it."""
    pos = [model.predict_proba(featurise(i)) for i in items if i.is_worth_chasing]
    neg = [model.predict_proba(featurise(i)) for i in items if not i.is_worth_chasing]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


class PlattCalibrator:
    """A one-dimensional logistic fitted on the model's own logit.

    Worth having because the capacity ranking multiplies a predicted probability
    by money at stake. That product is not monotonic in p, so a probability that
    is systematically off reorders the queue -- and at a tight budget the
    ordering at the top IS the decision. Measured at +32% net recovery at a
    budget of 20, and nothing at all by 100.

    Two parameters, fitted by gradient descent on train. Deliberately the
    smallest possible correction: with a few hundred points, isotonic regression
    would fit the noise in the reliability curve and report a better ECE for it.
    """

    def __init__(self, base: LogisticModel) -> None:
        self.base = base
        self.a, self.b = 1.0, 0.0

    def _logit(self, x) -> float:
        p = min(max(self.base.predict_proba(x), 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def fit(self, items, epochs: int = 3000, lr: float = 0.05) -> "Platt":
        zs = [self._logit(featurise(i)) for i in items]
        ys = [1 if i.is_worth_chasing else 0 for i in items]
        n = len(zs)
        for _ in range(epochs):
            ga = gb = 0.0
            for z, y in zip(zs, ys):
                t = max(-30.0, min(30.0, self.a * z + self.b))
                err = 1 / (1 + math.exp(-t)) - y
                ga += err * z
                gb += err
            self.a -= lr * ga / n
            self.b -= lr * gb / n
        return self

    def predict_proba(self, x) -> float:
        t = max(-30.0, min(30.0, self.a * self._logit(x) + self.b))
        return 1 / (1 + math.exp(-t))


def split(items: list[AtRiskItem], seed: int = 7, test_fraction: float = 0.3):
    """Split by customer, not by item.

    Splitting by item would put the same customer's behaviour on both sides and
    quietly inflate the test numbers, because customer history is a feature.
    """
    rng = random.Random(seed)
    customers = sorted({it.customer_id for it in items})
    rng.shuffle(customers)
    cut = int(len(customers) * (1 - test_fraction))
    train_cust = set(customers[:cut])
    train = [it for it in items if it.customer_id in train_cust]
    test = [it for it in items if it.customer_id not in train_cust]
    return train, test


def load_stream(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parents[1] / "data" / "events.json"
    return json.loads(path.read_text())
