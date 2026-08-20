"""A learned diagnosis tier, between the rules table and the model.

The diagnoser was two tiers: an exact-match rules table, then a language model
for everything it had never seen. That leaves an obvious gap. A merchant running
this for a month has thousands of *labelled* failures -- ops resolved them, the
outcome is known -- and none of that was being used. The language model was being
asked to reason from scratch about cases the historical data already answers.

So: rules -> learned classifier -> model, in ascending order of cost.

The classifier sees only observable fields. Critically it one-hot encodes the
error reason, which means a reason that never appeared in training contributes
nothing and the prediction falls back on method, recurrence, failure history and
note tokens. That is the correct behaviour rather than a limitation: on genuinely
novel reasons the classifier *should* be less certain, and its confidence is what
routes those cases onward to the model.

Multinomial logistic regression, written out for the same reason the rest of the
repo is: no dependency, and every number traceable to code here.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .models import FailedPayment, RootCause

# Classes the classifier can predict. UNKNOWN is not among them -- abstention is
# expressed through low confidence, not through a class, so that the routing
# threshold is a single tunable rather than a second mechanism.
CLASSES = [c for c in RootCause if c is not RootCause.UNKNOWN]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

METHODS = ["upi", "card", "netbanking", "wallet"]
CODES = ["BAD_REQUEST_ERROR", "GATEWAY_ERROR"]

# Note tokens worth a feature. Chosen from what merchants actually write about
# failures, not fitted -- a vocabulary selected on the training labels would leak
# the labels into the feature space.
NOTE_TOKENS = ["salary", "payday", "otp", "expired", "closed", "new card", "limit",
               "cap", "risk", "fraud", "outage", "bank", "approve", "balance",
               "revoked", "later", "tomorrow", "dead"]


def vocabulary(rows: list[FailedPayment]) -> list[str]:
    """Error reasons seen in training. Anything else encodes as all-zeros."""
    return sorted({r.error_reason for r in rows})


def featurise(p: FailedPayment, vocab: list[str]) -> list[float]:
    note = (p.merchant_note or "").lower()
    feats = [1.0]                                        # bias
    feats += [1.0 if p.error_reason == v else 0.0 for v in vocab]
    feats += [1.0 if p.method == m else 0.0 for m in METHODS]
    feats += [1.0 if p.error_code == c else 0.0 for c in CODES]
    feats.append(1.0 if p.is_recurring else 0.0)
    feats.append(min(p.customer_prior_failures, 3) / 3.0)
    feats.append(math.log10(max(p.amount_inr, 1.0)) / 6.0)
    feats.append(1.0 if note else 0.0)
    feats += [1.0 if t in note else 0.0 for t in NOTE_TOKENS]
    # Whether the reason is one the training set ever contained. The single most
    # informative feature for routing: it tells the classifier when it is out of
    # its depth, which is exactly when the model tier should take over.
    feats.append(1.0 if p.error_reason in vocab else 0.0)
    return feats


@dataclass
class Prediction:
    cause: RootCause
    confidence: float
    margin: float          # gap to the runner-up; a better abstention signal
                           # than the top probability on its own


class SoftmaxClassifier:
    """Multinomial logistic regression by batch gradient descent, with L2."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.w = [[0.0] * n_features for _ in range(n_classes)]

    def _scores(self, x: list[float]) -> list[float]:
        return [sum(wi * xi for wi, xi in zip(row, x)) for row in self.w]

    def predict_proba(self, x: list[float]) -> list[float]:
        z = self._scores(x)
        m = max(z)
        e = [math.exp(v - m) for v in z]
        total = sum(e)
        return [v / total for v in e]

    def fit(self, X: list[list[float]], y: list[int], *, epochs: int = 300,
            lr: float = 0.5, l2: float = 1e-3) -> "SoftmaxClassifier":
        n = len(X)
        n_classes = len(self.w)
        for _ in range(epochs):
            grads = [[0.0] * len(self.w[0]) for _ in range(n_classes)]
            for xi, yi in zip(X, y):
                probs = self.predict_proba(xi)
                for k in range(n_classes):
                    err = probs[k] - (1.0 if k == yi else 0.0)
                    if err == 0.0:
                        continue
                    row = grads[k]
                    for j, v in enumerate(xi):
                        if v:
                            row[j] += err * v
            for k in range(n_classes):
                for j in range(len(self.w[k])):
                    reg = 0.0 if j == 0 else l2 * self.w[k][j]
                    self.w[k][j] -= lr * (grads[k][j] / n + reg)
        return self

    def log_loss(self, X, y) -> float:
        eps = 1e-12
        return -sum(math.log(max(self.predict_proba(xi)[yi], eps))
                    for xi, yi in zip(X, y)) / len(X)


class LearnedDiagnoser:
    """Trained on labelled history. Abstains by margin, not by top probability."""

    name = "learned"

    def __init__(self, model: SoftmaxClassifier, vocab: list[str],
                 min_margin: float = 0.25) -> None:
        self.model = model
        self.vocab = vocab
        self.min_margin = min_margin

    def predict(self, p: FailedPayment) -> Prediction:
        probs = self.model.predict_proba(featurise(p, self.vocab))
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        top, second = order[0], order[1]
        return Prediction(CLASSES[top], probs[top], probs[top] - probs[second])

    def confident(self, pred: Prediction) -> bool:
        """Margin, not probability.

        A prediction of 0.45 that beats the runner-up by 0.40 is a confident
        call; one of 0.55 that beats it by 0.02 is a coin flip wearing a high
        number. Routing on the top probability alone sends the wrong cases to
        the expensive tier.
        """
        return pred.margin >= self.min_margin


def train(rows: list[FailedPayment], *, l2: float = 1e-3, epochs: int = 300
          ) -> LearnedDiagnoser:
    vocab = vocabulary(rows)
    X = [featurise(p, vocab) for p in rows]
    y = [CLASS_INDEX[p.true_root_cause] for p in rows]
    model = SoftmaxClassifier(len(X[0]), len(CLASSES)).fit(X, y, l2=l2, epochs=epochs)
    return LearnedDiagnoser(model, vocab)
