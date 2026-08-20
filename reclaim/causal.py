"""Measuring what the agent actually caused, using the shield's own refusals.

THE PROBLEM
-----------
Every recovery product reports gross recovery: "we recovered 70% of failed
payments". That number is close to meaningless, because a large share of those
customers would have paid anyway -- topped up the balance, retried in the app,
settled the invoice on day three. Advertising went through this exact reckoning
and found that measured incremental lift runs far below platform-reported
numbers. Payment recovery has not had its reckoning yet.

The fix in advertising is a holdout: withhold treatment from a random slice and
difference the outcomes. In payments that is expensive and awkward -- you are
deliberately not recovering real money to learn something, and product owners
hate it, so it does not get run.

THE IDEA
--------
We already have a system that refuses to intervene: the compliance kernel. And
crucially, several of its refusals key on facts that have nothing to do with the
customer:

    an NPCI peak window closes at 13:00. A mandate that fails at 12:59 cannot be
    retried immediately. One that fails at 13:01 can.

Nobody chooses their failure minute, and the two customers are otherwise alike.
So the regulation has run a randomised experiment on our behalf, for free, and
the shield is where the assignment is recorded. In the safe-RL literature a
shield is pure cost -- it only ever removes options. Here it also *produces the
identification strategy*, which is the part I have not seen done before.

That makes this a sharp regression discontinuity: treatment (immediate retry)
flips deterministically at a threshold in a running variable (time of failure),
and we estimate the jump in the outcome at the threshold.

WHAT THIS MODULE MUST EARN
--------------------------
An estimator is only worth having if it recovers the right answer. Because the
data generator plants a known true lift, eval/run_causal_eval.py can check the
estimate against ground truth -- and run placebo tests at cutoffs where no rule
changes, where the honest answer is zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RDResult:
    """A regression-discontinuity estimate with its uncertainty."""

    estimate: float
    std_error: float
    n_left: int
    n_right: int
    bandwidth: float
    cutoff: float

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.estimate - 1.96 * self.std_error,
                self.estimate + 1.96 * self.std_error)

    @property
    def significant(self) -> bool:
        lo, hi = self.ci95
        return lo > 0 or hi < 0

    def __str__(self) -> str:
        lo, hi = self.ci95
        return (f"{self.estimate:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                f"n={self.n_left}+{self.n_right}  h={self.bandwidth:.2f}")


def _triangular(u: float) -> float:
    """Weight observations by closeness to the cutoff.

    A triangular kernel is the standard choice for sharp RD: it is the boundary-
    optimal weighting for local linear regression, and it makes the estimate
    depend mostly on points where the comparability assumption is most credible.
    """
    return max(0.0, 1.0 - abs(u))


def _wls_intercept_at_zero(xs: list[float], ys: list[float], ws: list[float]
                           ) -> tuple[float, float, float]:
    """Weighted local-linear fit; return the intercept, its variance, and n.

    Fitting a line rather than taking a mean is what removes the bias from any
    smooth trend in the running variable. Recovery genuinely varies over the day;
    a difference of means at the boundary would attribute that trend to the
    treatment. The slope absorbs it.
    """
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), n
    sw = sum(ws)
    swx = sum(w * x for w, x in zip(ws, xs))
    swxx = sum(w * x * x for w, x in zip(ws, xs))
    swy = sum(w * y for w, y in zip(ws, ys))
    swxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))

    det = sw * swxx - swx * swx
    if abs(det) < 1e-12:
        return float("nan"), float("nan"), n

    intercept = (swxx * swy - swx * swxy) / det
    slope = (sw * swxy - swx * swy) / det

    # Heteroskedasticity-robust variance of the intercept. Binary outcomes are
    # heteroskedastic by construction, so the classical formula would understate.
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    meat = sum((w * (swxx - swx * x)) ** 2 * r * r
               for w, x, r in zip(ws, xs, resid))
    var = meat / (det * det)
    return intercept, var, n


def sharp_rd(running: list[float], outcome: list[float], *, cutoff: float,
             bandwidth: float, treat_above: bool = True) -> RDResult:
    """Estimate the jump in `outcome` at `cutoff`.

    `running` is centred on the cutoff, each side is fitted separately with a
    local linear regression, and the estimate is the difference of the two
    boundary intercepts -- the treated side minus the untreated side.
    """
    lx, ly, lw, rx, ry, rw = [], [], [], [], [], []
    for x, y in zip(running, outcome):
        d = x - cutoff
        if abs(d) > bandwidth:
            continue
        w = _triangular(d / bandwidth)
        if w <= 0:
            continue
        if d < 0:
            lx.append(d); ly.append(y); lw.append(w)
        else:
            rx.append(d); ry.append(y); rw.append(w)

    a_l, v_l, n_l = _wls_intercept_at_zero(lx, ly, lw)
    a_r, v_r, n_r = _wls_intercept_at_zero(rx, ry, rw)

    if any(math.isnan(v) for v in (a_l, a_r, v_l, v_r)):
        return RDResult(float("nan"), float("nan"), n_l, n_r, bandwidth, cutoff)

    est = (a_r - a_l) if treat_above else (a_l - a_r)
    return RDResult(est, math.sqrt(max(v_l + v_r, 0.0)), n_l, n_r, bandwidth, cutoff)


def naive_difference(treated: list[float], untreated: list[float]) -> float:
    """What a dashboard reports: treated recovery minus untreated recovery.

    Biased whenever treatment correlates with anything that also drives recovery
    -- which, since the shield assigns treatment by time of day and by decline
    code, it always does here.
    """
    if not treated or not untreated:
        return float("nan")
    return sum(treated) / len(treated) - sum(untreated) / len(untreated)


def gross_recovery_rate(treated: list[float]) -> float:
    """What the industry reports: the share of chased payments that came back.

    Attributes every self-cure to the intervention.
    """
    return sum(treated) / len(treated) if treated else float("nan")


def density_continuity(running: list[float], cutoff: float, bandwidth: float,
                       bins: int = 10) -> float:
    """A McCrary-style check for manipulation of the running variable.

    If units could sort around the threshold, the design is dead. Returns the
    ratio of observation density just above to just below; anything near 1.0 is
    consistent with nobody being able to choose their side. Here that should hold
    by construction -- customers do not select the minute their payment fails --
    but it is checked rather than assumed, because that is the assumption on
    which the whole estimate rests.
    """
    lo = [x for x in running if cutoff - bandwidth <= x < cutoff]
    hi = [x for x in running if cutoff <= x <= cutoff + bandwidth]
    if not lo or not hi:
        return float("nan")
    return (len(hi) / bandwidth) / (len(lo) / bandwidth)
