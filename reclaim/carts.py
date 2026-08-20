"""Abandoned checkout recovery, on a surface where targeting does not work.

The detector measured an AUC of 0.53 on abandoned carts -- a coin flip, at
convergence, with a feature ceiling rather than a training problem. Session
metadata does not know whether somebody meant to buy.

Most systems respond to that by messaging everyone, three times. This module
takes the finding seriously instead, because the honest consequence of "we cannot
tell who is worth chasing" is not "chase everyone", it is "chase only where the
expected value survives being wrong about half the time".

THE COST THAT IS NOT POSTAGE
----------------------------
A cart reminder costs 70 paise to send, which makes almost any cart look worth
chasing. That arithmetic is what produces three-message sequences.

It is the wrong arithmetic. Each message carries a probability that the customer
opts out of being contacted at all -- and an opt-out is permanent, removes every
future recovery opportunity for that customer, and rises steeply with each
message in a sequence. The third message in a cart sequence is not competing
against 70 paise. It is competing against every future cart, failed payment and
renewal for that customer, forever.

So the objective is incremental revenue net of the lifetime value that the
messaging itself destroys.
"""
from __future__ import annotations

from dataclasses import dataclass

MESSAGE_COST_INR = 0.70

# Conversion uplift from the nth message to a customer who was NOT going to
# return on their own. Diminishing, and fast.
UPLIFT_BY_MESSAGE = [0.11, 0.035, 0.012]

# Probability the nth message makes the customer opt out of contact entirely.
# Rising, because a sequence reads as pressure in a way one message does not.
OPT_OUT_BY_MESSAGE = [0.008, 0.021, 0.048]

# What an opt-out costs: the recovery revenue this customer would have generated
# over the remaining relationship, which we can no longer reach them for.
FUTURE_RECOVERABLE_EVENTS = 5.0
FUTURE_RECOVERY_UPLIFT = 0.10


def opt_out_cost(customer_value_inr: float) -> float:
    """Lifetime recovery value forfeited when a customer stops accepting contact."""
    return customer_value_inr * FUTURE_RECOVERABLE_EVENTS * FUTURE_RECOVERY_UPLIFT


@dataclass
class Cart:
    cart_id: str
    customer_id: str
    amount_inr: float
    customer_value_inr: float        # typical order value for this customer
    stall_hour: int
    is_new_customer: bool
    reached_payment_page: bool
    # ground truth, eval only
    would_self_recover: bool = False


def expected_value_of_message(cart: Cart, n: int) -> float:
    """Net expected rupees from sending the nth message to this cart.

    Deliberately ignores any per-cart targeting score. The detector established
    there isn't one worth having on this surface, so pretending otherwise here
    would be the same overclaim in a different file. What is left -- and what
    actually decides it -- is the size of the cart against the cost of losing
    the customer.
    """
    if n >= len(UPLIFT_BY_MESSAGE):
        return float("-inf")
    gain = UPLIFT_BY_MESSAGE[n] * cart.amount_inr
    loss = OPT_OUT_BY_MESSAGE[n] * opt_out_cost(cart.customer_value_inr)
    return gain - loss - MESSAGE_COST_INR


def messages_worth_sending(cart: Cart, max_messages: int = 3) -> int:
    """How many messages clear the bar. Usually zero or one."""
    n = 0
    while n < max_messages and expected_value_of_message(cart, n) > 0:
        n += 1
    return n
