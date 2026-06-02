"""
influence.py — iota functions: (tau, net_signal) -> new assumption strength.

net_signal = support_signal - attack_signal  (can be negative)
tau        = base score (fixed anchor for this assumption)

All functions must return a value in [0, 1].
"""


class LinearInfluence:
    """
    DF-QuAD linear influence (Baroni et al. 2019).

    w >= 0: strength pulled toward 1, scaled by (1 - tau).
    w <  0: strength pulled toward 0, scaled by tau.
    conservativeness k=1 is the standard choice.
    """

    name = "LinearInfluence"

    def __init__(self, conservativeness: float = 1.0):
        self.conservativeness = conservativeness

    def __call__(self, tau: float, w: float) -> float:
        if w >= 0:
            return tau + (1.0 - tau) / self.conservativeness * w
        else:
            return tau + tau / self.conservativeness * w


class QuadraticInfluence:
    """
    Quadratic-maximum influence (Rago et al. 2016 / QE semantics).

    Uses h = (w/k)^2 / (1 + (w/k)^2) as a smooth squashing factor.
    """

    name = "QuadraticInfluence"

    def __init__(self, conservativeness: float = 1.0):
        self.conservativeness = conservativeness

    def __call__(self, tau: float, w: float) -> float:
        scaled = w / self.conservativeness
        h = scaled ** 2 / (1.0 + scaled ** 2)
        if w >= 0:
            return tau + h * (1.0 - tau)
        else:
            return tau - h * tau
