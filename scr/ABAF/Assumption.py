from typing import Callable
from .Sentence import Sentence, _alpha_max


def _iota_lin(tau: float, w: float, k: float = 1.0) -> float:
    if w >= 0:
        return tau + (1.0 - tau) / k * w
    else:
        return tau + tau / k * w


class Assumption(Sentence):
    def __init__(
        self,
        name: str,
        contrary: str = None,
        tau: float = 0.5,
        iota: Callable = _iota_lin,
        alpha: Callable = _alpha_max,
    ):
        super().__init__(name, alpha=alpha)
        self.contrary = contrary  # name of the contrary sentence
        self.tau = tau            # base strength (fixed across iterations)
        self.iota = iota          # influence fn: (tau, w) -> new_strength

    def __eq__(self, other):
        if not isinstance(other, Assumption):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Assumption({self.name}, contrary={self.contrary}, tau={self.tau:.4f})"
