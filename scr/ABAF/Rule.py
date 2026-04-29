from typing import Callable, List
from .Sentence import Sentence


def _zeta_product(body_vals: List[float]) -> float:
    if not body_vals:
        return 1.0
    result = 1.0
    for v in body_vals:
        result *= v
    return result


class Rule:
    def __init__(
        self,
        head: Sentence,
        body: List[Sentence] = None,
        name: str = None,
        tau: float = 1.0,
        zeta: Callable = _zeta_product,
    ):
        if head is None:
            raise ValueError("Head must be specified.")
        self.name = name
        self.head = head
        self.body = body if body is not None else []
        self.tau = tau    # rule reliability (fixed across iterations)
        self.zeta = zeta  # body aggregation fn: List[float] -> float

    def __repr__(self):
        body_str = ", ".join(s.name for s in self.body) if self.body else "∅"
        return f"{self.name}[τ={self.tau}]: {self.head.name} ← {body_str}"
