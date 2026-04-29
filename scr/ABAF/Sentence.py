from typing import Callable, List


def _alpha_max(rule_strengths: List[float]) -> float:
    return max(rule_strengths) if rule_strengths else 0.0


class Sentence:
    def __init__(self, name: str, alpha: Callable = _alpha_max):
        self.name = name
        self.alpha = alpha  # aggregates rule strengths deriving this sentence

    def __eq__(self, other):
        if not isinstance(other, Sentence):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Sentence({self.name})"
