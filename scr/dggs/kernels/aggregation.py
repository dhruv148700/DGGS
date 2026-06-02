"""
aggregation.py — alpha functions: aggregating rule strengths into a single signal.

Used in two distinct slots:
  claim_agg   — how rules that derive a claim combine into claim strength
  support_agg — how rules that directly derive an assumption combine into support signal
               (non-flat frameworks only; returns 0.0 when no such rules exist)

Both slots accept the same interface: List[float] -> float.
Empty list must return 0.0 (no rules → no signal).
"""


class MaxAggregation:
    """Maximum rule strength. Standard disjunctive (DF-QuAD) choice."""

    name = "MaxAggregation"

    def __call__(self, rule_strengths: list[float]) -> float:
        return max(rule_strengths) if rule_strengths else 0.0


class SumAggregation:
    """Sum of rule strengths. Unbounded; suitable when downstream is clamped."""

    name = "SumAggregation"

    def __call__(self, rule_strengths: list[float]) -> float:
        return sum(rule_strengths) if rule_strengths else 0.0


class MeanAggregation:
    """Mean of rule strengths. Stays in [0, 1] when inputs are in [0, 1]."""

    name = "MeanAggregation"

    def __call__(self, rule_strengths: list[float]) -> float:
        return sum(rule_strengths) / len(rule_strengths) if rule_strengths else 0.0
