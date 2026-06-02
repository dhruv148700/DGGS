"""
body.py — zeta functions: body atom aggregation → rule strength.

Applied as: rule_strength = tau_r * zeta(body_atom_strengths)
Empty body (fact rule) must return 1.0 (Factuality axiom).
"""


class ProductBody:
    """Product of body atom strengths. Satisfies Factuality, Non-negativity, Identity."""

    name = "ProductBody"

    def __call__(self, values: list[float]) -> float:
        if not values:
            return 1.0
        result = 1.0
        for v in values:
            result *= v
        return result


class MinBody:
    """Minimum of body atom strengths. More conservative than product."""

    name = "MinBody"

    def __call__(self, values: list[float]) -> float:
        return min(values) if values else 1.0
