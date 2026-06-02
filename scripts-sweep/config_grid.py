"""
config_grid.py — Enumerate the valid kernel configurations and expand by k.

Config grid
-----------
  body_agg     ∈ {ProductBody, MinBody}            (2)
  claim_agg    ∈ {MaxAggregation, SumAggregation, MeanAggregation}  (3)
  support_agg  ∈ {MaxAggregation, SumAggregation, MeanAggregation}  (3)
  influence    ∈ {LinearInfluence, QuadraticInfluence}               (2)

Constraint 1 — Sum→Quadratic:
  If claim_agg OR support_agg is Sum, influence MUST be Quadratic.
  SumAggregation is unbounded; LinearInfluence + Sum can produce σ outside [0,1].

Constraint 2 — Linear requires k ≥ 1:
  LinearInfluence(k) = τ ± (τ or 1−τ)/k · w.  For k < 1 the scaling factor > 1,
  so a net signal w = ±1 overshoots (σ > 1 or σ < 0).  Exactly at k = 1, |w| = 1
  lands on the [0,1] boundary — the docstring's "standard choice."  k > 1 is more
  conservative (smaller response to w).  k < 1 leaves the codomain and breaks the
  convergence guarantees of gradual semantics → excluded.
  QuadraticInfluence uses h = (w/k)² / (1+(w/k)²) ∈ [0,1) for any k > 0, so no
  lower bound on k is needed there.

Counting:
  Base configs (before k): 26
    linear base:    2 × 2 × 2 × 1 = 8   (no Sum allowed under Linear by C1)
    quadratic base: 26 − 8        = 18
  k-sweep:
    Linear:    k ∈ {1.0, 2.0}       → 8 × 2 = 16
    Quadratic: k ∈ {0.5, 1.0, 2.0}  → 18 × 3 = 54
  Total = 70 runs per ABAF.

Config ID format (human-readable, filesystem-safe):
  "{body}·{claim}·{support}·{iota}·k{k}"
  e.g. "prod·max·max·lin·k1.0",  "prod·max·sum·quad·k0.5"

Usage
-----
    from scripts_sweep.config_grid import all_configs, build_kernels
    configs = all_configs()          # list of 70 dicts
    body, claim, support, inf = build_kernels(configs[0])
"""

import itertools
from typing import List, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scr.dggs.kernels import (
    ProductBody, MinBody,
    MaxAggregation, SumAggregation, MeanAggregation,
    LinearInfluence, QuadraticInfluence,
)

# ---------------------------------------------------------------------------
# Symbolic constants used in config dicts
# ---------------------------------------------------------------------------

_BODY      = {"prod": ProductBody,    "min": MinBody}
_AGG       = {"max": MaxAggregation,  "sum": SumAggregation, "mean": MeanAggregation}
_INFLUENCE = {"lin": LinearInfluence, "quad": QuadraticInfluence}

# k ≥ 1 required for Linear (see docstring); Quadratic is safe for any k > 0
_K_LINEAR    = [1.0, 2.0]
_K_QUADRATIC = [0.5, 1.0, 2.0]


def _k_str(k: float) -> str:
    return f"k{k:.1f}"


def _config_id(body: str, claim: str, support: str, iota: str, k: float) -> str:
    return f"{body}·{claim}·{support}·{iota}·{_k_str(k)}"


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def base_configs() -> List[Dict]:
    """Return the 26 valid base configs (no k expansion).

    Asserts the count is exactly 26; raises AssertionError if the constraint
    logic is wrong.
    """
    configs = []
    for body, claim, support, iota in itertools.product(
        _BODY.keys(), _AGG.keys(), _AGG.keys(), _INFLUENCE.keys()
    ):
        # Constraint: Sum aggregation requires Quadratic influence
        if (claim == "sum" or support == "sum") and iota == "lin":
            continue
        configs.append({
            "body":    body,
            "claim":   claim,
            "support": support,
            "iota":    iota,
        })

    assert len(configs) == 26, (
        f"Expected 26 valid base configs, got {len(configs)}. "
        "Check the Sum→Quadratic constraint logic."
    )
    return configs


def all_configs() -> List[Dict]:
    """Return all 70 configs (8 Linear × 2 k + 18 Quadratic × 3 k), each with a unique config_id."""
    expanded = []
    for cfg in base_configs():
        k_values = _K_LINEAR if cfg["iota"] == "lin" else _K_QUADRATIC
        for k in k_values:
            entry = dict(cfg)
            entry["k"]         = k
            entry["config_id"] = _config_id(cfg["body"], cfg["claim"],
                                             cfg["support"], cfg["iota"], k)
            expanded.append(entry)

    assert len(expanded) == 70, (
        f"Expected 70 total configs, got {len(expanded)}. "
        "(8 Linear×2k + 18 Quadratic×3k)"
    )
    return expanded


def config_by_id(config_id: str) -> Dict:
    """Look up a config dict by its config_id string. Raises KeyError if not found."""
    for cfg in all_configs():
        if cfg["config_id"] == config_id:
            return cfg
    raise KeyError(f"Unknown config_id: {config_id!r}")


def config_by_index(idx: int) -> Dict:
    """Look up a config dict by its 0-based index in all_configs()."""
    configs = all_configs()
    if not 0 <= idx < len(configs):
        raise IndexError(f"Config index {idx} out of range [0, {len(configs)-1}]")
    return configs[idx]


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def build_kernels(cfg: Dict):
    """Instantiate kernel objects from a config dict.

    Returns (body_agg, claim_agg, support_agg, influence) ready to pass to
    DGGSRunner.
    """
    k = cfg["k"]
    return (
        _BODY[cfg["body"]](),
        _AGG[cfg["claim"]](),
        _AGG[cfg["support"]](),
        _INFLUENCE[cfg["iota"]](conservativeness=k),
    )


# ---------------------------------------------------------------------------
# CLI — print the full grid (useful for inspection / debugging)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bases = base_configs()
    print(f"Base configs (no k): {len(bases)}  (asserted == 26)")
    print()

    configs = all_configs()
    print(f"Total configs (with k sweep): {len(configs)}  (asserted == 70)")
    print()
    print(f"{'#':>3}  {'config_id':<35}  {'body':<5}  {'claim':<5}  {'support':<7}  {'iota':<5}  {'k':>5}")
    print("-" * 75)
    for i, c in enumerate(configs):
        print(f"{i:>3}  {c['config_id']:<35}  {c['body']:<5}  {c['claim']:<5}  "
              f"{c['support']:<7}  {c['iota']:<5}  {c['k']:>5g}")
