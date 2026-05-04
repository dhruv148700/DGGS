"""
DGGS (Dependency Graph Gradual Semantics) — Support Cycle Example
==================================================================

This script runs DGGS on a single ABAF containing two disjoint support
cycles so that both behaviours can be observed in one run.

ABAF:
  Assumptions: A = {a, b, c, d}
  Contraries:  a_bar = p   (only a has a derivable contrary)
               b, c, d have no contrary derived by any rule
  Rules:
    r1: a <- b     (b supports a — non-flat)
    r2: b <- a     (a supports b — non-flat; together r1+r2 form cycle 1)
    r3: p <-       (fact — attacks a)

    r4: c <- d     (d supports c — non-flat)
    r5: d <- c     (c supports d — non-flat; together r4+r5 form cycle 2)

Behaviour we expect:
  - Cycle 1 (a, b) — support cycle with one assumption under attack.
      a is pulled down by the fact attack via p. b has no attack, only
      support from a. Both converge to a stable fixed point below 1.
  - Cycle 2 (c, d) — pure support cycle, no attacks.
      Mutual reinforcement drives both to their maximum strength 1.0
      (or asymptotically toward it).

Analytic fixed points:
  CYCLE 1 (tau_A(a)=0.6, tau_A(b)=0.5):
      w_a* = b* - 1   (attack fact + support b*),  a* = 0.6 + 0.6*(b* - 1) = 0.6*b*
      w_b* = a*       (pure support),              b* = 0.5 + 0.5*a*
      Solving: b* = 0.5 + 0.5*(0.6*b*) = 0.5 + 0.3*b*  =>  0.7*b* = 0.5
               b* = 5/7 ≈ 0.7143
               a* = 0.6 * 5/7 = 3/7 ≈ 0.4286

  CYCLE 2 (tau_A(c)=0.7, tau_A(d)=0.8):
      w_c* = d*,  c* = 0.7 + 0.3*d*
      w_d* = c*,  d* = 0.8 + 0.2*c*
      Solving: c* = 0.7 + 0.3*(0.8 + 0.2*c*) = 0.94 + 0.06*c*  =>  0.94*c* = 0.94
               c* = 1.0,   d* = 0.8 + 0.2 = 1.0

Kernel used (same as the main DGGS script):
  zeta_body  = zeta_pi (product)
  alpha_head = max
  alpha_att  = identity (unique contrary)
  alpha_sup  = max
  iota_A     = iota_lin (DF-QuAD, k=1)
"""

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def zeta_pi(values: List[float]) -> float:
    """Product aggregation. Empty body (fact) => 1.0 by Factuality."""
    if not values:
        return 1.0
    result = 1.0
    for v in values:
        result *= v
    return result


def iota_lin(tau: float, w: float, k: float = 1.0) -> float:
    """
    DF-QuAD linear influence function.
      w >= 0 : support pulls strength UP, scaled by (1 - tau) — ceiling at 1.
      w  < 0 : attack pulls strength DOWN, scaled by tau — floor at 0.
    """
    if w >= 0:
        return tau + (1.0 - tau) / k * w
    else:
        return tau + tau / k * w


def alpha_head(rule_strengths: List[float]) -> float:
    """Max over all rules deriving a claim."""
    return max(rule_strengths) if rule_strengths else 0.0


def alpha_sup(rule_strengths: List[float]) -> float:
    """Max over rules directly deriving an assumption (non-flat support)."""
    return max(rule_strengths) if rule_strengths else 0.0


# ---------------------------------------------------------------------------
# Framework specification
# ---------------------------------------------------------------------------

TAU_A = {
    'a': 0.6,
    'b': 0.5,
    'c': 0.7,
    'd': 0.8,
}

TAU_R = {
    'r1': 1.0, 'r2': 1.0, 'r3': 1.0, 'r4': 1.0, 'r5': 1.0,
}

RULES = {
    'r1': {'head': 'a', 'body': ['b']},   # b supports a
    'r2': {'head': 'b', 'body': ['a']},   # a supports b
    'r3': {'head': 'p', 'body': []},      # fact attacking a
    'r4': {'head': 'c', 'body': ['d']},   # d supports c
    'r5': {'head': 'd', 'body': ['c']},   # c supports d
}

# Contrary function: only a has a derivable contrary (p)
# b, c, d have no derived contraries in this framework
CONTRARY = {
    'a': 'p',
}

# Derived classification
ASSUMPTIONS = set(TAU_A.keys())
RULE_NODES  = set(RULES.keys())
# Claims = rule heads that are NOT assumptions
CLAIMS = {spec['head'] for spec in RULES.values()} - ASSUMPTIONS

# Pre-computed lookup tables
RULES_DERIVING     = {c: [r for r, s in RULES.items() if s['head'] == c] for c in CLAIMS}
RULES_SUPPORTING   = {a: [r for r, s in RULES.items() if s['head'] == a] for a in ASSUMPTIONS}


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

def initialise_state() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Initial state at t=0:
      - Assumptions   : base scores tau_A
      - Claims        : 0.0 (nothing derived yet)
      - Rules         : pre-initialised from current body values,
                        using tau_A for assumption bodies and 0 for claim
                        bodies (prevents spurious non-flat support pollution).
    """
    asm_state   = dict(TAU_A)
    claim_state = {c: 0.0 for c in CLAIMS}

    rule_state = {}
    for r, spec in RULES.items():
        body_vals = []
        for atom in spec['body']:
            if atom in ASSUMPTIONS:
                body_vals.append(TAU_A[atom])
            else:
                body_vals.append(0.0)
        rule_state[r] = TAU_R[r] * zeta_pi(body_vals)

    return asm_state, rule_state, claim_state


def step(asm: Dict[str, float],
         rule: Dict[str, float],
         claim: Dict[str, float]
         ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Perform one synchronous DGGS update over all node types.
    Every node at t+1 is computed from the full state at t.
    """
    # --- Rules: r(t+1) = tau_R * zeta_pi(body at t) ---
    new_rule = {}
    for r, spec in RULES.items():
        body_vals = []
        for atom in spec['body']:
            if atom in ASSUMPTIONS:
                body_vals.append(asm[atom])
            else:
                body_vals.append(claim[atom])
        new_rule[r] = TAU_R[r] * zeta_pi(body_vals)

    # --- Claims: c(t+1) = max over rules deriving c ---
    new_claim = {c: alpha_head([rule[r] for r in RULES_DERIVING[c]]) for c in CLAIMS}

    # --- Assumptions: iota_lin(tau_A, alpha_sup - alpha_att) ---
    new_asm = {}
    for a in ASSUMPTIONS:
        attack_signal  = claim[CONTRARY[a]] if a in CONTRARY else 0.0
        support_signal = alpha_sup([rule[r] for r in RULES_SUPPORTING[a]])
        w = support_signal - attack_signal
        new_asm[a] = iota_lin(TAU_A[a], w)

    return new_asm, new_rule, new_claim


def max_change(old: Dict[str, float], new: Dict[str, float]) -> float:
    """Largest absolute change in any assumption strength."""
    return max(abs(new[k] - old[k]) for k in old)


def run_dggs(max_iterations: int = 200,
             epsilon: float = 1e-3,
             stability_window: int = 5,
             verbose: bool = True
             ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], int, str]:
    """
    Run DGGS with two stopping conditions:
      1. max_iterations  : hard cap on total iterations (safety net).
      2. convergence     : max assumption change < epsilon for
                           `stability_window` consecutive iterations
                           (the criterion used in Rapberger et al.).

    Returns: (assumptions, rules, claims, iterations_run, stop_reason)
    """
    asm, rule, claim = initialise_state()

    if verbose:
        _print_header()
        _print_state(0, asm, rule, claim)

    recent_changes: List[float] = []

    for t in range(1, max_iterations + 1):
        prev_asm = asm
        asm, rule, claim = step(asm, rule, claim)

        delta = max_change(prev_asm, asm)
        recent_changes.append(delta)
        if len(recent_changes) > stability_window:
            recent_changes.pop(0)

        if verbose:
            _print_state(t, asm, rule, claim, delta=delta)

        if (len(recent_changes) == stability_window
                and max(recent_changes) < epsilon):
            return asm, rule, claim, t, (
                f"converged (all changes < {epsilon} over last "
                f"{stability_window} steps)"
            )

    return asm, rule, claim, max_iterations, (
        f"max iterations reached ({max_iterations}) without convergence"
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_header():
    print(f"{'t':>4} | "
          f"{'a':>7} {'b':>7} {'c':>7} {'d':>7} | "
          f"{'r1':>7} {'r2':>7} {'r4':>7} {'r5':>7} | "
          f"{'p':>7} | {'Δmax':>8}")
    print("-" * 90)


def _print_state(t, asm, rule, claim, delta=None):
    d_str = f"{delta:8.5f}" if delta is not None else "    —   "
    print(f"{t:>4} | "
          f"{asm['a']:7.4f} {asm['b']:7.4f} {asm['c']:7.4f} {asm['d']:7.4f} | "
          f"{rule['r1']:7.4f} {rule['r2']:7.4f} {rule['r4']:7.4f} {rule['r5']:7.4f} | "
          f"{claim['p']:7.4f} | {d_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asm, rule, claim, iters, reason = run_dggs(
        max_iterations=200,
        epsilon=1e-3,
        stability_window=5,
        verbose=True,
    )

    print(f"\nStopped after {iters} iterations — {reason}\n")

    print("Final assumption strengths:")
    for a in sorted(asm):
        print(f"  sigma({a}) = {asm[a]:.4f}")