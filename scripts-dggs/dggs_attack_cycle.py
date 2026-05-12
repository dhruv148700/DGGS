"""
DGGS (Dependency Graph Gradual Semantics) Iteration
====================================================

This script computes assumption strengths under DGGS for the non-flat ABAF
example with an attack cycle between a and d, a fact attack on b, a DLS
chain attacking c, and a non-flat support of b by c.

ABAF:
  Assumptions: A = {a, b, c, d}
  Contraries:  a = p, b = q, c = m, d = t
  Rules:
    r1: p <- b, c      (multi-body attack on a)
    r2: p <- d         (alternative attack on a)
    r3: q <-           (fact attacking b)
    r4: m <- s         (DLS step 1)
    r5: s <- d         (DLS step 2)
    r6: t <- a         (a attacks d)
    r7: b <- c         (non-flat: c supports b)

Kernel choices:
  zeta_body = product (zeta_pi)
  iota_R    = multiplicative: s(r) = tau_R(r) * zeta_pi(body)
  alpha_head = max over rules deriving the claim
  alpha_att  = identity (unique contrary convention)
  alpha_sup  = max over rules deriving the assumption
  iota_A     = DF-QuAD linear influence (k=1)
"""

from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def zeta_pi(values):
    """Product aggregation for rule bodies. Satisfies OR, F, N, ID, WL, V."""
    if not values:
        return 1.0            # Factuality: empty body (fact) gives max strength
    result = 1.0
    for v in values:
        result *= v
    return result


def iota_lin(tau: float, w: float, k: float = 1.0) -> float:
    """
    DF-QuAD-style linear influence function.
      - w >= 0: support pulls strength up toward 1, scaled by (1 - tau).
      - w  < 0: attack pulls strength down toward 0, scaled by tau.
    """
    if w >= 0:
        return tau + (1.0 - tau) / k * w
    else:
        return tau + tau / k * w


def alpha_head(rule_strengths):
    """Disjunctive aggregation over rules deriving a claim. Returns 0 if empty."""
    return max(rule_strengths) if rule_strengths else 0.0


def alpha_sup(rule_strengths):
    """Max aggregation over rules directly deriving an assumption (non-flat)."""
    return max(rule_strengths) if rule_strengths else 0.0


# ---------------------------------------------------------------------------
# Framework specification
# ---------------------------------------------------------------------------

# Assumption base scores tau_A
TAU_A = {
    'a': 0.6,
    'b': 0.5,
    'c': 0.7,
    'd': 0.8,
}

# Rule intrinsic reliabilities tau_R (all 1 for clean BSAF comparison)
TAU_R = {
    'r1': 1.0, 'r2': 1.0, 'r3': 1.0, 'r4': 1.0,
    'r5': 1.0, 'r6': 1.0, 'r7': 1.0,
}

# Rule structure
# Each rule has: head (atom), body (list of atoms)
# Atoms can be assumptions ('a','b','c','d') or claims ('p','q','m','s','t')
RULES = {
    'r1': {'head': 'p', 'body': ['b', 'c']},
    'r2': {'head': 'p', 'body': ['d']},
    'r3': {'head': 'q', 'body': []},
    'r4': {'head': 'm', 'body': ['s']},
    'r5': {'head': 's', 'body': ['d']},
    'r6': {'head': 't', 'body': ['a']},
    'r7': {'head': 'b', 'body': ['c']},
}

# Contrary function: contrary[assumption] = claim
CONTRARY = {
    'a': 'p',
    'b': 'q',
    'c': 'm',
    'd': 't',
}

# Node-type classification
ASSUMPTIONS = set(TAU_A.keys())
RULE_NODES  = set(RULES.keys())
# Claims = all atoms that appear as rule heads but are NOT assumptions
CLAIMS = {RULES[r]['head'] for r in RULES} - ASSUMPTIONS


# ---------------------------------------------------------------------------
# Pre-computation helpers (derived from the rule structure)
# ---------------------------------------------------------------------------

# For each claim, which rules derive it? (for alpha_head)
RULES_DERIVING = {c: [r for r in RULES if RULES[r]['head'] == c] for c in CLAIMS}

# For each assumption, which rules directly derive it? (for alpha_sup, non-flat)
RULES_SUPPORTING_ASM = {
    asm: [r for r in RULES if RULES[r]['head'] == asm] for asm in ASSUMPTIONS
}


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

def initialise_state() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Initialise node strengths at t=0.
      - Assumptions: base scores tau_A
      - Rules: pre-initialised using current body values (tau_A for
               assumptions, 0 for claims since claims start at 0). This
               prevents spurious support signals in the non-flat case
               where a rule directly derives an assumption.
      - Claims: 0 (no derivation has happened yet)
    """
    asm_state = dict(TAU_A)
    claim_state = {c: 0.0 for c in CLAIMS}

    rule_state = {}
    for r, spec in RULES.items():
        body_vals = []
        for atom in spec['body']:
            if atom in ASSUMPTIONS:
                body_vals.append(TAU_A[atom])
            else:                         # claim body atom starts at 0
                body_vals.append(0.0)
        rule_state[r] = TAU_R[r] * zeta_pi(body_vals)

    return asm_state, rule_state, claim_state


def step(asm: Dict[str, float],
         rule: Dict[str, float],
         claim: Dict[str, float]
         ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Perform one synchronous DGGS update over all nodes.
    Each node at t+1 is computed from the full state at t.
    """
    # --- Rules ---
    new_rule = {}
    for r, spec in RULES.items():
        body_vals = []
        for atom in spec['body']:
            if atom in ASSUMPTIONS:
                body_vals.append(asm[atom])
            else:
                body_vals.append(claim[atom])
        new_rule[r] = TAU_R[r] * zeta_pi(body_vals)

    # --- Claims ---
    new_claim = {}
    for c in CLAIMS:
        new_claim[c] = alpha_head([rule[r] for r in RULES_DERIVING[c]])

    # --- Assumptions ---
    new_asm = {}
    for a in ASSUMPTIONS:
        contrary_claim = CONTRARY[a]
        attack_signal = claim[contrary_claim]                 # alpha_att = identity
        support_signal = alpha_sup(
            [rule[r] for r in RULES_SUPPORTING_ASM[a]]
        )
        w = support_signal - attack_signal
        new_asm[a] = iota_lin(TAU_A[a], w)

    return new_asm, new_rule, new_claim


def max_change(old: Dict[str, float], new: Dict[str, float]) -> float:
    """Maximum absolute change across a dict of strengths."""
    return max(abs(new[k] - old[k]) for k in old)


def run_dggs(max_iterations: int = 100,
             epsilon: float = 1e-3,
             stability_window: int = 5,
             verbose: bool = True
             ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], int, str]:
    """
    Run the DGGS iteration with two stopping conditions:
      1. Convergence: max change over the last `stability_window` iterations
         is below `epsilon` for ALL assumption strengths.
      2. Iteration cap: `max_iterations` reached without convergence.

    Parameters
    ----------
    max_iterations : int
        Hard cap on number of iterations.
    epsilon : float
        Convergence tolerance.
    stability_window : int
        Number of consecutive iterations the max change must stay below
        epsilon for convergence to be declared.
    verbose : bool
        If True, print the state at every iteration.

    Returns
    -------
    (asm_state, rule_state, claim_state, iterations_used, stop_reason)
    """
    asm, rule, claim = initialise_state()

    if verbose:
        _print_header()
        _print_state(0, asm, rule, claim)

    recent_changes = []        # rolling log of max assumption changes

    for t in range(1, max_iterations + 1):
        prev_asm = asm
        asm, rule, claim = step(asm, rule, claim)

        delta = max_change(prev_asm, asm)
        recent_changes.append(delta)
        if len(recent_changes) > stability_window:
            recent_changes.pop(0)

        if verbose:
            _print_state(t, asm, rule, claim, delta=delta)

        # Convergence check: window is full AND all recent changes are tiny
        if (len(recent_changes) == stability_window
                and max(recent_changes) < epsilon):
            return asm, rule, claim, t, f"converged (all changes < {epsilon} over last {stability_window} steps)"

    return asm, rule, claim, max_iterations, f"max iterations reached ({max_iterations})"


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def _print_header():
    print(f"{'t':>3} | "
          f"{'a':>6} {'b':>6} {'c':>6} {'d':>6} | "
          f"{'r1':>6} {'r2':>6} {'r4':>6} {'r5':>6} {'r6':>6} {'r7':>6} | "
          f"{'p':>6} {'m':>6} {'s':>6} {'t_cl':>6} | {'Δmax':>7}")
    print("-" * 125)


def _print_state(t, asm, rule, claim, delta=None):
    d_str = f"{delta:7.5f}" if delta is not None else "    —  "
    print(f"{t:>3} | "
          f"{asm['a']:6.4f} {asm['b']:6.4f} {asm['c']:6.4f} {asm['d']:6.4f} | "
          f"{rule['r1']:6.4f} {rule['r2']:6.4f} {rule['r4']:6.4f} "
          f"{rule['r5']:6.4f} {rule['r6']:6.4f} {rule['r7']:6.4f} | "
          f"{claim['p']:6.4f} {claim['m']:6.4f} {claim['s']:6.4f} {claim['t']:6.4f} | "
          f"{d_str}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asm, rule, claim, iters, reason = run_dggs(
        max_iterations=100,
        epsilon=1e-3,
        stability_window=5,
        verbose=True,
    )

    print(f"\nStopped after {iters} iterations — {reason}")
    print("\nFinal assumption strengths:")
    for a in sorted(asm):
        print(f"  sigma({a}) = {asm[a]:.4f}")