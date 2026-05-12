#!/usr/bin/env python3
"""
run_dggs.py — General DGGS runner for any .aba file.

Parses a .aba framework, bridges to weighted ABAF objects, then runs
DGGS gradual-semantics iteration until convergence or the iteration cap.

Usage
-----
    python run_dggs.py framework.aba [OPTIONS]

Options
-------
    --tau-a   JSON dict of assumption base scores  (default 0.5 each)
              e.g. '{"a": 0.6, "b": 0.5, "c": 0.7}'
    --tau-r   JSON dict of rule reliabilities keyed by rule index int (default 1.0 each)
              e.g. '{"1": 1.0, "2": 0.8}'
    --max-iter   Hard iteration cap           (default 100)
    --epsilon    Convergence tolerance         (default 1e-3)
    --window     Stability window size         (default 5)
    --verbose    Print per-iteration Δmax table
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scr.dependency_graph import DependencyGraph
from scr.ABAF import ABAF, Assumption, Sentence


# ---------------------------------------------------------------------------
# Index precomputation  (done once before the iteration loop)
# ---------------------------------------------------------------------------

class _Index:
    """Precomputed per-framework lookups to avoid repeated linear scans in step()."""

    def __init__(self, abaf: ABAF):
        # rules whose head is each claim sentence
        self.rules_for_claim: Dict[str, List] = {
            s.name: [r for r in abaf.rules if r.head is s]
            for s in abaf.sentences
        }
        # rules whose head is each assumption (non-flat support)
        self.rules_for_asm: Dict[str, List] = {
            a.name: [r for r in abaf.rules if r.head is a]
            for a in abaf.assumptions
        }
        # fast assumption-name lookup for body-atom dispatch
        self.asm_names: frozenset = frozenset(a.name for a in abaf.assumptions)


# ---------------------------------------------------------------------------
# DGGS iteration
# ---------------------------------------------------------------------------

def initialise_state(
    abaf: ABAF,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    t=0 state:
      - assumptions  → their base scores tau_a
      - claims       → 0.0  (nothing derived yet)
      - rules        → tau_r * zeta(body), treating claim body atoms as 0.0
    """
    asm_state   = {a.name: a.tau for a in abaf.assumptions}
    claim_state = {s.name: 0.0   for s in abaf.sentences}

    rule_state = {}
    for rule in abaf.rules:
        body_vals = [
            asm_state[atom.name] if isinstance(atom, Assumption) else 0.0
            for atom in rule.body
        ]
        rule_state[rule.name] = rule.tau * rule.zeta(body_vals)

    return asm_state, rule_state, claim_state


def step(
    abaf: ABAF,
    idx: _Index,
    asm_state: Dict[str, float],
    rule_state: Dict[str, float],
    claim_state: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """One synchronous DGGS update. All t+1 values computed from t state."""

    # --- Rules: tau_r * zeta(body) ---
    new_rule: Dict[str, float] = {}
    for rule in abaf.rules:
        body_vals = [
            asm_state[atom.name] if isinstance(atom, Assumption) else claim_state.get(atom.name, 0.0)
            for atom in rule.body
        ]
        new_rule[rule.name] = rule.tau * rule.zeta(body_vals)

    # --- Claims: alpha over rules that derive them ---
    new_claim: Dict[str, float] = {
        s.name: s.alpha([rule_state[r.name] for r in idx.rules_for_claim[s.name]])
        for s in abaf.sentences
    }

    # --- Assumptions: iota(tau, support - attack) ---
    new_asm: Dict[str, float] = {}
    for asm in abaf.assumptions:
        attack  = claim_state.get(asm.contrary, 0.0) if asm.contrary else 0.0
        support = asm.alpha([rule_state[r.name] for r in idx.rules_for_asm[asm.name]])
        new_asm[asm.name] = asm.iota(asm.tau, support - attack)

    return new_asm, new_rule, new_claim


def _max_change(old: Dict[str, float], new: Dict[str, float]) -> float:
    return max(abs(new[k] - old[k]) for k in old)


def run_dggs(
    abaf: ABAF,
    max_iterations: int = 100,
    epsilon: float = 1e-3,
    stability_window: int = 5,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], int, str]:
    """
    Run DGGS until convergence or the iteration cap.

    Convergence: max assumption change over the last `stability_window`
    iterations is below `epsilon`.

    Returns (asm_state, rule_state, claim_state, iterations_used, stop_reason).
    """
    idx = _Index(abaf)
    asm, rule, claim = initialise_state(abaf)

    if verbose:
        print(f"{'t':>5} | {'Δmax':>12}")
        print("-" * 22)
        print(f"{'0':>5} | {'—':>12}")

    recent_changes: List[float] = []

    for t in range(1, max_iterations + 1):
        prev_asm = asm
        asm, rule, claim = step(abaf, idx, asm, rule, claim)

        delta = _max_change(prev_asm, asm)
        recent_changes.append(delta)
        if len(recent_changes) > stability_window:
            recent_changes.pop(0)

        if verbose:
            print(f"{t:>5} | {delta:12.8f}")

        if len(recent_changes) == stability_window and max(recent_changes) < epsilon:
            reason = f"converged (all changes < {epsilon} over last {stability_window} steps)"
            return asm, rule, claim, t, reason

    return asm, rule, claim, max_iterations, f"max iterations reached ({max_iterations})"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _stable_hash(name: str) -> int:
    """First two bytes of SHA-256(name) combined into a 16-bit int."""
    digest = hashlib.sha256(name.encode()).digest()
    return (digest[0] << 8) | digest[1]


def main():
    parser = argparse.ArgumentParser(
        description="Run DGGS gradual semantics on a .aba framework file."
    )
    parser.add_argument("aba_file", help="Path to the .aba framework file")
    parser.add_argument(
        "--tau-a", default="{}",
        help='JSON dict of assumption base scores, e.g. \'{"a": 0.6, "b": 0.5}\'. '
             'Overrides --init for named assumptions.'
    )
    parser.add_argument(
        "--tau-r", default="{}",
        help='JSON dict of rule reliabilities by rule index, e.g. \'{"1": 1.0, "2": 0.8}\''
    )
    parser.add_argument(
        "--init", choices=["fixed", "random"], default="fixed",
        help="Base score initialisation for assumptions not in --tau-a: "
             "'fixed' = 0.5 (default), 'random' = uniform [0,1] with --seed"
    )
    parser.add_argument("--seed",     type=int,   default=42,
                        help="Base random seed for --init random (default 42)")
    parser.add_argument("--max-iter",  type=int,   default=100,  dest="max_iter")
    parser.add_argument("--epsilon",   type=float, default=1e-3)
    parser.add_argument("--window",    type=int,   default=5)
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    tau_a_override: Dict[str, float] = json.loads(args.tau_a)
    tau_r: Dict[int, float] = {int(k): v for k, v in json.loads(args.tau_r).items()}

    dg = DependencyGraph()
    dg.create_from_file(args.aba_file)

    # Build tau_a: explicit overrides take priority; remaining assumptions use
    # fixed 0.5 or per-assumption reproducible random draw depending on --init.
    tau_a: Dict[str, float] = {}
    for name in sorted(dg.assumptions):
        if name in tau_a_override:
            tau_a[name] = tau_a_override[name]
        elif args.init == "random":
            random.seed(args.seed + _stable_hash(name) % 1000)
            tau_a[name] = random.uniform(0.0, 1.0)
        # else: leave absent → ABAF.from_dependency_graph defaults to 0.5

    abaf = ABAF.from_dependency_graph(dg, tau_a=tau_a, tau_r=tau_r)
    init_label = f"random (seed={args.seed})" if args.init == "random" else "fixed 0.5"
    print(f"Loaded: {abaf}")
    print(f"  {len(abaf.assumptions)} assumptions, "
          f"{len(abaf.sentences)} claims, "
          f"{len(abaf.rules)} rules  [init={init_label}]\n")

    asm, rule, claim, iters, reason = run_dggs(
        abaf,
        max_iterations=args.max_iter,
        epsilon=args.epsilon,
        stability_window=args.window,
        verbose=args.verbose,
    )

    print(f"\nStopped after {iters} iteration(s) — {reason}")
    print("\nFinal assumption strengths:")
    for name in sorted(asm):
        print(f"  sigma({name}) = {asm[name]:.6f}")


if __name__ == "__main__":
    main()
