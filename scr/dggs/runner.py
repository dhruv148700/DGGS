"""
runner.py — DGGSRunner: the central DGGS iteration engine.

Owns all four kernel slots so they can be swapped independently:
  body_agg    (zeta)    — how rule body atoms combine into rule strength
  claim_agg   (alpha_c) — how rules deriving a claim combine into claim strength
  support_agg (alpha_s) — how rules directly deriving an assumption combine into support
                          (non-flat frameworks; can differ from claim_agg)
  influence   (iota)    — how net signal (support - attack) maps to new assumption strength

ABAF objects are pure data containers; no kernel logic lives on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from scr.ABAF import ABAF

from .kernels import (
    MaxAggregation,
    ProductBody,
    LinearInfluence,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class State:
    assumptions: Dict[str, float]
    rules: Dict[str, float]
    claims: Dict[str, float]


@dataclass
class RunResult:
    final_state: State
    n_iterations: int
    converged: bool
    stop_reason: str


# ---------------------------------------------------------------------------
# Pre-computed index (built once per framework, reused across iterations)
# ---------------------------------------------------------------------------

class _Index:
    """Per-framework lookups to avoid repeated linear scans in iterate()."""

    def __init__(self, abaf: "ABAF"):
        self.rules_for_claim: Dict[str, List] = {
            s.name: [r for r in abaf.rules if r.head is s]
            for s in abaf.sentences
        }
        self.rules_for_asm: Dict[str, List] = {
            a.name: [r for r in abaf.rules if r.head is a]
            for a in abaf.assumptions
        }
        self.asm_names: frozenset = frozenset(a.name for a in abaf.assumptions)


# ---------------------------------------------------------------------------
# DGGSRunner
# ---------------------------------------------------------------------------

class DGGSRunner:
    """
    Runs DGGS gradual semantics on a weighted ABAF.

    Parameters
    ----------
    abaf        : parsed and weighted ABAF (assumptions, rules, sentences)
    body_agg    : zeta — body atom aggregation (default: product)
    claim_agg   : alpha for claims — rule aggregation into claim strength (default: max)
    support_agg : alpha for non-flat support — rule aggregation into assumption support
                  (default: max; can differ from claim_agg)
    influence   : iota — maps (tau, net_signal) to new assumption strength (default: linear)
    max_iter    : hard iteration cap
    epsilon     : convergence tolerance
    window      : number of consecutive iterations all deltas must stay below epsilon
    """

    def __init__(
        self,
        abaf: "ABAF",
        body_agg=None,
        claim_agg=None,
        support_agg=None,
        influence=None,
        max_iter: int = 200,
        epsilon: float = 1e-3,
        window: int = 5,
    ):
        self.abaf        = abaf
        self.body_agg    = body_agg    or ProductBody()
        self.claim_agg   = claim_agg   or MaxAggregation()
        self.support_agg = support_agg or MaxAggregation()
        self.influence   = influence   or LinearInfluence()
        self.max_iter    = max_iter
        self.epsilon     = epsilon
        self.window      = window
        self._idx        = _Index(abaf)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialise(self) -> State:
        """
        Build the t=0 state.
          - assumptions  → their base scores tau_a
          - claims       → 0.0  (nothing derived yet)
          - rules        → tau_r * body_agg(body), treating claim body atoms as 0.0
        """
        asm   = {a.name: a.tau for a in self.abaf.assumptions}
        claim = {s.name: 0.0   for s in self.abaf.sentences}
        rule  = {}
        for r in self.abaf.rules:
            body_vals = [
                asm[atom.name] if atom.name in self._idx.asm_names else 0.0
                for atom in r.body
            ]
            rule[r.name] = r.tau * self.body_agg(body_vals)
        return State(assumptions=asm, rules=rule, claims=claim)

    def iterate(self, state: State) -> State:
        """One synchronous DGGS update. All t+1 values computed from t state."""
        abaf = self.abaf
        idx  = self._idx

        # Rules: tau_r * body_agg(body at t)
        new_rule: Dict[str, float] = {}
        for r in abaf.rules:
            body_vals = [
                state.assumptions[atom.name]
                if atom.name in idx.asm_names
                else state.claims.get(atom.name, 0.0)
                for atom in r.body
            ]
            new_rule[r.name] = r.tau * self.body_agg(body_vals)

        # Claims: claim_agg over rules that derive each claim
        new_claim: Dict[str, float] = {
            s.name: self.claim_agg([state.rules[r.name] for r in idx.rules_for_claim[s.name]])
            for s in abaf.sentences
        }

        # Assumptions: influence(tau, support_agg(support_rules) - attack)
        new_asm: Dict[str, float] = {}
        for a in abaf.assumptions:
            attack  = state.claims.get(a.contrary, 0.0) if a.contrary else 0.0
            support = self.support_agg([state.rules[r.name] for r in idx.rules_for_asm[a.name]])
            new_asm[a.name] = self.influence(a.tau, support - attack)

        return State(assumptions=new_asm, rules=new_rule, claims=new_claim)

    def run(self, initial_state: State = None) -> RunResult:
        """
        Run DGGS until convergence or max_iter.

        Convergence criterion: max assumption delta over last `window` iterations
        is below `epsilon`.

        Returns a RunResult with the final state, iteration count, and stop reason.
        """
        state = initial_state if initial_state is not None else self.initialise()
        recent_deltas: List[float] = []

        for t in range(1, self.max_iter + 1):
            prev_asm = state.assumptions
            state    = self.iterate(state)

            delta = max(abs(state.assumptions[k] - prev_asm[k]) for k in prev_asm)
            recent_deltas.append(delta)
            if len(recent_deltas) > self.window:
                recent_deltas.pop(0)

            if len(recent_deltas) == self.window and max(recent_deltas) < self.epsilon:
                return RunResult(
                    final_state   = state,
                    n_iterations  = t,
                    converged     = True,
                    stop_reason   = (
                        f"converged (max delta < {self.epsilon} "
                        f"over last {self.window} steps)"
                    ),
                )

        return RunResult(
            final_state  = state,
            n_iterations = self.max_iter,
            converged    = False,
            stop_reason  = f"max iterations reached ({self.max_iter})",
        )
