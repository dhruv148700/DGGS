"""
dggs_extension_generator.py — iterative DGGS-based extension builder.

Replaces the GNN oracle in extension_generator.py with DGGSRunner.
No hetero-graph construction needed; DependencyGraph -> ABAF directly.
"""

import argparse
import re
import time

from scr.dependency_graph import DependencyGraph
from scr.ABAF.ABAF import ABAF
from scr.dggs.runner import DGGSRunner
from scr.dggs.kernels import ProductBody, MaxAggregation, LinearInfluence
from scr.dggs.tau_utils import load_scores, build_tau_a, build_tau_r


_marginal_indep_re = re.compile(r"^indep_(\d+)_(\d+)__$")
_cond_dep_re       = re.compile(r"^-indep_(\d+)_(\d+)__(\d+)$")


def _assumption_priority(name: str, sigma: float) -> tuple:
    """
    Selection key for argmax over non-dummy assumptions.

    Primary: sigma (higher wins unconditionally).
    Secondary tier:  indep_* (2) > arr_*/noe_* (1) > everything else (0).
    Tertiary:  arr_ beats noe_ within the arr/noe tier.
    Quaternary: lexicographic name for determinism.

    Rationale: indep_* must be committed before v-structure rules fire;
    blocked_path_* are auxiliary with no direct evidential content.
    """
    if name.startswith("indep_"):
        tier = 1
    elif name.startswith("arr_") or name.startswith("noe_"):
        tier = 1
    else:
        tier = 0
    arr_bias = 1 if name.startswith("arr_") else 0
    return (tier, sigma, arr_bias, name)


def _inject_vstructure_rules(dep_graph, verbose: bool) -> int:
    """
    Inject v-structure support rules into dep_graph after file load.

    For each (X, Y, Z) where indep_X_Y__ is an assumption (marginal independence)
    and -indep_X_Y__Z is a fact head (conditional dependence given Z):
      Z is a collider; add  arr_X_Z <- indep_X_Y__, -indep_X_Y__Z
                       and  arr_Y_Z <- indep_X_Y__, -indep_X_Y__Z

    This creates a support path from CI evidence into arrow assumptions so DGGS
    can propagate CI scores to break the symmetric arr_i_j / arr_j_i deadlock.
    """
    # (X, Y) -> assumption name
    marginal: dict[tuple, str] = {}
    for asm in dep_graph.assumptions:
        m = _marginal_indep_re.match(asm)
        if m:
            marginal[(m.group(1), m.group(2))] = asm

    # (X, Y, Z) -> contrary-of-conditional-indep head name, from empty-body rules only
    cond_dep: dict[tuple, str] = {}
    for _, (head, body) in dep_graph.rules.items():
        if body:
            continue
        m = _cond_dep_re.match(head)
        if m:
            cond_dep[(m.group(1), m.group(2), m.group(3))] = head

    next_idx = max(dep_graph.rules.keys()) + 1
    n_injected = 0
    for (x, y, z), dep_head in cond_dep.items():
        if (x, y) not in marginal:
            continue
        indep_asm = marginal[(x, y)]
        body = sorted([indep_asm, dep_head])

        for arr in (f"arr_{x}_{z}", f"arr_{y}_{z}"):
            if arr not in dep_graph.assumptions:
                continue
            dep_graph.rules[next_idx] = (arr, body)
            if verbose:
                print(f"[vstructure] +rule  {arr} <- {body}  (collider: {z})")
            next_idx += 1
            n_injected += 1

    if verbose:
        print(f"[vstructure] injected {n_injected} rules for "
              f"{len(cond_dep)} conditional-dep facts, "
              f"{len(marginal)} marginal-indep assumptions")
    return n_injected


def build_dggs_extension(
    aba_file_path: str,
    scores_path: str = None,
    body_agg=None,
    claim_agg=None,
    support_agg=None,
    influence=None,
    max_iter: int = 200,
    epsilon: float = 1e-3,
    window: int = 5,
    verbose: bool = True,
    reject_edge_on_indep: bool = True,
) -> tuple[set, set]:
    """
    Iteratively ground assumptions using DGGS fixed points.

    At each step: run DGGS on the current reduct with fresh tau (CI-derived if
    scores_path is given, uniform 0.5/1.0 otherwise), accept the argmax-sigma
    assumption, mutate the framework via remove_accepted_assumption(), repeat.

    Stops when remove_accepted_assumption() returns False (structural rejection)
    or dep_graph.assumptions is empty.

    Returns (extension, all_assumptions): accepted assumption names and the full
    original assumption set before any grounding.
    """
    # Default kernels: sweep winner prod·max·max·lin·k1.0
    body_agg    = body_agg    or ProductBody()
    claim_agg   = claim_agg   or MaxAggregation()
    support_agg = support_agg or MaxAggregation()
    influence   = influence   or LinearInfluence(conservativeness=1.0)

    # Load CI scores once; reused each iteration over the changing reduct.
    plain_scores, rule_scores = {}, {}
    if scores_path is not None:
        plain_scores, rule_scores = load_scores(scores_path)
        if verbose:
            print(f"[dggs] loaded scores from {scores_path} ({len(plain_scores)} plain, {len(rule_scores)} rule entries)")

    dep_graph = DependencyGraph(reject_edge_on_indep=reject_edge_on_indep)
    dep_graph.create_from_file(aba_file_path)
    all_assumptions = dep_graph.assumptions.copy()

    if verbose:
        print(f"\n[dggs] injecting v-structure rules ...")
    n_vstructure_rules = _inject_vstructure_rules(dep_graph, verbose=verbose)
    # _inject_vstructure_rules mutates dep_graph.rules directly; rebuild indices.
    dep_graph._init_indices()

    extension: set[str] = set()
    step = 0

    while dep_graph.assumptions:
        step += 1
        if verbose:
            print(f"\n[step {step}] assumptions remaining: {len(dep_graph.assumptions)}")

        # Fresh tau each iteration from the same CI scores; arr_*/noe_* absent from
        # scores fall back to tau_a=0.5, tau_r=1.0 (ABAF.from_dependency_graph defaults).
        tau_a = build_tau_a(dep_graph, plain_scores)
        tau_r = build_tau_r(dep_graph, rule_scores)
        abaf = ABAF.from_dependency_graph(dep_graph, tau_a=tau_a, tau_r=tau_r)
        result = DGGSRunner(
            abaf,
            body_agg=body_agg,
            claim_agg=claim_agg,
            support_agg=support_agg,
            influence=influence,
            max_iter=max_iter,
            epsilon=epsilon,
            window=window,
        ).run()

        # Non-convergence is not fatal; the current fixed-point state is still used.
        if not result.converged:
            print(f"[step {step}] warning: did not converge ({result.stop_reason})")

        if verbose:
            print(f"[step {step}] converged={result.converged} in {result.n_iterations} iters")
            ranked = sorted(
                result.final_state.assumptions.items(), key=lambda kv: kv[1], reverse=True
            )
            print(f"[step {step}] sigma ranking:")
            for name, sigma in ranked:
                marker = " <-- argmax" if name == ranked[0][0] else ""
                print(f"           {sigma:.4f}  {name}{marker}")

        # Exclude dummy assumptions — reduct artifacts, never eligible for acceptance.
        non_dummy = {
            k: v for k, v in result.final_state.assumptions.items()
            if not k.startswith("dummy")
        }
        if not non_dummy:
            if verbose:
                print(f"[step {step}] STOP — no real assumptions remain (only dummies)")
            break

        # Stop early if only blocked_path_ assumptions remain (no CI/arrow content left).
        if all(k.startswith("blocked_path_") for k in non_dummy):
            if verbose:
                print(f"[step {step}] STOP — only blocked_path_ assumptions remain")
            break

        # Argmax with type-priority tiebreak (see _assumption_priority).
        best = max(non_dummy, key=lambda k: _assumption_priority(k, non_dummy[k]))

        ok = dep_graph.remove_accepted_assumption(best)
        if not ok:
            # Structural rejection: grounding best would make the reduct inconsistent.
            if verbose:
                print(f"[step {step}] STOP — structural rejection on remove_accepted_assumption({best})")
            break

        extension.add(best)
        if verbose:
            print(f"[step {step}] ACCEPTED: {best}  |  extension so far: {sorted(extension)}")

    if verbose and not dep_graph.assumptions:
        print(f"\n[step {step}] STOP — framework exhausted")

    return extension, all_assumptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a DGGS-grounded extension from an ABA framework."
    )
    parser.add_argument("--aba_file", type=str, required=True, help="Path to .aba file")
    parser.add_argument("--scores", type=str, default=None, help="Path to .scores.json file")
    args = parser.parse_args()

    start = time.time()
    extension, assumptions = build_dggs_extension(args.aba_file, scores_path=args.scores)
    elapsed = time.time() - start

    print(f"Generation time: {elapsed:.4f}s")
    print(f"{extension=}")
    print(f"{assumptions=}")
