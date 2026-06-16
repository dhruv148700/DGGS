"""
gnn_causal_extension_generator.py — iterative GNN-based causal extension builder.

Mirrors dggs_extension_generator.py but uses ABAInferenceEngine (GCN/GAT) as the
oracle instead of DGGSRunner.  Key design differences vs extension_generator.py:

  - Prioritises arr_*/noe_*/indep_* assumptions over blocked_path_* at each step.
  - Stops when all remaining non-dummy assumptions are blocked_path_.
  - Injects v-structure support rules into the dependency graph before the main loop
    so CI evidence can propagate into arrow assumptions.
  - Supports reject_edge_on_indep (removes arr/noe when an indep is committed).
"""

import argparse
import time

from scr.dependency_graph import DependencyGraph
from scr.hetero_graph_utils import update_graph
from scr.aba_inference import ABAInferenceEngine
from scr.causal_extension_utils import assumption_priority, inject_vstructure_rules


def build_gnn_extension(
    aba_file_path: str,
    model_type: str,
    model_path: str = None,
    enumeration_threshold: float = None,
    inference_engine=None,
    verbose: bool = True,
    reject_edge_on_indep: bool = True,
    scores_path: str = None,
    max_steps: int = 500,
) -> tuple[set, set]:
    """
    Iteratively ground assumptions using GNN predictions.

    At each step: run GNN inference on the current hetero graph, select the
    highest-probability accepted arr_*/noe_*/indep_* assumption (falling back to
    blocked_path_* only if that tier is exhausted), commit it via
    remove_accepted_assumption(), rebuild the hetero graph, and repeat.

    Stops when:
      - All non-dummy assumptions are blocked_path_ (structural stop), or
      - No non-dummy, non-blocked-path assumption is accepted by the GNN, or
      - remove_accepted_assumption() returns False (structural rejection).

    Returns (extension, all_assumptions).
    """
    if inference_engine is None:
        if model_path is None:
            model_path = f"results_final_{model_type}/trained_model.pt"
        inference_engine = ABAInferenceEngine(model_type, model_path, enumeration_threshold)

    # Load CI-test scores for scored models (3-feature graphs).
    atom_scores = None
    if scores_path is not None:
        try:
            import json as _json
            with open(scores_path) as _f:
                raw = _json.load(_f)
            atom_scores = {k: v for k, v in raw.items() if "|" not in k}
        except Exception:
            atom_scores = None

    dep_graph = DependencyGraph(reject_edge_on_indep=reject_edge_on_indep)
    dep_graph.create_from_file(aba_file_path)
    all_assumptions = dep_graph.assumptions.copy()

    if verbose:
        print(f"\n[gnn] injecting v-structure rules ...")
    inject_vstructure_rules(dep_graph, verbose=verbose)
    dep_graph._init_indices()

    dep_graph.create_dependency_graph()
    hetero_graph, dep_graph, assmpt_mapping = update_graph(dep_graph, atom_scores=atom_scores)

    extension: set[str] = set()
    step = 0

    while dep_graph.assumptions:
        step += 1
        if max_steps is not None and step > max_steps:
            print(f"[gnn] WARNING: max_steps={max_steps} reached — returning partial extension "
                  f"({len(extension)} committed, {len(dep_graph.assumptions)} assumptions remaining)")
            break
        if verbose:
            print(f"\n[step {step}] assumptions remaining: {len(dep_graph.assumptions)}")

        predictions = inference_engine.inference(hetero_graph, assmpt_mapping)

        # Filter out dummy assumptions — reduct artifacts, never eligible.
        non_dummy = [
            (name, prob, acc) for name, prob, acc in predictions
            if not name.startswith("dummy")
        ]
        if not non_dummy:
            if verbose:
                print(f"[step {step}] STOP — no real assumptions remain (only dummies)")
            break

        # Stop when only blocked_path_ assumptions remain (no CI/arrow content).
        if all(name.startswith("blocked_path_") for name, _, _ in non_dummy):
            if verbose:
                print(f"[step {step}] STOP — only blocked_path_ assumptions remain")
            break

        # Prefer arr_*/noe_*/indep_* over blocked_path_*; pure argmax, no threshold.
        preferred = [
            (name, prob) for name, prob, _ in non_dummy
            if not name.startswith("blocked_path_")
        ]

        if verbose:
            ranked = sorted(non_dummy, key=lambda t: t[1], reverse=True)
            print(f"[step {step}] GNN ranking (top non-dummy):")
            for name, prob, acc in ranked:
                status = "ACCEPTED" if acc else "rejected"
                print(f"           {prob:.4f}  {name}  [{status}]")

        best = max(preferred, key=lambda kv: assumption_priority(kv[0], kv[1]))[0]

        ok = dep_graph.remove_accepted_assumption(best)
        if not ok:
            if verbose:
                print(f"[step {step}] STOP — structural rejection on remove_accepted_assumption({best})")
            break

        extension.add(best)
        if verbose:
            print(f"[step {step}] ACCEPTED: {best}  |  extension so far: {sorted(extension)}")

        if not dep_graph.assumptions:
            break

        dep_graph.create_dependency_graph()
        hetero_graph, dep_graph, assmpt_mapping = update_graph(dep_graph, atom_scores=atom_scores)

    if verbose and not dep_graph.assumptions:
        print(f"\n[step {step}] STOP — framework exhausted")

    return extension, all_assumptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a GNN-grounded causal extension from an ABA framework."
    )
    parser.add_argument("--aba_file", type=str, required=True, help="Path to .aba file")
    parser.add_argument("--model_type", type=str, required=True, choices=["gcn", "gat"])
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model .pt file")
    parser.add_argument("--threshold", type=float, default=None, help="Acceptance threshold override")
    parser.add_argument("--no-reject-edge-on-indep", action="store_true",
                        help="Disable arr/noe removal when an indep is committed")
    args = parser.parse_args()

    start = time.time()
    extension, assumptions = build_gnn_extension(
        args.aba_file,
        model_type=args.model_type,
        model_path=args.model_path,
        enumeration_threshold=args.threshold,
        reject_edge_on_indep=not args.no_reject_edge_on_indep,
    )
    elapsed = time.time() - start

    print(f"Generation time: {elapsed:.4f}s")
    print(f"{extension=}")
    print(f"{assumptions=}")
