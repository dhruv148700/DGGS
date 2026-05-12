"""
run_single_seed_synthetic_aba.py
────────────────────────────────
Single-seed end-to-end trace for synthetic DAGs, *bypassing the GNN*:
  simulate_dag (ER / SF) → simulate_data_and_run_PC → facts_from_sepset
  → get_credulous_assumptions_from_facts (ASPforABA, all extensions)
  → save_credulous_assumptions

Mirrors run_single_seed_synthetic.py up through fact extraction, then takes
the alternative path: enumerate every stable extension via ASPforABA and emit
the union of assumptions accepted in at least one extension (credulous
acceptance) as GNN-training labels.

All orchestration helpers are imported from scr/causal_aba/abapc.py.

Place at the GNN4ABA repo root.

    python run_single_seed_synthetic_aba.py
"""

import os
import sys
import logging
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scr"))

# ─── Config ───────────────────────────────────────────────────────────────────

SEED        = 42
D           = 5          # number of nodes (keep small: extensions grow fast)
S0          = 5          # expected number of edges
ALPHA       = 0.05
GRAPH_TYPES = ["ER", "SF"]

# ─── Imports ──────────────────────────────────────────────────────────────────

from ArgCausalDisco.utils.data_utils    import simulate_dag, simulate_data_and_run_PC
from scr.causal_aba.enums               import SemanticEnum
from scr.causal_aba.abapc               import (
    facts_from_sepset,
    save_credulous_assumptions,
)

# Surface logger.info(...) from abapc so the fact-removal loop and credulous
# stats appear on stdout.
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ─── Run ──────────────────────────────────────────────────────────────────────

for graph_type in GRAPH_TYPES:

    print("=" * 70)
    print(f"Synthetic ABA trace  |  graph={graph_type}  d={D}  s0={S0}  seed={SEED}")
    print("=" * 70)

    CREDULOUS_OUT = f"synthetic_{graph_type.lower()}_credulous.txt"

    # ── [1] Simulate DAG ──────────────────────────────────────────────────────
    print("\n[1] Simulating DAG")
    print("-" * 70)

    np.random.seed(SEED)
    B_true = simulate_dag(d=D, s0=S0, graph_type=graph_type)
    n_nodes = B_true.shape[0]

    G_true = nx.from_numpy_array(B_true, create_using=nx.DiGraph)
    G_true = nx.relabel_nodes(G_true, {i: f"X{i+1}" for i in range(n_nodes)})

    print(f"  n_nodes       : {n_nodes}")
    print(f"  n edges (true): {int(B_true.sum())}")
    print(f"  Is DAG        : {nx.is_directed_acyclic_graph(G_true)}")
    print(f"  B_true:\n{B_true}")

    # ── [2] Simulate data and run PC (cg reused, no second PC call) ──────────
    print("\n[2] Simulating discrete data and running PC")
    print("-" * 70)

    data, cg = simulate_data_and_run_PC(G_true, alpha=ALPHA, seed=SEED)

    print(f"  data shape    : {data.shape}")
    print(f"  CPDAG type    : {type(cg).__name__}")

    # ── [3] Extract LP facts from cg.sepset ──────────────────────────────────
    print("\n[3] Extracting LP facts from cg.sepset")
    print("-" * 70)

    facts = facts_from_sepset(cg, n_nodes, ALPHA)

    print(f"  n facts       : {len(facts)}")
    print(f"  first 5 facts :")
    for f in facts[:5]:
        print(f"    {f}")

    # ── [4] Enumerate extensions + write credulous assumptions ────────────────
    #
    # save_credulous_assumptions wraps:
    #   get_extensions_from_facts (factory.create_solver + enumerate_extensions
    #     with fact-removal loop because PC output is noisy)
    #   → union over model.assumptions across all extensions
    #   → write sorted, one per line.
    #
    # Returns the credulous set (all assumption types: arr_*, indep, blocked_path).
    print("\n[4] Enumerating extensions and writing credulous assumptions")
    print("-" * 70)

    credulous = save_credulous_assumptions(
        facts=facts,
        n_nodes=n_nodes,
        output_path=CREDULOUS_OUT,
        semantics=SemanticEnum.ST,
    )

    arr_count   = sum(1 for a in credulous if a.startswith("arr_"))
    indep_count = sum(1 for a in credulous if a.startswith("indep"))
    bp_count    = sum(1 for a in credulous if a.startswith("blocked_path"))
    other_count = len(credulous) - arr_count - indep_count - bp_count

    print(f"  file written     : {CREDULOUS_OUT}  ({os.path.getsize(CREDULOUS_OUT)} bytes)")
    print(f"  total credulous  : {len(credulous)}")
    print(f"    arr_*          : {arr_count}")
    print(f"    indep          : {indep_count}")
    print(f"    blocked_path   : {bp_count}")
    print(f"    other          : {other_count}")
    print(f"  sample (first 15):")
    for a in sorted(credulous)[:15]:
        print(f"    {a}")

    print("\n" + "=" * 70)
    print(f"ABA trace complete for {graph_type}.")
    print("=" * 70 + "\n")
