"""
run_single_seed_synthetic.py
────────────────────────────
Single-seed end-to-end trace for synthetic DAGs:
  simulate_dag (ER / BA) → simulate_discrete_data → PC (reuse cg)
  → LP facts → .aba → GNN → B_est → metrics

Place at the GNN4ABA repo root.

    python run_single_seed_synthetic.py
"""

import os
import sys
import re
import numpy as np
import networkx as nx
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Config ───────────────────────────────────────────────────────────────────

SEED        = 42
D           = 4       # number of nodes (keep small: CausalABA assumptions grow exponentially)
S0          = 5       # expected number of edges
SAMPLE_SIZE = 10_000
ALPHA       = 0.05
GRAPH_TYPES = ["ER", "SF"]   # (graph_type arg)

# ─── Imports ──────────────────────────────────────────────────────────────────

from ArgCausalDisco.utils.data_utils    import simulate_dag, simulate_data_and_run_PC
from ArgCausalDisco.utils.graph_utils   import initial_strength, DAGMetrics
from scr.causal_aba.enums               import Fact, RelationEnum
from scr.causal_aba.lp_to_aba_translator import lp_facts_to_aba_file
from scr.causal_aba.utils               import get_matrix_from_arrow_set
from scr.extension_generator            import build_extension

# ─── Helper: extract LP facts from an existing cg (Option 2, no PC re-run) ───

def facts_from_sepset(cg, n_nodes, alpha):
    facts = []
    for node1, node2 in combinations(range(n_nodes), 2):
        for sep_set, p in cg.sepset[node1, node2]:
            dep_type = "indep" if p > alpha else "dep"
            score    = initial_strength(p, len(sep_set), alpha, 0.5, n_nodes)
            fact     = Fact(
                relation=RelationEnum(dep_type),
                node1=node1, node2=node2,
                node_set=set(sep_set), score=score,
            )
            if fact not in facts:
                facts.append(fact)
    return facts

# ─── Run ──────────────────────────────────────────────────────────────────────

for graph_type in GRAPH_TYPES:

    print("=" * 70)
    print(f"Synthetic trace  |  graph={graph_type}  d={D}  s0={S0}  seed={SEED}")
    print("=" * 70)

    ABA_OUT = f"synthetic_{graph_type.lower()}.aba"

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

    # ── [2] Simulate data and run PC (cg reused, no second PC call) ───────────
    print("\n[2] Simulating discrete data and running PC")
    print("-" * 70)

    data, cg = simulate_data_and_run_PC(G_true, alpha=ALPHA, seed=SEED)

    print(f"  data shape    : {data.shape}")
    print(f"  CPDAG type    : {type(cg).__name__}")

    # ── [3] Extract LP facts from cg.sepset ────────────────────────
    print("\n[3] Extracting LP facts from cg.sepset")
    print("-" * 70)

    facts = facts_from_sepset(cg, n_nodes, ALPHA)

    print(f"  n facts       : {len(facts)}")
    print(f"  first 5 facts :")
    for f in facts[:5]:
        print(f"    {f}")

    # ── [4] Write .aba file ───────────────────────────────────────────────────
    print("\n[4] Writing .aba file")
    print("-" * 70)

    lp_facts_to_aba_file(facts, n_nodes=n_nodes, out_path=ABA_OUT, optimise_remove_edges=True)

    aba_size = os.path.getsize(ABA_OUT)
    with open(ABA_OUT) as fh:
        aba_lines = fh.readlines()

    print(f"  file written  : {ABA_OUT}  ({aba_size} bytes, {len(aba_lines)} lines)")
    print(f"  first 15 lines:")
    for line in aba_lines[:15]:
        print(f"    {line.rstrip()}")

    arr_count   = sum(1 for ln in aba_lines if ln.startswith("a arr_"))
    noe_count   = sum(1 for ln in aba_lines if ln.startswith("a noe_"))
    indep_count = sum(1 for ln in aba_lines if ln.startswith("a indep_"))
    bp_count    = sum(1 for ln in aba_lines if ln.startswith("a blocked_path_"))
    print(f"  assumption counts: arr={arr_count}  noe={noe_count}  "
          f"indep={indep_count}  blocked_path={bp_count}")

    # ── [5] GNN inference ─────────────────────────────────────────────────────
    print("\n[5] Running GNN inference")
    print("-" * 70)

    extension, all_assumptions = build_extension(
        aba_file_path=ABA_OUT,
        enumeration_threshold=None,
        model_type="gcn",
    )

    print(f"  total assumptions: {len(all_assumptions)}")
    print(f"  extension size   : {len(extension)}")
    print(f"  extension sample (first 15):")
    for a in list(extension)[:15]:
        print(f"    {a}")

    # ── [6] Build B_est ───────────────────────────────────────────────────────
    print("\n[6] Building B_est")
    print("-" * 70)

    arr_pattern = re.compile(r"^arr_(\d+)_(\d+)$")
    arrow_set   = set()
    for name in extension:
        m = arr_pattern.match(name)
        if m:
            arrow_set.add((int(m.group(1)), int(m.group(2))))

    print(f"  n arrow tuples  : {len(arrow_set)}")
    print(f"  arrows          : {sorted(arrow_set)}")

    if not arrow_set:
        print("\n  !!! WARNING: no arr_* assumptions accepted.")
        suspicious = [a for a in extension if 'arr' in a.lower()][:10]
        if suspicious:
            print(f"      Arrow-like names: {suspicious}")

    B_est = get_matrix_from_arrow_set(arrow_set, n_nodes)
    print(f"  B_est shape     : {B_est.shape}")
    print(f"  B_est:\n{B_est}")
    print(f"  n edges (est)   : {int((B_est != 0).sum())}")

    # ── [7] Metrics ───────────────────────────────────────────────────────────
    print("\n[7] Computing metrics")
    print("-" * 70)

    metrics      = DAGMetrics(B_est, B_true).metrics
    n_true_edges = max(int(B_true.sum()), 1)

    print(f"  TPR / recall    : {metrics.get('tpr'):.3f}")
    print(f"  precision       : {metrics.get('precision'):.3f}")
    print(f"  F1              : {metrics.get('F1'):.3f}")
    print(f"  SHD             : {metrics.get('shd')}")
    print(f"  SID             : {metrics.get('sid')}")
    print(f"  SID normalised  : {metrics.get('sid') / n_true_edges:.3f}")

    print("\n" + "=" * 70)
    print(f"Trace complete for {graph_type}.")
    print("=" * 70 + "\n")
