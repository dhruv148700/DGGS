"""
check_determinism.py
--------------------
Run the simulate_dag → simulate_data_and_run_PC pipeline twice with
identical params/seeds and verify all outputs match.

Usage:
    python check_determinism.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scr"))

import numpy as np
import networkx as nx

from ArgCausalDisco.utils.data_utils import simulate_dag, simulate_data_and_run_PC
from ArgCausalDisco.utils.helpers import random_stability

N       = 5
DENSITY = 0.5
ALPHA   = 0.1
SEED    = 42

S0 = max(N - 1, int(DENSITY * N * (N - 1) / 2))   # mirrors er_s0()


def run_once():
    random_stability(SEED)
    B = simulate_dag(d=N, s0=S0, graph_type="ER")
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, {i: f"X{i+1}" for i in range(N)})
    data, cg = simulate_data_and_run_PC(G, alpha=ALPHA, seed=SEED)
    return B, data, cg.G.graph.copy()


print(f"Params: n={N}, density={DENSITY}, s0={S0}, alpha={ALPHA}, seed={SEED}\n")

B1, data1, pc1 = run_once()
B2, data2, pc2 = run_once()

dag_ok  = np.array_equal(B1, B2)
data_ok = np.array_equal(data1, data2)
pc_ok   = np.array_equal(pc1, pc2)

print(f"DAG adjacency matrix identical : {dag_ok}")
print(f"Simulated data identical       : {data_ok}")
print(f"PC skeleton/orientation matrix : {pc_ok}")

if dag_ok and data_ok and pc_ok:
    print("\nPASS — pipeline is fully deterministic.")
else:
    print("\nFAIL — non-determinism detected.")
    if not dag_ok:
        print("\nDAG run 1:\n", B1)
        print("DAG run 2:\n", B2)
    if not data_ok:
        n_diff = int((data1 != data2).sum())
        print(f"\nData: {n_diff} of {data1.size} cells differ")
    if not pc_ok:
        print("\nPC graph run 1:\n", pc1)
        print("PC graph run 2:\n", pc2)
