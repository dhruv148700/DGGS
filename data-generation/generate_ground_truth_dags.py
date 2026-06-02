"""
generate_ground_truth_dags.py
─────────────────────────────
Re-simulate every ground-truth DAG from the causal data generation grid and
save each B_true as a .npy file in dag_ground_truth/.

Reuses the identical seed formula and parameters from generate_data_causal.py
so every B_true is bit-for-bit reproducible.

Output filename per instance (no role suffix — B_true is shared across probes):
    dag_ground_truth/dag_{graph_type}_n{n}_{d_str}_{m_str}_a{alpha}_i{i}.npy

    python generate_ground_truth_dags.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_data_causal import (
    BASE_SEED,
    SAMPLES_PER_CELL,
    _CELL_OFFSETS,
    er_s0,
    iter_cells,
)
from ArgCausalDisco.utils.data_utils import simulate_dag
from ArgCausalDisco.utils.helpers import random_stability

OUTPUT_DIR = "dag_ground_truth"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def dag_fname(graph_type, n, density, m, alpha, i):
    d_str = f"d{density}" if density is not None else "dna"
    m_str = f"m{m}"       if m       is not None else "mna"
    return f"dag_{graph_type}_n{n}_{d_str}_{m_str}_a{alpha}_i{i}.npy"


total = 0
skipped = 0

for graph_type, n, density, m, alpha in iter_cells():
    density_or_m = density if graph_type == "er" else m
    dag_kind = "ER" if graph_type == "er" else "SF"
    s0 = er_s0(n, density_or_m) if graph_type == "er" else int(density_or_m) * n

    for i in range(SAMPLES_PER_CELL):
        out_path = os.path.join(OUTPUT_DIR, dag_fname(graph_type, n, density, m, alpha, i))

        if os.path.exists(out_path):
            skipped += 1
            continue

        seed = BASE_SEED + _CELL_OFFSETS[(graph_type, n, density_or_m, alpha)] + i
        random_stability(seed)
        B_true = simulate_dag(d=n, s0=s0, graph_type=dag_kind)
        np.save(out_path, B_true)
        total += 1

print(f"Done. Written: {total}  Skipped (already exist): {skipped}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
