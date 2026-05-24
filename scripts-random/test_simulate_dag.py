"""
Test script for simulate_dag and simulate_discrete_data using ER and BA (SF) graph types.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ArgCausalDisco.utils.data_utils import simulate_dag, simulate_discrete_data

SEED = 42
np.random.seed(SEED)

D = 10   # number of nodes
S0 = 20  # expected number of edges


def edges_from_adj(B):
    return list(zip(*np.where(B == 1)))


def print_dag_stats(name, B):
    edges = edges_from_adj(B)
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    print(f"\n--- {name} ---")
    print(f"  Nodes:       {B.shape[0]}")
    print(f"  Edges:       {len(edges)}")
    print(f"  Is DAG:      {nx.is_directed_acyclic_graph(G)}")
    in_deg  = np.array([d for _, d in G.in_degree()])
    out_deg = np.array([d for _, d in G.out_degree()])
    print(f"  In-degree  : mean={in_deg.mean():.2f}, max={in_deg.max()}")
    print(f"  Out-degree : mean={out_deg.mean():.2f}, max={out_deg.max()}")
    return G


def simulate_and_print_data(name, B, sample_size=500):
    edges = edges_from_adj(B)
    if len(edges) == 0:
        print(f"\n  [{name}] No edges — skipping data simulation.")
        return None
    data = simulate_discrete_data(
        num_of_nodes=B.shape[0],
        sample_size=sample_size,
        truth_DAG_directed_edges=edges,
        random_seed=SEED,
    )
    print(f"\n  [{name}] Simulated data shape: {data.shape}")
    print(f"  Unique states per variable: {[len(np.unique(data[:, i])) for i in range(data.shape[1])]}")
    return data


np.random.seed(SEED)
B_er = simulate_dag(d=D, s0=S0, graph_type='ER')
np.random.seed(SEED + 1)
B_sf = simulate_dag(d=D, s0=S0, graph_type='SF')

G_er = print_dag_stats("Erdos-Renyi (ER)", B_er)
G_sf = print_dag_stats("Barabasi-Albert (SF)", B_sf)

data_er = simulate_and_print_data("ER", B_er, sample_size=1000)
data_sf = simulate_and_print_data("SF/BA", B_sf, sample_size=1000)

# --- Plot adjacency matrices and degree distributions ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, B, title in zip(axes[0], [B_er, B_sf], ["ER Adjacency", "BA (SF) Adjacency"]):
    ax.imshow(B, cmap='Blues', interpolation='none')
    ax.set_title(title)
    ax.set_xlabel("To node")
    ax.set_ylabel("From node")
    plt.colorbar(ax.images[0], ax=ax)

for ax, G, title in zip(axes[1], [G_er, G_sf], ["ER In-degree dist.", "BA (SF) In-degree dist."]):
    in_degs = [d for _, d in G.in_degree()]
    ax.hist(in_degs, bins=range(0, max(in_degs) + 2), align='left', rwidth=0.8, color='steelblue')
    ax.set_title(title)
    ax.set_xlabel("In-degree")
    ax.set_ylabel("Count")

plt.tight_layout()
out_path = "figures/test_simulate_dag.png"
plt.savefig(out_path, dpi=120)
print(f"\nPlot saved to {out_path}")
plt.show()
