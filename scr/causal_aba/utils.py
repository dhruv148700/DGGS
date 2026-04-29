from itertools import chain, combinations, product
from scr.causal_aba.enums import RelationEnum, Fact
from networkx.algorithms.d_separation import is_d_separator
import networkx as nx
import pandas as pd
import numpy as np
import igraph as ig

def is_unique(ary):
    return len(ary) == len(set(ary))


def powerset(s):
    s = sorted(list(s))
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def unique_product(elements, repeat: int):
    '''
    Generate all combinations of elements without duplicates.
    '''
    for element_set in product(elements, repeat=repeat):
        if is_unique(element_set):
            yield element_set

def parse_fact_line(line):
    line = line.strip()
    relation, rest = line.split('(', 1)
    rest = rest.split(')')[0]
    node1, node2, node_set = rest.split(',')
    node1 = int(node1.strip())
    node2 = int(node2.strip())
    node_set = node_set.strip()
    if node_set == 'empty':
        node_set = {}
    else:
        node_set = node_set.strip()[1:].split('y')
        node_set = {int(x) for x in node_set if x.strip()}

    return Fact(
        relation=RelationEnum[relation.strip()],
        node1=node1,
        node2=node2,
        node_set=node_set,
        score=1,
    )


def facts_from_file(filename):
    facts = []
    with open(filename, 'r') as f:
        for line in f:
            facts.append(parse_fact_line(line))
    return facts

def randomG(n_nodes, edge_per_node=2, graph_type="ER", seed=2024, mec_check=True):
    scenario = "randomG"
    output_name = f"{scenario}_{n_nodes}_{edge_per_node}_{graph_type}_{seed}"
    facts_location = f"data/{output_name}.lp"
    print(f"n_nodes={n_nodes}, edge_per_node={edge_per_node}, graph_type={graph_type}, seed={seed}")
    s0 = int(n_nodes*edge_per_node)
    if s0 > int(n_nodes*(n_nodes-1)/2):
        s0 = int(n_nodes*(n_nodes-1)/2)
    B_true = simulate_dag(d=n_nodes, s0=s0, graph_type=graph_type)
    G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
    true_seplist = find_all_d_separations_sets(G_true, verbose=False)
    with open(facts_location, "w") as f:
        for s in true_seplist:
            f.write(s + "\n")

### From notears repo: https://github.com/xunzheng/notears
def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    return B_perm


def find_all_d_separations_sets(G, verbose=True, debug=False):
    no_of_var = len(G.nodes)
    septests = []
    for comb in combinations(range(no_of_var), 2):
        if comb[0] != comb[1]:
            x = comb[0]
            y = comb[1]
            depth = 0
            while no_of_var-1 > depth:
                Neigh_x_noy = [f"X{k+1}" for k in range(no_of_var) if k != x and k != y]
                for S in combinations(Neigh_x_noy, depth):
                    s = set([int(s.replace('X',''))-1 for s in S])
                    s_str = 'empty' if len(S)==0 else 's'+'y'.join([str(i) for i in s])
                    if is_d_separator(G, {f"X{x+1}"}, {f"X{y+1}"}, set(S)):
                        septests.append(f"indep({x},{y},{s_str}).")
                    else:
                        # logging.info(f"X{x+1} and X{y+1} are not d-separated by {S}")
                        septests.append(f"dep({x},{y},{s_str}).")
                depth += 1
    return septests

def get_matrix_from_arrow_set(arrow_set, n_nodes):
    """
    Get the adjacency matrix from the arrow set.
    Args:
        arrow_set: set
            The arrow set to be converted to an adjacency matrix
        n_nodes: int
            The number of nodes in the graph
    Returns:
        B_est: np.array
            The adjacency matrix of the graph
    """
    B_est = np.zeros((n_nodes, n_nodes), dtype=int)
    for node1, node2 in arrow_set:
        B_est[node1, node2] = 1
    return B_est