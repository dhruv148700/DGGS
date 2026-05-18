"""
abapc.py
────────
Training-data generation pipeline:

  simulate_dag → simulate_data_and_run_PC → facts_from_sepset
  → get_extensions_from_facts (fact-removal loop + enumeration)
  → get_credulous_assumptions_from_facts
  → save_credulous_assumptions / generate_training_sample

Contrast with run_single_seed_synthetic.py (inference pipeline), which goes:
  simulate_dag → simulate_data_and_run_PC → facts_from_sepset
  → lp_facts_to_aba_file → build_extension (GNN)

Here we enumerate ALL extensions via ASPforABA and collect the union of
accepted assumptions across extensions (credulous acceptance) as GNN labels.
"""

import logging
from itertools import combinations

import networkx as nx

from ArgCausalDisco.cd_algorithms.PC import pc
from ArgCausalDisco.utils.data_utils import simulate_dag, simulate_data_and_run_PC
from ArgCausalDisco.utils.graph_utils import initial_strength
from ArgCausalDisco.utils.helpers import random_stability
from scr.causal_aba.enums import Fact, RelationEnum, SemanticEnum
from scr.causal_aba.factory import ABASPSolverFactory

logger = logging.getLogger(__name__)

def get_cg_and_facts(data,
                     alpha=0.01,
                     indep_test='gsq',
                     uc_rule=5,
                     stable=True):
    n_nodes = data.shape[1]
    cg = pc(data=data, alpha=alpha, indep_test=indep_test, uc_rule=uc_rule,
            stable=stable, show_progress=False, verbose=False)
    facts = []

    for node1, node2 in combinations(range(n_nodes), 2):
        test_PC = [t for t in cg.sepset[node1, node2]]
        for sep_set, p in test_PC:
            dep_type_PC = "indep" if p > alpha else "dep"
            init_strength_value = initial_strength(p, len(sep_set), alpha, 0.5, n_nodes)

            fact = Fact(
                relation=RelationEnum(dep_type_PC),
                node1=node1,
                node2=node2,
                node_set=set(sep_set),
                score=init_strength_value
            )

            if fact not in facts:
                facts.append(fact)
    return cg, facts

# ─── Step 1: Extract facts from cg.sepset ─────────────────────────────────────
#
# This mirrors get_cg_and_facts from the original abapc but does NOT re-run PC.
# It reads the CI test trace that PC already populated into cg.sepset[i, j]:
#   - each entry is (sep_set, p) where sep_set is the conditioning set S
#     and p is the p-value from the CI test
#   - classifies as indep/dep based on p vs alpha
#   - computes initial_strength so the fact-removal loop knows what to drop first
#
# You get one Fact per (node1, node2, sep_set) triple — NOT one per pair.
# PC runs many CI tests per pair; the ABA encoding wants the full trace.

def facts_from_sepset(cg, n_nodes, alpha=0.05):
    """
    Extract Fact objects from cg.sepset populated by PC.

    Args:
        cg:      CausalGraph object returned by simulate_data_and_run_PC
        n_nodes: int, number of nodes
        alpha:   float, significance level used during PC

    Returns:
        facts: list of Fact objects, one per (node1, node2, sep_set) CI test
    """
    facts = []
    for node1, node2 in combinations(range(n_nodes), 2):
        for sep_set, p in cg.sepset[node1, node2]:
            dep_type = "indep" if p > alpha else "dep"

            # initial_strength encodes how much to trust this fact:
            #   - p far from alpha → stronger (more trustworthy)
            #   - large conditioning set → weaker (less data per cell)
            # Used by the fact-removal loop to decide what to drop first.
            score = initial_strength(p, len(sep_set), alpha, 0.5, n_nodes)

            fact = Fact(
                relation=RelationEnum(dep_type),
                node1=node1,
                node2=node2,
                node_set=set(sep_set),
                score=score,
            )
            if fact not in facts:
                facts.append(fact)
    return facts

# ─── Step 2: Enumerate extensions with fact-removal loop ──────────────────────
#
# PC on real/simulated data frequently produces contradictory facts, e.g.:
#   dep(X, Y, {})       — marginal test says dependent
#   indep(X, Y, {Z})    — conditional test says independent given Z
#
# These can't both hold under any DAG. The ABA solver then finds no valid
# extensions (unsatisfiable encoding). The recovery strategy is:
#   sort facts by strength descending → drop weakest → retry
#
# The sort key (score, node1, node2, str(sorted(node_set))) is NOT cosmetic:
# it makes the drop order fully deterministic even when scores tie, so two
# runs on the same data always drop the same facts in the same order.
# This is essential for reproducible GNN training data.

def get_extensions_from_facts(facts, n_nodes, semantics=SemanticEnum.ST):
    """
    Build the ABA encoding from facts and enumerate all stable extensions,
    dropping the weakest fact and retrying if the encoding is unsatisfiable.

    Args:
        facts:     list of Fact objects from facts_from_sepset
        n_nodes:   int, number of nodes
        semantics: SemanticEnum, ABA semantics to use (default: stable)

    Returns:
        models:    list of AssumptionSet extensions from ASPforABA
        fact_idx:  int, how many facts were used (len(facts) if no removal needed)
    """
    # Sort descending by strength so we drop from the end (weakest first).
    # Tiebreaker on (node1, node2, node_set) makes order fully deterministic.
    sorted_facts = sorted(
        facts,
        key=lambda x: (x.score, x.node1, x.node2, str(sorted(list(x.node_set)))),
        reverse=True,
    )

    factory = ABASPSolverFactory(n_nodes=n_nodes)
    fact_idx = len(sorted_facts)

    while fact_idx > 0:
        # factory.create_solver is non-skippable: it builds the full ABA encoding.
        # For each fact (X, Y, S) it finds all simple paths X→Y in the graph,
        # and per path adds path/blocked_path/non_blocking rules plus the indep
        # assumption with its contrary. This is the d-separation→ABA encoding.
        solver = factory.create_solver(sorted_facts[:fact_idx])

        # enumerate_extensions is the ASPforABA call.
        # k caps the number of extensions — with >6 nodes they can blow up.
        models = solver.enumerate_extensions(semantics.value, k=50000)
        del solver

        only_empty_model = (
            models is not None
            and len(models) == 1
            and len(models[0].assumptions) == 0
        )
        # Valid result: at least one non-empty extension found
        break_condition = (
            models is not None
            and len(models) > 0
            and not only_empty_model
        )
        if break_condition:
            break

        fact_idx -= 1
        logger.info(f"Encoding unsatisfiable, retrying with top {fact_idx} facts")

    if fact_idx <= 0:
        raise RuntimeError(
            "No satisfiable extension found even after dropping all facts. "
            "Check your data or alpha setting."
        )

    return models, fact_idx

# ─── Step 3: Collect credulously accepted assumptions ─────────────────────────
#
# Credulous acceptance: an assumption is accepted if it appears in AT LEAST
# ONE extension. We union model.assumptions across all extensions.
#
# model.assumptions contains ALL assumption types:
#   arr_i_j         — directed edge i→j
#   indep(x,y,S)    — independence claim
#   blocked_path... — per-path blocking
#
# We return the full union here. Filter on startswith("arr_") downstream
# if you only want directed-edge assumptions as GNN labels.

def get_credulous_assumptions_from_facts(facts, n_nodes, semantics=SemanticEnum.ST):
    """
    Enumerate all extensions and return the set of credulously accepted
    assumptions (union over all extensions).

    Args:
        facts:     list of Fact objects
        n_nodes:   int, number of nodes
        semantics: SemanticEnum

    Returns:
        credulous: set of assumption name strings
        models:    list of AssumptionSet extensions (for downstream use)
        fact_idx:  int, number of facts used after removal loop
    """
    models, fact_idx = get_extensions_from_facts(facts, n_nodes, semantics)

    credulous = set()
    for model in models:
        credulous.update(model.assumptions)

    logger.info(
        f"Credulous assumptions: {len(credulous)} total "
        f"({sum(1 for a in credulous if a.startswith('arr_'))} arr_*) "
        f"from {len(models)} extensions using {fact_idx} facts"
    )
    return credulous, models, fact_idx

# ─── Step 4: Full training sample generator ───────────────────────────────────
#
# Orchestrates one complete pipeline run for a single DAG:
#   simulate_dag → simulate_data_and_run_PC → facts_from_sepset
#   → get_credulous_assumptions_from_facts
#
# seed is applied via random_stability before PC so the CI tests are
# reproducible. Log the seed alongside the output for traceability.

def generate_training_sample(
    d=5,
    s0=5,
    graph_type="ER",
    alpha=0.05,
    seed=42,
    semantics=SemanticEnum.ST,
):
    """
    Generate one training sample: a DAG and its credulously accepted assumptions.

    Args:
        d:           int, number of nodes
        s0:          int, expected number of edges
        graph_type:  str, "ER" or "SF"
        alpha:       float, significance level for PC
        seed:        int, random seed (log this for reproducibility)
        semantics:   SemanticEnum, ABA semantics

    Returns:
        B_true:      np.ndarray (d, d), ground-truth adjacency matrix
        credulous:   set of credulously accepted assumption strings
        models:      list of AssumptionSet extensions
        fact_idx:    int, facts used after removal loop
    """
    random_stability(seed)
    logger.info(f"Generating training sample | graph={graph_type} d={d} s0={s0} seed={seed}")

    # Simulate DAG
    B_true = simulate_dag(d=d, s0=s0, graph_type=graph_type)
    n_nodes = B_true.shape[0]
    G_true = nx.from_numpy_array(B_true, create_using=nx.DiGraph)
    G_true = nx.relabel_nodes(G_true, {i: f"X{i+1}" for i in range(n_nodes)})

    # Simulate data and run PC. simulate_data_and_run_PC hardcodes 10_000
    # samples internally — no override available — so sample size is fixed.
    data, cg = simulate_data_and_run_PC(G_true, alpha=alpha, seed=seed)

    # Extract facts from cg.sepset
    facts = facts_from_sepset(cg, n_nodes, alpha)
    logger.info(f"  Extracted {len(facts)} facts from cg.sepset")

    # Enumerate extensions and collect credulous assumptions
    credulous, models, fact_idx = get_credulous_assumptions_from_facts(
        facts, n_nodes, semantics
    )

    return B_true, credulous, models, fact_idx

def save_credulous_assumptions(facts, n_nodes, output_path, semantics=SemanticEnum.ST):
    """
    Enumerate extensions for a list of Fact objects and write the credulously
    accepted assumptions (union over all extensions) to output_path, one per line.

    Args:
        facts:       list of Fact objects (e.g. from facts_from_sepset)
        n_nodes:     int, number of nodes
        output_path: str, path to write the sorted assumption list to
        semantics:   SemanticEnum

    Returns:
        credulous: set of credulously accepted assumption strings
    """
    credulous, _, _ = get_credulous_assumptions_from_facts(facts, n_nodes, semantics)
    with open(output_path, 'w') as fh:
        for assumption in sorted(credulous):
            fh.write(assumption + '\n')
    return credulous
