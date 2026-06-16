"""
causal_extension_utils.py — shared utilities for causal extension generators.

Used by both dggs_extension_generator.py and gnn_causal_extension_generator.py.
"""

import re


_indep_asm_re  = re.compile(r"^indep_(\d+)_(\d+)__(.*)$")
_indep_fact_re = re.compile(r"^-indep_(\d+)_(\d+)__(.*)$")


def _parse_indep_s(s_str: str) -> frozenset:
    """Parse the conditioning-set portion of an indep name into a frozenset of node strings."""
    return frozenset(s_str.split("_")) - {""}


def assumption_priority(name: str, score: float) -> tuple:
    """
    Selection key for argmax over non-dummy assumptions.

    Primary:   score (higher wins unconditionally).
    Secondary: arr_*/noe_*/indep_* (tier 1) > everything else (tier 0).
    Tertiary:  arr_ beats noe_ within the arr/noe tier.
    Quaternary: lexicographic name for determinism.

    Rationale: indep_* must be committed before v-structure rules fire;
    blocked_path_* are auxiliary with no direct evidential content.
    """
    if name.startswith("indep_") or name.startswith("arr_") or name.startswith("noe_"):
        tier = 1
    else:
        tier = 0
    arr_bias = 1 if name.startswith("arr_") else 0
    return (tier, score, arr_bias, name)


def inject_vstructure_rules(dep_graph, verbose: bool) -> int:
    """
    Inject v-structure support rules into dep_graph after file load.

    Generalised criterion: if indep_X_Y__S is an assumption (X⊥Y|S) and
    -indep_X_Y__S_Z is an empty-body fact head (X⊬Y|S∪{Z}), then Z is a
    collider on a path between X and Y given S.  Inject:
        arr_X_Z <- indep_X_Y__S, -indep_X_Y__S_Z
        arr_Y_Z <- indep_X_Y__S, -indep_X_Y__S_Z

    Handles any conditioning set size (S=∅ is the marginal case).

    Mutates dep_graph.rules directly; caller must call dep_graph._init_indices()
    afterwards to rebuild inverted indices.
    """
    # (X, Y, frozenset(S)) -> assumption name
    indep_asms: dict[tuple, str] = {}
    for asm in dep_graph.assumptions:
        m = _indep_asm_re.match(asm)
        if m:
            key = (m.group(1), m.group(2), _parse_indep_s(m.group(3)))
            indep_asms[key] = asm

    # (X, Y, frozenset(S)) -> fact head name, from empty-body rules only
    indep_facts: dict[tuple, str] = {}
    for _, (head, body) in dep_graph.rules.items():
        if body:
            continue
        m = _indep_fact_re.match(head)
        if m:
            key = (m.group(1), m.group(2), _parse_indep_s(m.group(3)))
            indep_facts[key] = head

    # Group facts by (X, Y) to avoid O(n_asms * n_facts) cross-product
    facts_by_xy: dict[tuple, list] = {}
    for (x, y, s_neg), dep_head in indep_facts.items():
        facts_by_xy.setdefault((x, y), []).append((s_neg, dep_head))

    next_idx = max(dep_graph.rules.keys()) + 1
    n_injected = 0
    for (x, y, s_pos), indep_asm in indep_asms.items():
        for s_neg, dep_head in facts_by_xy.get((x, y), []):
            if len(s_neg) != len(s_pos) + 1:
                continue
            diff = s_neg - s_pos
            if len(diff) != 1:
                continue
            z = next(iter(diff))
            body = sorted([indep_asm, dep_head])
            for arr in (f"arr_{x}_{z}", f"arr_{y}_{z}"):
                if arr not in dep_graph.assumptions:
                    continue
                dep_graph.rules[next_idx] = (arr, body)
                if verbose:
                    print(f"[vstructure] +rule  {arr} <- {body}  (collider: {z}, S={sorted(s_pos)})")
                next_idx += 1
                n_injected += 1

    if verbose:
        print(f"[vstructure] injected {n_injected} rules from "
              f"{len(indep_asms)} indep assumptions, "
              f"{len(indep_facts)} dep facts")
    return n_injected
