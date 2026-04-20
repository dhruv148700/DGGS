# lp_to_aba_translator code

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

import scr.causal_aba.atoms as atoms
import scr.causal_aba.assumptions as assums
from scr.causal_aba.enums import Fact, RelationEnum
from scr.causal_aba.utils import unique_product, powerset

Rule = Tuple[str, List[str]]  # (head, body_tokens)

@dataclass
class ABAFramework:
    assumptions: Set[str] = field(default_factory=set)
    contrary: Dict[str, str] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)

    def all_elements(self) -> Set[str]:
        elems: Set[str] = set()
        elems |= set(self.assumptions)
        elems |= set(self.contrary.keys())
        elems |= set(self.contrary.values())
        for h, body in self.rules:
            elems.add(h)
            elems |= set(body)
        return elems


class _ABATextCollector:
    def __init__(self) -> None:
        self.framework = ABAFramework()
        self._rule_set: Set[Tuple[str, Tuple[str, ...]]] = set()  # normalized dedupe

    @staticmethod
    def _tok(x) -> str:
        return str(x)

    def add_assumption(self, a) -> None:
        self.framework.assumptions.add(self._tok(a))

    def add_contrary(self, a, c) -> None:
        a_tok, c_tok = self._tok(a), self._tok(c)
        self.framework.contrary[a_tok] = c_tok

    def add_rule(self, head, body: Sequence) -> None:
        h = self._tok(head)
        body_toks = [self._tok(b) for b in body]

        # Mirror GNN loader behavior: body dedupe via set (orderless)
        body_toks = list(set(body_toks))

        # Dedupe rules robustly (since body order is irrelevant after set())
        key = (h, tuple(sorted(body_toks)))
        if key in self._rule_set:
            return
        self._rule_set.add(key)

        self.framework.rules.append((h, body_toks))


class CoreToABABuilder:
    """
    Re-implements CoreABASPSolverFactory.create_core_solver(...) logic,
    but writes to _ABATextCollector instead of an ABASolver.
    """
    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes

    @staticmethod
    def _add_graph_edge_assumptions(collector: _ABATextCollector, X, Y) -> None:
        for assumption in [assums.arr(X, Y), assums.arr(Y, X), assums.noe(X, Y)]:
            collector.add_assumption(assumption)
            collector.add_contrary(assumption, assums.contrary(assumption))

        for assumption1, assumption2 in unique_product(
            [assums.arr(X, Y), assums.arr(Y, X), assums.noe(X, Y)], repeat=2
        ):
            collector.add_rule(assums.contrary(assumption2), [assumption1])

        collector.add_rule(atoms.dpath(X, Y), [assums.arr(X, Y)])
        collector.add_rule(atoms.dpath(Y, X), [assums.arr(Y, X)])

        collector.add_rule(atoms.edge(X, Y), [assums.arr(X, Y)])
        collector.add_rule(atoms.edge(X, Y), [assums.arr(Y, X)])

    @staticmethod
    def _add_acyclicity_rules(collector: _ABATextCollector, X, Y) -> None:
        collector.add_rule(assums.contrary(assums.arr(Y, X)), [atoms.dpath(X, Y)])

    @staticmethod
    def _add_non_blocking_rules(collector: _ABATextCollector, X, Y, S, n_nodes: int) -> None:
        for N in S:
            if N not in {X, Y}:
                collector.add_rule(atoms.non_blocking(N, X, Y, S), [atoms.collider(X, N, Y)])

        for N in set(range(n_nodes)) - set(S):
            if N not in {X, Y}:
                collector.add_rule(atoms.non_blocking(N, X, Y, S), [atoms.not_collider(X, N, Y)])

                for Z in S:
                    if Z not in {X, Y, N}:
                        collector.add_rule(
                            atoms.non_blocking(N, X, Y, S),
                            [atoms.collider(X, N, Y), atoms.descendant_of_collider(Z, X, N, Y)],
                        )

    @staticmethod
    def _add_direct_path_definition_rules(collector: _ABATextCollector, X, Y, Z) -> None:
        collector.add_rule(atoms.dpath(X, Y), [assums.arr(X, Z), atoms.dpath(Z, Y)])

    @staticmethod
    def _add_collider_definition_rules(collector: _ABATextCollector, X, Y, Z) -> None:
        collector.add_rule(atoms.collider(X, Y, Z), [assums.arr(X, Y), assums.arr(Z, Y)])

        collector.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(X, Y), assums.arr(Y, Z)])
        collector.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(Y, X), assums.arr(Y, Z)])
        collector.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(Z, Y), assums.arr(Y, X)])

    @staticmethod
    def _add_collider_descendant_definition_rules(collector: _ABATextCollector, X, Y, Z, N) -> None:
        collector.add_rule(
            atoms.descendant_of_collider(N, X, Y, Z),
            [atoms.collider(X, Y, Z), atoms.dpath(Y, N)],
        )

    def build_core(self, collector: _ABATextCollector, edges_to_remove: Set[FrozenSet[int]]) -> None:
        # Mirrors CoreABASPSolverFactory.create_core_solver

        for X, Y in unique_product(range(self.n_nodes), repeat=2):
            if frozenset({X, Y}) in edges_to_remove:
                continue

            if X < Y:
                self._add_graph_edge_assumptions(collector, X, Y)

            self._add_acyclicity_rules(collector, X, Y)

        for X, Y, Z in unique_product(range(self.n_nodes), repeat=3):
            if frozenset({X, Z}) not in edges_to_remove:
                self._add_direct_path_definition_rules(collector, X, Y, Z)

            if frozenset({Y, X}) not in edges_to_remove and frozenset({Y, Z}) not in edges_to_remove:
                if X < Z:
                    self._add_collider_definition_rules(collector, X, Y, Z)

                    for N in range(self.n_nodes):
                        if N not in {X, Y, Z}:
                            self._add_collider_descendant_definition_rules(collector, X, Y, Z, N)

        for X, Y in combinations(range(self.n_nodes), 2):
            for S in powerset(range(self.n_nodes)):
                self._add_non_blocking_rules(collector, X, Y, S, self.n_nodes)


class LPToABATranslator:
    """
    Full translation:
      - core factory content
      - plus ABASPSolverFactory fact-dependent content
    """
    def __init__(self, n_nodes: int, optimise_remove_edges: bool = True) -> None:
        self.n_nodes = n_nodes
        self.optimise_remove_edges = optimise_remove_edges
        self._core_builder = CoreToABABuilder(n_nodes=n_nodes)

    # ---- mirrors ABASPSolverFactory methods, but writes to collector ----

    @staticmethod
    def _add_path_definition_rules(collector: _ABATextCollector, paths, X, Y) -> None:
        for path_id, my_path in enumerate(paths):
            collector.add_rule(
                atoms.path(X, Y, path_id),
                [atoms.edge(my_path[i], my_path[i + 1]) for i in range(len(my_path) - 1)],
            )

    @staticmethod
    def _add_indep_assumptions(collector: _ABATextCollector, X, Y, S) -> None:
        collector.add_assumption(assums.indep(X, Y, S))
        collector.add_contrary(assums.indep(X, Y, S), assums.contrary(assums.indep(X, Y, S)))

    @staticmethod
    def _add_independence_rules(collector: _ABATextCollector, paths, X, Y, S) -> None:
        indep_body = [assums.blocked_path(X, Y, path_id, S) for path_id in range(len(paths))]
        collector.add_rule(assums.indep(X, Y, S), indep_body)

    @staticmethod
    def _add_blocked_path_assumptions(collector: _ABATextCollector, path_id, X, Y, S) -> None:
        collector.add_assumption(assums.blocked_path(X, Y, path_id, S))
        collector.add_contrary(
            assums.blocked_path(X, Y, path_id, S),
            assums.contrary(assums.blocked_path(X, Y, path_id, S)),
        )

    @staticmethod
    def _add_active_path_rules(collector: _ABATextCollector, path_id, path_nodes, X, Y, S) -> None:
        non_blocking_body = [
            atoms.non_blocking(path_nodes[i], path_nodes[i - 1], path_nodes[i + 1], S)
            for i in range(1, len(path_nodes) - 1)
        ]
        collector.add_rule(
            assums.contrary(assums.blocked_path(X, Y, path_id, S)),
            [atoms.path(X, Y, path_id), *non_blocking_body],
        )

    @staticmethod
    def _add_dependence_rules(collector: _ABATextCollector, path_id, X, Y, S) -> None:
        collector.add_rule(
            assums.contrary(assums.indep(X, Y, S)),
            [assums.contrary(assums.blocked_path(X, Y, path_id, S))],
        )

    @staticmethod
    def _add_fact(collector: _ABATextCollector, fact: Fact) -> None:
        if fact.relation == RelationEnum.dep:
            collector.add_rule(assums.contrary(assums.indep(fact.node1, fact.node2, fact.node_set)), [])
        elif fact.relation == RelationEnum.indep:
            collector.add_rule(assums.indep(fact.node1, fact.node2, fact.node_set), [])

    def translate_facts(self, facts: List[Fact]) -> ABAFramework:
        collector = _ABATextCollector()

        # compute edges_to_remove exactly as in ABASPSolverFactory._create_solver
        edges_to_remove: Set[FrozenSet[int]] = set()
        if self.optimise_remove_edges:
            for fact in facts:
                if fact.relation == RelationEnum.indep:
                    edges_to_remove.add(frozenset({fact.node1, fact.node2}))

        # 1) core rules/assumptions
        self._core_builder.build_core(collector, edges_to_remove)

        # 2) fact-dependent additions (paths, indep assumptions, blocked_path, etc.)
        graph = nx.complete_graph(self.n_nodes)
        if self.optimise_remove_edges:
            graph.remove_edges_from({tuple(e) for e in edges_to_remove})  # safe conversion

        all_paths: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}
        for fact in facts:
            X, Y = fact.node1, fact.node2
            if (X, Y) not in all_paths:
                all_paths[(X, Y)] = [tuple(p) for p in nx.all_simple_paths(graph, source=X, target=Y)]

        for fact in facts:
            X, Y, S = fact.node1, fact.node2, fact.node_set
            paths = all_paths.get((X, Y))
            if paths is None:
                raise RuntimeError(f"No paths found between {X} and {Y} in the graph.")

            self._add_path_definition_rules(collector, paths, X, Y)
            self._add_indep_assumptions(collector, X, Y, S)

            for path_id, my_path in enumerate(paths):
                self._add_blocked_path_assumptions(collector, path_id, X, Y, S)
                self._add_active_path_rules(collector, path_id, my_path, X, Y, S)
                self._add_dependence_rules(collector, path_id, X, Y, S)

            self._add_independence_rules(collector, paths, X, Y, S)
            self._add_fact(collector, fact)

        return collector.framework


def write_aba_file(framework: ABAFramework, out_path: str) -> None:
    all_elems = framework.all_elements()
    n = len(all_elems)

    lines: List[str] = [f"p aba {n}"]

    for a in sorted(framework.assumptions):
        lines.append(f"a {a}")

    for a in sorted(framework.contrary.keys()):
        lines.append(f"c {a} {framework.contrary[a]}")

    for head, body in framework.rules:
        if body:
            lines.append("r " + " ".join([head, *body]))
        else:
            lines.append("r " + head)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def lp_facts_to_aba_file(
    facts: List[Fact],
    *,
    n_nodes: int,
    out_path: str,
    optimise_remove_edges: bool = True,
) -> None:
    translator = LPToABATranslator(n_nodes=n_nodes, optimise_remove_edges=optimise_remove_edges)
    fw = translator.translate_facts(facts)
    write_aba_file(fw, out_path)