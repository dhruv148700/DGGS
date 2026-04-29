from typing import Dict, Iterable, Optional

from .Assumption import Assumption
from .Rule import Rule
from .Sentence import Sentence


class ABAF:
    def __init__(
        self,
        assumptions: Iterable[Assumption] = (),
        sentences: Iterable[Sentence] = (),
        rules: Iterable[Rule] = (),
    ):
        self.assumptions: set[Assumption] = set(assumptions)
        self.sentences: set[Sentence] = set(sentences)   # non-assumption atoms (claims)
        self.rules: list[Rule] = list(rules)

    @classmethod
    def from_dependency_graph(
        cls,
        dep_graph,
        tau_a: Optional[Dict[str, float]] = None,
        tau_r: Optional[Dict[int, float]] = None,
    ) -> "ABAF":
        """
        Build a weighted ABAF from a parsed DependencyGraph.

        dep_graph fields used:
          .assumptions  - set of assumption name strings
          .contrary     - dict {assumption_name: contrary_name}
          .rules        - dict {index (int): (head_name, body_name_list)}

        tau_a: maps assumption name -> base strength (default 0.5)
        tau_r: maps rule index     -> rule reliability (default 1.0)
        """
        tau_a = tau_a or {}
        tau_r = tau_r or {}

        # collect every atom name that appears anywhere
        all_names: set[str] = set(dep_graph.assumptions)
        for _, (head, body) in dep_graph.rules.items():
            all_names.add(head)
            all_names.update(body)
        all_names.update(dep_graph.contrary.values())

        claim_names = all_names - dep_graph.assumptions

        # create Sentence objects for non-assumption atoms
        sentences = {name: Sentence(name) for name in claim_names}

        # create Assumption objects
        assumptions = {
            name: Assumption(
                name=name,
                contrary=dep_graph.contrary.get(name),
                tau=tau_a.get(name, 0.5),
            )
            for name in dep_graph.assumptions
        }

        # unified name → atom lookup (Assumption is a subclass of Sentence)
        atoms: dict[str, Sentence] = {**sentences, **assumptions}

        # create Rule objects preserving the index-based naming from dep_graph
        rules = [
            Rule(
                head=atoms[head_name],
                body=[atoms[b] for b in body_names],
                name=f"r{idx}",
                tau=tau_r.get(idx, 1.0),
            )
            for idx, (head_name, body_names) in dep_graph.rules.items()
        ]

        return cls(
            assumptions=assumptions.values(),
            sentences=sentences.values(),
            rules=rules,
        )

    def __repr__(self):
        asms = ", ".join(sorted(a.name for a in self.assumptions))
        return f"ABAF(assumptions=[{asms}], rules={len(self.rules)})"
