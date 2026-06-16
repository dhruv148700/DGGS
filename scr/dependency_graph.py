import os
import re
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

# Matches indep_X_Y__<conditioning_set> — X and Y are always ordered X < Y
# by assums.indep(), so arr_X_Y, arr_Y_X, and noe_X_Y can be built directly.
_INDEP_RE = re.compile(r"^indep_(\d+)_(\d+)__")

class DependencyGraph:
    def __init__(self, reject_edge_on_indep: bool = False):
        self.dummy_var_counter = 0
        self.assumptions = set()
        self.contrary = dict()
        self.rules = dict()
        self.all_elements = set()
        self.filename = ""
        self.graph = None
        self.reject_edge_on_indep = reject_edge_on_indep
        # Inverted indices — populated by _init_indices()
        self._head_to_rules: defaultdict = defaultdict(set)
        self._body_elem_to_rules: defaultdict = defaultdict(set)
        self._empty_rules: set = set()
        self._contrary_reverse: defaultdict = defaultdict(list)

    def _init_indices(self):
        """Rebuild all inverted indices from current self.rules and self.contrary."""
        self._head_to_rules = defaultdict(set)
        self._body_elem_to_rules = defaultdict(set)
        self._empty_rules = set()
        self._contrary_reverse = defaultdict(list)
        for idx, (head, body) in self.rules.items():
            self._head_to_rules[head].add(idx)
            for elem in body:
                self._body_elem_to_rules[elem].add(idx)
            if not body:
                self._empty_rules.add(idx)
        for asm, contrary in self.contrary.items():
            self._contrary_reverse[contrary].append(asm)

    def create_from_file(self, framework_filename):
        self.filename = framework_filename
        with open(framework_filename, "r") as f:
            text = f.read().split("\n")
        self.non_assumptions = set()

        rule_index = 1
        for line in text:
            if line.startswith("a "):
                self.assumptions.add(str(line.split()[1]))
            if line.startswith("c "):
                components = line.split()
                element = str(components[1])
                contrary = components[2]
                self.contrary[element] = contrary
                self.all_elements.add(element)
                self.all_elements.add(contrary)
            if line.startswith("r "):
                components = line.split()[1:]
                head, body = str(components[0]), components[1:]
                body = sorted(set(body))
                rule = (head, tuple(body))
                if rule not in self.rules.values():
                    self.rules[rule_index] = (head, body)
                    self.all_elements.add(head)
                    for item in body:
                        self.all_elements.add(str(item))
                    rule_index += 1

        self.non_assumptions = self.all_elements - self.assumptions
        self._init_indices()


    def create_dependency_graph(self, print_graph = False):
        nxg = nx.DiGraph()
        for asmpt in self.assumptions:
            nxg.add_node(asmpt)

        for asmpt, contrary in self.contrary.items():
            nxg.add_node(asmpt)
            nxg.add_node(contrary)
            nxg.add_edge(contrary, asmpt, label="-")

        for index, (head, body) in self.rules.items():
            nxg.add_node(head)
            rule_node = f"r{index}"
            nxg.add_node(rule_node)
            nxg.add_nodes_from(list(body))
            for elem in body:
                nxg.add_edge(elem, rule_node, label="+")
            nxg.add_edge(rule_node, head, label='d')

        if print_graph:
            edges = nxg.edges(data=True)
            edge_labels = {(u, v): f"{d['label']}" for u, v, d in edges}
            pos = nx.circular_layout(nxg)
            plt.figure(figsize=(10, 10))
            nx.draw(nxg, pos, with_labels=True, node_size=3000, node_color="white", font_size=12, edgecolors='black')
            nx.draw_networkx_edge_labels(nxg, pos, edge_labels=edge_labels, font_size=12)
            filename = os.path.basename(self.filename)
            val = filename.split(".")[0]
            plt.savefig(f"dependency_graph_{val}")

        self.graph = nxg

    def remove_rejected_assumption(self, attacked_assmpt):
        self.assumptions.remove(attacked_assmpt)

        # Remove contrary mapping for attacked_assmpt
        contrary_val = self.contrary.pop(attacked_assmpt, None)
        if contrary_val is not None:
            rev = self._contrary_reverse.get(contrary_val)
            if rev is not None:
                try:
                    rev.remove(attacked_assmpt)
                except ValueError:
                    pass

        # Rules involving attacked_assmpt (as head or body element)
        affected = (
            self._head_to_rules.get(attacked_assmpt, set()) |
            self._body_elem_to_rules.get(attacked_assmpt, set())
        ).copy()

        for rule_index in affected:
            if rule_index not in self.rules:
                continue
            head, body = self.rules.pop(rule_index)
            self._head_to_rules.get(head, set()).discard(rule_index)
            for elem in body:
                self._body_elem_to_rules.get(elem, set()).discard(rule_index)
            self._empty_rules.discard(rule_index)
            # Clean up any dummy elements that were exclusive to this rule
            for dummy_element in [item for item in body if item.startswith("dummy")]:
                self.assumptions.discard(dummy_element)
                dc = self.contrary.pop(dummy_element, None)
                if dc is not None:
                    rev = self._contrary_reverse.get(dc)
                    if rev is not None:
                        try:
                            rev.remove(dummy_element)
                        except ValueError:
                            pass
                    self.non_assumptions.discard(dc)

        self._head_to_rules.pop(attacked_assmpt, None)
        self._body_elem_to_rules.pop(attacked_assmpt, None)

        # Remove contrary entries where the VALUE == attacked_assmpt
        # (assumptions that had attacked_assmpt as their contrary lose their contrary mapping)
        for asm in list(self._contrary_reverse.pop(attacked_assmpt, [])):
            self.contrary.pop(asm, None)

    def remove_accepted_assumption(self, assumption):
        # ---- STEP 1 ----
        # Remove the contrary mapping of the committed assumption and transform
        # every rule whose head is that contrary into a dummy-gated rule.
        contrary = self.contrary.pop(assumption, None)
        if contrary is not None:
            rev = self._contrary_reverse.get(contrary)
            if rev is not None:
                try:
                    rev.remove(assumption)
                except ValueError:
                    pass

        for i in list(self._head_to_rules.get(contrary, set()) if contrary else []):
            if i not in self.rules:
                continue
            head, body = self.rules[i]

            new_dummy_elem = f"dummy_{self.dummy_var_counter}"
            new_dummy_contrary = f"dummy_contrary_{self.dummy_var_counter}"
            self.dummy_var_counter += 1
            new_body = list(body) + [new_dummy_elem]
            new_head = new_dummy_contrary

            # Move rule from old head to new head in index
            self._head_to_rules[contrary].discard(i)
            self._head_to_rules[new_head].add(i)
            # Update body indices: remove old body elements, add new body elements
            for elem in body:
                self._body_elem_to_rules[elem].discard(i)
            self._empty_rules.discard(i)
            for elem in new_body:
                self._body_elem_to_rules[elem].add(i)
            # new_body always contains new_dummy_elem, so never empty

            self.rules[i] = (new_head, new_body)
            self.assumptions.add(new_dummy_elem)
            self.non_assumptions.add(new_dummy_contrary)
            self.contrary[new_dummy_elem] = new_dummy_contrary
            self._contrary_reverse[new_dummy_contrary].append(new_dummy_elem)

        # ---- STEP 2 ----
        # Remove the assumption itself and all rules it participates in.
        self.assumptions.remove(assumption)

        # Remove rules whose head IS the assumption
        for i in list(self._head_to_rules.pop(assumption, set())):
            if i not in self.rules:
                continue
            _, body = self.rules.pop(i)
            for elem in body:
                self._body_elem_to_rules.get(elem, set()).discard(i)
            self._empty_rules.discard(i)

        # Remove assumption from rule bodies; detect unsatisfiable residuals
        for i in list(self._body_elem_to_rules.pop(assumption, set())):
            if i not in self.rules:
                continue
            head, body = self.rules[i]
            new_body = [item for item in body if item != assumption]
            if len(new_body) == 1 and new_body[0].startswith("dummy"):
                print("reached invalid ABAF")
                return False
            self.rules[i] = (head, new_body)
            if not new_body:
                self._empty_rules.add(i)

        # ---- STEP 3 ----
        # Any assumption whose contrary IS the assumption being committed is now attacked.
        attacked_assmpts = list(self._contrary_reverse.get(assumption, []))
        for attacked_assmpt in attacked_assmpts:
            if attacked_assmpt in self.assumptions:
                self.remove_rejected_assumption(attacked_assmpt)
        self._contrary_reverse.pop(assumption, None)

        # ---- STEP 4 ----
        # Propagate empty-body rules (derived facts) until fixpoint.
        while self._empty_rules:
            rule_index = next(iter(self._empty_rules))
            self._empty_rules.discard(rule_index)

            if rule_index not in self.rules:
                continue
            fact, _ = self.rules.pop(rule_index)
            self._head_to_rules.get(fact, set()).discard(rule_index)

            if fact not in self.non_assumptions:
                continue
            self.non_assumptions.remove(fact)

            # Remove all remaining rules with head == fact (now redundant)
            for ri in list(self._head_to_rules.pop(fact, set())):
                if ri not in self.rules:
                    continue
                _, rbody = self.rules.pop(ri)
                for elem in rbody:
                    self._body_elem_to_rules.get(elem, set()).discard(ri)
                self._empty_rules.discard(ri)

            # Remove fact from all rule bodies; newly empty rules become facts
            for ri in list(self._body_elem_to_rules.pop(fact, set())):
                if ri not in self.rules:
                    continue
                h, b = self.rules[ri]
                new_b = [item for item in b if item != fact]
                self.rules[ri] = (h, new_b)
                if not new_b:
                    self._empty_rules.add(ri)

            # Attack assumptions whose contrary is this derived fact
            attacked = list(self._contrary_reverse.get(fact, []))
            for attacked_assmpt in attacked:
                if attacked_assmpt in self.assumptions:
                    self.remove_rejected_assumption(attacked_assmpt)
            self._contrary_reverse.pop(fact, None)

        # If we just committed an indep_X_Y__S assumption, a direct edge X-Y is
        # structurally incompatible (a length-1 path is never blockable by any S),
        # so arr_X_Y, arr_Y_X, and noe_X_Y cannot appear in any consistent extension
        # that includes this assumption.  Reject all three directly so the scaffold
        # is cleaned up without adding anything extra to the extension.
        if self.reject_edge_on_indep:
            m = _INDEP_RE.match(assumption)
            if m:
                x, y = m.group(1), m.group(2)
                for asm in (f"arr_{x}_{y}", f"arr_{y}_{x}", f"noe_{x}_{y}"):
                    if asm in self.assumptions:
                        self.remove_rejected_assumption(asm)

        return True

    def calculate_node_features(self, mapping=None):
        raw_features = {}
        nodes = mapping if mapping else self.graph.nodes()
        for node in nodes:
           in_degree = len(self.graph.in_edges(node, data=True))
           out_degree = len(self.graph.out_edges(node, data=True))
           raw_features[node] = [in_degree, out_degree]

        scaler = StandardScaler()

        indegree_values = [node_data[0] for node_data in raw_features.values()]
        scaled_indegree_values = scaler.fit_transform([
            [value] for value in indegree_values
        ]).flatten()
        scaled_indegree_dict = {
            node: scaled_indegree_values[i] for i, node in enumerate(raw_features.keys())
        }

        outdegree_values = [node_data[1] for node_data in raw_features.values()]
        scaled_outdegree_values = scaler.fit_transform([
            [value] for value in outdegree_values
        ]).flatten()
        scaled_outdegree_dict = {
            node: scaled_outdegree_values[i] for i, node in enumerate(raw_features.keys())
        }

        normalized_features = {}
        for node in nodes:
            indegree_encoded = scaled_indegree_dict[node]
            outdegree_encoded = scaled_outdegree_dict[node]
            node_feature_vector = np.array([indegree_encoded, outdegree_encoded])
            normalized_features[node] = node_feature_vector

        return normalized_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='' , help='file')
    args = parser.parse_args()

    dep_graph = DependencyGraph()
    dep_graph.create_from_file(args.filepath)
    dep_graph.create_dependency_graph()
