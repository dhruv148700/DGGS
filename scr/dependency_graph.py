import os
import argparse 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as  np 
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

class DependencyGraph:
    def __init__(self):
        self.dummy_var_counter = 0
        self.assumptions = set()
        self.contrary = dict()
        self.rules = dict()
        self.all_elements = set()
        self.filename = ""
        self.graph = None

    def create_from_file(self, framework_filename):
        self.filename = framework_filename
        with open(framework_filename, "r") as f:
            text = f.read().split("\n")        
        # Create a set to store all non assumption elements 
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
                body = list(set(body))
                rule = (head, body)

                if rule not in self.rules.values():
                    self.rules[rule_index] = (head, body)
                    self.all_elements.add(head)
                    for item in body:
                        self.all_elements.add(str(item))
                    rule_index += 1
        
        self.non_assumptions = self.all_elements - self.assumptions
        # print(f"{self.assumptions=}")
        # print(f"{self.contrary=}")
        # print(f"{self.rules=}")


    def create_dependency_graph(self, print_graph = False):
        nxg = nx.DiGraph()
        for asmpt in self.assumptions:
            nxg.add_node(asmpt)
        
        for asmpt, contrary in self.contrary.items():
            nxg.add_node(asmpt)
            nxg.add_node(contrary)
            nxg.add_edge(contrary, asmpt, label="-")
            #print(f"adding edge {contrary} -> {asmpt} label -")
        
        # print(list(nxg.nodes(data=True)))
        
        for index, (head, body) in self.rules.items():
            nxg.add_node(head)
            rule_node = f"r{index}"
            nxg.add_node(rule_node)
            nxg.add_nodes_from(list(body))
            for elem in body:
                nxg.add_edge(elem, rule_node, label="+")
                #print(f"adding edge {elem} -> {rule_node} label +")
            
            nxg.add_edge(rule_node, head, label='d')
            #print(f"adding edge {elem} -> {rule_node} label +")
        
        # print(list(nxg.nodes(data=True)))
        
        if print_graph:
            edges = nxg.edges(data=True)
            # print(f"{edges=}")
            edge_labels = {(u, v): f"{d['label']}" for u, v, d in edges}

            # Define positions using Graphviz layout
            pos = nx.circular_layout(nxg)  # Use 'dot' for hierarchical layout

            # Draw the graph
            plt.figure(figsize=(10, 10))
            nx.draw(nxg, pos, with_labels=True, node_size=3000, node_color="white", font_size=12, edgecolors='black')
            nx.draw_networkx_edge_labels(nxg, pos, edge_labels=edge_labels, font_size=12)
            filename = os.path.basename(self.filename)
            val = filename.split(".")[0]
            plt.savefig(f"dependency_graph_{val}")
        
        self.graph = nxg
    
    def remove_rejected_assumption(self, attacked_assmpt):
        #print(f"removing attacked assmpt {attacked_assmpt}")
        # remove the attacked assumption 
        self.assumptions.remove(attacked_assmpt)
        # remove the contrary of the attacked assumption 
        del self.contrary[attacked_assmpt]
        
        # remove any rule that had the attacked assumption within it 
        rules_with_attacked_assmpts = [(rule_index, head, body) for rule_index, (head, body) in self.rules.items() if attacked_assmpt in body] 
        for (rule_index, head, body) in rules_with_attacked_assmpts:
            del self.rules[rule_index]
            # since a dummy element is created for a rule, when the rule is deleted, 
            # the dummy element and its contrary should also be deleted. 
            dummy_element = next((item for item in body if item.startswith("dummy")), None)
            if dummy_element:
                self.assumptions.remove(dummy_element)
                dummy_contrary = self.contrary[dummy_element]
                del self.contrary[dummy_element]
                self.non_assumptions.remove(dummy_contrary)
        
        # remove any contrary relationships that had the attacked assumption as the contrary
        self.contrary = {
            assmpt: contrary for assmpt, contrary in self.contrary.items() if contrary != attacked_assmpt
        }

    def remove_accepted_assumption(self, assumption):
        # print(f"starting assumptions {self.assumptions}")
        # print(f"starting contraries {self.contrary}")
        # print(f"starting rules {self.rules}")
        # print(f"starting non assumptions {self.non_assumptions}")

        # ---- STEP 1 ----
        # remove the contrary of the assumption if it exists
        contrary = self.contrary.pop(assumption, None)

        # find each rule r with head(r)=contrary(a)
        matching_with_indices = [(i, head, body) for i, (head, body) in self.rules.items() if head == contrary]
        # print(f"rules with contrary {matching_with_indices}")
        
        for (i, head, body) in matching_with_indices:
            # r' = body(r) \cup {d_r} ----> contrary(d_r)
            new_dummy_elem = f"dummy_{self.dummy_var_counter}"
            new_dummy_contrary = f"dummy_contrary_{self.dummy_var_counter}"
            self.dummy_var_counter += 1
            body.append(new_dummy_elem)
            new_head = new_dummy_contrary
            self.rules[i] = (new_head, body)
            # the dummy variable is an assumption and the dummy contrary is a non_assumption 
            self.assumptions.add(new_dummy_elem)
            self.non_assumptions.add(new_dummy_contrary)
            self.contrary[new_dummy_elem] = new_dummy_contrary
        
        #print(f"rules with dummies {self.contrary}")

        # ---- STEP 2----
        # remove the assumption from the assumptions 
        self.assumptions.remove(assumption)
        
        # remove the assumption from the body of any rule 
        assmpt_in_body = [(i, head, body) for i, (head, body) in self.rules.items() if assumption in body]

        #print(f"assumption in bodies {assmpt_in_body}")
        
        for (i, head, body) in assmpt_in_body:
            body.remove(assumption)
            # if the body only consists of dummy elements now, remove all traces of dummy elements 
            # and delete the rule. 
            if len(body) == 1 and body[0].startswith("dummy"):
                print("reached invalid ABAF")
                # in this case, we are basically left with a rule that derives the contrary
                # so ABAF is unsatisfiable and we return false. 
                return False
            else:
                self.rules[i] = (head, body)
        

        # print("rules without assumption:", self.rules)
        # print()
        
        # ---- STEP 3 ----
        # if the assumption is the contrary of another assumption
        attacked_assmpts = [assmpt for assmpt, contrary in self.contrary.items() if contrary == assumption]
        for attacked_assmpt in attacked_assmpts:
            self.remove_rejected_assumption(attacked_assmpt)
        
        # ---- STEP 4 ----
        # Get indices of rules where body is empty, i.e. the facts 
        empty_body_indices = [rule_index for rule_index, (head, body) in self.rules.items() if not body]

        while len(empty_body_indices) > 0:
            # print(f"{empty_body_indices=}")
            for rule_index in empty_body_indices:
                #remove rule with empty body
                rule = self.rules.pop(rule_index, None)
                if not rule:
                    continue
                else:
                    (fact, body) = rule 
                # print(f"{fact=}")
                # since the head is a fact, remove the head from non_assumptions
                self.non_assumptions.remove(fact)
                # Remove any rule that concludes the fact, since it is not necessary, and
                # remove the head from all rule bodies. 
                self.rules = {
                    rule_index: (head, [item for item in body if item != fact])
                    for rule_index, (head, body) in self.rules.items()
                    if head != fact
                }

                # print(f"rules without head: {self.rules}")

                # if head is the contrary of an assumption, that assumption is attacked 
                attacked_assmpts = [assmpt for assmpt, contrary in self.contrary.items() if contrary == fact]
                # print(f"{attacked_assmpts=}")
                 
                for attacked_assmpt in attacked_assmpts:
                    self.remove_rejected_assumption(attacked_assmpt)
                
                #print(f"rules without attacked assumption: {self.rules}")

            # repeat this until there are no more facts left in the ABAF. 
            empty_body_indices = [rule_index for rule_index, (head, body) in self.rules.items() if not body]
        
        # print(f"final assumptions {self.assumptions}")
        # print(f"final contraries {self.contrary}")
        # print(f"final rules {self.rules}")
        # print(f"final non assumptions {self.non_assumptions}")
        # print()
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
    # dep_graph.calculate_node_features()

                
    