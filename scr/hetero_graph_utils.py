
import dgl
import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from dependency_graph import DependencyGraph
from plot_graphs import plot_hetero_graph

def reindex_nodes(dep_graph):
    """Reindex nodes by type for heterograph creation"""
    graph = dep_graph.graph
    # First map: nodes that begin with 'r'
    rule_nodes = [node.strip() for node in graph.nodes() if str(node).strip().startswith('r')]
    rule_mapping = {node: index for index, node in enumerate(rule_nodes)}
    
    # Second map: assumption nodes
    assmpt_nodes = [node.strip() for node in graph.nodes() if str(node).strip() in dep_graph.assumptions]
    assmpt_mapping = {node: index for index, node in enumerate(assmpt_nodes)}

    non_assmpt_nodes = [node.strip() for node in graph.nodes() if str(node).strip() in dep_graph.non_assumptions]
    non_assmpt_mapping = {node: index for index, node in enumerate(non_assmpt_nodes)}
    
    return rule_mapping, assmpt_mapping, non_assmpt_mapping

def create_hetero_graph(graph, rule_mapping, assmpt_mapping, non_assmpt_mapping):
    """Create heterogeneous DGL graph from dependency graph"""
    # tuples containing a list of the source nodes and list of the respective target nodes of each edge type
    support_assmpt_rule = ([], [])
    support_non_assmpt_rule = ([], [])
    attack_non_assmpt_assmpt = ([], [])
    attack_assmpt_assmpt = ([], [])
    derive_rule_non_assmpt = ([], [])
    derive_rule_assmpt = ([], [])

    # Collect all node IDs of each type
    assmpt_nodes = set(assmpt_mapping.values())
    rule_nodes = set(rule_mapping.values())
    non_assmpt_nodes = set(non_assmpt_mapping.values())

    # Create self-connections for type 'assmpt', 'rule' and 'non_assmpt' nodes
    self_support_assmpt = (list(assmpt_nodes), list(assmpt_nodes))
    self_support_rule = (list(rule_nodes), list(rule_nodes))
    self_support_non_assmpt = (list(non_assmpt_nodes), list(non_assmpt_nodes))

    for u, v, d in graph.edges(data=True):
        if d.get('label') == "+":
            if u in assmpt_mapping:
                support_assmpt_rule[0].append(assmpt_mapping[u])
                support_assmpt_rule[1].append(rule_mapping[v])
            elif u in non_assmpt_mapping:
                support_non_assmpt_rule[0].append(non_assmpt_mapping[u])
                support_non_assmpt_rule[1].append(rule_mapping[v])
            else:
                print(f"ERROR: Invalid nodes for + edge: {u} -> {v}")
                return None

        elif d.get('label') == '-':
            if u in assmpt_mapping:
                attack_assmpt_assmpt[0].append(assmpt_mapping[u])
                attack_assmpt_assmpt[1].append(assmpt_mapping[v])
            elif u in non_assmpt_mapping:
                attack_non_assmpt_assmpt[0].append(non_assmpt_mapping[u])
                attack_non_assmpt_assmpt[1].append(assmpt_mapping[v])
            else:
                print(f"ERROR: Invalid nodes for - edge: {u} -> {v}")
                return None

        elif d.get('label') == 'd':
            if v in assmpt_mapping:
                derive_rule_assmpt[0].append(rule_mapping[u])
                derive_rule_assmpt[1].append(assmpt_mapping[v])
            elif v in non_assmpt_mapping:
                derive_rule_non_assmpt[0].append(rule_mapping[u])
                derive_rule_non_assmpt[1].append(non_assmpt_mapping[v])
            else:
                print(f"ERROR: Invalid nodes for d edge: {u} -> {v}")
                return None

        else:
            print(f"ERROR: Invalid edge label: {d.get('label')}")
            return None
            
    data_dict = {
        # supports relationships 
        ('assmpt', 'supports', 'rule'): support_assmpt_rule,
        ('non_assmpt', 'supports', 'rule'): support_non_assmpt_rule,
        # attacks relationships 
        ('non_assmpt', 'attacks', 'assmpt'): attack_non_assmpt_assmpt,
        ('assmpt', 'attacks', 'assmpt'): attack_assmpt_assmpt,
        # derives relationships 
        ('rule', 'derives', 'non_assmpt'): derive_rule_non_assmpt,
        ('rule', 'derives', 'assmpt'): derive_rule_assmpt,
        # Add self-connections for 'assmpt' nodes of type '+'
        ('assmpt', 'supports', 'assmpt'): self_support_assmpt,
        # Add self-connections for 'rule' nodes of type '+'
        ('rule', 'supports', 'rule'): self_support_rule,
        # Add self-connections for 'rule' nodes of type '+'
        ('non_assmpt', 'supports', 'non_assmpt'): self_support_non_assmpt
    }

    return data_dict

def create_graph(aba_file_path, plot_graph=False):
    """
    Process an ABA file and convert it to a heterogeneous DGL graph.
    
    Args:
        aba_file_path (str): Path to the ABA file
        
    Returns:
        tuple: (hetero_graph, assmpt_mapping) where assmpt_mapping maps 
                assumption names to their indices in the graph
    """
    if not os.path.exists(aba_file_path):
        raise FileNotFoundError(f"ABA file not found: {aba_file_path}")
    
    # Create dependency graph from ABA file
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(aba_file_path)
    dep_graph.create_dependency_graph()
    
    # Reindex nodes by type
    rules_mapping, assmpt_mapping, non_assmpt_mapping = reindex_nodes(dep_graph)
    # print(f"{rules_mapping=}")
    # print(f"{assmpt_mapping=}")
    # print(f"{non_assmpt_mapping=}")
    
    if len(assmpt_mapping) == 0:
        raise ValueError("No assumptions found in the ABA file")
    
    # Calculate node features
    features = dep_graph.calculate_node_features(assmpt_mapping | non_assmpt_mapping)
    
    # Create heterogeneous graph
    data_dict = create_hetero_graph(
        dep_graph.graph, 
        rules_mapping, 
        assmpt_mapping, 
        non_assmpt_mapping
    )

    hetero_graph = dgl.heterograph(data_dict)
    
    if hetero_graph is None:
        raise ValueError("Failed to create heterogeneous graph")
    
    # Add node features to the graph
    # Assumption features
    assmpt_feat_arr = np.empty((len(assmpt_mapping), 2))
    for key in assmpt_mapping:
        assmpt_feat_arr[assmpt_mapping[key], :] = features[key]
    hetero_graph.nodes['assmpt'].data['features'] = torch.tensor(assmpt_feat_arr, dtype=torch.float32)

    # Non-assumption features
    non_assmpt_feat_arr = np.empty((len(non_assmpt_mapping), 2))
    for key in non_assmpt_mapping:
        non_assmpt_feat_arr[non_assmpt_mapping[key], :] = features[key]
    hetero_graph.nodes['non_assmpt'].data['features'] = torch.tensor(non_assmpt_feat_arr, dtype=torch.float32)
    
    # Rule features (random as in training)
    rules_arr = np.random.randn(len(rules_mapping), 2)
    hetero_graph.nodes['rule'].data['features'] = torch.tensor(rules_arr, dtype=torch.float32)

    if plot_graph:
        plot_hetero_graph(data_dict, rules_mapping, assmpt_mapping, non_assmpt_mapping)
    
    #print_hetero_graph(hetero_graph)
    
    return hetero_graph, dep_graph, assmpt_mapping


def update_graph(dep_graph, plot_graph=False, id=1):
    rules_mapping, assmpt_mapping, non_assmpt_mapping = reindex_nodes(dep_graph)
    # print(f"{rules_mapping=}")
    # print(f"{assmpt_mapping=}")
    # print(f"{non_assmpt_mapping=}")
    
    if len(assmpt_mapping) == 0:
        raise ValueError("No assumptions found in the ABA file")
    
    # Calculate node features
    features = dep_graph.calculate_node_features(assmpt_mapping | non_assmpt_mapping)
    
    # Create heterogeneous graph
    data_dict = create_hetero_graph(
        dep_graph.graph, 
        rules_mapping, 
        assmpt_mapping, 
        non_assmpt_mapping
    )
    
    hetero_graph = dgl.heterograph(data_dict)

    if hetero_graph is None:
        raise ValueError("Failed to create heterogeneous graph")
    
    # Add node features to the graph
    # Assumption features
    assmpt_feat_arr = np.empty((len(assmpt_mapping), 2))
    for key in assmpt_mapping:
        assmpt_feat_arr[assmpt_mapping[key], :] = features[key]
    hetero_graph.nodes['assmpt'].data['features'] = torch.tensor(assmpt_feat_arr, dtype=torch.float32)

    # Non-assumption features
    non_assmpt_feat_arr = np.empty((len(non_assmpt_mapping), 2))
    for key in non_assmpt_mapping:
        non_assmpt_feat_arr[non_assmpt_mapping[key], :] = features[key]
    hetero_graph.nodes['non_assmpt'].data['features'] = torch.tensor(non_assmpt_feat_arr, dtype=torch.float32)
    
    # Rule features (random as in training)
    rules_arr = np.random.randn(len(rules_mapping), 2)
    hetero_graph.nodes['rule'].data['features'] = torch.tensor(rules_arr, dtype=torch.float32)

    if plot_graph:
        plot_hetero_graph(data_dict, rules_mapping, assmpt_mapping, non_assmpt_mapping, id=id)
    
    #print_hetero_graph(hetero_graph)
    
    return hetero_graph, dep_graph, assmpt_mapping


def print_hetero_graph(g):
    # If you have node features, print them
    print("\nNode features (if available):")
    for ntype in g.ntypes:
        if g.nodes[ntype].data:  # Check if features exist
            for feature_name, feature_tensor in g.nodes[ntype].data.items():
                print(f"Node type '{ntype}', feature '{feature_name}':")
                for i in range(g.number_of_nodes(ntype)):
                    print(f"  Node {i}: {feature_tensor[i]}")
        
    
    #Print nodes along with their connections
    print("\nNode connections:")
    for canonical_etype in g.canonical_etypes:
        src_type, rel_type, dst_type = canonical_etype
        src, dst = g.edges(etype=canonical_etype)
        print(f"Relation '{src_type}-{rel_type}->{dst_type}':")
        for i in range(len(src)):
            print(f"  {src_type}_{src[i].item()} -> {dst_type}_{dst[i].item()}")