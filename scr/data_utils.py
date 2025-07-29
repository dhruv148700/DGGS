import numpy as np
import random
import torch 
import dgl 
import os 
import sys
sys.path.append('../')  # Adjust the path as necessary to import from the parent directory
from dependency_graph import DependencyGraph
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def set_seeds(seed):
    """
    Set seeds for reproducibility across all libraries. 
    Args:
        seed: Integer seed value 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    os.environ['PYTHONHASSEED'] = str(seed)

def split_data(flat_data, non_flat_data, option=0):
    """
    option0: only have flat data to work with, create train/val/test sets out of that
    option1: only train and validate on flat data and test accuracy on non_flat data 
    option2: train and validate on both flat and non flat data and test accuracy on both data 
    Both options ensure equal amounts of flat/non-flat data in each split where applicable.
    """
    if option == 0:
        random.shuffle(flat_data)
        n = len(flat_data)
        # have a 70 / 15 / 15 split for training evaluation and test data 
        split1 = int(n * 0.9)

        training = flat_data[:split1]
        validation = flat_data[split1:]

    elif option == 1:
        random.shuffle(flat_data)
        n = len(flat_data)
    
        # use 0.7 and 0.3 for a 70/30 split  
        split1 = int(n * 0.7)
        training = flat_data[:split1]
        validation = flat_data[split1:]

        # test set is all non flat data
        test = non_flat_data

    elif option == 2:
        # Shuffle both datasets separately
        random.shuffle(flat_data)
        random.shuffle(non_flat_data)

        n_flat = len(flat_data)
        flat_split_1 = int(n_flat*0.7)
        flat_split_2 = int(n_flat*0.85)
        n_non_flat = len(non_flat_data)
        non_flat_split_1 = int(n_non_flat*0.7)
        non_flat_split_2 = int(n_non_flat*0.85)
        
        flat_train = flat_data[:flat_split_1]
        flat_val = flat_data[flat_split_1:flat_split_2]
        flat_test = flat_data[flat_split_2:]

        non_flat_train = non_flat_data[:non_flat_split_1]
        non_flat_val = non_flat_data[non_flat_split_1:non_flat_split_2]
        non_flat_test = non_flat_data[non_flat_split_2:]

        # Combine the datasets
        training = flat_train + non_flat_train
        validation = flat_val + non_flat_val
        test = flat_test + non_flat_test
        
        # Shuffle the combined datasets
        random.shuffle(training)
        random.shuffle(validation)
        random.shuffle(test)

    else:
        raise ValueError("Option must be either 1 or 2")


    # with open(f"../{sub_folder}/output_seed_{seed}.txt", "a") as file: 
    #     file.write(f"Split with seed {seed}: {len(training)} training, {len(validation)} validation, {len(test)} test\n")

    #     if option == 2:
    #         flat_count_train = sum(1 for item in training if item in flat_data)
    #         non_flat_count_train = len(training) - flat_count_train
    #         flat_count_val = sum(1 for item in validation if item in flat_data)
    #         non_flat_count_val = len(validation) - flat_count_val
    #         flat_count_test = sum(1 for item in test if item in flat_data)
    #         non_flat_count_test = len(test) - flat_count_test
            
    #         file.write(f"Balance - Training: {flat_count_train} flat, {non_flat_count_train} non-flat\n")
    #         file.write(f"Balance - Validation: {flat_count_val} flat, {non_flat_count_val} non-flat\n")
    #         file.write(f"Balance - Test: {flat_count_test} flat, {non_flat_count_test} non-flat\n")

    return (training, validation)

# Calculate confidence intervals for each metric
def get_confidence_interval(scores):
    scores = np.array(scores)
    mean = np.mean(scores)
    std_dev = np.std(scores, ddof=1)
    
    # For normal approximation
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * std_dev
    
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, std_dev, ci_lower, ci_upper

# how are extensions computed 
def load_dataset(input_directory, output_directory, dataset_files = None):
    if dataset_files:
        print("exists")
        input_files = []
        with open(dataset_files, 'r') as file:
            for line in file:
                # Strip whitespace and newlines and add to list if not empty
                clean_line = line.strip()
                if clean_line:
                    input_files.append(clean_line)
    else:
        input_files = os.listdir(input_directory)
    
    graphs = []
    for filename in input_files:
        print(filename)
        f_input = os.path.join(input_directory, filename)
        output_file = f"output_{filename}"
        f_output = os.path.join(output_directory, output_file)

        # checking if it is a file
        if not os.path.isfile(f_input) or not os.path.isfile(f_output):
            print("skipping file")
            continue 

        dep_graph = DependencyGraph()
        dep_graph.create_from_file(f_input)
        # print(f"{dep_graph.assumptions=}")
        # print(f"{dep_graph.contrary=}")
        # print(f"{dep_graph.non_assumptions=}")
        # print(f"{dep_graph.rules=}")
        dep_graph.create_dependency_graph()
        
        (rules_mapping, assmpt_mapping, non_assmpt_mapping) = reindex_nodes(dep_graph) 
        # print(rules_mapping)
        # print(assmpt_mapping)
        # print(non_assmpt_mapping)
        # print()
        
        # TODO handle scaling of features
        features = dep_graph.calculate_node_features(assmpt_mapping | non_assmpt_mapping)
        # print("features", features)
        # print()
        hetero_graph, _ = create_hetero_graph(
            dep_graph.graph, 
            rules_mapping, 
            assmpt_mapping, 
            non_assmpt_mapping,
        )
        
        assmpt_feat_arr = np.empty((len(assmpt_mapping), 2))
        for key in assmpt_mapping:
            assmpt_feat_arr[assmpt_mapping[key], :] = features[key]
        
        hetero_graph.nodes['assmpt'].data['features'] = torch.tensor(assmpt_feat_arr, dtype=torch.float32)

        non_assmpt_feat_arr = np.empty((len(non_assmpt_mapping), 2))
        for key in non_assmpt_mapping:
            non_assmpt_feat_arr[non_assmpt_mapping[key], :] = features[key]
        
        hetero_graph.nodes['non_assmpt'].data['features'] = torch.tensor(non_assmpt_feat_arr, dtype=torch.float32)
        
        rules_arr = np.random.randn(len(rules_mapping), 2)
        hetero_graph.nodes['rule'].data['features'] = torch.tensor(rules_arr, dtype=torch.float32)

        label_vector = create_label_vector(f_output, assmpt_mapping)
        
        hetero_graph.nodes['assmpt'].data['label'] = torch.tensor(label_vector, dtype=torch.float32)
        #print_hetero_graph(hetero_graph)
        graphs.append(hetero_graph)
            
    print(len(graphs))
    return graphs

def reindex_nodes(dep_graph):
    graph = dep_graph.graph
    # First map: nodes that begin with 'r'
    rule_nodes = [node.strip() for node in graph.nodes() if str(node).strip().startswith('r')]
    rule_mapping = {node: index for index, node in enumerate(rule_nodes)}
    
    # Second map: assumption nodes
    assmpt_nodes = [node.strip() for node in graph.nodes() if str(node).strip() in dep_graph.assumptions]
    assmpt_mapping = {node: index for index, node in enumerate(assmpt_nodes)}

    non_assmpt_nodes = [node.strip() for node in graph.nodes() if str(node).strip() in dep_graph.non_assumptions]
    non_assmpt_mapping = {node: index for index, node in enumerate(non_assmpt_nodes)}
    # print(r_mapping)
    # print(a_mapping)
    
    # Return the relabeled graph and both mappings
    return rule_mapping, assmpt_mapping, non_assmpt_mapping


def create_hetero_graph(graph, rule_mapping, assmpt_mapping, non_assmpt_mapping, print_hetero_graph=False):
    # tuples containing a list of the source nodes and list of the respective target nodes of each
    # edge type. 
    support_assmpt_rule = ([],[])
    support_non_assmpt_rule = ([],[])
    attack_non_assmpt_assmpt = ([],[])
    attack_assmpt_assmpt= ([],[])
    derive_rule_non_assmpt = ([],[])
    derive_rule_assmpt = ([],[])

    # Collect all node IDs of each type
    assmpt_nodes = set(assmpt_mapping.values())
    rule_nodes = set(rule_mapping.values())
    non_assmpt_nodes = set(non_assmpt_mapping.values())

    # Create self-connections for type 'assmpt', 'rule' and 'non_assmpt' nodes
    self_support_assmpt = (list(assmpt_nodes), list(assmpt_nodes))
    self_support_rule = (list(rule_nodes), list(rule_nodes))
    self_support_non_assmpt = (list(non_assmpt_nodes), list(non_assmpt_nodes))

    
    for u, v, d in graph.edges(data=True):
        # print("edge:", u, v, d)
        if d.get('label') == "+":
            if u in assmpt_mapping:
                support_assmpt_rule[0].append(assmpt_mapping[u])
                support_assmpt_rule[1].append(rule_mapping[v])
            elif u in non_assmpt_mapping:
                support_non_assmpt_rule[0].append(non_assmpt_mapping[u])
                support_non_assmpt_rule[1].append(rule_mapping[v])
            else:
                #TODO make this into a proper error
                print("ERROR PRODUCING GRAPH - INVALID NODES FOR + EDGE")
                return

        elif d.get('label') == '-':
            if u in assmpt_mapping:
                attack_assmpt_assmpt[0].append(assmpt_mapping[u])
                attack_assmpt_assmpt[1].append(assmpt_mapping[v])
            elif u in non_assmpt_mapping:
                attack_non_assmpt_assmpt[0].append(non_assmpt_mapping[u])
                attack_non_assmpt_assmpt[1].append(assmpt_mapping[v])
            else:
                #TODO make this into a proper error
                print("ERROR PRODUCING GRAPH - INVALID NODES FOR + EDGE")
                return

        elif d.get('label') == 'd':
            if v in assmpt_mapping:
                derive_rule_assmpt[0].append(rule_mapping[u])
                derive_rule_assmpt[1].append(assmpt_mapping[v])
            elif v in non_assmpt_mapping:
                derive_rule_non_assmpt[0].append(rule_mapping[u])
                derive_rule_non_assmpt[1].append(non_assmpt_mapping[v])
            else:
                print("ERROR PRODUCING GRAPH - INVALID NODES FOR d EDGE")
                return

        else:
            #TODO make this into a proper error
            print("ERROR PRODUCING GRAPH - INVALID LABEL")
            return
            
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

    # print(f"{data_dict=}")

    return dgl.heterograph(data_dict), data_dict 


def create_label_vector(file, mapping):
    array = np.zeros(len(mapping.keys()))
    with open(file, "r") as f:
        text = f.read().split("\n")
    
    for line in text:
        if line != '':
            index = mapping[line.strip()]
            array[index] = 1
    
    return array


def print_hetero_graph(g):
    # If you have node features, print them
    print("\nNode features (if available):")
    for ntype in g.ntypes:
        if g.nodes[ntype].data:  # Check if features exist
            for feature_name, feature_tensor in g.nodes[ntype].data.items():
                print(f"Node type '{ntype}', feature '{feature_name}':")
                for i in range(g.number_of_nodes(ntype)):
                    print(f"  Node {i}: {feature_tensor[i]}")
    
    # Print nodes labels for node type a 
    print("\nLabels for nodes of type 'assmpt':")    
    labels = g.nodes['assmpt'].data['label']
    for i in range(g.number_of_nodes('assmpt')):
        print(f"Node {i}: {labels[i]}")
        
    
    #Print nodes along with their connections
    print("\nNode connections:")
    for canonical_etype in g.canonical_etypes:
        src_type, rel_type, dst_type = canonical_etype
        src, dst = g.edges(etype=canonical_etype)
        print(f"Relation '{src_type}-{rel_type}->{dst_type}':")
        for i in range(len(src)):
            print(f"  {src_type}_{src[i].item()} -> {dst_type}_{dst[i].item()}")

if __name__ == "__main__":
    print("generating train dataset graphs")
    train_graphs1 = load_dataset(input_directory="./input_data_iccma", output_directory="./output_data_iccma", dataset_files="./train_test_splits/train_25_100_iccma.csv")
    train_graphs2 = load_dataset(input_directory="./input_data_iccma", output_directory="./output_data_iccma", dataset_files="./train_test_splits/train_rest_iccma.csv")
    dgl.save_graphs('train_iccma.bin', train_graphs1 + train_graphs2)
    print("generating train dataset graphs 2")
    train_graphs3 = load_dataset(input_directory="./input_data_generated", output_directory="./output_data_generated", dataset_files="./train_test_splits/train_generated.csv")
    train_graphs = train_graphs1 + train_graphs2 + train_graphs3
    dgl.save_graphs('train_all.bin', train_graphs)

    print("generating test dataset graphs")
    test_graphs1 = load_dataset(input_directory="./input_data_iccma", output_directory="./output_data_iccma", dataset_files="./train_test_splits/test_25_100_iccma.csv")
    dgl.save_graphs('test_25_100.bin', test_graphs1)
    test_graphs2 = load_dataset(input_directory="./input_data_iccma", output_directory="./output_data_iccma", dataset_files="./train_test_splits/test_rest_iccma.csv")
    dgl.save_graphs('test_iccma.bin', test_graphs1 + test_graphs2)
    print("generating test dataset graphs 2")
    test_graphs3 = load_dataset(input_directory="./input_data_generated", output_directory="./output_data_generated", dataset_files="./train_test_splits/test_generated.csv")
    test_graphs = test_graphs1 + test_graphs2 + test_graphs3
    dgl.save_graphs('test_all.bin', test_graphs)

