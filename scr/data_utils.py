import json
import numpy as np
import random
import torch
import dgl
import os
import sys
from pathlib import Path
from scr.dependency_graph import DependencyGraph
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


def create_causal_label_vector(file, mapping):
    """
    Credulous-acceptance label vector for causal ABA data.

    Label files written by write_label_file in generate_data_causal.py have
    one line per stable extension, with assumption names comma-separated.
    An assumption is labelled 1 if it appears in ANY extension (credulous).
    Assumption names absent from mapping are silently skipped (they may be
    non-assumption atoms or names not present after graph construction).
    """
    array = np.zeros(len(mapping.keys()))
    with open(file, "r") as f:
        text = f.read().split("\n")

    for line in text:
        line = line.strip()
        if not line:
            continue
        for assumption in line.split(","):
            assumption = assumption.strip()
            if assumption and assumption in mapping:
                array[mapping[assumption]] = 1

    return array


def create_tier_label_vector(tier_path: str, mapping: dict) -> np.ndarray:
    """Binary label vector from a pre-computed tier_labels JSON.

    skeptical or credulous → 1  (credulously accepted)
    rejected or no_ext     → 0
    """
    with open(tier_path) as fh:
        tiers = json.load(fh)
    array = np.zeros(len(mapping))
    for name, idx in mapping.items():
        if tiers.get(name, "rejected") in ("skeptical", "credulous"):
            array[idx] = 1
    return array


def _load_atom_scores(f_scores: str) -> dict:
    """Load the atom-level scores from a .scores.json file.

    Only string-keyed entries are returned (rule-tuple keys use '|' as
    separator and map to rule structures, not atoms).
    Returns an empty dict if the file is missing or unreadable.
    """
    try:
        import json as _json
        with open(f_scores) as fh:
            raw = _json.load(fh)
        return {k: v for k, v in raw.items() if "|" not in k}
    except Exception:
        return {}


def _build_causal_graph(f_abaf: str, f_labels: str, f_scores: str = None, f_tier: str = None):
    """Build one DGL heterograph from a single .aba / labels file pair.

    When f_scores is provided the CI-test reliability score for each atom is
    appended as a third node feature (0.0 for atoms not present in the score
    map, i.e. arr_*, noe_*, and structural non-assumptions). Node feature
    tensors are then shape (n, 3) instead of (n, 2).

    Raises on any parse or graph-construction error so the caller can
    record a structured failure rather than crashing the whole build.
    """
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(f_abaf)
    dep_graph.create_dependency_graph()

    rules_mapping, assmpt_mapping, non_assmpt_mapping = reindex_nodes(dep_graph)
    features = dep_graph.calculate_node_features(assmpt_mapping | non_assmpt_mapping)
    hetero_graph, _ = create_hetero_graph(
        dep_graph.graph, rules_mapping, assmpt_mapping, non_assmpt_mapping
    )

    atom_scores = _load_atom_scores(f_scores) if f_scores else {}
    n_feat = 3 if f_scores is not None else 2

    assmpt_feat_arr = np.zeros((len(assmpt_mapping), n_feat))
    for key in assmpt_mapping:
        idx = assmpt_mapping[key]
        assmpt_feat_arr[idx, :2] = features[key]
        if f_scores is not None:
            assmpt_feat_arr[idx, 2] = atom_scores.get(key, 0.0)
    hetero_graph.nodes["assmpt"].data["features"] = torch.tensor(
        assmpt_feat_arr, dtype=torch.float32
    )

    non_assmpt_feat_arr = np.zeros((len(non_assmpt_mapping), n_feat))
    for key in non_assmpt_mapping:
        idx = non_assmpt_mapping[key]
        non_assmpt_feat_arr[idx, :2] = features[key]
        if f_scores is not None:
            non_assmpt_feat_arr[idx, 2] = atom_scores.get(key, 0.0)
    hetero_graph.nodes["non_assmpt"].data["features"] = torch.tensor(
        non_assmpt_feat_arr, dtype=torch.float32
    )

    hetero_graph.nodes["rule"].data["features"] = torch.tensor(
        np.random.randn(len(rules_mapping), n_feat), dtype=torch.float32
    )

    if f_tier is not None:
        label_vector = create_tier_label_vector(f_tier, assmpt_mapping)
    else:
        label_vector = create_causal_label_vector(f_labels, assmpt_mapping)
    hetero_graph.nodes["assmpt"].data["label"] = torch.tensor(
        label_vector, dtype=torch.float32
    )
    return hetero_graph


def load_causal_dataset_from_manifest(
    entries: list,
    base_dir: str = ".",
    use_scores: bool = False,
    tier_dir: str = None,
) -> tuple:
    """Build DGL heterographs for a list of manifest entries.

    Uses the exact 'abaf' and 'labels' paths from each entry rather than
    inferring the label filename from the input filename.

    Args:
        entries:    list of manifest dicts with at minimum 'abaf' and 'labels'.
        base_dir:   root for resolving relative paths (default: current dir).
        use_scores: if True and the entry has a 'scores' field, pass the
                    .scores.json path to _build_causal_graph so CI-test
                    reliability scores are added as a 3rd node feature.
                    Graphs will have feature tensors of shape (n, 3) instead
                    of (n, 2). Default False for backwards compatibility.
        tier_dir:   directory containing per-instance tier_labels JSONs
                    (default: <base_dir>/dataset/tier_labels).  When a
                    tier JSON exists for an entry it is used for labels
                    (skeptical|credulous→1, rejected→0) instead of the raw
                    extension file, so output_data_causal is not needed.

    Returns:
        (graphs, metadata_list, failed_entries) where graphs and metadata_list
        are parallel lists (same order, failures excluded), and failed_entries
        is a list of {'entry', 'error', 'stage'} dicts.
    """
    if tier_dir is None:
        tier_dir = os.path.join(base_dir, "dataset", "tier_labels")

    graphs: list = []
    metadata_list: list = []
    failed_entries: list = []
    n_total = len(entries)

    for idx, entry in enumerate(entries):
        if idx % 100 == 0:
            print(f"  [{idx}/{n_total}] {entry.get('instance_id', '?')}")

        def _resolve(p):
            return p if os.path.isabs(p) else os.path.join(base_dir, p)

        f_abaf   = _resolve(entry["abaf"])
        f_labels = _resolve(entry["labels"])
        f_scores = _resolve(entry["scores"]) if use_scores and entry.get("scores") else None

        key    = Path(f_abaf).stem
        f_tier = os.path.join(tier_dir, f"{key}.json")
        if not os.path.isfile(f_tier):
            f_tier = None  # fall back to raw labels file

        missing = []
        if not os.path.isfile(f_abaf):
            missing.append((f_abaf, "abaf"))
        if f_tier is None and not os.path.isfile(f_labels):
            missing.append((f_labels, "labels"))
        if missing:
            failed_entries.append({
                "entry": entry,
                "error": "; ".join(f"not found: {p}" for p, _ in missing),
                "stage": "file_check",
            })
            continue

        try:
            g = _build_causal_graph(f_abaf, f_labels, f_scores, f_tier)
            graphs.append(g)
            metadata_list.append(entry)
        except Exception as exc:
            failed_entries.append({
                "entry": entry,
                "error": str(exc),
                "stage": "graph_build",
            })

    print(
        f"load_causal_dataset_from_manifest: "
        f"{len(graphs)} built, {len(failed_entries)} failed / {n_total} total"
    )
    return graphs, metadata_list, failed_entries


def load_causal_dataset(input_directory, output_directory, dataset_files=None):
    """
    Like load_dataset but uses create_causal_label_vector for credulous-
    acceptance labels written in the comma-separated-per-extension format
    produced by generate_data_causal.py.
    """
    if dataset_files:
        input_files = []
        with open(dataset_files, "r") as fh:
            for line in fh:
                clean = line.strip()
                if clean:
                    input_files.append(clean)
    else:
        input_files = os.listdir(input_directory)

    graphs = []
    n_total = len(input_files)
    for idx, filename in enumerate(input_files):
        if idx % 50 == 0:
            print(f"  [{idx}/{n_total}] {filename}")
        f_input  = os.path.join(input_directory, filename)
        f_output = os.path.join(output_directory, f"output_{filename}")

        if not os.path.isfile(f_input) or not os.path.isfile(f_output):
            print(f"skipping {filename}")
            continue

        dep_graph = DependencyGraph()
        dep_graph.create_from_file(f_input)
        dep_graph.create_dependency_graph()

        rules_mapping, assmpt_mapping, non_assmpt_mapping = reindex_nodes(dep_graph)
        features = dep_graph.calculate_node_features(assmpt_mapping | non_assmpt_mapping)
        hetero_graph, _ = create_hetero_graph(
            dep_graph.graph, rules_mapping, assmpt_mapping, non_assmpt_mapping
        )

        assmpt_feat_arr = np.empty((len(assmpt_mapping), 2))
        for key in assmpt_mapping:
            assmpt_feat_arr[assmpt_mapping[key], :] = features[key]
        hetero_graph.nodes["assmpt"].data["features"] = torch.tensor(
            assmpt_feat_arr, dtype=torch.float32
        )

        non_assmpt_feat_arr = np.empty((len(non_assmpt_mapping), 2))
        for key in non_assmpt_mapping:
            non_assmpt_feat_arr[non_assmpt_mapping[key], :] = features[key]
        hetero_graph.nodes["non_assmpt"].data["features"] = torch.tensor(
            non_assmpt_feat_arr, dtype=torch.float32
        )

        hetero_graph.nodes["rule"].data["features"] = torch.tensor(
            np.random.randn(len(rules_mapping), 2), dtype=torch.float32
        )

        label_vector = create_causal_label_vector(f_output, assmpt_mapping)
        hetero_graph.nodes["assmpt"].data["label"] = torch.tensor(
            label_vector, dtype=torch.float32
        )
        graphs.append(hetero_graph)

    print(f"Loaded {len(graphs)} causal graphs")
    return graphs


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

