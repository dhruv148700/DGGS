import torch
import dgl
import argparse
from dependency_graph import DependencyGraph
from data_utils import reindex_nodes, create_hetero_graph, set_seeds
from aba_inference import ABAInferenceEngine
import numpy as np

def predict_cred_accept(model_type, aba_file, print_result=True, seed=42):
    set_seeds(seed)
    MODEL_PATH = f'results_final_{model_type}/trained_model.pt'

    # --- LOAD AND PROCESS ABA FRAMEWORK ---
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(aba_file)
    dep_graph.create_dependency_graph()
    rules_mapping, assmpt_mapping, non_assmpt_mapping = reindex_nodes(dep_graph)

    # Calculate node features
    features = dep_graph.calculate_node_features({**assmpt_mapping, **non_assmpt_mapping})

    # Build DGL heterograph
    hetero_graph, _ = create_hetero_graph(
        dep_graph.graph,
        rules_mapping,
        assmpt_mapping,
        non_assmpt_mapping
    )

    hetero_graph.nodes['assmpt'].data['features'] = torch.tensor(
        np.array([features[k] for k in assmpt_mapping]), dtype=torch.float32
    )
    hetero_graph.nodes['non_assmpt'].data['features'] = torch.tensor(
        np.array([features[k] for k in non_assmpt_mapping]), dtype=torch.float32
    )
    hetero_graph.nodes['rule'].data['features'] = torch.randn(len(rules_mapping), 2)  # Random features for rules

    # --- INFERENCE ENGINE ---
    engine = ABAInferenceEngine(model_type, MODEL_PATH)
    results = engine.inference(hetero_graph, assmpt_mapping)

    # Extract accepted assumptions
    if results is None:
        accepted_assumptions = []
    else:
        # If engine.inference returns a list of tuples, filter for accepted
        if isinstance(results, list):
            accepted_assumptions = [assump for assump, prob, accepted in results if accepted]
        else:
            # If engine.inference returns a single tuple (for top result)
            accepted_assumptions = [results[0]] if results[2] else []

    if print_result:
        print("Accepted assumptions:")
        print(accepted_assumptions)
    return accepted_assumptions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict accepted assumptions from an ABA file using a trained GCN or GAT model.")
    parser.add_argument('--model_type', type=str, required=True, choices=['gcn', 'gat'], help="Model type: 'gcn' or 'gat'")
    parser.add_argument('--aba_file', type=str, required=True, help="Path to the ABA framework file")
    parser.add_argument('--no-print', action='store_true', help="Do not print the accepted assumptions, only return them.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    predict_cred_accept(args.model_type, args.aba_file, print_result=not args.no_print, seed=args.seed)