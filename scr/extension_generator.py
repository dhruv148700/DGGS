from aba_inference import ABAInferenceEngine
from hetero_graph_utils import create_graph, update_graph
from metrics import find_best_matching_extension, aggregate_metrics, print_detailed_results, print_aggregate_results
import os
import time
import json
import argparse

def build_extension(aba_file_path, enumeration_threshold, model_type):
    model_path = f"results_final_{model_type}/trained_model.pt"
    aba_inference_engine = ABAInferenceEngine(model_type, model_path, enumeration_threshold)
    hetero_graph, dependency_graph, assmpt_mapping = create_graph(aba_file_path)
    all_assumptions = dependency_graph.assumptions.copy()

    print(f"Graph stats: {hetero_graph.number_of_nodes()} nodes total")
    for ntype in hetero_graph.ntypes:
        print(f"  - {ntype}: {hetero_graph.number_of_nodes(ntype)} nodes")
    
    predictions = aba_inference_engine.inference(hetero_graph, assmpt_mapping)

    extension = set()

    if not predictions:
        print(f"No assumptions accepted, {extension=}")
        return extension, all_assumptions
    else: 
        (assumption_name, probability, accepted) = predictions if isinstance(predictions, tuple) else predictions[0]
        if not accepted:
            print(f"No assumptions accepted, {extension=}")
            return extension, all_assumptions

    i = 1
    while True:
        if assumption_name.startswith('dummy'):
            break
        
        extension.add(assumption_name)
        print(f"assumption to remove {assumption_name}")
        result = dependency_graph.remove_accepted_assumption(assumption_name)

        if not result:
            break

        extension.add(assumption_name)

        if (not dependency_graph.assumptions):
            break

        if not dependency_graph.rules:
            print("NO RULES")

        dependency_graph.create_dependency_graph()
        hetero_graph, dependency_graph, assmpt_mapping = update_graph(dependency_graph)
        i+=1
        predictions = aba_inference_engine.inference(hetero_graph, assmpt_mapping)
        if not predictions:
            print("remaining predictions are too low")
            break
        else:
            (assumption_name, probability, accepted) = predictions if isinstance(predictions, tuple) else predictions[0]
            if not accepted:
                print(f"No further assumptions accepted, {extension=}")
                break

    return extension, all_assumptions
    

def extract_extensions(output_file_path):
    extensions = []
    
    with open(output_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            if not line:
                continue
            
            # Split by comma, strip whitespace, and filter out empty strings
            elements = [item.strip() for item in line.split(',')]
            elements = [item for item in elements if item]  # Remove empty strings
            line_set = set(elements)
            
            # Only add non-empty sets
            if line_set:
                extensions.append(line_set)
                    
    return extensions if len(extensions) > 0 else None

def save_aggregated_results(aggregate_results, json_file="threshold_results.json"):
    """Save or append aggregated results to JSON file."""
    
    # Try to load existing results
    existing_results = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        print(f"Loaded {len(existing_results)} existing results from {json_file}")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Creating new results file: {json_file}")
    
    # Check if this threshold already exists
    threshold = aggregate_results["threshold"]
    existing_thresholds = [r.get("threshold") for r in existing_results]
    
    if threshold in existing_thresholds:
        print(f"WARNING: Threshold {threshold} already exists. Replacing existing entry.")
        # Remove existing entry with same threshold
        existing_results = [r for r in existing_results if r.get("threshold") != threshold]
    
    # Add new results
    existing_results.append(aggregate_results)
    
    # Sort by threshold for easier reading
    existing_results.sort(key=lambda x: x.get("threshold", 0))
    
    # Save back to file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"Results saved to {json_file}. Total thresholds: {len(existing_results)}")
    return existing_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate extensions using ABAInferenceEngine with GCN or GAT model.")
    parser.add_argument('--aba_file', type=str, default="data/example1.aba", help="Path to the ABA file to process")
    parser.add_argument('--inclusion_threshold', type=float, default=None, help="Inclusion threshold")
    parser.add_argument('--model_type', type=str, required=True, choices=['gcn', 'gat'], help="Model type: 'gcn' or 'gat'")
    args = parser.parse_args()

    aba_file_path = args.aba_file
    inclusion_threshold = args.inclusion_threshold
    model_type = args.model_type

    start_time = time.time()
    extension, assumptions = build_extension(aba_file_path, inclusion_threshold, model_type)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")
    print(f"{extension=}")
    print(f"{assumptions=}")

        
