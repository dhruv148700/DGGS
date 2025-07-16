
import os

def jaccard_index(set1, set2):
    """Calculate Jaccard Index between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def calculate_instance_metrics(pred_set, gt_set, universe_set):
    """Calculate comprehensive metrics for a single instance."""
    # Basic set operations
    intersection = pred_set.intersection(gt_set)
    union = pred_set.union(gt_set)
    
    # Jaccard Index
    jaccard = jaccard_index(pred_set, gt_set)
    
    # Exact match (completely correct)
    exact_match = pred_set == gt_set
    
    # False positives and false negatives
    true_positives = intersection
    false_positives = pred_set - gt_set  # Predicted but not in ground truth
    false_negatives = gt_set - pred_set  # In ground truth but not predicted
    true_negatives = universe_set - pred_set - gt_set
    
    # Counts
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    tp_count = len(true_positives)
    tn_count = len(true_negatives)

    #calculate precision, recall, accuracy, F1
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    accuracy = (tp_count + tn_count) / (tp_count + fp_count + fn_count + tn_count) if (tp_count + fp_count + fn_count + tn_count) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Subset relationships
    pred_subset_of_gt = pred_set.issubset(gt_set)
    gt_subset_of_pred = gt_set.issubset(pred_set)
    
    return {
        'jaccard_index': jaccard,
        'exact_match': exact_match,
        'true_positives': tp_count,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'true_negatives': tn_count,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1,
        'pred_subset_of_gt': pred_subset_of_gt,
        'gt_subset_of_pred': gt_subset_of_pred,
    }


def find_best_matching_extension(predicted, expected_extensions, universe_set):
    """Find the ground truth extension that gives the best metrics for the predicted extension."""
    
    best_metrics = None
    best_index = -1
    best_score = -1

    print(f"total assumptions: {len(universe_set)}")
    
    for i, expected_extension in enumerate(expected_extensions):
        metrics = calculate_instance_metrics(predicted, expected_extension, universe_set)
        
        if metrics['jaccard_index'] > best_score:
            best_score = metrics['jaccard_index']
            best_metrics = metrics
            best_index = i
    
    return best_metrics, best_index

def aggregate_metrics(all_metrics):
    """Aggregate metrics across all instances."""
    if not all_metrics:
        return {}
    
    # Metrics to average
    avg_metrics = [
        'jaccard_index', 'true_positives', 'false_positives', 'false_negatives', 'true_negatives',
        'precision', 'recall', 'accuracy', 'f1_score', 'runtime'
    ]
    
    # Metrics to count/sum
    exact_matches = sum(1 for m in all_metrics if m['exact_match'])
    pred_subset_count = sum(1 for m in all_metrics if m['pred_subset_of_gt'])
    gt_subset_count = sum(1 for m in all_metrics if m['gt_subset_of_pred'])
    
    aggregated = {
        'total_instances': len(all_metrics),
        'exact_matches': exact_matches,
        'exact_match_rate': exact_matches / len(all_metrics),
        'pred_subset_of_gt_count': pred_subset_count,
        'pred_subset_of_gt_rate': pred_subset_count / len(all_metrics),
        'gt_subset_of_pred_count': gt_subset_count,
        'gt_subset_of_pred_rate': gt_subset_count / len(all_metrics),
    }
    
    # Calculate averages
    for metric in avg_metrics:
        values = [m[metric] for m in all_metrics]
        aggregated[f'avg_{metric}'] = sum(values) / len(values)
        aggregated[f'total_{metric}'] = sum(values)
    
    return aggregated

def print_detailed_results(instance_metrics, filename):
    """Print detailed results for a single instance."""
    print(f"\n=== Results for {filename} ===")
    print(f"Jaccard Index: {instance_metrics['jaccard_index']:.4f}")
    print(f"Exact Match: {'Yes' if instance_metrics['exact_match'] else 'No'}")
    print(f"True Positives: {instance_metrics['true_positives']}")
    print(f"False Positives: {instance_metrics['false_positives']}")
    print(f"False Negatives: {instance_metrics['false_negatives']}")
    print(f"True Negatives: {instance_metrics['true_negatives']}")
    print(f"Runtime: {instance_metrics['runtime']}")
    print(f"Precision: {instance_metrics['precision']:.4f}")
    print(f"Recall: {instance_metrics['recall']:.4f}")
    print(f"Accuracy: {instance_metrics['accuracy']:.4f}")
    print(f"F1 Score: {instance_metrics['f1_score']:.4f}")
    print(f"Predicted is subset of ground truth: {'Yes' if instance_metrics['pred_subset_of_gt'] else 'No'}")
    print(f"Ground truth is subset of predicted: {'Yes' if instance_metrics['gt_subset_of_pred'] else 'No'}")

def print_aggregate_results(aggregate_metrics):
    """Print aggregate results across all instances."""
    print("\n" + "="*50)
    print("AGGREGATE RESULTS ACROSS ALL INSTANCES")
    print("="*50)
    print(f"Total instances processed: {int(aggregate_metrics['total_instances'])}")
    print(f"Exact matches: {int(aggregate_metrics['exact_matches'])} ({aggregate_metrics['exact_match_rate']:.2%})")
    print(f"Predicted subset of ground truth: {int(aggregate_metrics['pred_subset_of_gt_count'])} ({aggregate_metrics['pred_subset_of_gt_rate']:.2%})")
    print(f"Ground truth subset of predicted: {int(aggregate_metrics['gt_subset_of_pred_count'])} ({aggregate_metrics['gt_subset_of_pred_rate']:.2%})")
    
    print(f"\nAverage Metrics:")
    print(f"  Jaccard Index: {aggregate_metrics['avg_jaccard_index']:.4f}")
    print(f"  Precision: {aggregate_metrics['avg_precision']:.4f}")
    print(f"  Recall: {aggregate_metrics['avg_recall']:.4f}")
    print(f"  Accuracy: {aggregate_metrics['avg_accuracy']:.4f}")
    print(f"  F1 Score: {aggregate_metrics['avg_f1_score']:.4f}")
    print(f"  Runtime: {aggregate_metrics['avg_runtime']:.4f}")
    
    print(f"\nTotal Counts:")
    print(f"  True Positives: {int(aggregate_metrics['total_true_positives'])}")
    print(f"  False Positives: {int(aggregate_metrics['total_false_positives'])}")
    print(f"  False Negatives: {int(aggregate_metrics['total_false_negatives'])}")
    print(f"  True Negatives: {int(aggregate_metrics['total_true_negatives'])}")

# Example usage:
if __name__ == "__main__":
    # Example with universe set
    pred_set = {1, 2, 3}
    gt_set = {2, 3, 4}
    universe_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  # All possible elements
    
    metrics = calculate_instance_metrics(pred_set, gt_set, universe_set)
    print_detailed_results(metrics, "example")