import sys
import os

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import dgl
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from dgl.dataloading import GraphDataLoader

from scr.data_utils import set_seeds
from scr.GCN_learnable import GCNLearnableModel

# Configuration
RESULTS_DIR = "results/results_full_gcn_a40_b64"
MODEL_PATH = f"{RESULTS_DIR}/best_model.pt"
RESULTS_JSON = f"{RESULTS_DIR}/results.json"

# Load hyperparameters from results.json
with open(RESULTS_JSON, 'r') as f:
    results_data = json.load(f)
    hyperparams = results_data['hyperparams']

EMBEDDING_DIM = hyperparams['embedding_dim']
HIDDEN_DIM = hyperparams['hidden_dim']
NUM_LAYERS = hyperparams['num_layers']
DROPOUT = hyperparams['dropout']
BATCH_SIZE = 64
NUM_WORKERS = 0

# Device setup
USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Set seed for reproducibility
set_seeds(42)

# Load model
model = GCNLearnableModel(
    in_features=2,
    hidden_features=HIDDEN_DIM,
    out_features=1,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Load validation data directly
val_set, _ = dgl.load_graphs('splits/val.bin')

# Create validation DataLoader
val_loader = GraphDataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS
)

# Collect all predictions and labels
all_logits = []
all_labels = []

print("Generating predictions on validation set...")
with torch.no_grad():
    for batched_graph in val_loader:
        batched_graph = batched_graph.to(device)

        inputs = {
            'assmpt': batched_graph.nodes['assmpt'].data['features'],
            'rule': batched_graph.nodes['rule'].data['features'],
            'non_assmpt': batched_graph.nodes['non_assmpt'].data['features']
        }

        labels = batched_graph.nodes['assmpt'].data['label']

        outputs = model(batched_graph, inputs)
        logits = outputs['assmpt'].squeeze(1)

        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_logits = np.array(all_logits)
all_labels = np.array(all_labels)
all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid

print(f"Validation set size: {len(all_labels)}")
print(f"Positive class ratio: {all_labels.mean():.4f}")

# Define threshold points for sweep
# Use dense sampling around the typical range
thresholds = np.arange(0.0, 1.01, 0.05)  # 0.0, 0.05, 0.10, ..., 1.0

# Store results
threshold_results = []

print("\nThreshold Sweep Results:")
print("=" * 80)
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
print("=" * 80)

for threshold in thresholds:
    predictions = (all_probs > threshold).astype(int)

    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    accuracy = (predictions == all_labels).mean()

    threshold_results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    })

    print(f"{threshold:<12.3f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {accuracy:<12.4f}")

print("=" * 80)

# Find optimal threshold by F1
best_idx = np.argmax([r['f1'] for r in threshold_results])
best_result = threshold_results[best_idx]
print(f"\nBest F1 threshold: {best_result['threshold']:.3f} with F1 = {best_result['f1']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")

# Create precision-recall plot with F1 scores annotated
fig, ax = plt.subplots(figsize=(10, 8))

precisions = [r['precision'] for r in threshold_results]
recalls = [r['recall'] for r in threshold_results]
f1_scores_list = [r['f1'] for r in threshold_results]
thresholds_list = [r['threshold'] for r in threshold_results]

# Plot PR curve
ax.plot(recalls, precisions, 'b-', linewidth=2, marker='o', markersize=6, label='PR Curve')

# Annotate each point with its threshold and F1 score
for i, (recall, prec, f1_val, thr) in enumerate(zip(recalls, precisions, f1_scores_list, thresholds_list)):
    ax.annotate(
        f"t={thr:.2f}\nF1={f1_val:.4f}",
        xy=(recall, prec),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
    )

# Highlight the best F1 point
best_recall = recalls[best_idx]
best_prec = precisions[best_idx]
ax.plot(best_recall, best_prec, 'r*', markersize=20, label=f'Best F1 (t={best_result["threshold"]:.3f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve with Threshold Sweep (GCN Full A40 B64)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/threshold_tuning_pr_curve.png", dpi=150, bbox_inches='tight')
print(f"\nPrecision-Recall plot saved to: {RESULTS_DIR}/threshold_tuning_pr_curve.png")

# Save results to JSON
with open(f"{RESULTS_DIR}/threshold_tuning_results.json", 'w') as f:
    json.dump({
        'threshold_sweep': threshold_results,
        'best_threshold': best_result['threshold'],
        'best_f1': best_result['f1'],
        'sweep_parameters': {
            'start': float(thresholds[0]),
            'end': float(thresholds[-1]),
            'step': float(thresholds[1] - thresholds[0]),
            'total_points': len(thresholds)
        }
    }, f, indent=2)
print(f"Results saved to: {RESULTS_DIR}/threshold_tuning_results.json")

plt.show()
