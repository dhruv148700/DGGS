import sys
import os

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import dgl
import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dgl.dataloading import GraphDataLoader

from scr.data_utils import set_seeds
from scr.GCN_learnable import GCNLearnableModel

# Configuration
RESULTS_DIR = "results/results_full_gcn_a40_b64"
MODEL_PATH = f"{RESULTS_DIR}/best_model.pt"
RESULTS_JSON = f"{RESULTS_DIR}/results.json"
THRESHOLD = 0.650  # Best F1 threshold from tuning

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

# Load test data
test_set, _ = dgl.load_graphs('splits/test.bin')

# Create test DataLoader
test_loader = GraphDataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS
)

# Collect all predictions and labels
all_logits = []
all_labels = []

print(f"Evaluating on test set with threshold={THRESHOLD}...")
with torch.no_grad():
    for batched_graph in test_loader:
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

predictions = (all_probs > THRESHOLD).astype(int)

# Calculate metrics
precision = precision_score(all_labels, predictions, zero_division=0)
recall = recall_score(all_labels, predictions, zero_division=0)
f1 = f1_score(all_labels, predictions, zero_division=0)
accuracy = accuracy_score(all_labels, predictions)

print(f"\nTest Set Results (Threshold={THRESHOLD}):")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("=" * 60)

# Save results
results = {
    'threshold': THRESHOLD,
    'test_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'n_predictions': len(predictions),
        'n_positive': int(np.sum(predictions)),
        'n_positive_true': int(np.sum(all_labels))
    }
}

with open(f"{RESULTS_DIR}/test_eval_threshold_{THRESHOLD}.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {RESULTS_DIR}/test_eval_threshold_{THRESHOLD}.json")
