import sys
import os

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import dgl
import numpy as np

# Load training set
train_set, _ = dgl.load_graphs('splits/train.bin')

# Collect all labels from training set
all_labels = []

print("Collecting labels from training set...")
for graph in train_set:
    labels = graph.nodes['assmpt'].data['label'].numpy()
    all_labels.extend(labels)

all_labels = np.array(all_labels)

# Calculate statistics
n_total = len(all_labels)
n_positive = np.sum(all_labels)
n_negative = n_total - n_positive
pos_ratio = n_positive / n_total
neg_ratio = n_negative / n_total

# Calculate pos_weight
# pos_weight = n_negative / n_positive (standard way to balance classes)
pos_weight = n_negative / n_positive

print("\n" + "="*60)
print("Training Set Class Distribution")
print("="*60)
print(f"Total samples:     {n_total:,}")
print(f"Positive (1):      {n_positive:,} ({pos_ratio:.4f} = {pos_ratio*100:.2f}%)")
print(f"Negative (0):      {n_negative:,} ({neg_ratio:.4f} = {neg_ratio*100:.2f}%)")
print("="*60)
print(f"Calculated pos_weight: {pos_weight:.4f}")
print("="*60)
