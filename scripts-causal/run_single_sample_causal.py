"""
run_single_sample_causal.py
───────────────────────────
Single-sample end-to-end trace using a pre-generated .aba from
input_data_causal/ and its ground-truth extension from output_data_causal/.

Place at the GNN4ABA repo root or run from scripts-causal/:

    python scripts-causal/run_single_sample_causal.py
"""

import os
import sys
import re
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scr"))

# ─── Setup ────────────────────────────────────────────────────────────────────

SAMPLE_NAME = "causal_ba_n5_dna_m1_a0.01_i0_bsat"

INPUT_DIR  = os.path.join(REPO_ROOT, "input_data_causal")
TIER_LABELS_DIR = os.path.join(REPO_ROOT, "dataset", "tier_labels")
DAG_DIR    = os.path.join(REPO_ROOT, "dag_ground_truth")

ABA_PATH = os.path.join(INPUT_DIR,  f"{SAMPLE_NAME}.aba")
GT_TIER_LABELS_PATH = os.path.join(TIER_LABELS_DIR, f"{SAMPLE_NAME}.json")

# dag_ground_truth files strip the leading "causal_" and trailing role suffix (e.g. "_full")
_dag_core = re.sub(r"^causal_", "", SAMPLE_NAME)
_dag_core = re.sub(r"_[^_]+$", "", _dag_core)   # strip role suffix (_full, _train, etc.)
DAG_PATH  = os.path.join(DAG_DIR, f"dag_{_dag_core}.npy")

arr_pattern = re.compile(r"^arr_(\d+)_(\d+)$")
n_nodes_re  = re.compile(r"_n(\d+)_")

print("=" * 70)
print(f"Single-sample trace  |  sample={SAMPLE_NAME}")
print("=" * 70)

# ─── CHECKPOINT 1 ─────────────────────────────────────────────────────────────
print("\n[1] Loading input .aba file")
print("-" * 70)

with open(ABA_PATH) as fh:
    aba_lines = fh.readlines()

aba_size = os.path.getsize(ABA_PATH)

m_n = n_nodes_re.search(SAMPLE_NAME)
if not m_n:
    raise ValueError(f"Cannot parse n_nodes from filename: {SAMPLE_NAME!r}")
n_nodes = int(m_n.group(1))

arr_count   = sum(1 for ln in aba_lines if ln.startswith("a arr_"))
noe_count   = sum(1 for ln in aba_lines if ln.startswith("a noe_"))
indep_count = sum(1 for ln in aba_lines if ln.startswith("a indep_"))
bp_count    = sum(1 for ln in aba_lines if ln.startswith("a blocked_path_"))

print(f"  file             : {ABA_PATH}")
print(f"  size             : {aba_size} bytes, {len(aba_lines)} lines")
print(f"  n_nodes          : {n_nodes}  (from filename)")
print(f"  assumption counts: arr={arr_count}  noe={noe_count}  "
      f"indep={indep_count}  blocked_path={bp_count}")
print(f"  first 15 lines:")
for line in aba_lines[:15]:
    print(f"    {line.rstrip()}")

# ─── CHECKPOINT 2 ─────────────────────────────────────────────────────────────
# Load tier labels from JSON. Extract assumptions labeled as "skeptical" or "credulous"
# (ground truth in extension). Other tiers ("no_ext", "rejected") are treated as not in extension.
print("\n[2] Loading ground-truth extensions")
print("-" * 70)

import json
with open(GT_TIER_LABELS_PATH) as fh:
    tier_labels = json.load(fh)

gt_extension = frozenset(a for a, tier in tier_labels.items() if tier in ("skeptical", "credulous"))

print(f"  file                : {GT_TIER_LABELS_PATH}")
print(f"  total assumptions   : {len(tier_labels)}")
print(f"  in extension (skeptical+credulous): {len(gt_extension)}")
if not gt_extension:
    print("  (UNSAT — no assumptions in extension)")
else:
    arrows = sorted(a for a in gt_extension if arr_pattern.match(a))
    print(f"  extension size={len(gt_extension)}  arrows={arrows}")

gt_extensions = [gt_extension]  # Wrap in list for compatibility with downstream code

# ─── CHECKPOINT 3 ─────────────────────────────────────────────────────────────
print("\n[3] Loading ground-truth DAG (B_true)")
print("-" * 70)

B_true = np.load(DAG_PATH)

print(f"  file   : {DAG_PATH}")
print(f"  B_true :\n{B_true}")

# ─── CHECKPOINT 4 ─────────────────────────────────────────────────────────────
print("\n[4] Running GNN inference")
print("-" * 70)

from scr.extension_generator import build_extension

extension, all_assumptions = build_extension(
    aba_file_path=ABA_PATH,
    enumeration_threshold=None,
    model_type="gcn",
    model_path=os.path.join(REPO_ROOT, "results", "results_full_gcn_a40_b64", "best_model.pt"),
)

print(f"  total assumptions : {len(all_assumptions)}")
print(f"  extension size    : {len(extension)}")
print(f"  extension sample (first 15):")
for a in list(extension)[:15]:
    print(f"    {a}")

# ─── CHECKPOINT 5 ─────────────────────────────────────────────────────────────
print("\n[5] Extension-level evaluation vs ground truth")
print("-" * 70)

if not gt_extensions:
    print("  Ground truth is UNSAT — skipping extension match.")
else:
    for idx, gt_ext in enumerate(gt_extensions):
        match = frozenset(extension) == gt_ext
        print(f"  Match ext[{idx}]: {match}")
        if not match:
            extra   = frozenset(extension) - gt_ext
            missing = gt_ext - frozenset(extension)
            print(f"    In GNN but not GT : {sorted(extra)[:10]}")
            print(f"    In GT  but not GNN: {sorted(missing)[:10]}")

# ─── CHECKPOINT 6 ─────────────────────────────────────────────────────────────
print("\n[6] Arrow-matrix comparison (GNN vs B_true)")
print("-" * 70)

from scr.causal_aba.utils import get_matrix_from_arrow_set

def arrows_from_ext(ext):
    return {(int(m.group(1)), int(m.group(2)))
            for a in ext if (m := arr_pattern.match(a))}

gnn_arrows = arrows_from_ext(extension)
B_est = get_matrix_from_arrow_set(gnn_arrows, n_nodes)
print(f"  GNN arrows : {sorted(gnn_arrows)}")
print(f"  B_est:\n{B_est}")

from ArgCausalDisco.utils.graph_utils import DAGMetrics

n_ref_edges = max(int(B_true.sum()), 1)
metrics = DAGMetrics(B_est, B_true).metrics
print(f"\n  vs B_true")
print(f"    TPR/recall : {metrics.get('tpr'):.3f}")
print(f"    precision  : {metrics.get('precision'):.3f}")
print(f"    F1         : {metrics.get('F1'):.3f}")
print(f"    SHD        : {metrics.get('shd')}")
print(f"    SID        : {metrics.get('sid')}  "
      f"(norm {metrics.get('sid') / n_ref_edges:.3f})")

print("\n" + "=" * 70)
print("Single-sample trace complete.")
print("=" * 70)
