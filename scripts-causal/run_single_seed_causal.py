"""
run_single_seed.py
──────────────────
Single-seed end-to-end trace: bnlearn BIF → PC facts → .aba → GNN → B_est → SID.

Place at the GNN4ABA repo root.

    python run_single_seed.py
"""

import os
import sys
import re
import numpy as np

# Files inside scr/ use bare imports (e.g. `from dependency_graph import ...`)
# that assume scr/ itself is on sys.path. Adding it here lets those work without
# having to rewrite every import site.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scr"))

# ─── Setup ────────────────────────────────────────────────────────────────────

SEED = 0
DATASET = "cancer"
SAMPLE_SIZE = 5000
ABA_OUT = "single_seed.aba"

# load_bn_from_BIF joins: DATA_PATH / 'bayesian' / <size> / <name>.bif / <name>.bif
# so DATA_PATH must be the parent of bayesian/, not bayesian/ itself.
DATA_PATH = "ArgCausalDisco/datasets"

np.random.seed(SEED)

print("="*70)
print(f"Single-seed trace  |  dataset={DATASET}  seed={SEED}")
print("="*70)

# ─── CHECKPOINT 1 ─────────────────────────────────────────────────────────────
print("\n[1] Loading BIF and simulating data")
print("-"*70)

from ArgCausalDisco.utils.data_utils import load_bnlearn_data_dag

X_s, B_true = load_bnlearn_data_dag(
    dataset_name=DATASET,
    data_path=DATA_PATH,
    sample_size=SAMPLE_SIZE,
    seed=SEED,
    standardise=True,
    print_info=True,
)

n_nodes = B_true.shape[0]

print(f"  X_s shape       : {X_s.shape}")
print(f"  n_nodes         : {n_nodes}")
print(f"  B_true:\n{B_true}")
print(f"  n edges (true)  : {int(B_true.sum())}")

# ─── CHECKPOINT 2 ─────────────────────────────────────────────────────────────
print("\n[2] Running PC and extracting facts")
print("-"*70)

from scr.causal_aba.abapc import get_cg_and_facts

cg, facts = get_cg_and_facts(X_s)

print(f"  CPDAG object    : {type(cg).__name__}")
print(f"  n facts         : {len(facts)}")
print(f"  first 5 facts:")
for f in facts[:5]:
    print(f"    {f}")

# ─── CHECKPOINT 3 ─────────────────────────────────────────────────────────────
print("\n[3] Writing .aba file")
print("-"*70)

from scr.causal_aba.lp_to_aba_translator import lp_facts_to_aba_file

lp_facts_to_aba_file(
    facts,
    n_nodes=n_nodes,
    out_path=ABA_OUT,
    optimise_remove_edges=True,
)

aba_size = os.path.getsize(ABA_OUT)
with open(ABA_OUT) as fh:
    aba_lines = fh.readlines()

print(f"  file written    : {ABA_OUT}  ({aba_size} bytes, {len(aba_lines)} lines)")
print(f"  first 15 lines:")
for line in aba_lines[:15]:
    print(f"    {line.rstrip()}")

arr_count   = sum(1 for ln in aba_lines if ln.startswith("a arr_"))
noe_count   = sum(1 for ln in aba_lines if ln.startswith("a noe_"))
indep_count = sum(1 for ln in aba_lines if ln.startswith("a indep_"))
bp_count    = sum(1 for ln in aba_lines if ln.startswith("a blocked_path_"))
print(f"  assumption counts: arr={arr_count}  noe={noe_count}  "
      f"indep={indep_count}  blocked_path={bp_count}")

# ─── CHECKPOINT 4 ─────────────────────────────────────────────────────────────
print("\n[4] Running GNN inference")
print("-"*70)

from scr.extension_generator import build_extension

# build_extension returns (extension_set, all_assumptions)
# enumeration_threshold=None uses the model's default cutoff
extension, all_assumptions = build_extension(
    aba_file_path=ABA_OUT,
    enumeration_threshold=None,
    model_type="gcn",
)

print(f"  total assumptions: {len(all_assumptions)}")
print(f"  extension size   : {len(extension)}")
print(f"  extension sample (first 15):")
for a in list(extension)[:15]:
    print(f"    {a}")

# ─── CHECKPOINT 5 ─────────────────────────────────────────────────────────────
print("\n[5] Filtering arrow assumptions → integer tuples")
print("-"*70)

# get_matrix_from_arrow_set expects (i, j) integer tuples, not strings.
# Parse assumption names like "arr_0_1" → (0, 1).
arr_pattern = re.compile(r"^arr_(\d+)_(\d+)$")

arrow_set = set()
for name in extension:
    m = arr_pattern.match(name)
    if m:
        i, j = int(m.group(1)), int(m.group(2))
        arrow_set.add((i, j))

print(f"  n arrow tuples  : {len(arrow_set)}")
print(f"  arrows          : {sorted(arrow_set)}")

if len(arrow_set) == 0:
    print("\n  !!! WARNING: no arr_* assumptions accepted.")
    print("      Inspect the full extension above to check naming convention.")
    suspicious = [a for a in extension if 'arr' in a.lower()][:10]
    if suspicious:
        print(f"      Arrow-like names in extension: {suspicious}")

# ─── CHECKPOINT 6 ─────────────────────────────────────────────────────────────
print("\n[6] Building B_est")
print("-"*70)

from scr.causal_aba.utils import get_matrix_from_arrow_set

B_est = get_matrix_from_arrow_set(arrow_set, n_nodes)

print(f"  B_est shape     : {B_est.shape}")
print(f"  B_est:\n{B_est}")
print(f"  n edges (est)   : {int((B_est != 0).sum())}")

# ─── CHECKPOINT 7 ─────────────────────────────────────────────────────────────
print("\n[7] Computing metrics")
print("-"*70)

from ArgCausalDisco.utils.graph_utils import DAGMetrics

metrics = DAGMetrics(B_est, B_true).metrics
n_true_edges = max(int(B_true.sum()), 1)

print(f"  TPR / recall    : {metrics.get('tpr'):.3f}")
print(f"  precision       : {metrics.get('precision'):.3f}")
print(f"  F1              : {metrics.get('F1'):.3f}")
print(f"  SHD             : {metrics.get('shd')}")
print(f"  SID             : {metrics.get('sid')}")
print(f"  SID normalised  : {metrics.get('sid') / n_true_edges:.3f}")

print("\n" + "="*70)
print("Single-seed trace complete.")
print("="*70)