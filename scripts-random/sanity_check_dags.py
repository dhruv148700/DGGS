"""
sanity_check_dags.py
────────────────────
End-to-end reproducibility check for a handful of seeds.

For each chosen instance the full generate_data_causal pipeline is re-run
from scratch (simulate_dag → PC → facts → probes → .aba + label).
Three things are compared against what is already on disk:

  1. B_true        vs  dag_ground_truth/<dag>.npy
  2. generated .aba vs  input_data_causal/<instance>.aba   (per probe/role)
  3. generated label vs output_data_causal/<instance>.aba  (per probe/role)

    python scripts-causal/sanity_check_dags.py
"""

import itertools
import os
import sys
import tempfile

import networkx as nx
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from generate_ground_truth_dags import dag_fname
from generate_data_causal import (
    BASE_SEED,
    _CELL_OFFSETS,
    er_s0,
    fname,
    write_label_file,
    INPUT_DIR,
    OUTPUT_DIR,
    ROLE_CODES,
)
from ArgCausalDisco.utils.data_utils import simulate_dag, simulate_data_and_run_PC
from ArgCausalDisco.utils.helpers import random_stability
from scr.causal_aba.abapc import facts_from_sepset, get_probes_from_facts
from scr.causal_aba.enums import SemanticEnum
from scr.causal_aba.lp_to_aba_translator import lp_facts_to_aba_file

DAG_DIR = os.path.join(REPO_ROOT, "dag_ground_truth")

# Pick a handful of diverse seeds to check.
# Each entry: (graph_type, n, density_or_m, alpha, i)
# density_or_m is density for "er", m for "ba".
INSTANCES = [
    ("ba", 3, 1,   0.01, 0),
    ("er", 4, 0.3, 0.05, 7),
    ("ba", 4, 2,   0.1,  15),
    ("er", 5, 0.3, 0.1,  3),
]


def files_equal(path_a, path_b):
    with open(path_a, "rb") as a, open(path_b, "rb") as b:
        return a.read() == b.read()


def labels_equal(path_a, path_b):
    """Compare label files as sets of extensions — line order is arbitrary."""
    def load(path):
        with open(path) as f:
            return {frozenset(ln.strip().split(",")) for ln in f if ln.strip()}
    return load(path_a) == load(path_b)


overall_pass = True
failed = []

for graph_type, n, density_or_m, alpha, i in INSTANCES:
    seed = BASE_SEED + _CELL_OFFSETS[(graph_type, n, density_or_m, alpha)] + i
    dag_kind = "ER" if graph_type == "er" else "SF"
    s0 = er_s0(n, density_or_m) if graph_type == "er" else int(density_or_m) * n
    density = density_or_m if graph_type == "er" else None
    m       = density_or_m if graph_type == "ba" else None

    print("=" * 70)
    print(f"Instance: {graph_type} n={n} {'d' if graph_type == 'er' else 'm'}="
          f"{density_or_m} a={alpha} i={i}  seed={seed}")
    print("=" * 70)

    # ── full pipeline, capturing B_true ───────────────────────────────────
    random_stability(seed)

    B_true = simulate_dag(d=n, s0=s0, graph_type=dag_kind)

    G_true = nx.from_numpy_array(B_true, create_using=nx.DiGraph)
    G_true = nx.relabel_nodes(G_true, {k: f"X{k+1}" for k in range(n)})

    _data, cg = simulate_data_and_run_PC(G_true, alpha=alpha, seed=seed)
    facts  = facts_from_sepset(cg, n, alpha)
    probes = get_probes_from_facts(facts, n, semantics=SemanticEnum.ST)

    print(f"  probes generated : {[p.role for p in probes]}")

    # ── 1. Compare B_true ─────────────────────────────────────────────────
    dag_file = dag_fname(graph_type, n, density, m, alpha, i)
    dag_path = os.path.join(DAG_DIR, dag_file)

    instance_failures = []

    if not os.path.exists(dag_path):
        print(f"  [FAIL] DAG file missing: {dag_file}")
        instance_failures.append("DAG file missing")
    else:
        B_stored = np.load(dag_path)
        dag_ok = np.array_equal(B_true, B_stored)
        print(f"  [{'PASS' if dag_ok else 'FAIL'}] B_true vs dag_ground_truth/{dag_file}")
        if not dag_ok:
            instance_failures.append("B_true mismatch")
            print(f"    B_true:\n{B_true}")
            print(f"    B_stored:\n{B_stored}")

    # ── 2+3. Compare .aba and label per probe ─────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        for probe in probes:
            role      = probe.role
            role_code = ROLE_CODES[role]

            orig_aba   = os.path.join(REPO_ROOT, INPUT_DIR,
                             fname(graph_type, n, density, m, alpha, i, role))
            orig_label = os.path.join(REPO_ROOT, OUTPUT_DIR,
                             "output_" + fname(graph_type, n, density, m, alpha, i, role))

            new_aba   = os.path.join(tmpdir, f"{role_code}.aba")
            new_label = os.path.join(tmpdir, f"{role_code}_label.aba")

            lp_facts_to_aba_file(probe.fact_subset, n_nodes=n, out_path=new_aba)
            write_label_file(probe.models or [], new_label)

            aba_ok   = files_equal(orig_aba,  new_aba)
            label_ok = labels_equal(orig_label, new_label)

            print(f"  [{'PASS' if aba_ok   else 'FAIL'}] role={role_code:5s}  .aba")
            print(f"  [{'PASS' if label_ok else 'FAIL'}] role={role_code:5s}  label")

            if not aba_ok:
                instance_failures.append(f"role={role_code} .aba mismatch")
                with open(orig_aba) as f: o = f.readlines()
                with open(new_aba)  as f: n_ = f.readlines()
                print(f"    orig {len(o)} lines / new {len(n_)} lines")
                for idx, (lo, ln) in enumerate(itertools.zip_longest(o, n_, fillvalue="<missing>")):
                    if lo != ln:
                        print(f"    first diff at line {idx+1}:")
                        print(f"      orig: {lo.rstrip()}")
                        print(f"      new : {ln.rstrip()}")
                        break

            if not label_ok:
                instance_failures.append(f"role={role_code} label mismatch")
                def _load_exts(path):
                    with open(path) as f:
                        return {frozenset(ln.strip().split(",")) for ln in f if ln.strip()}
                orig_exts = _load_exts(orig_label)
                new_exts  = _load_exts(new_label)
                print(f"    only in orig : {orig_exts - new_exts}")
                print(f"    only in new  : {new_exts  - orig_exts}")

    if instance_failures:
        overall_pass = False
        failed.append((f"{graph_type} n={n} "
                       f"{'d' if graph_type == 'er' else 'm'}={density_or_m} "
                       f"a={alpha} i={i}",
                       instance_failures))

print("\n" + "=" * 70)
if overall_pass:
    print("ALL CHECKS PASSED")
else:
    print("FAILED INSTANCES:")
    for instance_label, reasons in failed:
        print(f"  {instance_label}")
        for r in reasons:
            print(f"    - {r}")
print("=" * 70)
