"""
generate_data_causal.py
───────────────────────
Bulk causal-ABA data generation for GNN training.

Pipeline (per instance):
  simulate_dag → simulate_data_and_run_PC (discrete, 10_000 samples,
       single PC call — sepset reused) → facts_from_sepset
  → lp_facts_to_aba_file              (writes input_data_causal/*.aba)
  → get_credulous_assumptions_from_facts (ASPforABA, ST semantics)
  → write one extension per line       (writes output_data_causal/*.aba)

Parameter grid is iterated separately for ER (over densities) and BA
(over m-values). A 2:1 accepted:none balancing pass is applied after
the full sweep. Output filenames follow

  input_data_causal/causal_{gt}_n{n}_d{d}_m{m}_a{alpha}_i{i}.aba
  output_data_causal/output_causal_{gt}_n{n}_d{d}_m{m}_a{alpha}_i{i}.aba

so the existing load_dataset() pairing convention in scr/data_utils.py
applies directly.

Place at the GNN4ABA repo root.

    python generate_data_causal.py            # full sweep
    python generate_data_causal.py --dry-run  # 2 / cell, n<=6, 600s timeout, summary
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scr"))

from ArgCausalDisco.utils.data_utils import (
    simulate_dag,
    simulate_data_and_run_PC,
)
from ArgCausalDisco.utils.helpers import random_stability
from scr.causal_aba.abapc import (
    facts_from_sepset,
    get_credulous_assumptions_from_facts,
)
from scr.causal_aba.enums import SemanticEnum
from scr.causal_aba.lp_to_aba_translator import lp_facts_to_aba_file

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_SEED         = 42

NODE_COUNTS       = [3, 4, 5, 6]
ER_EDGE_DENSITIES = [0.3, 0.5, 0.7]
BA_M_VALUES       = [1, 2, 3]
ALPHA_LEVELS      = [0.01, 0.05, 0.1]
SAMPLES_PER_CELL  = 300

INPUT_DIR     = "input_data_causal"
OUTPUT_DIR    = "output_data_causal"
MANIFEST_PATH = "causal_manifest.json"

# Silence verbose INFO chatter from the fact-removal loop during bulk runs.
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# ─── Timeout helper ───────────────────────────────────────────────────────────

class _InstanceTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """SIGALRM-based wall-clock cap; no-op when seconds falsy."""
    if not seconds:
        yield
        return

    def _handler(signum, frame):
        raise _InstanceTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def er_s0(n, density):
    return max(n - 1, int(density * n * (n - 1) / 2))


def run_instance(graph_type, n, density_or_m, alpha, seed):
    random_stability(seed)

    if graph_type == "er":
        s0 = er_s0(n, density_or_m)
        dag_kind = "ER"
    elif graph_type == "ba":
        # simulate_dag SF branch: ig.Graph.Barabasi uses m = round(s0 / d).
        # So set s0 = m * n to recover the requested BA parameter exactly.
        s0 = int(density_or_m) * n
        dag_kind = "SF"
    else:
        raise ValueError(graph_type)

    B_true = simulate_dag(d=n, s0=s0, graph_type=dag_kind)

    # simulate_data_and_run_PC expects a nx.DiGraph with "X{i+1}" labels —
    # mirrors the convention in run_single_seed_synthetic_aba.py.
    G_true = nx.from_numpy_array(B_true, create_using=nx.DiGraph)
    G_true = nx.relabel_nodes(G_true, {i: f"X{i+1}" for i in range(n)})

    # Single PC call (discrete data, 10_000 samples hardcoded inside the
    # helper); cg.sepset feeds facts_from_sepset — no second pass.
    _data, cg = simulate_data_and_run_PC(G_true, alpha=alpha, seed=seed)

    facts = facts_from_sepset(cg, n, alpha)

    t_solve = time.time()
    credulous, models, fact_idx = get_credulous_assumptions_from_facts(
        facts, n, semantics=SemanticEnum.ST,
    )
    solve_seconds = time.time() - t_solve

    return facts, credulous, models, fact_idx, solve_seconds


def write_label_file(models, path):
    """One line per extension; comma-separated accepted assumption names."""
    with open(path, "w") as fh:
        for model in models:
            fh.write(",".join(sorted(model.assumptions)) + "\n")


def fname(graph_type, n, density, m, alpha, i):
    d_str = f"d{density}" if density is not None else "dna"
    m_str = f"m{m}"       if m       is not None else "mna"
    return f"causal_{graph_type}_n{n}_{d_str}_{m_str}_a{alpha}_i{i}.aba"


def iter_cells():
    for n in NODE_COUNTS:
        for alpha in ALPHA_LEVELS:
            for density in ER_EDGE_DENSITIES:
                yield "er", n, density, None, alpha
            for m in BA_M_VALUES:
                yield "ba", n, None, m, alpha


# ─── Dry-run summary ──────────────────────────────────────────────────────────

def print_dry_run_summary(per_node):
    print("\n" + "=" * 82)
    print("DRY-RUN SUMMARY (per node count)")
    print("=" * 82)
    header = (
        f"{'n':>3}  {'mean_total':>10}  {'mean_solve':>10}  "
        f"{'mean_assums':>11}  {'timeout_rate':>12}  flag"
    )
    print(header)
    print("-" * len(header))

    flagged = False
    for n in sorted(per_node):
        s = per_node[n]
        mean_total = float(np.mean(s["total_times"]))   if s["total_times"]   else float("nan")
        mean_solve = float(np.mean(s["solve_times"]))   if s["solve_times"]   else float("nan")
        mean_asm   = float(np.mean(s["n_assumptions"])) if s["n_assumptions"] else float("nan")
        tor        = s["n_timeouts"] / max(s["n_attempted"], 1)
        flag       = ""
        if not flagged and tor > 0.10:
            flag = "<-- FIRST >10% timeout"
            flagged = True
        print(
            f"{n:>3}  {mean_total:>10.3f}  {mean_solve:>10.3f}  "
            f"{mean_asm:>11.1f}  {tor:>12.1%}  {flag}"
        )


# ─── Main sweep ───────────────────────────────────────────────────────────────

def run(dry_run=False):
    os.makedirs(INPUT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    instances_per_cell = 2 if dry_run else SAMPLES_PER_CELL
    timeout_seconds    = 600 if dry_run else 0

    per_node = defaultdict(lambda: {
        "total_times":   [],
        "solve_times":   [],
        "n_assumptions": [],
        "n_timeouts":    0,
        "n_attempted":   0,
    })

    manifest = []

    for graph_type, n, density, m, alpha in iter_cells():
        if dry_run and n > 6:
            continue
        density_or_m = density if graph_type == "er" else m
        cell_label = (
            f"{graph_type.upper()} n={n} "
            f"{'density' if graph_type == 'er' else 'm'}={density_or_m} "
            f"alpha={alpha}"
        )
        print(f"[cell] {cell_label}  ({instances_per_cell} instances)")

        for i in range(instances_per_cell):
            seed = BASE_SEED + i
            stats = per_node[n]
            stats["n_attempted"] += 1
            print(f"  i={i} seed={seed} ...", end=" ", flush=True)

            t_start = time.time()
            try:
                with time_limit(timeout_seconds):
                    facts, credulous, models, _, solve_s = run_instance(
                        graph_type, n, density_or_m, alpha, seed,
                    )
            except _InstanceTimeout:
                stats["n_timeouts"] += 1
                print("TIMEOUT", flush=True)
                continue
            total_s = time.time() - t_start
            print(f"ok ({total_s:.2f}s, solve {solve_s:.2f}s)", flush=True)

            input_path  = os.path.join(INPUT_DIR,  fname(graph_type, n, density, m, alpha, i))
            output_path = os.path.join(OUTPUT_DIR, "output_" + fname(graph_type, n, density, m, alpha, i))

            fw = lp_facts_to_aba_file(facts, n_nodes=n, out_path=input_path)
            write_label_file(models, output_path)

            stats["total_times"].append(total_s)
            stats["solve_times"].append(solve_s)
            stats["n_assumptions"].append(len(fw.assumptions))

            arr_credulous = [a for a in credulous if a.startswith("arr_")]
            manifest.append({
                "graph_type":      graph_type,
                "n_nodes":         n,
                "density_or_m":    density_or_m,
                "alpha":           alpha,
                "abaf":            input_path,
                "labels":          output_path,
                "has_accepted":    len(credulous) > 0,
                "n_atoms":         len(fw.all_elements()),
                "n_assumptions":   len(fw.assumptions),
                "n_credulous":     len(credulous),
                "n_credulous_arr": len(arr_credulous),
            })

    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest written: {MANIFEST_PATH}  ({len(manifest)} instances)")

    if dry_run:
        print_dry_run_summary(per_node)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run", action="store_true",
        help="3 instances per cell, 60s timeout, print per-node summary table.",
    )
    args = p.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
