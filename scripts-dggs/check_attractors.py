#!/usr/bin/env python3
"""
check_attractors.py — Pairwise convergence divergence across DGGS and BSAF runs.

For every pair of PKL datasets, prints:
  - frameworks where A converged but B did not
  - frameworks where B converged but A did not

Usage:
    python scripts/check_attractors.py
"""

import pickle
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "dggs_fixed":      ROOT / "convergence_results_dggs/dggs_e3_d5_s5000.pkl",
    "dggs_rand":       ROOT / "convergence_results_dggs/dggs_e3_d5_s5000_randinit.pkl",
    "bsaf_fixed_min":  ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_min.pkl",
    "bsaf_fixed_prod": ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_prod.pkl",
    "bsaf_rand_min":   ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_randinitall_min.pkl",
    "bsaf_rand_prod":  ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_randinitall_prod.pkl",
}

SEP  = "=" * 78
SEP2 = "-" * 60


def load_converged(path: Path) -> set[str]:
    with open(path, "rb") as f:
        records = pickle.load(f)
    return {
        r["file"]
        for r in records
        if r.get("global_converged") and not r.get("timeout")
    }


def run():
    datasets = {}
    for label, path in DATASETS.items():
        if not path.exists():
            print(f"[WARN] {path} not found — skipping {label}")
            continue
        datasets[label] = load_converged(path)
        print(f"Loaded {label:20s}  converged={len(datasets[label])}")

    print()

    def _skip(la, lb):
        return (
            (la.startswith("bsaf") and lb.startswith("bsaf")) or
            (la == "dggs_rand"  and lb.startswith("bsaf_fixed")) or
            (lb == "dggs_rand"  and la.startswith("bsaf_fixed")) or
            (la == "dggs_fixed" and lb.startswith("bsaf_rand"))  or
            (lb == "dggs_fixed" and la.startswith("bsaf_rand"))
        )

    for (la, a_conv), (lb, b_conv) in combinations(datasets.items(), 2):
        if _skip(la, lb):
            continue
        only_a = sorted(a_conv - b_conv)
        only_b = sorted(b_conv - a_conv)

        print(SEP)
        print(f"{la}  vs  {lb}")
        print(SEP)

        print(f"\n  {la} converged, {lb} did not  ({len(only_a)}):")
        if only_a:
            for f in only_a:
                print(f"    {f}")
        else:
            print("    (none)")

        print(f"\n  {lb} converged, {la} did not  ({len(only_b)}):")
        if only_b:
            for f in only_b:
                print(f"    {f}")
        else:
            print("    (none)")

        print()


if __name__ == "__main__":
    run()
