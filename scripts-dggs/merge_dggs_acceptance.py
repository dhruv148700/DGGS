"""
merge_dggs_acceptance.py
────────────────────────
Merges per-chunk JSON results from analyse_dggs_acceptance.py and prints
the aggregated summary table.

Usage:
    python scripts-dggs/merge_dggs_acceptance.py --out-dir analysis_results/
"""

# Role: aggregates per-chunk JSON results from analyse_dggs_acceptance.py into a single summary table.
# No DGGS iteration; no kernel fns. Pure list concatenation and numpy stats.
# Refactor target: no logic changes needed.

import argparse
import json
import numpy as np
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True,
                   help="Directory containing acceptance_chunk_*.json files")
    args = p.parse_args()

    base    = Path(__file__).parent.parent
    out_dir = base / args.out_dir

    chunk_files = sorted(out_dir.glob("acceptance_chunk_*.json"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {out_dir}")

    skeptical_means      = []
    credulous_only_means = []
    rejected_means       = []
    skipped_missing      = 0
    skipped_no_ext       = 0
    n_no_rejected        = 0
    n_no_skeptical       = 0

    for path in chunk_files:
        with open(path) as f:
            r = json.load(f)
        skeptical_means      += r["skeptical_means"]
        credulous_only_means += r["credulous_only_means"]
        rejected_means       += r["rejected_means"]
        skipped_missing      += r["skipped_missing"]
        skipped_no_ext       += r["skipped_no_ext"]
        n_no_rejected        += r["n_no_rejected"]
        n_no_skeptical       += r["n_no_skeptical"]

    print(f"Chunks merged:             {len(chunk_files)}")
    print(f"Skipped (missing files):   {skipped_missing}")
    print(f"Skipped (no extensions):   {skipped_no_ext}")
    print(f"Instances w/ no rejected:  {n_no_rejected}")
    print(f"Instances w/ no skeptical: {n_no_skeptical}")
    print()

    header = f"{'Group':<40} {'N':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))

    for label, means in [
        ("Skeptically accepted",               skeptical_means),
        ("Credulously accepted (not skeptic)", credulous_only_means),
        ("Credulously rejected",               rejected_means),
    ]:
        if means:
            arr = np.array(means)
            print(f"{label:<40} {len(arr):>6} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")
        else:
            print(f"{label:<40} {'N/A':>6}")


if __name__ == "__main__":
    main()
