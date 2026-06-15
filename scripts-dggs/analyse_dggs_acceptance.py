"""
analyse_dggs_acceptance.py
──────────────────────────
For each ABAF instance, partition assumptions into three acceptance groups
and report the mean DGGS fixed-point score per group:

  1. Skeptically accepted  — in every extension (intersection of all lines)
  2. Credulously accepted  — in some but not all extensions
  3. Credulously rejected  — in no extension (all_assumptions - union)

Can run standalone (no chunk args) or as a Condor array chunk:

    python scripts-dggs/analyse_dggs_acceptance.py          # all instances
    python scripts-dggs/analyse_dggs_acceptance.py \\
        --chunk-id 0 --n-chunks 16 --out-dir analysis_results/
"""

# Role: post-processing only — reads pre-computed .dggs.json scores and extension files.
# Partitions assumptions into skeptically accepted / credulously only / credulously rejected.
# No DGGS iteration here; kernel fns are not used. Supports Condor chunked mode.
# Refactor target: minimal changes needed — just ensure dggs score format is stable.

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

DGGS_DIR   = "dggs_scores"
INPUT_DIR  = "input_data_causal"
OUTPUT_DIR = "output_data_causal"


def parse_assumptions(aba_path):
    assumptions = set()
    with open(aba_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == "a":
                assumptions.add(parts[1])
    return assumptions


def parse_extensions(output_path):
    extensions = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                extensions.append(set(line.split(",")))
    return extensions


def group_mean(assumption_set, dggs_scores):
    scores = [dggs_scores[a] for a in assumption_set if a in dggs_scores]
    return float(np.mean(scores)) if scores else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-id",  type=int, default=None)
    p.add_argument("--n-chunks",  type=int, default=None)
    p.add_argument("--out-dir",   type=str, default=None,
                   help="Directory to write per-chunk result JSON (chunked mode only)")
    args = p.parse_args()

    chunked = args.chunk_id is not None
    if chunked and (args.n_chunks is None or args.out_dir is None):
        p.error("--chunk-id requires --n-chunks and --out-dir")

    base = Path(__file__).parent.parent
    dggs_dir   = base / DGGS_DIR
    input_dir  = base / INPUT_DIR
    output_dir = base / OUTPUT_DIR

    all_files = sorted(dggs_dir.glob("*.dggs.json"))
    if chunked:
        chunk_files = [f for i, f in enumerate(all_files) if i % args.n_chunks == args.chunk_id]
        desc = f"Chunk {args.chunk_id}/{args.n_chunks}"
    else:
        chunk_files = all_files
        desc = "Analysing instances"

    skeptical_means     = []
    credulous_only_means = []
    rejected_means      = []
    skipped_missing     = 0
    skipped_no_ext      = 0
    n_no_rejected       = 0
    n_no_skeptical      = 0

    for dggs_path in tqdm(chunk_files, desc=desc):
        stem        = dggs_path.name.replace(".dggs.json", "")
        input_path  = input_dir  / f"{stem}.aba"
        output_path = output_dir / f"output_{stem}.aba"

        if not input_path.exists() or not output_path.exists():
            skipped_missing += 1
            continue

        with open(dggs_path) as f:
            dggs_scores = json.load(f)

        all_assumptions = parse_assumptions(input_path)
        extensions      = parse_extensions(output_path)

        if not extensions:
            skipped_no_ext += 1
            continue

        union        = set.union(*extensions)
        intersection = set.intersection(*extensions)

        skeptically_accepted = intersection
        credulously_only     = union - intersection
        credulously_rejected = all_assumptions - union

        if not credulously_rejected:
            n_no_rejected += 1
        if not skeptically_accepted:
            n_no_skeptical += 1

        s = group_mean(skeptically_accepted, dggs_scores)
        c = group_mean(credulously_only, dggs_scores)
        r = group_mean(credulously_rejected, dggs_scores)

        if s is not None:
            skeptical_means.append(s)
        if c is not None:
            credulous_only_means.append(c)
        if r is not None:
            rejected_means.append(r)

    result = {
        "skeptical_means":      skeptical_means,
        "credulous_only_means": credulous_only_means,
        "rejected_means":       rejected_means,
        "skipped_missing":      skipped_missing,
        "skipped_no_ext":       skipped_no_ext,
        "n_no_rejected":        n_no_rejected,
        "n_no_skeptical":       n_no_skeptical,
    }

    if chunked:
        out_dir = base / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"acceptance_chunk_{args.chunk_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"Saved {out_path}")
    else:
        _print_summary(result)


def _print_summary(result):
    print(f"\nSkipped (missing files):   {result['skipped_missing']}")
    print(f"Skipped (no extensions):   {result['skipped_no_ext']}")
    print(f"Instances w/ no rejected:  {result['n_no_rejected']}")
    print(f"Instances w/ no skeptical: {result['n_no_skeptical']}")
    print()

    header = f"{'Group':<40} {'N':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))

    for label, means in [
        ("Skeptically accepted",              result["skeptical_means"]),
        ("Credulously accepted (not skeptic)", result["credulous_only_means"]),
        ("Credulously rejected",              result["rejected_means"]),
    ]:
        if means:
            arr = np.array(means)
            print(f"{label:<40} {len(arr):>6} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")
        else:
            print(f"{label:<40} {'N/A':>6}")


if __name__ == "__main__":
    main()
