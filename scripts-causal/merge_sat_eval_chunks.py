"""
merge_sat_eval_chunks.py
────────────────────────
Merge individual chunk results from parallel evaluation into a single JSON file.

Usage:
  python scripts-causal/merge_sat_eval_chunks.py
  python scripts-causal/merge_sat_eval_chunks.py --output final_results.json --num-chunks 50
"""

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def merge_chunks(num_chunks: int = 50, output_path: str = None):
    """Merge all chunk JSON files into one."""
    if output_path is None:
        output_path = os.path.join(REPO_ROOT, "results_sat_eval.json")

    chunk_pattern = "results_sat_eval_chunk_*.json"
    chunk_dir = REPO_ROOT

    # Find all chunk files
    chunk_files = sorted(Path(chunk_dir).glob(chunk_pattern))
    chunk_files = [f for f in chunk_files if f.is_file()]

    if not chunk_files:
        print(f"ERROR: No chunk files found matching {chunk_pattern}")
        print(f"  Searched in: {chunk_dir}")
        return False

    print(f"Found {len(chunk_files)} chunk files")

    # Load and merge results
    all_results = []
    for i, chunk_file in enumerate(chunk_files):
        print(f"  Loading {chunk_file.name}...", end=" ", flush=True)
        try:
            with open(chunk_file) as f:
                chunk_results = json.load(f)
            all_results.extend(chunk_results)
            print(f"({len(chunk_results)} samples)")
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    # Write merged results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Merged {len(all_results)} results into {output_path}")

    # Summary statistics
    successful = sum(1 for r in all_results if r.get("status") == "success")
    failed = len(all_results) - successful
    print(f"  Successful: {successful}/{len(all_results)}")
    print(f"  Failed: {failed}/{len(all_results)}")

    if successful > 0:
        import numpy as np
        success_results = [r for r in all_results if r.get("status") == "success"]
        avg_f1 = np.mean([r["dag_metrics"]["f1"] for r in success_results])
        avg_tpr = np.mean([r["dag_metrics"]["tpr"] for r in success_results])
        avg_precision = np.mean([r["dag_metrics"]["precision"] for r in success_results])
        avg_shd = np.mean([r["dag_metrics"]["shd"] for r in success_results])
        avg_sid_low = np.mean([r["dag_metrics"]["sid_low"] for r in success_results])
        avg_sid_high = np.mean([r["dag_metrics"]["sid_high"] for r in success_results])
        avg_sid_low_norm = np.mean([r["dag_metrics"]["sid_low_normalized"] for r in success_results])
        avg_sid_high_norm = np.mean([r["dag_metrics"]["sid_high_normalized"] for r in success_results])

        print(f"\nCPDAG Aggregate metrics:")
        print(f"  F1:                  {avg_f1:.4f}")
        print(f"  TPR:                 {avg_tpr:.4f}")
        print(f"  Precision:           {avg_precision:.4f}")
        print(f"  SHD:                 {avg_shd:.2f}")
        print(f"  SID (low):           {avg_sid_low:.2f}")
        print(f"  SID (high):          {avg_sid_high:.2f}")
        print(f"  SID (low, norm):     {avg_sid_low_norm:.4f}")
        print(f"  SID (high, norm):    {avg_sid_high_norm:.4f}")

    print(f"{'='*70}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "results_sat_eval.json"),
        help="Output JSON file (default: results_sat_eval.json)"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=50,
        help="Expected number of chunks (default: 50)"
    )
    args = parser.parse_args()

    success = merge_chunks(args.num_chunks, args.output)
    exit(0 if success else 1)
