"""
merge_bnlearn_eval.py
─────────────────────
Merge per-chunk JSON results from the condor bnlearn eval run into a single
summary table, split by condition (reject_on vs reject_off).

Usage:
  python scripts-causal/merge_bnlearn_eval.py
  python scripts-causal/merge_bnlearn_eval.py --results-dir results_bnlearn_eval
  python scripts-causal/merge_bnlearn_eval.py --out merged_bnlearn_eval.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

DATASETS = ["cancer", "earthquake", "survey"]
CONDITIONS = ["reject_on", "reject_off"]


def _nan_to_zero(x):
    return 0.0 if (x is None or np.isnan(x)) else x


def load_chunks(results_dir: str) -> dict:
    """Returns {condition: {dataset: [result_dicts]}}."""
    combined = {c: {d: [] for d in DATASETS} for c in CONDITIONS}

    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        print(f"No JSON files found in {results_dir}")
        return combined

    for fpath in files:
        with open(fpath) as f:
            payload = json.load(f)
        condition = payload.get("condition")
        if condition not in CONDITIONS:
            print(f"  Skipping {os.path.basename(fpath)}: unknown condition {condition!r}")
            continue
        for dataset, results in payload.get("results", {}).items():
            if dataset in combined[condition]:
                combined[condition][dataset].extend(results)

    return combined


def print_condition_table(condition: str, combined: dict) -> None:
    print(f"\n{'='*80}")
    print(f"  Condition: {condition}")
    print(f"{'='*80}")
    print(f"  {'dataset':<12}  {'ok':>4}  {'avg F1':>8}  {'avg SHD':>8}  "
          f"{'SIDn_lo':>9}  {'SIDn_hi':>9}  {'avg t(s)':>9}")
    print(f"  {'─'*68}")

    all_ok = []
    for dataset in DATASETS:
        rows = combined[condition][dataset]
        ok = [r for r in rows if r.get("status") == "ok"]
        all_ok.extend(ok)
        if not ok:
            print(f"  {dataset:<12}  {len(ok):>4}  (no data)")
            continue
        avg = lambda key, rs=ok: np.mean([_nan_to_zero(r[key]) for r in rs])
        print(f"  {dataset:<12}  {len(ok):>4}  {avg('f1'):>8.3f}  "
              f"{avg('shd'):>8.2f}  {avg('sid_low_n'):>9.3f}  "
              f"{avg('sid_high_n'):>9.3f}  {avg('elapsed'):>9.1f}")

    if all_ok:
        avg = lambda key: np.mean([_nan_to_zero(r[key]) for r in all_ok])
        print(f"  {'─'*68}")
        print(f"  {'ALL':<12}  {len(all_ok):>4}  {avg('f1'):>8.3f}  "
              f"{avg('shd'):>8.2f}  {avg('sid_low_n'):>9.3f}  "
              f"{avg('sid_high_n'):>9.3f}  {avg('elapsed'):>9.1f}")


def print_comparison(combined: dict) -> None:
    print(f"\n{'='*80}")
    print("  Side-by-side comparison (reject_on vs reject_off)")
    print(f"{'='*80}")
    header = (f"  {'dataset':<12}  "
              f"{'F1_on':>7}  {'SHD_on':>7}  {'SID_on':>7}  "
              f"{'F1_off':>7}  {'SHD_off':>7}  {'SID_off':>7}  "
              f"{'ΔF1':>7}  {'ΔSHD':>7}  {'ΔSID':>7}")
    print(header)
    print(f"  {'─'*90}")

    for dataset in DATASETS:
        on  = [r for r in combined["reject_on"][dataset]  if r.get("status") == "ok"]
        off = [r for r in combined["reject_off"][dataset] if r.get("status") == "ok"]
        if not on or not off:
            print(f"  {dataset:<12}  (missing data for one or both conditions)")
            continue
        avg_on  = lambda k: np.mean([_nan_to_zero(r[k]) for r in on])
        avg_off = lambda k: np.mean([_nan_to_zero(r[k]) for r in off])
        f1_on,  shd_on,  sid_on  = avg_on("f1"),  avg_on("shd"),  avg_on("sid_low_n")
        f1_off, shd_off, sid_off = avg_off("f1"), avg_off("shd"), avg_off("sid_low_n")
        print(f"  {dataset:<12}  "
              f"{f1_on:>7.3f}  {shd_on:>7.2f}  {sid_on:>7.3f}  "
              f"{f1_off:>7.3f}  {shd_off:>7.2f}  {sid_off:>7.3f}  "
              f"{f1_on-f1_off:>+7.3f}  {shd_on-shd_off:>+7.2f}  {sid_on-sid_off:>+7.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results_bnlearn_eval"),
                   help="Directory containing per-chunk JSON files")
    p.add_argument("--out-dir", default=None,
                   help="Save one merged JSON per condition into this directory")
    args = p.parse_args()

    print(f"Loading chunks from: {args.results_dir}")
    combined = load_chunks(args.results_dir)

    n_files = len(glob.glob(os.path.join(args.results_dir, "*.json")))
    total_ok = sum(
        sum(1 for r in combined[c][d] if r.get("status") == "ok")
        for c in CONDITIONS for d in DATASETS
    )
    print(f"Loaded {n_files} chunk files, {total_ok} successful evaluations total")

    for condition in CONDITIONS:
        print_condition_table(condition, combined)

    print_comparison(combined)
    print()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for condition in CONDITIONS:
            out_path = os.path.join(args.out_dir, f"merged_{condition}.json")
            with open(out_path, "w") as f:
                json.dump(combined[condition], f, indent=2)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
