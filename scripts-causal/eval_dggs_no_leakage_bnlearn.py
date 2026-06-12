"""
eval_dggs_no_leakage_bnlearn.py
────────────────────────────────
Run DGGS extension generator on the no-leakage bnlearn causal dataset
(no_leakage_data_bnlearn_causal/aba/) and print per-sample + per-dataset
summary tables to the console.

Files consumed:
  no_leakage_data_bnlearn_causal/aba/causal_bnlearn_{ds}_n{n}_a0.01_s{seed}_full.aba
  no_leakage_data_bnlearn_causal/aba/causal_bnlearn_{ds}_n{n}_a0.01_s{seed}_full.scores.json
  no_leakage_data_bnlearn_causal/dag/dag_bnlearn_{ds}_n{n}.npy

Usage:
  python scripts-causal/eval_dggs_no_leakage_bnlearn.py
  python scripts-causal/eval_dggs_no_leakage_bnlearn.py --seeds-per-dataset 5
  python scripts-causal/eval_dggs_no_leakage_bnlearn.py --datasets cancer earthquake
  python scripts-causal/eval_dggs_no_leakage_bnlearn.py --no-scores
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from scr.dggs_extension_generator import build_dggs_extension
from ArgCausalDisco.utils.graph_utils import DAGMetrics, dag2cpdag
from ArgCausalDisco.utils.helpers import random_stability

arr_pattern = re.compile(r"^arr_(\d+)_(\d+)$")

ABA_DIR = os.path.join(REPO_ROOT, "no_leakage_data_bnlearn_causal", "aba")
DAG_DIR = os.path.join(REPO_ROOT, "no_leakage_data_bnlearn_causal", "dag")
ALPHA   = 0.01
N_RUNS  = 50


def _nan_to_zero(x: float) -> float:
    return 0.0 if np.isnan(x) else x


def compute_dag_metrics(B_true: np.ndarray, extension: set, n_nodes: int) -> dict:
    B_est = np.zeros((n_nodes, n_nodes))
    for a in extension:
        m = arr_pattern.match(a)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            B_est[i, j] = 1

    B_est_cpdag = dag2cpdag(B_est)
    metrics = DAGMetrics(B_est_cpdag, B_true).metrics

    n_ref = max(int(B_true.sum()), 1)
    sid = metrics.get("sid", (0.0, 0.0))
    sid_low, sid_high = sid if isinstance(sid, tuple) else (sid, sid)

    return {
        "nnz":        int(metrics.get("nnz", 0)),
        "fdr":        round(float(metrics.get("fdr", 0.0)), 4),
        "tpr":        round(float(metrics.get("tpr", 0.0)), 4),
        "fpr":        round(float(metrics.get("fpr", 0.0)), 4),
        "precision":  round(_nan_to_zero(float(metrics.get("precision", 0.0))), 4),
        "recall":     round(float(metrics.get("recall", 0.0)), 4),
        "f1":         round(_nan_to_zero(float(metrics.get("F1", 0.0))), 4),
        "shd":        int(metrics.get("shd", 0)),
        "sid_low_n":  round(float(sid_low) / n_ref, 4),
        "sid_high_n": round(float(sid_high) / n_ref, 4),
    }


def discover_n_nodes(dataset_name: str) -> int | None:
    matches = glob.glob(os.path.join(DAG_DIR, f"dag_bnlearn_{dataset_name}_n*.npy"))
    if not matches:
        return None
    return np.load(matches[0]).shape[0]


def evaluate_dataset(dataset_name: str, seeds: list, use_scores: bool, use_reject_edge: bool = True) -> list:
    n_nodes = discover_n_nodes(dataset_name)
    if n_nodes is None:
        print(f"  No DAG file found for {dataset_name} — skipping")
        return []

    dag_path = os.path.join(DAG_DIR, f"dag_bnlearn_{dataset_name}_n{n_nodes}.npy")
    B_true = np.load(dag_path)

    results = []
    for seed_idx, seed in enumerate(seeds):
        stem        = f"bnlearn_{dataset_name}_n{n_nodes}_a{ALPHA}_s{seed}"
        aba_path    = os.path.join(ABA_DIR, f"causal_{stem}_full.aba")
        scores_path = os.path.join(ABA_DIR, f"causal_{stem}_full.scores.json")

        if not os.path.exists(aba_path):
            print(f"  [{seed_idx+1}/{len(seeds)}] seed={seed} — .aba not found, skipping")
            results.append({"status": "missing", "seed": seed})
            continue

        if not use_scores or not os.path.exists(scores_path):
            scores_path = None

        print(f"  [{seed_idx+1}/{len(seeds)}] seed={seed:4d} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            extension, _ = build_dggs_extension(
                aba_path, scores_path=scores_path, verbose=False,
                reject_edge_on_indep=use_reject_edge,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"status": "error", "seed": seed, "error": str(e)})
            continue
        elapsed = time.time() - t0

        dag_metrics = compute_dag_metrics(B_true, extension, n_nodes)
        r = {"status": "ok", "seed": seed, "elapsed": round(elapsed, 2), **dag_metrics}
        results.append(r)
        print(f"F1={r['f1']:.3f}  prec={r['precision']:.3f}  rec={r['recall']:.3f}  "
              f"SHD={r['shd']}  t={elapsed:.1f}s")

    return results


def print_dataset_table(dataset_name: str, results: list) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    n_miss = sum(1 for r in results if r["status"] == "missing")
    n_err  = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'─'*90}")
    print(f"  {dataset_name}  ({len(ok)}/{len(results)} ok  missing={n_miss}  error={n_err})")
    print(f"{'─'*90}")
    if not ok:
        print("  (no successful evaluations)")
        return

    print(f"  {'seed':>5}  {'F1':>6}  {'prec':>6}  {'rec':>6}  "
          f"{'SHD':>5}  {'SIDn_lo':>8}  {'SIDn_hi':>8}  {'t(s)':>6}")
    print(f"  {'─'*75}")
    for r in results:
        if r["status"] != "ok":
            print(f"  {r['seed']:>5}  {'–':>6}  (skipped/error)")
            continue
        print(f"  {r['seed']:>5}  {r['f1']:>6.3f}  {r['precision']:>6.3f}  "
              f"{r['recall']:>6.3f}  {r['shd']:>5}  "
              f"{r['sid_low_n']:>8.3f}  {r['sid_high_n']:>8.3f}  {r['elapsed']:>6.1f}")

    avg = lambda key: np.mean([r[key] for r in ok])
    print(f"  {'─'*75}")
    print(f"  {'AVG':>5}  {avg('f1'):>6.3f}  {avg('precision'):>6.3f}  "
          f"{avg('recall'):>6.3f}  {avg('shd'):>5.1f}  "
          f"{avg('sid_low_n'):>8.3f}  {avg('sid_high_n'):>8.3f}  {avg('elapsed'):>6.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+",
                   default=["cancer", "earthquake", "survey"])
    p.add_argument("--seeds-per-dataset", type=int, default=50,
                   help="How many seeds to evaluate per dataset (default: 50)")
    p.add_argument("--no-scores", action="store_true",
                   help="Use neutral DGGS initialisation instead of score maps")
    p.add_argument("--no-reject-edge-on-indep", action="store_true",
                   help="Disable the optimisation that rejects arr/noe assumptions when an indep is committed")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Start index into the global seed list (default: 0)")
    p.add_argument("--out-file", type=str, default=None,
                   help="Save per-seed results as JSON to this path (for later merging)")
    args = p.parse_args()

    # Reproduce the same ordered seed list as the generation scripts
    random_stability(2024)
    all_seeds = np.random.randint(0, 10000, (N_RUNS,)).tolist()
    seeds = all_seeds[args.seed_offset:args.seed_offset + args.seeds_per_dataset]

    use_reject_edge = not args.no_reject_edge_on_indep
    condition = "reject_on" if use_reject_edge else "reject_off"
    print(f"Datasets: {args.datasets}")
    print(f"Seeds [{args.seed_offset}:{args.seed_offset + args.seeds_per_dataset}]  "
          f"(first seed: {seeds[0] if seeds else 'none'})")
    print(f"Use scores: {not args.no_scores}")
    print(f"Reject edge on indep: {use_reject_edge}")

    all_results = {}
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        results = evaluate_dataset(dataset_name, seeds, use_scores=not args.no_scores,
                                   use_reject_edge=use_reject_edge)
        all_results[dataset_name] = results

    for dataset_name in args.datasets:
        print_dataset_table(dataset_name, all_results.get(dataset_name, []))

    print(f"\n{'='*72}")
    print("  Overall summary")
    print(f"{'='*72}")
    print(f"  {'dataset':<12}  {'ok':>4}  {'avg F1':>8}  {'avg SHD':>8}  "
          f"{'SIDn_lo':>9}  {'SIDn_hi':>9}  {'avg t(s)':>9}")
    print(f"  {'─'*65}")
    all_ok = []
    for dataset_name in args.datasets:
        ok = [r for r in all_results.get(dataset_name, []) if r.get("status") == "ok"]
        all_ok.extend(ok)
        if ok:
            avg = lambda key, rows=ok: np.mean([r[key] for r in rows])
            print(f"  {dataset_name:<12}  {len(ok):>4}  {avg('f1'):>8.3f}  "
                  f"{avg('shd'):>8.2f}  {avg('sid_low_n'):>9.3f}  "
                  f"{avg('sid_high_n'):>9.3f}  {avg('elapsed'):>9.1f}")
    if all_ok:
        avg = lambda key: np.mean([r[key] for r in all_ok])
        print(f"  {'─'*65}")
        print(f"  {'ALL':<12}  {len(all_ok):>4}  {avg('f1'):>8.3f}  "
              f"{avg('shd'):>8.2f}  {avg('sid_low_n'):>9.3f}  "
              f"{avg('sid_high_n'):>9.3f}  {avg('elapsed'):>9.1f}")
    print()

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
        payload = {
            "condition":   condition,
            "seed_offset": args.seed_offset,
            "seeds_per_dataset": args.seeds_per_dataset,
            "datasets":    args.datasets,
            "results":     all_results,
        }
        with open(args.out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.out_file}")


if __name__ == "__main__":
    main()
