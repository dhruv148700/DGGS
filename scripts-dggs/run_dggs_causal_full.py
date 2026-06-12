"""
run_dggs_causal_full.py
───────────────────────
Batch DGGS runner for the full causal dataset.  Intended to be run as a
Condor array job where each task processes a non-overlapping subset of the
manifest determined by round-robin chunk assignment:

    entry belongs to chunk C  iff  (manifest_index % n_chunks) == chunk_id

Round-robin interleaving ensures each chunk gets a representative mix of
n_nodes values, preventing any single job from receiving all large (slow)
frameworks.

Output per entry
────────────────
  {stem}.dggs.json  — flat {assumption_name: float} dict of final DGGS
                      strengths for every assumption in the framework.
                      Claims and rules are omitted (not used downstream).

  dggs_chunk_{chunk_id}_report.json  — per-chunk convergence report written
                                        to --out-dir when the chunk finishes.

tau initialisation
──────────────────
  tau_a  : CI score from plain keys in .scores.json; 0.5 for scaffold
            assumptions absent from the score map.
  tau_r  : CI score from rule-tuple keys ("head|body1|body2") in
            .scores.json; 1.0 for scaffold rules absent from the score map.

Usage
─────
    python scripts-dggs/run_dggs_causal_full.py --chunk-id 0 --n-chunks 8
    python scripts-dggs/run_dggs_causal_full.py --chunk-id 3 --n-chunks 8 \\
        --manifest causal_manifest.json --out-dir dggs_results/
"""

# Role: Condor array-job batch runner over causal_manifest.json.
# Uses DGGSRunner from scr.dggs; tau init helpers from scr.dggs.tau_utils.

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scr.dependency_graph import DependencyGraph
from scr.ABAF import ABAF
from scr.dggs import DGGSRunner
from scr.dggs.tau_utils import load_scores, build_tau_a, build_tau_r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_path(abaf_path: str, dggs_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(abaf_path))[0]
    return os.path.join(dggs_dir, stem + ".dggs.json")


def _already_done(abaf_path: str, dggs_dir: str) -> bool:
    return os.path.exists(_output_path(abaf_path, dggs_dir))


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(entry: dict, repo_root: str, dggs_dir: str, runner_kwargs: dict) -> dict:
    abaf_path   = os.path.join(repo_root, entry["abaf"])
    scores_path = os.path.join(repo_root, entry.get("scores", ""))
    out_path    = _output_path(abaf_path, dggs_dir)

    record = {
        "instance_id":   entry["instance_id"],
        "probe_role":    entry["probe_role"],
        "n_nodes":       entry["n_nodes"],
        "n_assumptions": entry["n_assumptions"],
        "converged":     None,
        "n_iterations":  None,
        "time_s":        None,
        "skipped":       False,
        "error":         None,
    }

    if _already_done(abaf_path, dggs_dir):
        record["skipped"] = True
        return record

    try:
        plain_scores, rule_scores = load_scores(scores_path)

        dg = DependencyGraph()
        dg.create_from_file(abaf_path)

        abaf = ABAF.from_dependency_graph(
            dg,
            tau_a=build_tau_a(dg, plain_scores),
            tau_r=build_tau_r(dg, rule_scores),
        )

        runner = DGGSRunner(abaf, **runner_kwargs)

        t0     = time.perf_counter()
        result = runner.run()
        elapsed = time.perf_counter() - t0

        with open(out_path, "w") as fh:
            json.dump(result.final_state.assumptions, fh)

        record["converged"]    = result.converged
        record["n_iterations"] = result.n_iterations
        record["time_s"]       = round(elapsed, 4)

    except Exception as exc:
        record["error"] = str(exc)

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-id", type=int, required=True)
    p.add_argument("--n-chunks", type=int, required=True)
    p.add_argument("--manifest", default=str(REPO_ROOT / "causal_manifest.json"))
    p.add_argument("--dggs-dir", default=str(REPO_ROOT / "dggs_scores"),
                   help="Directory where .dggs.json files are written")
    p.add_argument("--out-dir",  default=str(REPO_ROOT / "dggs_results"),
                   help="Directory for per-chunk convergence reports")
    p.add_argument("--max-iter", type=int,   default=200)
    p.add_argument("--epsilon",  type=float, default=1e-3)
    p.add_argument("--window",   type=int,   default=5)
    args = p.parse_args()

    if args.chunk_id >= args.n_chunks:
        raise ValueError(f"chunk_id={args.chunk_id} >= n_chunks={args.n_chunks}")

    os.makedirs(args.dggs_dir, exist_ok=True)
    os.makedirs(args.out_dir,  exist_ok=True)

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    chunk_entries = [e for i, e in enumerate(manifest) if i % args.n_chunks == args.chunk_id]

    print(
        f"Chunk {args.chunk_id}/{args.n_chunks}  "
        f"entries={len(chunk_entries)}  "
        f"max_iter={args.max_iter}  epsilon={args.epsilon}"
    )

    runner_kwargs = dict(max_iter=args.max_iter, epsilon=args.epsilon, window=args.window)

    records: List[dict] = []
    n_total = len(chunk_entries)
    n_done = n_skipped = n_errors = 0

    for entry in chunk_entries:
        rec = process_entry(entry, str(REPO_ROOT), args.dggs_dir, runner_kwargs)
        records.append(rec)
        n_done += 1

        if rec["skipped"]:
            n_skipped += 1
            status = "SKIP"
        elif rec["error"]:
            n_errors += 1
            status = f"ERROR: {rec['error']}"
        else:
            status = (
                f"{'OK' if rec['converged'] else 'NO-CONV'}  "
                f"iters={rec['n_iterations']}  t={rec['time_s']:.3f}s"
            )

        if n_done % 100 == 0 or rec["error"]:
            print(f"  [{n_done}/{n_total}]  {rec['instance_id']}  {status}")

    report = {
        "chunk_id":        args.chunk_id,
        "n_chunks":        args.n_chunks,
        "n_total":         n_total,
        "n_skipped":       n_skipped,
        "n_errors":        n_errors,
        "n_converged":     sum(1 for r in records if r.get("converged")),
        "n_not_converged": sum(1 for r in records if r.get("converged") is False),
        "records":         records,
    }
    report_path = os.path.join(args.out_dir, f"dggs_chunk_{args.chunk_id}_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    ok    = [r for r in records if not r["skipped"] and not r["error"]]
    times = sorted(r["time_s"] for r in ok)
    n_ok  = len(ok)
    if times:
        print(
            f"\nChunk {args.chunk_id} done  "
            f"ok={n_ok}  skipped={n_skipped}  errors={n_errors}\n"
            f"  time: med={times[n_ok//2]:.3f}s  "
            f"p90={times[int(n_ok*0.9)]:.3f}s  "
            f"max={times[-1]:.3f}s  "
            f"total={sum(times):.1f}s"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
