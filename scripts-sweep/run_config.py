"""
run_config.py — Run one kernel config over all ABAFs.

Writes two output files per config:

  results/raw/{config_id}.parquet     — per-assumption sigma values
    abaf_id, assumption_id, sigma_final (float32), converged (bool), n_iter (int16)

  results/timing/{config_id}.parquet  — per-ABAF timing
    abaf_id, config_id, construction_time_s (float32), semantics_time_s (float32),
    timed_out (bool)

If the sigma parquet already exists it is skipped (correct fixed points are already
stored); the timing parquet is always written fresh.  This allows a timing-only
re-run without re-computing sigma values for configs that fully converged.

Construction time: load_scores + DependencyGraph.create_from_file +
                   ABAF.from_dependency_graph + DGGSRunner.__init__
Semantics time:    runner.run()  (capped at --timeout seconds via SIGALRM)

Usage
-----
    python scripts-sweep/run_config.py --config-id "prod·max·max·lin·k1.0"
    python scripts-sweep/run_config.py --config-index 0
    python scripts-sweep/run_config.py --config-index 0 \\
        --manifest causal_manifest.json \\
        --out-dir results/raw/ --timing-dir results/timing/ \\
        --max-iter 5000 --epsilon 1e-3 --window 5 --timeout 600
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scr.dependency_graph import DependencyGraph
from scr.ABAF import ABAF
from scr.dggs import DGGSRunner
from scr.dggs.tau_utils import load_scores, build_tau_a, build_tau_r

sys.path.insert(0, str(REPO_ROOT / "scripts-sweep"))
from config_grid import config_by_id, config_by_index, build_kernels

_SIGMA_SCHEMA = pa.schema([
    pa.field("abaf_id",       pa.string()),
    pa.field("assumption_id", pa.string()),
    pa.field("sigma_final",   pa.float32()),
    pa.field("converged",     pa.bool_()),
    pa.field("n_iter",        pa.int16()),
])

_TIMING_SCHEMA = pa.schema([
    pa.field("abaf_id",             pa.string()),
    pa.field("config_id",           pa.string()),
    pa.field("construction_time_s", pa.float32()),
    pa.field("semantics_time_s",    pa.float32()),
    pa.field("timed_out",           pa.bool_()),
])


# ---------------------------------------------------------------------------
# Timeout helpers (SIGALRM, Unix only)
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass


def _arm(seconds: int) -> None:
    def _handler(signum, frame):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)


def _disarm() -> None:
    signal.alarm(0)


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(entry: dict, cfg: dict, runner_kwargs: dict, timeout_s: int) -> tuple:
    """
    Run DGGS for one manifest entry.

    Returns (sigma_columns | None, timing_record) where sigma_columns is a
    dict of lists ready for pa.record_batch (None on construction error or
    timeout), and timing_record always has construction_time_s, semantics_time_s,
    timed_out, converged, n_iter.
    """
    abaf_id     = Path(entry["abaf"]).stem
    aba_path    = REPO_ROOT / entry["abaf"]
    scores_path = REPO_ROOT / entry["scores"]

    timing = {
        "abaf_id":             abaf_id,
        "config_id":           cfg["config_id"],
        "construction_time_s": float("nan"),
        "semantics_time_s":    float("nan"),
        "timed_out":           False,
        "error":               None,
    }

    # --- Construction ---
    try:
        t0 = time.perf_counter()
        plain_scores, rule_scores = load_scores(str(scores_path))
        dg = DependencyGraph()
        dg.create_from_file(str(aba_path))
        abaf = ABAF.from_dependency_graph(
            dg,
            tau_a=build_tau_a(dg, plain_scores),
            tau_r=build_tau_r(dg, rule_scores),
        )
        body_agg, claim_agg, support_agg, influence = build_kernels(cfg)
        runner = DGGSRunner(
            abaf,
            body_agg    = body_agg,
            claim_agg   = claim_agg,
            support_agg = support_agg,
            influence   = influence,
            **runner_kwargs,
        )
        timing["construction_time_s"] = round(time.perf_counter() - t0, 6)
    except Exception as exc:
        timing["error"] = f"construction: {exc}"
        return None, timing

    # --- Semantics (with timeout) ---
    try:
        _arm(timeout_s)
        t0 = time.perf_counter()
        result = runner.run()
        timing["semantics_time_s"] = round(time.perf_counter() - t0, 6)
        _disarm()
    except _Timeout:
        _disarm()
        timing["timed_out"]        = True
        timing["semantics_time_s"] = float(timeout_s)
        return None, timing
    except Exception as exc:
        _disarm()
        timing["error"] = f"semantics: {exc}"
        return None, timing

    asm_names = sorted(result.final_state.assumptions)
    n = len(asm_names)
    sigma_columns = {
        "abaf_id":       [abaf_id] * n,
        "assumption_id": asm_names,
        "sigma_final":   [result.final_state.assumptions[a] for a in asm_names],
        "converged":     [result.converged] * n,
        "n_iter":        [result.n_iterations] * n,
    }
    return sigma_columns, timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    id_group = p.add_mutually_exclusive_group(required=True)
    id_group.add_argument("--config-id",    type=str)
    id_group.add_argument("--config-index", type=int)
    p.add_argument("--manifest",    default=str(REPO_ROOT / "causal_manifest.json"))
    p.add_argument("--out-dir",     default=str(REPO_ROOT / "results" / "raw"))
    p.add_argument("--timing-dir",  default=str(REPO_ROOT / "results" / "timing"))
    p.add_argument("--max-iter",    type=int,   default=5000, dest="max_iter")
    p.add_argument("--epsilon",     type=float, default=1e-3)
    p.add_argument("--window",      type=int,   default=5)
    p.add_argument("--timeout",     type=int,   default=600,
                   help="Per-ABAF wall-clock timeout in seconds for runner.run()")
    args = p.parse_args()

    cfg       = (config_by_id(args.config_id) if args.config_id
                 else config_by_index(args.config_index))
    config_id = cfg["config_id"]

    out_dir    = Path(args.out_dir);    out_dir.mkdir(parents=True, exist_ok=True)
    timing_dir = Path(args.timing_dir); timing_dir.mkdir(parents=True, exist_ok=True)

    out_sigma  = out_dir    / f"{config_id}.parquet"
    out_timing = timing_dir / f"{config_id}.parquet"

    if out_timing.exists():
        print(f"Timing already exists: {out_timing} — delete it to re-run.")
        sys.exit(0)

    write_sigma = not out_sigma.exists()
    print(f"Config:       {config_id}")
    print(f"Sigma parquet: {'WRITE' if write_sigma else 'SKIP (already exists)'}")
    print(f"Timing parquet: {out_timing}")

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    print(f"Manifest:     {len(manifest)} entries  max_iter={args.max_iter}\n")

    runner_kwargs = dict(max_iter=args.max_iter, epsilon=args.epsilon, window=args.window)

    sigma_writer  = (pq.ParquetWriter(str(out_sigma),  _SIGMA_SCHEMA,  compression="snappy")
                     if write_sigma else None)
    timing_writer = pq.ParquetWriter(str(out_timing), _TIMING_SCHEMA, compression="snappy")

    n_total = len(manifest)
    n_done  = n_errors = n_timeouts = 0
    t_start = time.perf_counter()

    for entry in manifest:
        sigma_cols, timing = process_entry(entry, cfg, runner_kwargs, args.timeout)
        n_done += 1

        if timing["timed_out"]:
            n_timeouts += 1
            print(f"  [TIMEOUT] {timing['abaf_id']}", flush=True)
        elif timing["error"]:
            n_errors += 1
            if n_errors <= 10:
                print(f"  [ERROR] {timing['abaf_id']}: {timing['error']}", flush=True)

        if sigma_cols is not None and sigma_writer is not None:
            sigma_writer.write_batch(pa.record_batch(sigma_cols, schema=_SIGMA_SCHEMA))

        timing_writer.write_batch(pa.record_batch({
            "abaf_id":             [timing["abaf_id"]],
            "config_id":           [timing["config_id"]],
            "construction_time_s": [timing["construction_time_s"]],
            "semantics_time_s":    [timing["semantics_time_s"]],
            "timed_out":           [timing["timed_out"]],
        }, schema=_TIMING_SCHEMA))

        if n_done % 500 == 0:
            elapsed = time.perf_counter() - t_start
            rate    = n_done / elapsed
            eta_s   = (n_total - n_done) / rate if rate > 0 else 0
            print(
                f"  [{n_done}/{n_total}]  timeouts={n_timeouts}  errors={n_errors}"
                f"  {rate:.1f}/s  ETA {eta_s/60:.1f}min",
                flush=True,
            )

    if sigma_writer:
        sigma_writer.close()
    timing_writer.close()

    elapsed_total = time.perf_counter() - t_start
    if write_sigma:
        print(f"\nSigma  → {out_sigma}  ({out_sigma.stat().st_size/1e6:.1f} MB)")
    print(f"Timing → {out_timing}  ({out_timing.stat().st_size/1e6:.1f} MB)")
    print(f"Done.  entries={n_total}  timeouts={n_timeouts}  errors={n_errors}"
          f"  time={elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
