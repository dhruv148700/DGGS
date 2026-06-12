"""
run_neutral_grid.py — Constant-tau grid over the winning kernel.

Fixed kernel : prod·max·max·lin·k1.0
Conditions   : tau_a ∈ TAU_A_VALS × tau_r ∈ TAU_R_VALS  (36 conditions)
Config ID    : prod·max·max·lin·k1.0·a{tau_a:.2f}·r{tau_r:.2f}

All assumptions share the same constant tau_a; all rules share the same
constant tau_r.  The neutral·neutral baseline (a=0.50, r=1.00) is copied
from the existing ablation parquets rather than recomputed.

TAU GRID
--------
  tau_a : 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90   (9 values)
  tau_r : 0.25 0.50 0.75 1.00                              (4 values)
  Total : 36 conditions  (a0.50·r1.00 = neutral·neutral, copied)

OUTPUTS (all under results/neutral_perturbations/)
  raw/{condition_id}.parquet     — per-assumption sigma (same schema as sweep)
  timing/{condition_id}.parquet  — per-ABAF timing      (same schema as sweep)

Compute metrics afterwards with:
  python scripts-sweep/compute_metrics.py --all \\
      --raw-dir results/neutral_perturbations/raw \\
      --metrics-dir results/neutral_perturbations/metrics

Usage
-----
    python scripts-sweep/run_neutral_grid.py --condition a0.50·r1.00
    python scripts-sweep/run_neutral_grid.py --condition all
    python scripts-sweep/run_neutral_grid.py --list
"""

import argparse
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scr.dependency_graph import DependencyGraph
from scr.ABAF import ABAF
from scr.dggs import DGGSRunner

sys.path.insert(0, str(REPO_ROOT / "scripts-sweep"))
from config_grid import config_by_id, build_kernels

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KERNEL_ID = "prod·max·max·lin·k1.0"

TAU_A_VALS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
TAU_R_VALS = [0.25, 0.50, 0.75, 1.00]

NEUTRAL_A = 0.50
NEUTRAL_R = 1.00

ALL_CONDITIONS = [f"a{a:.2f}·r{r:.2f}" for a in TAU_A_VALS for r in TAU_R_VALS]

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
# Helpers
# ---------------------------------------------------------------------------

def condition_id(tau_a: float, tau_r: float) -> str:
    return f"{KERNEL_ID}·a{tau_a:.2f}·r{tau_r:.2f}"


def parse_condition(s: str):
    """Parse 'a0.50·r1.00' → (0.50, 1.00)."""
    parts = s.split("·")
    if len(parts) != 2 or not parts[0].startswith("a") or not parts[1].startswith("r"):
        raise argparse.ArgumentTypeError(
            f"Invalid condition {s!r}: expected 'a<float>·r<float>'"
        )
    try:
        return float(parts[0][1:]), float(parts[1][1:])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid condition {s!r}: could not parse floats"
        )


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
# Baseline: copy neutral·neutral, rewrite config_id in timing
# ---------------------------------------------------------------------------

def handle_neutral_neutral(
    ablation_raw_dir: Path,
    ablation_timing_dir: Path,
    out_dir: Path,
    timing_dir: Path,
) -> None:
    cond       = condition_id(NEUTRAL_A, NEUTRAL_R)
    src_raw    = ablation_raw_dir    / f"{KERNEL_ID}·neutral·neutral.parquet"
    src_timing = ablation_timing_dir / f"{KERNEL_ID}·neutral·neutral.parquet"
    dst_raw    = out_dir    / f"{cond}.parquet"
    dst_timing = timing_dir / f"{cond}.parquet"

    if not src_raw.exists():
        print(f"  [ERROR] source raw parquet not found: {src_raw}")
        return
    if not src_timing.exists():
        print(f"  [ERROR] source timing parquet not found: {src_timing}")
        return

    if dst_raw.exists():
        print(f"  Raw already exists: {dst_raw} — skipping.")
    else:
        shutil.copy2(src_raw, dst_raw)
        print(f"  Raw  → {dst_raw}")

    if dst_timing.exists():
        print(f"  Timing already exists: {dst_timing} — skipping.")
        return

    try:
        table = pq.read_table(str(src_timing))
        df = table.to_pandas()
        df["config_id"] = cond
        new_table = pa.Table.from_pandas(
            df[["abaf_id", "config_id", "construction_time_s",
                "semantics_time_s", "timed_out"]],
            schema=_TIMING_SCHEMA,
            preserve_index=False,
        )
        pq.write_table(new_table, str(dst_timing), compression="snappy")
        print(f"  Timing → {dst_timing}")
    except Exception as exc:
        print(f"  [ERROR] could not rewrite timing: {exc}")


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(
    entry: dict,
    tau_a_val: float,
    tau_r_val: float,
    kernel_cfg: dict,
    runner_kwargs: dict,
    timeout_s: int,
    cond_id: str,
) -> tuple:
    abaf_id  = Path(entry["abaf"]).stem
    aba_path = REPO_ROOT / entry["abaf"]

    timing = {
        "abaf_id":             abaf_id,
        "config_id":           cond_id,
        "construction_time_s": float("nan"),
        "semantics_time_s":    float("nan"),
        "timed_out":           False,
        "error":               None,
    }

    try:
        t0 = time.perf_counter()

        dg = DependencyGraph()
        dg.create_from_file(str(aba_path))

        tau_a = {name: tau_a_val for name in sorted(dg.assumptions)}
        # Pass {} for tau_r=1.0: ABAF defaults absent rules to 1.0 — identical result
        tau_r = ({} if abs(tau_r_val - 1.0) < 1e-9
                 else {idx: tau_r_val for idx in sorted(dg.rules.keys())})

        abaf = ABAF.from_dependency_graph(dg, tau_a=tau_a, tau_r=tau_r)
        body_agg, claim_agg, support_agg, influence = build_kernels(kernel_cfg)
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
# Run one condition
# ---------------------------------------------------------------------------

def run_condition(
    tau_a_val: float,
    tau_r_val: float,
    manifest: list,
    kernel_cfg: dict,
    runner_kwargs: dict,
    timeout_s: int,
    out_dir: Path,
    timing_dir: Path,
) -> None:
    cond       = condition_id(tau_a_val, tau_r_val)
    out_sigma  = out_dir    / f"{cond}.parquet"
    out_timing = timing_dir / f"{cond}.parquet"

    if out_timing.exists():
        print(f"Timing already exists: {out_timing} — delete it to re-run.")
        return

    write_sigma = not out_sigma.exists()
    print(f"\nCondition: {cond}")
    print(f"  tau_a={tau_a_val:.2f}  tau_r={tau_r_val:.2f}")
    print(f"  Sigma  : {'WRITE' if write_sigma else 'SKIP (already exists)'}")
    print(f"  Timing : {out_timing}")

    sorted_manifest = sorted(manifest, key=lambda e: Path(e["abaf"]).stem)

    sigma_writer = (
        pq.ParquetWriter(str(out_sigma), _SIGMA_SCHEMA, compression="snappy")
        if write_sigma else None
    )
    timing_writer = pq.ParquetWriter(str(out_timing), _TIMING_SCHEMA, compression="snappy")

    n_total = len(sorted_manifest)
    n_done = n_errors = n_timeouts = 0
    t_start = time.perf_counter()

    for entry in sorted_manifest:
        sigma_cols, timing = process_entry(
            entry, tau_a_val, tau_r_val,
            kernel_cfg, runner_kwargs, timeout_s, cond,
        )
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

    elapsed = time.perf_counter() - t_start
    if write_sigma:
        print(f"  Sigma  → {out_sigma}  ({out_sigma.stat().st_size/1e6:.1f} MB)")
    print(f"  Timing → {out_timing}  ({out_timing.stat().st_size/1e6:.1f} MB)")
    print(f"  Done.  entries={n_total}  timeouts={n_timeouts}  errors={n_errors}"
          f"  time={elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition",
                   help="Condition to run, e.g. 'a0.50·r1.00', or 'all'.")
    p.add_argument("--list", action="store_true",
                   help="Print all valid conditions and exit.")
    p.add_argument("--manifest",
                   default=str(REPO_ROOT / "causal_manifest.json"))
    p.add_argument("--out-dir",
                   default=str(REPO_ROOT / "results" / "neutral_perturbations" / "raw"))
    p.add_argument("--timing-dir",
                   default=str(REPO_ROOT / "results" / "neutral_perturbations" / "timing"))
    p.add_argument("--ablation-raw-dir",
                   default=str(REPO_ROOT / "results" / "ablations" / "raw"),
                   help="Source dir for the neutral·neutral parquets to copy.")
    p.add_argument("--ablation-timing-dir",
                   default=str(REPO_ROOT / "results" / "ablations" / "timing"),
                   help="Source dir for the neutral·neutral timing parquet to copy.")
    p.add_argument("--max-iter",  type=int,   default=5000, dest="max_iter")
    p.add_argument("--epsilon",   type=float, default=1e-3)
    p.add_argument("--window",    type=int,   default=5)
    p.add_argument("--timeout",   type=int,   default=600,
                   help="Per-ABAF wall-clock timeout in seconds.")
    args = p.parse_args()

    if args.list:
        print(f"All {len(ALL_CONDITIONS)} conditions:")
        for c in ALL_CONDITIONS:
            print(f"  {c}")
        return

    if not args.condition:
        p.error("--condition is required (or use --list to see options).")

    out_dir    = Path(args.out_dir);    out_dir.mkdir(parents=True, exist_ok=True)
    timing_dir = Path(args.timing_dir); timing_dir.mkdir(parents=True, exist_ok=True)
    ablation_raw_dir    = Path(args.ablation_raw_dir)
    ablation_timing_dir = Path(args.ablation_timing_dir)

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    print(f"Manifest: {len(manifest)} entries")
    print(f"Output  : {out_dir.parent}/")

    kernel_cfg    = config_by_id(KERNEL_ID)
    runner_kwargs = dict(max_iter=args.max_iter, epsilon=args.epsilon, window=args.window)

    conditions_to_run = ALL_CONDITIONS if args.condition == "all" else [args.condition]

    for cond_str in conditions_to_run:
        if cond_str not in ALL_CONDITIONS:
            print(f"[SKIP] Unknown condition: {cond_str!r}. Use --list to see valid conditions.")
            continue
        tau_a_val, tau_r_val = parse_condition(cond_str)

        is_baseline = abs(tau_a_val - NEUTRAL_A) < 1e-9 and abs(tau_r_val - NEUTRAL_R) < 1e-9
        if is_baseline:
            print(f"\nCondition: {condition_id(tau_a_val, tau_r_val)}  (copy from neutral·neutral)")
            handle_neutral_neutral(ablation_raw_dir, ablation_timing_dir, out_dir, timing_dir)
        else:
            run_condition(
                tau_a_val, tau_r_val,
                manifest, kernel_cfg, runner_kwargs, args.timeout,
                out_dir, timing_dir,
            )

    print("\nAll requested conditions complete.")
    print(f"Next step — compute metrics:")
    print(f"  python scripts-sweep/compute_metrics.py --all \\")
    print(f"      --raw-dir {out_dir} \\")
    print(f"      --metrics-dir {out_dir.parent / 'metrics'}")


if __name__ == "__main__":
    main()
