"""
run_ablation.py — tau initialisation ablation on the winning kernel.

Fixed kernel : prod·max·max·lin·k1.0
Conditions   : {asm_source}·{rule_source}  where asm_source ∈ {ci, neutral, random}
                                                   rule_source ∈ {ci, neutral, random, hybrid}
Config ID    : prod·max·max·lin·k1.0·{asm_source}·{rule_source}

THE GRID (3×3 original + 2 hybrid)
--------
  ci·ci        ci·neutral        ci·random        ci·hybrid
  neutral·ci   neutral·neutral   neutral·random   neutral·hybrid
  random·ci    random·neutral    random·random

tau_a sources
  ci       — base scores loaded from scores.json (same as sweep)
  neutral  — 0.5 for every assumption
  random   — U[0,1], drawn from rng_asm (seed=42)

tau_r sources
  ci       — rule reliability scores loaded from scores.json (same as sweep)
  neutral  — 1.0 for every rule (passed as empty dict; ABAF defaults to 1.0)
  random   — U[0,1], drawn from rng_rule (seed=123)
  hybrid   — 1.0 for empty-body rules (facts), CI scores for rules with bodies

SEEDING
-------
  Two independent numpy Generators, created fresh at the start of each condition
  run and advanced sequentially over ABAFs sorted by ABAF ID.

  Within each ABAF:
    1. Sort assumption names alphabetically.
    2. Draw len(assumptions) values from rng_asm in one call (if asm=="random").
    3. Sort rule indices; draw len(rules) values from rng_rule (if rule=="random").
    Neutral conditions consume no RNG state.

  ci·ci: reuses the existing sweep raw/timing parquets (copy + config_id rewrite).

OUTPUTS (all under results/ablations/)
  raw/{condition_id}.parquet     — per-assumption sigma (same schema as sweep)
  timing/{condition_id}.parquet  — per-ABAF timing    (same schema as sweep)

Compute metrics afterwards with:
  python scripts-sweep/compute_metrics.py --all \\
      --raw-dir results/ablations/raw \\
      --metrics-dir results/ablations/metrics

Usage
-----
    python scripts-sweep/run_ablation.py --condition neutral·neutral
    python scripts-sweep/run_ablation.py --condition all
    python scripts-sweep/run_ablation.py --condition ci·ci   # copy step only
    python scripts-sweep/run_ablation.py --list              # print conditions and exit
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
from scr.dggs.tau_utils import load_scores, build_tau_a, build_tau_r

sys.path.insert(0, str(REPO_ROOT / "scripts-sweep"))
from config_grid import config_by_id, build_kernels

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KERNEL_ID    = "prod·max·max·lin·k1.0"
ASM_SOURCES  = ["ci", "neutral", "random"]
RULE_SOURCES = ["ci", "neutral", "random", "hybrid"]
ALL_CONDITIONS = [f"{a}·{r}" for a in ASM_SOURCES for r in RULE_SOURCES]

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

def condition_id(asm_src: str, rule_src: str) -> str:
    return f"{KERNEL_ID}·{asm_src}·{rule_src}"


def parse_condition(s: str):
    parts = s.split("·")
    if len(parts) != 2 or parts[0] not in ASM_SOURCES or parts[1] not in RULE_SOURCES:
        raise argparse.ArgumentTypeError(
            f"Invalid condition {s!r}. Expected {{asm}}·{{rule}} where each "
            f"is one of {ASM_SOURCES}."
        )
    return parts[0], parts[1]


class _Timeout(Exception):
    pass


def _arm(seconds: int) -> None:
    def _handler(signum, frame):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)


def _disarm() -> None:
    signal.alarm(0)


def build_tau_r_hybrid(dg, rule_scores: dict) -> dict:
    """
    Hybrid rule initialization:
    - Rules with empty bodies (facts): explicitly set tau_r = 1.0
    - Rules with bodies: tau_r from rule_scores (CI values)
    """
    tau_r: dict = {}
    for idx, (head, body) in dg.rules.items():
        if not body:
            # Empty-body rule (fact): explicitly set to 1.0
            tau_r[idx] = 1.0
        else:
            # Rule with body: use CI score if available
            key = head + "|" + "|".join(body)
            if key in rule_scores:
                tau_r[idx] = rule_scores[key]
    return tau_r


# ---------------------------------------------------------------------------
# ci·ci: copy existing sweep parquets, rewrite config_id in timing
# ---------------------------------------------------------------------------

def handle_ci_ci(sweep_raw_dir: Path, sweep_timing_dir: Path,
                 out_dir: Path, timing_dir: Path) -> None:
    cond = condition_id("ci", "ci")
    src_raw    = sweep_raw_dir    / f"{KERNEL_ID}.parquet"
    src_timing = sweep_timing_dir / f"{KERNEL_ID}.parquet"
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

    # Rewrite timing parquet with corrected config_id
    try:
        table = pq.read_table(str(src_timing))
        df = table.to_pandas()
        df["config_id"] = cond
        new_table = pa.Table.from_pandas(df[["abaf_id", "config_id",
                                             "construction_time_s",
                                             "semantics_time_s",
                                             "timed_out"]],
                                         schema=_TIMING_SCHEMA,
                                         preserve_index=False)
        pq.write_table(new_table, str(dst_timing), compression="snappy")
        print(f"  Timing → {dst_timing}")
    except Exception as exc:
        print(f"  [ERROR] could not rewrite timing: {exc}")


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(
    entry: dict,
    asm_src: str,
    rule_src: str,
    rng_asm: np.random.Generator,
    rng_rule: np.random.Generator,
    kernel_cfg: dict,
    runner_kwargs: dict,
    timeout_s: int,
    cond_id: str,
) -> tuple:
    abaf_id     = Path(entry["abaf"]).stem
    aba_path    = REPO_ROOT / entry["abaf"]
    scores_path = REPO_ROOT / entry["scores"]

    timing = {
        "abaf_id":             abaf_id,
        "config_id":           cond_id,
        "construction_time_s": float("nan"),
        "semantics_time_s":    float("nan"),
        "timed_out":           False,
        "error":               None,
    }

    # --- Construction ---
    try:
        t0 = time.perf_counter()

        plain_scores: dict = {}
        rule_scores:  dict = {}
        if asm_src == "ci" or rule_src == "ci" or rule_src == "hybrid":
            plain_scores, rule_scores = load_scores(str(scores_path))

        dg = DependencyGraph()
        dg.create_from_file(str(aba_path))

        # tau_a
        sorted_asms = sorted(dg.assumptions)
        if asm_src == "ci":
            tau_a = build_tau_a(dg, plain_scores)
        elif asm_src == "neutral":
            tau_a = {name: 0.5 for name in sorted_asms}
        else:  # random
            vals = rng_asm.random(len(sorted_asms))
            tau_a = dict(zip(sorted_asms, vals))

        # tau_r
        sorted_rule_idxs = sorted(dg.rules.keys())
        if rule_src == "ci":
            tau_r = build_tau_r(dg, rule_scores)
        elif rule_src == "neutral":
            tau_r = {}  # ABAF defaults absent rules to 1.0
        elif rule_src == "hybrid":
            tau_r = build_tau_r_hybrid(dg, rule_scores)
        else:  # random
            vals = rng_rule.random(len(sorted_rule_idxs))
            tau_r = dict(zip(sorted_rule_idxs, vals))

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
# Run one condition
# ---------------------------------------------------------------------------

def run_condition(
    asm_src: str,
    rule_src: str,
    manifest: list,
    kernel_cfg: dict,
    runner_kwargs: dict,
    timeout_s: int,
    out_dir: Path,
    timing_dir: Path,
) -> None:
    cond      = condition_id(asm_src, rule_src)
    out_sigma  = out_dir    / f"{cond}.parquet"
    out_timing = timing_dir / f"{cond}.parquet"

    if out_timing.exists():
        print(f"Timing already exists: {out_timing} — delete it to re-run.")
        return

    write_sigma = not out_sigma.exists()
    print(f"\nCondition: {cond}")
    print(f"  Sigma  : {'WRITE' if write_sigma else 'SKIP (already exists)'}")
    print(f"  Timing : {out_timing}")

    # Fresh RNGs for every condition — same seeds, same advancement order
    rng_asm  = np.random.default_rng(seed=42)
    rng_rule = np.random.default_rng(seed=123)

    # Sort by ABAF ID for deterministic RNG state advancement
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
            entry, asm_src, rule_src,
            rng_asm, rng_rule,
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
                   help="Condition to run, e.g. 'neutral·random', or 'all'.")
    p.add_argument("--list", action="store_true",
                   help="Print all valid conditions and exit.")
    p.add_argument("--manifest",
                   default=str(REPO_ROOT / "causal_manifest.json"))
    p.add_argument("--out-dir",
                   default=str(REPO_ROOT / "results" / "ablations" / "raw"))
    p.add_argument("--timing-dir",
                   default=str(REPO_ROOT / "results" / "ablations" / "timing"))
    p.add_argument("--sweep-raw-dir",
                   default=str(REPO_ROOT / "results" / "raw"),
                   help="Source dir for ci·ci raw parquet (sweep output).")
    p.add_argument("--sweep-timing-dir",
                   default=str(REPO_ROOT / "results" / "timing"),
                   help="Source dir for ci·ci timing parquet (sweep output).")
    p.add_argument("--max-iter",  type=int,   default=5000, dest="max_iter")
    p.add_argument("--epsilon",   type=float, default=1e-3)
    p.add_argument("--window",    type=int,   default=5)
    p.add_argument("--timeout",   type=int,   default=600,
                   help="Per-ABAF wall-clock timeout in seconds.")
    args = p.parse_args()

    if args.list:
        print("Valid conditions:")
        for c in ALL_CONDITIONS:
            print(f"  {c}")
        return

    if not args.condition:
        p.error("--condition is required (or use --list to see options).")

    out_dir    = Path(args.out_dir);    out_dir.mkdir(parents=True, exist_ok=True)
    timing_dir = Path(args.timing_dir); timing_dir.mkdir(parents=True, exist_ok=True)
    sweep_raw_dir    = Path(args.sweep_raw_dir)
    sweep_timing_dir = Path(args.sweep_timing_dir)

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    print(f"Manifest: {len(manifest)} entries")
    print(f"Output  : {out_dir.parent}/")

    kernel_cfg    = config_by_id(KERNEL_ID)
    runner_kwargs = dict(max_iter=args.max_iter, epsilon=args.epsilon, window=args.window)

    conditions_to_run = ALL_CONDITIONS if args.condition == "all" else [args.condition]

    for cond_str in conditions_to_run:
        parts = cond_str.split("·")
        if len(parts) != 2 or parts[0] not in ASM_SOURCES or parts[1] not in RULE_SOURCES:
            print(f"[SKIP] Unknown condition: {cond_str!r}")
            continue
        asm_src, rule_src = parts

        if asm_src == "ci" and rule_src == "ci":
            print(f"\nCondition: {condition_id('ci', 'ci')}  (copy from sweep)")
            handle_ci_ci(sweep_raw_dir, sweep_timing_dir, out_dir, timing_dir)
        else:
            run_condition(
                asm_src, rule_src,
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
