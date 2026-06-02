"""
compute_metrics.py — Three views of per-tier DGGS sigma separation per config.

Tier pairs: skeptical-vs-rejected, skeptical-vs-credulous, credulous-vs-rejected.
no_ext assumptions excluded from all views.

VIEW 1 — POOLED AUC  (primary, used to rank the 70 configs)
  Pool all assumptions across all ABAFs; compute one AUC per pair on raw sigma.
  Cross-framework bias is constant across configs so relative ranking is valid.
  Also compute Cohen's d on pooled raw sigma.

VIEW 2 — MACRO AUC  (fairness check, NOT used for ranking)
  Compute per-ABAF AUC within each qualifying ABAF, then average.
  Each pair has its own qualifying set: an ABAF qualifies if it has >=1
  assumption in BOTH of that pair's tiers (independent per pair).
  Lead with MEDIAN; also report mean and IQR.

VIEW 3 — NORMALIZED-POOLED AUC  (removes between-ABAF scale differences)
  For each ABAF with >=5 labeled (non-no_ext) assumptions and sd > 0,
  z-score each assumption: z = (sigma - abaf_mean) / abaf_sd, where mean/sd
  are computed over that ABAF's labeled assumptions only.
  Pool z-values across qualifying ABAFs; compute one AUC per pair.
  Also compute Cohen's d on pooled z-values.

DIAGNOSTICS  (--diagnostics flag or once at start of --all)
  Dataset-level stats from tier_labels.json — no parquet needed:
  - no_ext counts (assumptions and ABAFs)
  - View 2 qualifying ABAFs per pair
  - View 3 qualifying ABAFs (>=5 labeled) and assumption count
  - Size distribution: qualifying ABAFs for rejected pairs vs full population

Usage
-----
    python scripts-sweep/compute_metrics.py --diagnostics
    python scripts-sweep/compute_metrics.py --config-id "prod·max·max·lin·k1.0"
    python scripts-sweep/compute_metrics.py --config-index 0
    python scripts-sweep/compute_metrics.py --all
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts-sweep"))

from config_grid import config_by_id, config_by_index

try:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import pyarrow.parquet as _pq
except ImportError as exc:
    print(f"ERROR: missing dependency — {exc}")
    sys.exit(1)

TIER_ORDER = ["skeptical", "credulous", "rejected"]
TIER_PAIRS = [
    ("skeptical", "rejected"),
    ("skeptical", "credulous"),
    ("credulous",  "rejected"),
]
VIEW3_MIN_LABELED = 5
METRICS_DIR       = REPO_ROOT / "results" / "metrics"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _auc(a: list, b: list) -> Optional[float]:
    """AUC: group a = label 1, group b = label 0. Returns None if either empty."""
    if not a or not b:
        return None
    scores = np.concatenate([
        np.array(a, dtype=np.float32),
        np.array(b, dtype=np.float32),
    ])
    labels = np.array([1] * len(a) + [0] * len(b))
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def _cohens_d(a: list, b: list) -> Optional[float]:
    """Pooled Cohen's d = (mean_a - mean_b) / pooled_sd."""
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    pooled_var = (
        ((len(a_arr) - 1) * a_arr.var(ddof=1) + (len(b_arr) - 1) * b_arr.var(ddof=1))
        / (len(a_arr) + len(b_arr) - 2)
    )
    pooled_sd = math.sqrt(float(pooled_var)) if pooled_var > 0 else 0.0
    return float(a_arr.mean() - b_arr.mean()) / pooled_sd if pooled_sd > 0 else None


def _r(v, digits: int = 6) -> Optional[float]:
    return round(float(v), digits) if v is not None else None


# ---------------------------------------------------------------------------
# Diagnostics (tier_labels.json only — no parquet)
# ---------------------------------------------------------------------------

def print_diagnostics(all_tiers: dict) -> None:
    from collections import Counter

    total_abafs = len(all_tiers)
    total_asms  = 0
    no_ext_asms = 0
    no_ext_abafs = 0

    abaf_tier_counts: List[Counter] = []

    for tier_map in all_tiers.values():
        c = Counter(tier_map.values())
        abaf_tier_counts.append(c)
        total_asms  += len(tier_map)
        no_ext_asms += c.get("no_ext", 0)
        if all(v == "no_ext" for v in tier_map.values()):
            no_ext_abafs += 1

    print("=" * 60)
    print("DATASET DIAGNOSTICS (from tier_labels.json)")
    print("=" * 60)
    print(f"Total ABAFs:             {total_abafs:>10,}")
    print(f"Total assumptions:       {total_asms:>10,}")
    pct_asms  = 100 * no_ext_asms  / total_asms  if total_asms  else 0
    pct_abafs = 100 * no_ext_abafs / total_abafs if total_abafs else 0
    print(f"no_ext assumptions:      {no_ext_asms:>10,}  ({pct_asms:.1f}%)")
    print(f"all-no_ext ABAFs:        {no_ext_abafs:>10,}  ({pct_abafs:.1f}%)")
    print()

    # View 2 qualifying ABAFs per pair (independent)
    print("View 2 qualifying ABAFs per pair (>=1 in each tier):")
    for tier_a, tier_b in TIER_PAIRS:
        n = sum(1 for c in abaf_tier_counts
                if c.get(tier_a, 0) >= 1 and c.get(tier_b, 0) >= 1)
        pct = 100 * n / total_abafs if total_abafs else 0
        print(f"  {tier_a:<12} vs {tier_b:<12}  {n:>8,}  ({pct:.1f}%)")
    print()

    # View 3 qualifying ABAFs (>=5 labeled assumptions)
    n_v3_abafs = 0
    n_v3_asms  = 0
    for c in abaf_tier_counts:
        labeled = sum(c.get(t, 0) for t in TIER_ORDER)
        if labeled >= VIEW3_MIN_LABELED:
            n_v3_abafs += 1
            n_v3_asms  += labeled
    pct_v3 = 100 * n_v3_abafs / total_abafs if total_abafs else 0
    print(f"View 3 qualifying ABAFs (>={VIEW3_MIN_LABELED} labeled):  {n_v3_abafs:>8,}  ({pct_v3:.1f}%)")
    print(f"View 3 qualifying assumptions:       {n_v3_asms:>8,}")
    print()

    # Size distribution: rejected-pair qualifying ABAFs vs full population
    all_labeled = np.array([
        sum(c.get(t, 0) for t in TIER_ORDER) for c in abaf_tier_counts
    ])
    rej_qualifying = np.array([
        sum(c.get(t, 0) for t in TIER_ORDER)
        for c in abaf_tier_counts
        if c.get("rejected", 0) >= 1
    ])

    def _dist(arr, label):
        if len(arr) == 0:
            print(f"  {label}: (empty)")
            return
        print(f"  {label} (n={len(arr):,}):  "
              f"median={np.median(arr):.0f}  "
              f"mean={arr.mean():.0f}  "
              f"p90={np.percentile(arr, 90):.0f}  "
              f"max={arr.max():.0f}")

    print("Size distribution of labeled assumptions per ABAF:")
    _dist(all_labeled,    "Full population")
    _dist(rej_qualifying, "ABAFs with >=1 rejected")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_metrics_for_config(
    config_id: str,
    raw_dir: Path,
    all_tiers: dict,
) -> Optional[Dict]:
    raw_parquet = raw_dir / f"{config_id}.parquet"
    if not raw_parquet.exists():
        return None

    # View 1: pooled raw sigma per tier
    pooled: Dict[str, List[float]] = {t: [] for t in TIER_ORDER}

    # View 2: per-ABAF AUC lists (independent per pair)
    per_abaf_aucs: Dict[Tuple, List[float]] = {p: [] for p in TIER_PAIRS}

    # View 3: pooled z-values per tier
    norm: Dict[str, List[float]] = {t: [] for t in TIER_ORDER}
    n_v3_abafs = n_v3_asms = 0

    convergence_by_abaf: Dict[str, bool] = {}

    pf = _pq.ParquetFile(str(raw_parquet))
    for batch in pf.iter_batches(
        columns=["abaf_id", "assumption_id", "sigma_final", "converged"]
    ):
        abaf_ids = batch.column("abaf_id").to_pylist()
        asm_ids  = batch.column("assumption_id").to_pylist()
        sigmas   = batch.column("sigma_final").to_pylist()
        convs    = batch.column("converged").to_pylist()

        if not abaf_ids:
            continue

        # Group rows by abaf_id within this batch — iter_batches() may split
        # or merge row groups across batch boundaries, so one batch ≠ one ABAF.
        batch_groups: Dict = defaultdict(lambda: {"asms": [], "sigmas": [], "conv": None})
        for abaf_id, asm, sigma, conv in zip(abaf_ids, asm_ids, sigmas, convs):
            g = batch_groups[abaf_id]
            g["asms"].append(asm)
            g["sigmas"].append(float(sigma))
            if g["conv"] is None:
                g["conv"] = conv

        for abaf_id, g in batch_groups.items():
            convergence_by_abaf[abaf_id] = g["conv"]
            tier_map = all_tiers.get(abaf_id, {})

            abaf_t: Dict[str, List[float]] = {t: [] for t in TIER_ORDER}
            for asm, sigma in zip(g["asms"], g["sigmas"]):
                tier = tier_map.get(asm, "unknown")
                if tier in TIER_ORDER:
                    pooled[tier].append(sigma)
                    abaf_t[tier].append(sigma)

            # View 2: within-ABAF AUC per pair (independent qualifying)
            for tier_a, tier_b in TIER_PAIRS:
                auc = _auc(abaf_t[tier_a], abaf_t[tier_b])
                if auc is not None:
                    per_abaf_aucs[(tier_a, tier_b)].append(auc)

            # View 3: z-score if >=5 labeled and sd > 0
            all_labeled = [s for t in TIER_ORDER for s in abaf_t[t]]
            if len(all_labeled) >= VIEW3_MIN_LABELED:
                abaf_mean = float(np.mean(all_labeled))
                abaf_sd   = float(np.std(all_labeled, ddof=1))
                if abaf_sd > 0:
                    n_v3_abafs += 1
                    for t in TIER_ORDER:
                        for s in abaf_t[t]:
                            norm[t].append((s - abaf_mean) / abaf_sd)
                            n_v3_asms += 1

    # --- convergence ---
    n_total   = len(convergence_by_abaf)
    conv_rate = sum(convergence_by_abaf.values()) / n_total if n_total else None

    # --- config metadata ---
    try:
        cfg = config_by_id(config_id)
    except KeyError:
        cfg = {"body": "?", "claim": "?", "support": "?", "iota": "?", "k": "?"}

    row: Dict = {
        "config_id":        config_id,
        "body":             cfg.get("body"),
        "claim":            cfg.get("claim"),
        "support":          cfg.get("support"),
        "iota":             cfg.get("iota"),
        "k":                cfg.get("k"),
        "convergence_rate": _r(conv_rate),
    }

    # Per-tier descriptive stats (raw sigma)
    for t in TIER_ORDER:
        arr = np.array(pooled[t])
        row[f"{t}_n"]    = len(arr)
        row[f"{t}_mean"] = _r(arr.mean())     if len(arr) > 0 else None
        row[f"{t}_sd"]   = _r(arr.std(ddof=1)) if len(arr) > 1 else None

    # View 1: pooled AUC + Cohen's d
    for tier_a, tier_b in TIER_PAIRS:
        row[f"v1_auc_{tier_a}_vs_{tier_b}"]    = _r(_auc(pooled[tier_a], pooled[tier_b]))
        row[f"v1_cohend_{tier_a}_vs_{tier_b}"] = _r(_cohens_d(pooled[tier_a], pooled[tier_b]))

    # View 2: macro AUC (median primary)
    for tier_a, tier_b in TIER_PAIRS:
        vals = per_abaf_aucs[(tier_a, tier_b)]
        if vals:
            arr = np.array(vals)
            row[f"v2_auc_median_{tier_a}_vs_{tier_b}"] = _r(np.median(arr))
            row[f"v2_auc_mean_{tier_a}_vs_{tier_b}"]   = _r(arr.mean())
            row[f"v2_auc_iqr_{tier_a}_vs_{tier_b}"]    = _r(np.percentile(arr, 75) - np.percentile(arr, 25))
        else:
            row[f"v2_auc_median_{tier_a}_vs_{tier_b}"] = None
            row[f"v2_auc_mean_{tier_a}_vs_{tier_b}"]   = None
            row[f"v2_auc_iqr_{tier_a}_vs_{tier_b}"]    = None
        row[f"v2_n_abafs_{tier_a}_vs_{tier_b}"] = len(vals)

    # View 3: normalized-pooled AUC + Cohen's d
    row["v3_n_abafs"] = n_v3_abafs
    row["v3_n_asms"]  = n_v3_asms
    for tier_a, tier_b in TIER_PAIRS:
        row[f"v3_auc_{tier_a}_vs_{tier_b}"]    = _r(_auc(norm[tier_a], norm[tier_b]))
        row[f"v3_cohend_{tier_a}_vs_{tier_b}"] = _r(_cohens_d(norm[tier_a], norm[tier_b]))

    return row


# ---------------------------------------------------------------------------
# Per-config metrics file — one JSON per config, no shared state
# ---------------------------------------------------------------------------

def write_metrics(row: Dict, metrics_dir: Path) -> None:
    """Write one config's metrics to results/metrics/{config_id}.json.

    Each Condor job writes its own file — no shared state, no race condition.
    Load all configs for analysis with:
        import json, glob, pandas as pd
        df = pd.DataFrame([json.load(open(f))
                           for f in sorted(glob.glob("results/metrics/*.json"))])
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{row['config_id']}.json"
    if out.exists():
        print(f"  Already exists: {out} — skipping.")
        return
    with open(out, "w") as fh:
        json.dump(row, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diagnostics",  action="store_true",
                      help="Print dataset diagnostics and exit (no parquet needed)")
    mode.add_argument("--config-id",    type=str)
    mode.add_argument("--config-index", type=int)
    mode.add_argument("--all",          action="store_true")
    p.add_argument("--raw-dir",     default=str(REPO_ROOT / "results" / "raw"))
    p.add_argument("--tier-labels", default=str(REPO_ROOT / "dataset" / "tier_labels.json"))
    p.add_argument("--metrics-dir", default=str(METRICS_DIR))
    args = p.parse_args()

    with open(args.tier_labels) as fh:
        all_tiers: dict = json.load(fh)
    print(f"Tier labels: {len(all_tiers):,} ABAFs\n")

    if args.diagnostics:
        print_diagnostics(all_tiers)
        return

    raw_dir     = Path(args.raw_dir)
    metrics_dir = Path(args.metrics_dir)

    if args.all:
        config_ids = [f.stem for f in sorted(raw_dir.glob("*.parquet"))]
        print(f"Processing {len(config_ids)} configs\n")
        print_diagnostics(all_tiers)
    elif args.config_id:
        config_ids = [args.config_id]
    else:
        config_ids = [config_by_index(args.config_index)["config_id"]]

    for config_id in config_ids:
        print(f"Computing: {config_id}")
        row = compute_metrics_for_config(config_id, raw_dir, all_tiers)
        if row is None:
            print(f"  Not found: {raw_dir}/{config_id}.parquet — skipping.")
            continue

        write_metrics(row, metrics_dir)

        def _f(k): return f"{row.get(k) or 'N/A'}"
        print(
            f"  V1 auc  sk/rej={_f('v1_auc_skeptical_vs_rejected')}  "
            f"sk/cred={_f('v1_auc_skeptical_vs_credulous')}  "
            f"cred/rej={_f('v1_auc_credulous_vs_rejected')}"
        )
        print(
            f"  V2 med  sk/rej={_f('v2_auc_median_skeptical_vs_rejected')} "
            f"(n={_f('v2_n_abafs_skeptical_vs_rejected')})  "
            f"sk/cred={_f('v2_auc_median_skeptical_vs_credulous')} "
            f"(n={_f('v2_n_abafs_skeptical_vs_credulous')})  "
            f"cred/rej={_f('v2_auc_median_credulous_vs_rejected')} "
            f"(n={_f('v2_n_abafs_credulous_vs_rejected')})"
        )
        print(
            f"  V3 auc  sk/rej={_f('v3_auc_skeptical_vs_rejected')}  "
            f"sk/cred={_f('v3_auc_skeptical_vs_credulous')}  "
            f"cred/rej={_f('v3_auc_credulous_vs_rejected')}  "
            f"(n_abafs={_f('v3_n_abafs')})"
        )
        print(f"  conv={_f('convergence_rate')}")

    print(f"\nMetrics → {metrics_dir}/")


if __name__ == "__main__":
    main()
