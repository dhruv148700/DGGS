"""
analyse_sweep.py — Full analysis of the 70-config DGGS kernel sweep.

Analyses (in order):
  1. Filter   — drop configs with convergence_rate < 0.9; report survivors
  2. Leaderboard — full table ranked by v1_auc_skeptical_vs_rejected;
                   flags configs where V3 and V1 ranks disagree by > RANK_FLAG_THRESH
  3. Scatter  — convergence_rate vs v1_auc_sk/rej for all loaded configs
  4. k curve  — V1 AUC vs k for prod·max·lin and prod·max·quad families
  5. Tier profile — all three AUC pairs for top 10 survivors
  6. Timing   — mean construction + semantics time, timeout rate per config

Configs without a metric JSON are silently skipped (incomplete/still-running jobs).
The same configs are skipped in the timing analysis.

Usage
-----
    python scripts-sweep/analyse_sweep.py
    python scripts-sweep/analyse_sweep.py --metrics-dir results/metrics \
        --timing-dir results/timing --out-dir results/figures
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import pyarrow.parquet as pq
except ImportError as exc:
    print(f"ERROR: missing dependency — {exc}")
    sys.exit(1)

CONV_THRESHOLD   = 0.9
RANK_FLAG_THRESH = 5   # flag if |rank_v1 - rank_v3| exceeds this

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    files = sorted(metrics_dir.glob("*.json"))
    rows = []
    for f in files:
        with open(f) as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    return df


def load_timing(timing_dir: Path, valid_configs: set) -> pd.DataFrame:
    """Load timing parquets for configs in valid_configs only."""
    rows = []
    for f in sorted(timing_dir.glob("*.parquet")):
        cfg = f.stem
        if cfg not in valid_configs:
            continue
        try:
            table = pq.read_table(str(f))
        except Exception as exc:
            print(f"  [WARN] skipping corrupted timing parquet: {f.name} ({exc})")
            continue
        pdf = table.to_pandas()
        agg = {
            "config_id":            cfg,
            "mean_construction_s":  pdf["construction_time_s"].mean(),
            "mean_semantics_s":      pdf["semantics_time_s"].mean(),
            "timeout_rate":          pdf["timed_out"].mean(),
            "n_abafs":               len(pdf),
        }
        rows.append(agg)
    return pd.DataFrame(rows)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# 1. Filter
# ---------------------------------------------------------------------------

def analysis_filter(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("1. FILTER  (convergence_rate >= {:.0%})".format(CONV_THRESHOLD))
    print("=" * 65)
    print(f"Loaded configs:    {len(df)}")
    dropped = df[df["convergence_rate"] < CONV_THRESHOLD]
    if len(dropped):
        print(f"Dropped ({len(dropped)}):")
        for _, r in dropped.sort_values("convergence_rate").iterrows():
            print(f"  {r['config_id']:<40}  conv={r['convergence_rate']:.3f}")
    survivors = df[df["convergence_rate"] >= CONV_THRESHOLD].copy()
    print(f"Survivors:         {len(survivors)}")

    # Compute rank_score: average of V2 median AUCs for sk/rej and cr/rej
    survivors["rank_score"] = (
        survivors["v2_auc_median_skeptical_vs_rejected"] +
        survivors["v2_auc_median_credulous_vs_rejected"]
    ) / 2

    return survivors


# ---------------------------------------------------------------------------
# 2. Leaderboard
# ---------------------------------------------------------------------------

def analysis_leaderboard(survivors: pd.DataFrame, missing: list) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("2. LEADERBOARD  (ranked by rank_score = mean(V2 median sk/rej, cr/rej))")
    print("=" * 65)

    df = survivors.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["rank_v3"] = df["v3_auc_skeptical_vs_rejected"].rank(ascending=False, method="min").astype(int)
    df["rank_diff"] = (df["rank"] - df["rank_v3"]).abs()
    df["flag"] = df["rank_diff"] > RANK_FLAG_THRESH

    cols = [
        "rank", "config_id",
        "rank_score",
        "v1_auc_skeptical_vs_rejected", "v1_auc_credulous_vs_rejected", "v1_auc_skeptical_vs_credulous",
        "v2_auc_median_skeptical_vs_rejected", "v2_auc_median_credulous_vs_rejected",
        "v3_auc_skeptical_vs_rejected",
        "v1_cohend_skeptical_vs_rejected",
        "convergence_rate",
        "rank_v3", "rank_diff", "flag",
    ]
    print(f"\n{'Rk':<4} {'Config':<40} {'Rank Score':>11} {'V1 sk/rej':>10} {'V1 cr/rej':>10} "
          f"{'V2 sk/rej':>10} {'V2 cr/rej':>10} {'V3 sk/rej':>10} {'Conv':>6} {'V3rk':>5} {'!':>2}")
    print("-" * 140)
    for _, r in df[cols].iterrows():
        flag_str = " *" if r["flag"] else "  "
        print(
            f"{int(r['rank']):<4} {r['config_id']:<40} "
            f"{r['rank_score']:>11.4f} "
            f"{r['v1_auc_skeptical_vs_rejected']:>10.4f} "
            f"{r['v1_auc_credulous_vs_rejected']:>10.4f} "
            f"{r['v2_auc_median_skeptical_vs_rejected']:>10.4f} "
            f"{r['v2_auc_median_credulous_vs_rejected']:>10.4f} "
            f"{r['v3_auc_skeptical_vs_rejected']:>10.4f} "
            f"{r['convergence_rate']:>6.3f} "
            f"{int(r['rank_v3']):>5}"
            f"{flag_str}"
        )

    if missing:
        print()
        print("Placeholder rows (no metric yet — still running or incomplete):")
        for m in sorted(missing):
            print(f"  {'—':<4} {m:<40}  {'N/A':>10}")

    flagged = df[df["flag"]]
    if len(flagged):
        print(f"\nFlagged configs (|rank_score - rank_v3| > {RANK_FLAG_THRESH}):")
        for _, r in flagged.iterrows():
            print(f"  {r['config_id']:<40}  rank_score-rank={int(r['rank'])}  V3-rank={int(r['rank_v3'])}  diff={int(r['rank_diff'])}")
    else:
        print(f"\nNo configs flagged for rank_score/V3 rank disagreement (threshold={RANK_FLAG_THRESH}).")

    return df


# ---------------------------------------------------------------------------
# 3. Scatter: convergence_rate vs v1_auc_sk/rej (all loaded configs)
# ---------------------------------------------------------------------------

def plot_scatter(df_all: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("3. SCATTER  convergence_rate vs rank_score")
    print("=" * 65)

    # Compute rank_score for all configs (including those not in survivors)
    df_all["rank_score"] = (
        df_all["v2_auc_median_skeptical_vs_rejected"] +
        df_all["v2_auc_median_credulous_vs_rejected"]
    ) / 2

    fig, ax = plt.subplots(figsize=(8, 5))

    body_colors = {"prod": "#1f77b4", "min": "#ff7f0e"}
    claim_markers = {"max": "o", "mean": "s", "sum": "^"}

    for _, r in df_all.iterrows():
        color  = body_colors.get(r["body"], "gray")
        marker = claim_markers.get(r["claim"], "x")
        ax.scatter(r["convergence_rate"], r["rank_score"],
                   c=color, marker=marker, s=60, alpha=0.75, edgecolors="none")

    ax.axvline(CONV_THRESHOLD, color="red", linestyle="--", linewidth=1, label=f"conv={CONV_THRESHOLD}")
    ax.set_xlabel("Convergence rate (fraction of ABAFs converged)")
    ax.set_ylabel("Rank Score  (mean V2 median AUCs)")
    ax.set_title("Reliability–performance frontier: all loaded configs")

    # Legend: body colour
    for body, col in body_colors.items():
        ax.scatter([], [], c=col, label=f"body={body}", s=60)
    # Legend: claim marker
    for claim, mk in claim_markers.items():
        ax.scatter([], [], c="gray", marker=mk, label=f"claim={claim}", s=60)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    save(fig, out_dir, "scatter_conv_vs_auc.png")


# ---------------------------------------------------------------------------
# 4. k sensitivity curve
# ---------------------------------------------------------------------------

def plot_k_curve(survivors: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("4. k SENSITIVITY CURVE  prod·max·lin and prod·max·quad")
    print("=" * 65)

    families = {
        "prod·max·lin":  survivors[(survivors["body"] == "prod") & (survivors["claim"] == "max") & (survivors["iota"] == "lin")],
        "prod·max·quad": survivors[(survivors["body"] == "prod") & (survivors["claim"] == "max") & (survivors["iota"] == "quad")],
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"prod·max·lin": "#1f77b4", "prod·max·quad": "#ff7f0e"}
    for name, sub in families.items():
        if sub.empty:
            print(f"  {name}: no surviving configs")
            continue
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["rank_score"],
                marker="o", label=name, color=colors[name])
        for _, r in sub.iterrows():
            ax.annotate(f"k={r['k']:.1f}", (r["k"], r["rank_score"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
        print(f"  {name}:")
        for _, r in sub.iterrows():
            print(f"    k={r['k']:.1f}  rank_score={r['rank_score']:.4f}")

    ax.set_xlabel("k  (conservativeness)")
    ax.set_ylabel("Rank Score  (mean V2 median AUCs)")
    ax.set_title("k sensitivity — prod·max family")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, out_dir, "k_sensitivity.png")


# ---------------------------------------------------------------------------
# 5. Tier-pair profile for top 10
# ---------------------------------------------------------------------------

def plot_tier_profile(ranked: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("5. TIER-PAIR PROFILE  top 10 survivors")
    print("=" * 65)

    top10 = ranked.head(10).copy()
    labels = [r["config_id"].replace("·", "·\n") for _, r in top10.iterrows()]

    pairs = [
        ("v1_auc_skeptical_vs_rejected",  "sk/rej",  "#1f77b4"),
        ("v1_auc_credulous_vs_rejected",  "cr/rej",  "#ff7f0e"),
        ("v1_auc_skeptical_vs_credulous", "sk/cr",   "#2ca02c"),
    ]

    x = np.arange(len(top10))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (col, label, color) in enumerate(pairs):
        vals = top10[col].values
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random (0.5)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("V1 AUC")
    ax.set_title("Tier-pair AUC profile — top 10 surviving configs")
    ax.legend(fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    save(fig, out_dir, "tier_profile_top10.png")


# ---------------------------------------------------------------------------
# 6. Timing
# ---------------------------------------------------------------------------

def analysis_timing(timing_df: pd.DataFrame, winner_id: str) -> None:
    print("\n" + "=" * 65)
    print("6. TIMING SUMMARY")
    print("=" * 65)

    t = timing_df.sort_values("mean_semantics_s")
    print(f"\n{'Config':<40} {'Constr(s)':>10} {'Seman(s)':>9} {'Timeout%':>9} {'n_ABAFs':>8}")
    print("-" * 80)
    for _, r in t.iterrows():
        marker = " ◀ WINNER" if r["config_id"] == winner_id else ""
        print(
            f"{r['config_id']:<40} "
            f"{r['mean_construction_s']:>10.3f} "
            f"{r['mean_semantics_s']:>9.3f} "
            f"{r['timeout_rate']*100:>8.1f}% "
            f"{int(r['n_abafs']):>8}"
            f"{marker}"
        )

    w = t[t["config_id"] == winner_id]
    if not w.empty:
        wr = w.iloc[0]
        print(f"\nWinner ({winner_id}):")
        print(f"  construction: {wr['mean_construction_s']:.3f}s/ABAF")
        print(f"  semantics:    {wr['mean_semantics_s']:.3f}s/ABAF")
        print(f"  timeout rate: {wr['timeout_rate']*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", default=str(REPO_ROOT / "results" / "metrics"))
    p.add_argument("--timing-dir",  default=str(REPO_ROOT / "results" / "timing"))
    p.add_argument("--out-dir",     default=str(REPO_ROOT / "results" / "figures"))
    args = p.parse_args()

    metrics_dir = Path(args.metrics_dir)
    timing_dir  = Path(args.timing_dir)
    out_dir     = Path(args.out_dir)

    # --- load ---
    df_all = load_metrics(metrics_dir)
    loaded_configs = set(df_all["config_id"].tolist())
    print(f"Loaded {len(df_all)} metric files from {metrics_dir}")

    # Identify configs with timing parquets but no metric (still running / incomplete)
    timing_configs = {f.stem for f in timing_dir.glob("*.parquet")}
    missing = sorted(timing_configs - loaded_configs)
    if missing:
        print(f"Skipped {len(missing)} configs with no metric (excluded from all analyses):")
        for m in missing:
            print(f"  {m}")

    # --- analyses ---
    survivors = analysis_filter(df_all)

    ranked = analysis_leaderboard(survivors, missing)

    winner_id = ranked.iloc[0]["config_id"] if len(ranked) else None
    print(f"\nWinner: {winner_id}")

    plot_scatter(df_all, out_dir)
    plot_k_curve(survivors, out_dir)
    plot_tier_profile(ranked, out_dir)

    timing_df = load_timing(timing_dir, loaded_configs)
    analysis_timing(timing_df, winner_id)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
