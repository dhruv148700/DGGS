"""
analyse_baselines.py — Head-to-head comparison of DGGS vs BSAF.

Loads top 2 DGGS configs (from full sweep, excluding sum) and the
2 surviving BSAF configs (convergence >= 90%), then plots a single
grouped bar chart comparing Macro AUC across all three tier pairs.

Usage
-----
    python analyse_baselines.py
    python analyse_baselines.py \
        --dggs-metrics-dir results/metrics \
        --bsaf-metrics-dir baselines/results/metrics \
        --out-dir baselines/results/figures_v2
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError as exc:
    print(f"ERROR: missing dependency — {exc}")
    sys.exit(1)

CONV_THRESHOLD = 0.9
METRIC         = "v2_auc_median_skeptical_vs_rejected"

PAIRS = [
    ("v2_auc_median_skeptical_vs_rejected",  "Skep. vs Rej.",  "#2166ac"),
    ("v2_auc_median_credulous_vs_rejected",  "Cred. vs Rej.",  "#4dac26"),
    ("v2_auc_median_skeptical_vs_credulous", "Skep. vs Cred.", "#d01c8b"),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_dggs(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} DGGS metric files from {metrics_dir}")

    # Drop sum aggregation
    sum_mask = (df["claim"] == "sum") | (df["support"] == "sum")
    df = df[~sum_mask].copy()

    # Drop low convergence
    df = df[df["convergence_rate"] >= CONV_THRESHOLD].copy()

    # Rank and take top 2
    df = df.sort_values(METRIC, ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["method"] = "DGGS"
    print(f"  After filtering: {len(df)} survivors — taking top 2")
    return df.head(2)


def load_bsaf(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} BSAF metric files from {metrics_dir}")

    # Filter convergence
    dropped = df[df["convergence_rate"] < CONV_THRESHOLD]
    if len(dropped):
        print(f"  Dropped ({len(dropped)} configs below {CONV_THRESHOLD} convergence):")
        for _, r in dropped.sort_values("convergence_rate").iterrows():
            print(f"    {r['config_id']:<40}  conv={r['convergence_rate']:.3f}")

    df = df[df["convergence_rate"] >= CONV_THRESHOLD].copy()
    df = df.sort_values(METRIC, ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["method"] = "BSAF"
    print(f"  After filtering: {len(df)} survivors")
    return df


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Plot: head-to-head grouped bar chart
# ---------------------------------------------------------------------------

def plot_head_to_head(dggs: pd.DataFrame, bsaf: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("PLOT  DGGS vs BSAF head-to-head")
    print("=" * 65)

    # Build ordered list: DGGS top2, then BSAF top2
    configs = pd.concat([dggs, bsaf], ignore_index=True)

    def short_label(row):
        cid = row["config_id"]
        if row["method"] == "DGGS":
            # e.g. prod·max·max·lin·k1.0
            p = cid.split("·")
            body_s = "P" if p[0] == "prod" else "M"
            return f"DGGS\n{body_s}·{p[1]}·{p[2]}·{p[3]}·{p[4]}"
        else:
            # e.g. bsaf_dfquad_product_random
            parts = cid.replace("bsaf_", "").split("_")
            return f"BSAF\n{'·'.join(parts)}"

    labels = [short_label(r) for _, r in configs.iterrows()]

    x = np.arange(len(configs))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (col, label, color) in enumerate(PAIRS):
        vals = configs[col].values
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Visual separator between DGGS and BSAF groups
    separator_x = 1.5 + width  # between config index 1 and 2
    ax.axvline(separator_x, color="black", linestyle="--",
               linewidth=1.2, alpha=0.5)
    ax.text(separator_x - 0.85, 1.02, "DGGS", transform=ax.get_xaxis_transform(),
            fontsize=11, fontweight="bold", ha="center", color="#2166ac")
    ax.text(separator_x + 0.85, 1.02, "BSAF", transform=ax.get_xaxis_transform(),
            fontsize=11, fontweight="bold", ha="center", color="#b2182b")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro AUC (median per-ABAF)", fontsize=11)
    ax.set_title("Tier-pair discrimination: DGGS vs BSAF", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0.45, 1.10)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save(fig, out_dir, "dggs_vs_bsaf_head_to_head.pdf")
    save(fig, out_dir, "dggs_vs_bsaf_head_to_head.png")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(dggs: pd.DataFrame, bsaf: pd.DataFrame, out_dir: Path) -> None:
    lines = []

    lines.append("=" * 65)
    lines.append("DGGS TOP 2  (ranked by Macro AUC sk/rej, sum excluded)")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"{'Rk':<4} {'Config':<45} {'sk/rej':>8} {'cr/rej':>8} {'sk/cr':>8} {'Conv':>6}")
    lines.append("-" * 85)
    for _, r in dggs.iterrows():
        lines.append(
            f"{int(r['rank']):<4} {r['config_id']:<45} "
            f"{r['v2_auc_median_skeptical_vs_rejected']:>8.4f} "
            f"{r['v2_auc_median_credulous_vs_rejected']:>8.4f} "
            f"{r['v2_auc_median_skeptical_vs_credulous']:>8.4f} "
            f"{r['convergence_rate']:>6.3f}"
        )
    lines.append("")

    lines.append("=" * 65)
    lines.append("BSAF SURVIVORS  (convergence >= 90%)")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"{'Rk':<4} {'Config':<45} {'sk/rej':>8} {'cr/rej':>8} {'sk/cr':>8} {'Conv':>6}")
    lines.append("-" * 85)
    for _, r in bsaf.iterrows():
        lines.append(
            f"{int(r['rank']):<4} {r['config_id']:<45} "
            f"{r['v2_auc_median_skeptical_vs_rejected']:>8.4f} "
            f"{r['v2_auc_median_credulous_vs_rejected']:>8.4f} "
            f"{r['v2_auc_median_skeptical_vs_credulous']:>8.4f} "
            f"{r['convergence_rate']:>6.3f}"
        )
    lines.append("")

    lines.append("=" * 65)
    lines.append("HEAD-TO-HEAD SUMMARY")
    lines.append("=" * 65)
    lines.append("")
    dggs_best = dggs.iloc[0]
    bsaf_best = bsaf.iloc[0]
    for col, label, _ in PAIRS:
        dv = dggs_best[col]
        bv = bsaf_best[col]
        diff = dv - bv
        lines.append(f"{label}:")
        lines.append(f"  DGGS best: {dv:.4f}  ({dggs_best['config_id']})")
        lines.append(f"  BSAF best: {bv:.4f}  ({bsaf_best['config_id']})")
        lines.append(f"  Delta:     {diff:+.4f}")
        lines.append("")

    text = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline_report.txt"
    path.write_text(text)
    print(f"\n  saved → {path}")
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dggs-metrics-dir", default="results/metrics")
    p.add_argument("--bsaf-metrics-dir", default="baselines/results/metrics")
    p.add_argument("--out-dir",          default="baselines/results/figures_v2")
    args = p.parse_args()

    dggs_dir = Path(args.dggs_metrics_dir)
    bsaf_dir = Path(args.bsaf_metrics_dir)
    out_dir  = Path(args.out_dir)

    dggs = load_dggs(dggs_dir)
    bsaf = load_bsaf(bsaf_dir)

    plot_head_to_head(dggs, bsaf, out_dir)
    write_report(dggs, bsaf, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()