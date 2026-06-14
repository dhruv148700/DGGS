"""
analyse_sweep_v2.py — Revised analysis for thesis plots.

Changes from v1:
  - Drops all configs where claim or support aggregation is 'sum'
  - Ranks by Macro AUC (skeptical vs rejected) as primary metric
  - Bar chart: top 5 configs, Macro AUC for all three tier pairs
  - Heatmap: body x claim-agg, split by iota, marginalised over support and k
  - Appendix table: full ranked list of all surviving non-sum configs
  - Text report: human-readable summary saved to sweep_report.txt

Usage
-----
    python analyse_sweep_v2.py
    python analyse_sweep_v2.py --metrics-dir results/metrics \
        --out-dir results/figures_v2
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_config_id(config_id: str) -> dict:
    """Parse 'prod·max·mean·lin·k1.0' into component fields."""
    parts = config_id.split("·")
    return {
        "body":    parts[0],
        "claim":   parts[1],
        "support": parts[2],
        "iota":    parts[3],
        "k":       float(parts[4].replace("k", "")),
    }


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} metric files from {metrics_dir}")
    return df


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# 1. Filter: drop sum, drop low convergence
# ---------------------------------------------------------------------------

def filter_configs(df: pd.DataFrame) -> pd.DataFrame:
    sum_mask = (df["claim"] == "sum") | (df["support"] == "sum")
    n_sum = sum_mask.sum()
    df = df[~sum_mask].copy()

    low_conv = df[df["convergence_rate"] < CONV_THRESHOLD].copy()
    df = df[df["convergence_rate"] >= CONV_THRESHOLD].copy()
    return df, n_sum, low_conv


# ---------------------------------------------------------------------------
# 2. Rank by Macro AUC sk/rej
# ---------------------------------------------------------------------------

def rank_configs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("v2_auc_median_skeptical_vs_rejected", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# 3. Bar chart: top 5 configs, Macro AUC all three tier pairs
# ---------------------------------------------------------------------------

def plot_top5_bar(ranked: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("3. BAR CHART  top 5 configs, Macro AUC all tier pairs")
    print("=" * 65)

    top5 = ranked.head(5).copy()

    def short_label(cfg_id):
        p = cfg_id.split("·")
        body_s = "P" if p[0] == "prod" else "M"
        return f"{body_s}·{p[1]}·{p[2]}·{p[3]}\n{p[4]}"

    labels = [short_label(r["config_id"]) for _, r in top5.iterrows()]

    pairs = [
        ("v2_auc_median_skeptical_vs_rejected",  "Skep. vs Rej.",  "#2166ac"),
        ("v2_auc_median_credulous_vs_rejected",  "Cred. vs Rej.",  "#4dac26"),
        ("v2_auc_median_skeptical_vs_credulous", "Skep. vs Cred.", "#d01c8b"),
    ]

    x = np.arange(len(top5))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (col, label, color) in enumerate(pairs):
        vals = top5[col].values
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Macro AUC (median per-ABAF)", fontsize=10)
    ax.set_title("Tier-pair discrimination — top 5 DGGS configurations\n"
                 "(P = ProductBody, M = MinBody; ranked by Skep. vs Rej.)", fontsize=10)
    # Move legend to upper right to avoid clashing with bars
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0.45, 1.05)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save(fig, out_dir, "top5_tier_bar.pdf")
    save(fig, out_dir, "top5_tier_bar.png")


# ---------------------------------------------------------------------------
# 4. Heatmap: body x claim, split by iota, marginalised over support and k
# ---------------------------------------------------------------------------

def plot_heatmap(ranked: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("4. HEATMAP  body x claim-agg, split by iota")
    print("=" * 65)

    metric = "v2_auc_median_skeptical_vs_rejected"

    grouped = (
        ranked.groupby(["body", "claim", "iota"])[metric]
        .mean()
        .reset_index()
    )

    bodies  = ["prod", "min"]
    claims  = ["max", "mean"]
    iotas   = ["lin", "quad"]

    body_labels  = {"prod": "Product", "min": "Minimum"}
    claim_labels = {"max": "Max", "mean": "Mean"}
    iota_labels  = {"lin": "Linear ($\\iota_{\\mathrm{lin}}$)",
                    "quad": "Quadratic ($\\iota_{\\mathrm{quad}}$)"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    vmin = grouped[metric].min() - 0.01
    vmax = grouped[metric].max() + 0.01

    for ax, iota in zip(axes, iotas):
        sub = grouped[grouped["iota"] == iota]

        mat = np.full((len(bodies), len(claims)), np.nan)
        for i, b in enumerate(bodies):
            for j, c in enumerate(claims):
                row = sub[(sub["body"] == b) & (sub["claim"] == c)]
                if not row.empty:
                    mat[i, j] = row[metric].values[0]

        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="Blues", aspect="auto")

        for i in range(len(bodies)):
            for j in range(len(claims)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if val > (vmin + vmax) / 2 else "black")

        ax.set_xticks(range(len(claims)))
        ax.set_xticklabels([claim_labels[c] for c in claims], fontsize=10)
        ax.set_yticks(range(len(bodies)))
        ax.set_yticklabels([body_labels[b] for b in bodies], fontsize=10)
        ax.set_xlabel("Claim aggregation ($\\alpha_c$)", fontsize=10)
        ax.set_title(iota_labels[iota], fontsize=11)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Mean Macro AUC (sk/rej)" if iota == iotas[-1] else "")

    axes[0].set_ylabel("Body kernel ($\\zeta$)", fontsize=10)
    fig.suptitle("Macro AUC (sk/rej) by kernel choice\n"
                 "(marginalised over support aggregation and $k$)", fontsize=11)
    fig.tight_layout()

    save(fig, out_dir, "heatmap_body_claim_iota.pdf")
    save(fig, out_dir, "heatmap_body_claim_iota.png")


# ---------------------------------------------------------------------------
# 5. Appendix table: full ranked list as CSV
# ---------------------------------------------------------------------------

def export_appendix_table(ranked: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("5. APPENDIX TABLE  (full ranked list → CSV)")
    print("=" * 65)

    cols = [
        "rank", "config_id",
        "v2_auc_median_skeptical_vs_rejected",
        "v2_auc_median_credulous_vs_rejected",
        "v2_auc_median_skeptical_vs_credulous",
        "v1_cohend_skeptical_vs_rejected",
        "convergence_rate",
    ]
    out = ranked[cols].copy()
    out.columns = [
        "Rank", "Config",
        "AUC sk/rej", "AUC cr/rej", "AUC sk/cr",
        "Cohen d", "Conv. rate"
    ]
    path = out_dir / "appendix_full_sweep.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.4f")
    print(f"  saved → {path}")
    print(out.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Text report
# ---------------------------------------------------------------------------

def write_report(df_all: pd.DataFrame, n_sum: int, low_conv: pd.DataFrame,
                 ranked: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append(f"Loaded {len(df_all)} metric files")
    lines.append("")

    lines.append("=" * 65)
    lines.append("1. FILTER")
    lines.append("=" * 65)
    lines.append(f"Loaded configs:    {len(df_all)}")
    lines.append(f"Dropped (sum agg): {n_sum}")
    if len(low_conv):
        lines.append(f"Dropped (conv < {CONV_THRESHOLD}):")
        for _, r in low_conv.sort_values("convergence_rate").iterrows():
            lines.append(f"  {r['config_id']:<40}  conv={r['convergence_rate']:.3f}")
    lines.append(f"Survivors:         {len(ranked)}")
    lines.append("")

    lines.append("=" * 65)
    lines.append("2. LEADERBOARD  (ranked by Macro AUC sk/rej)")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"{'Rk':<4} {'Config':<40} {'sk/rej':>8} {'cr/rej':>8} {'sk/cr':>8} {'Conv':>6}")
    lines.append("-" * 80)
    for _, r in ranked.iterrows():
        lines.append(
            f"{int(r['rank']):<4} {r['config_id']:<40} "
            f"{r['v2_auc_median_skeptical_vs_rejected']:>8.4f} "
            f"{r['v2_auc_median_credulous_vs_rejected']:>8.4f} "
            f"{r['v2_auc_median_skeptical_vs_credulous']:>8.4f} "
            f"{r['convergence_rate']:>6.3f}"
        )
    lines.append("")
    lines.append(f"Winner: {ranked.iloc[0]['config_id']}")
    lines.append("")

    lines.append("=" * 65)
    lines.append("3. TOP 5 CONFIGS")
    lines.append("=" * 65)
    for _, r in ranked.head(5).iterrows():
        lines.append(f"  {int(r['rank'])}. {r['config_id']}")
        lines.append(f"     Macro AUC  sk/rej={r['v2_auc_median_skeptical_vs_rejected']:.4f}  "
                     f"cr/rej={r['v2_auc_median_credulous_vs_rejected']:.4f}  "
                     f"sk/cr={r['v2_auc_median_skeptical_vs_credulous']:.4f}")
        lines.append(f"     Conv={r['convergence_rate']:.3f}")
        lines.append("")

    text = "\n".join(lines)
    path = out_dir / "sweep_report.txt"
    path.write_text(text)
    print(f"\n  saved → {path}")
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", default="results/metrics")
    p.add_argument("--out-dir",     default="results/figures_v2")
    args = p.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir     = Path(args.out_dir)

    df_all             = load_metrics(metrics_dir)
    filtered, n_sum, low_conv = filter_configs(df_all)
    ranked             = rank_configs(filtered)

    plot_top5_bar(ranked, out_dir)
    plot_heatmap(ranked, out_dir)
    export_appendix_table(ranked, out_dir)
    write_report(df_all, n_sum, low_conv, ranked, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()