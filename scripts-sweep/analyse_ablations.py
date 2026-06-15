"""
analyse_ablations.py — Analysis of score-initialisation ablations.

Fixed kernel: prod·max·max·lin·k1.0
Ablation grid: tau_a_init x tau_r_init in {neutral, ci, random} (3x3 = 9 configs)

Plots:
  1. Ranked bar chart — all 9 configs, Macro AUC for all three tier pairs
  2. Heatmap — tau_a x tau_r, Macro AUC sk/rej
  3. Marginal effect — tau_a effect and tau_r effect side by side

Also saves:
  - sweep_report.txt — human-readable summary

Usage
-----
    python analyse_ablations.py
    python analyse_ablations.py --metrics-dir results/ablations/metrics \
        --out-dir results/ablations/figures_v2
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


INIT_ORDER  = ["neutral", "ci", "random"]
INIT_LABELS = {"neutral": "Neutral (1.0)", "ci": "CI-scores", "random": "Random"}
METRIC      = "v2_auc_median_skeptical_vs_rejected"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_config_id(config_id: str) -> dict:
    """Parse 'prod·max·max·lin·k1.0·ci·random' into fields."""
    parts = config_id.split("·")
    return {
        "body":        parts[0],
        "claim":       parts[1],
        "support":     parts[2],
        "iota":        parts[3],
        "k":           float(parts[4].replace("k", "")),
        "tau_a_init":  parts[5],
        "tau_r_init":  parts[6],
    }


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        parsed = parse_config_id(d["config_id"])
        # Merge parsed fields (overwrite placeholder ?s from JSON)
        d.update(parsed)
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
# 1. Ranked bar chart: all 9 configs, Macro AUC all three tier pairs
# ---------------------------------------------------------------------------

def plot_ranked_bar(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("1. RANKED BAR CHART  all 9 configs, Macro AUC all tier pairs")
    print("=" * 65)

    ranked = df.sort_values(METRIC, ascending=False).reset_index(drop=True)

    def short_label(row):
        ta = row["tau_a_init"]
        tr = row["tau_r_init"]
        return f"τ_a={ta}\nτ_r={tr}"

    labels = [short_label(r) for _, r in ranked.iterrows()]

    pairs = [
        ("v2_auc_median_skeptical_vs_rejected",  "Skep. vs Rej.",  "#2166ac"),
        ("v2_auc_median_credulous_vs_rejected",  "Cred. vs Rej.",  "#4dac26"),
        ("v2_auc_median_skeptical_vs_credulous", "Skep. vs Cred.", "#d01c8b"),
    ]

    x = np.arange(len(ranked))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (col, label, color) in enumerate(pairs):
        vals = ranked[col].values
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro AUC (median per-ABAF)", fontsize=11)
    ax.set_title("Score-initialisation ablation — all 9 configurations\n"
                 "(fixed kernel: prod·max·max·lin·k=1.0; ranked by Skep. vs Rej.)", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0.45, 1.08)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save(fig, out_dir, "ablation_ranked_bar.pdf")
    save(fig, out_dir, "ablation_ranked_bar.png")


# ---------------------------------------------------------------------------
# 2. Heatmap: tau_a x tau_r, Macro AUC sk/rej
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("2. HEATMAP  tau_a x tau_r, Macro AUC sk/rej")
    print("=" * 65)

    mat = np.full((len(INIT_ORDER), len(INIT_ORDER)), np.nan)
    for i, ta in enumerate(INIT_ORDER):
        for j, tr in enumerate(INIT_ORDER):
            row = df[(df["tau_a_init"] == ta) & (df["tau_r_init"] == tr)]
            if not row.empty:
                mat[i, j] = row[METRIC].values[0]

    vmin = np.nanmin(mat) - 0.01
    vmax = np.nanmax(mat) + 0.01

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="Blues", aspect="auto")

    for i in range(len(INIT_ORDER)):
        for j in range(len(INIT_ORDER)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if val > (vmin + vmax) / 2 else "black")

    ax.set_xticks(range(len(INIT_ORDER)))
    ax.set_xticklabels([INIT_LABELS[t] for t in INIT_ORDER], fontsize=11)
    ax.set_yticks(range(len(INIT_ORDER)))
    ax.set_yticklabels([INIT_LABELS[t] for t in INIT_ORDER], fontsize=11)
    ax.set_xlabel("Rule score init ($\\tau_r$)", fontsize=12)
    ax.set_ylabel("Base score init ($\\tau_a$)", fontsize=12)
    ax.set_title("Macro AUC (sk/rej) by score initialisation\n"
                 "(fixed kernel: prod·max·max·lin·k=1.0)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro AUC (sk/rej)")
    fig.tight_layout()

    save(fig, out_dir, "ablation_heatmap.pdf")
    save(fig, out_dir, "ablation_heatmap.png")


# ---------------------------------------------------------------------------
# 3. Marginal effect: tau_a and tau_r side by side
# ---------------------------------------------------------------------------

def plot_marginal(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("3. MARGINAL EFFECT  tau_a and tau_r")
    print("=" * 65)

    # Marginalise over tau_r -> tau_a effect
    tau_a_mean = (
        df.groupby("tau_a_init")[METRIC].mean()
        .reindex(INIT_ORDER)
    )
    # Marginalise over tau_a -> tau_r effect
    tau_r_mean = (
        df.groupby("tau_r_init")[METRIC].mean()
        .reindex(INIT_ORDER)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    colors = ["#2166ac", "#4dac26", "#d01c8b"]

    for ax, (data, title, xlabel) in zip(axes, [
        (tau_a_mean, "Effect of $\\tau_a$ initialisation\n(marginalised over $\\tau_r$)", "Base score init ($\\tau_a$)"),
        (tau_r_mean, "Effect of $\\tau_r$ initialisation\n(marginalised over $\\tau_a$)", "Rule score init ($\\tau_r$)"),
    ]):
        bars = ax.bar(range(len(INIT_ORDER)),
                      data.values,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(INIT_ORDER)))
        ax.set_xticklabels([INIT_LABELS[t] for t in INIT_ORDER], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylim(0.45, 1.05)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean Macro AUC (sk/rej)", fontsize=11)
    fig.suptitle("Marginal effect of score initialisation on tier separation",
                 fontsize=12)
    fig.tight_layout()

    save(fig, out_dir, "ablation_marginal.pdf")
    save(fig, out_dir, "ablation_marginal.png")


# ---------------------------------------------------------------------------
# 4. Text report
# ---------------------------------------------------------------------------

def write_report(df: pd.DataFrame, out_dir: Path) -> None:
    ranked = df.sort_values(METRIC, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    tau_a_mean = df.groupby("tau_a_init")[METRIC].mean().reindex(INIT_ORDER)
    tau_r_mean = df.groupby("tau_r_init")[METRIC].mean().reindex(INIT_ORDER)

    lines = []
    lines.append(f"Loaded {len(df)} ablation configs")
    lines.append(f"Fixed kernel: prod·max·max·lin·k1.0")
    lines.append(f"Ablation: tau_a_init x tau_r_init in {{neutral, ci, random}}")
    lines.append("")

    lines.append("=" * 65)
    lines.append("1. LEADERBOARD  (ranked by Macro AUC sk/rej)")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"{'Rk':<4} {'Config':<45} {'sk/rej':>8} {'cr/rej':>8} {'sk/cr':>8} {'Conv':>6}")
    lines.append("-" * 85)
    for _, r in ranked.iterrows():
        lines.append(
            f"{int(r['rank']):<4} {r['config_id']:<45} "
            f"{r['v2_auc_median_skeptical_vs_rejected']:>8.4f} "
            f"{r['v2_auc_median_credulous_vs_rejected']:>8.4f} "
            f"{r['v2_auc_median_skeptical_vs_credulous']:>8.4f} "
            f"{r['convergence_rate']:>6.3f}"
        )
    lines.append("")
    lines.append(f"Winner: {ranked.iloc[0]['config_id']}")
    lines.append("")

    lines.append("=" * 65)
    lines.append("2. MARGINAL EFFECTS")
    lines.append("=" * 65)
    lines.append("")
    lines.append("tau_a effect (marginalised over tau_r):")
    for ta, val in tau_a_mean.items():
        lines.append(f"  tau_a={ta:<10}  Macro AUC={val:.4f}")
    lines.append("")
    lines.append("tau_r effect (marginalised over tau_a):")
    for tr, val in tau_r_mean.items():
        lines.append(f"  tau_r={tr:<10}  Macro AUC={val:.4f}")
    lines.append("")

    lines.append("=" * 65)
    lines.append("3. HEATMAP VALUES  (tau_a rows x tau_r cols, Macro AUC sk/rej)")
    lines.append("=" * 65)
    lines.append("")
    corner = 'tau_a / tau_r'
    header = f"{corner:<16}" + "".join(f"{INIT_LABELS[t]:>18}" for t in INIT_ORDER)
    lines.append(header)
    lines.append("-" * (16 + 18 * len(INIT_ORDER)))
    for ta in INIT_ORDER:
        row_vals = []
        for tr in INIT_ORDER:
            r = df[(df["tau_a_init"] == ta) & (df["tau_r_init"] == tr)]
            row_vals.append(f"{r[METRIC].values[0]:.4f}" if not r.empty else "N/A")
        lines.append(f"{INIT_LABELS[ta]:<16}" + "".join(f"{v:>18}" for v in row_vals))
    lines.append("")

    text = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_report.txt"
    path.write_text(text)
    print(f"\n  saved → {path}")
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", default="results/ablations/metrics")
    p.add_argument("--out-dir",     default="results/ablations/figures_v2")
    args = p.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir     = Path(args.out_dir)

    df = load_metrics(metrics_dir)

    plot_ranked_bar(df, out_dir)
    plot_heatmap(df, out_dir)
    plot_marginal(df, out_dir)
    write_report(df, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()