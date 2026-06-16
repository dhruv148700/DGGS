"""
plot_gnn_tier_auc.py
────────────────────
Grouped bar chart of V2 median per-ABAF AUC for the three tier pairs,
comparing the 4 causal GNN models against the best DGGS configuration
(prod·max·max·lin·k1.0·neutral·neutral).

Data sources:
  - GNN:  results/causal_gnn_metrics/results_causal_{gcn,gat,gcn_scored,gat_scored}.json
  - DGGS: results/ablations/metrics/prod·max·max·lin·k1.0·neutral·neutral.json

Output: results/causal_gnn_metrics/gnn_tier_auc.pdf
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent.parent
GNN_DIR   = REPO_ROOT / "results" / "causal_gnn_metrics"
DGGS_FILE = REPO_ROOT / "results" / "ablations" / "metrics" / "prod·max·max·lin·k1.0·neutral·neutral.json"
OUT_PATH  = GNN_DIR / "gnn_tier_auc.pdf"

MODELS = [
    ("ABA-GCN",  GNN_DIR / "results_causal_gcn.json",        "#00b4d8"),
    ("ABA-GAT",  GNN_DIR / "results_causal_gat.json",        "#f48c06"),
    ("ABA-GCN+", GNN_DIR / "results_causal_gcn_scored.json", "#5c4033"),
    ("ABA-GAT+", GNN_DIR / "results_causal_gat_scored.json", "#70b244"),
    ("DGGS",     DGGS_FILE,                                   "#e377c2"),
]

PAIRS = [
    ("v2_auc_median_skeptical_vs_rejected",  "Skep. vs Rej."),
    ("v2_auc_median_credulous_vs_rejected",  "Cred. vs Rej."),
    ("v2_auc_median_skeptical_vs_credulous", "Skep. vs Cred."),
]

FONTSIZE = 14


def main():
    rows = []
    for label, path, color in MODELS:
        if not Path(path).exists():
            print(f"WARNING: {path} not found — skipping {label}")
            continue
        with open(path) as f:
            d = json.load(f)
        vals = [d[col] for col, _ in PAIRS]
        rows.append((label, color, vals))

    n_models = len(rows)
    n_pairs  = len(PAIRS)
    width    = 0.14
    x        = np.arange(n_pairs)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, color, vals) in enumerate(rows):
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width * 0.9,
            label=label, color=color, alpha=0.85,
            edgecolor="black", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in PAIRS], fontsize=FONTSIZE)
    ax.set_ylabel("Median per-ABAF AUC (V2)", fontsize=FONTSIZE)
    ax.set_ylim(0.45, 1.15)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=FONTSIZE - 2, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    GNN_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")

    print(f"\n{'Model':<12}  " + "  ".join(f"{lbl:>18}" for _, lbl in PAIRS))
    print("─" * (12 + 3 * 20))
    for label, _, vals in rows:
        print(f"{label:<12}  " + "  ".join(f"{v:>18.4f}" for v in vals))


if __name__ == "__main__":
    main()
