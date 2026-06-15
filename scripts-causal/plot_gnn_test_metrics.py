"""
plot_gnn_test_metrics.py
────────────────────────
Grouped bar chart of F1 / Precision / Recall / Accuracy for the 4 trained
causal GNN models on the synthetic test set.

Data source: test.overall section of each model's results.json.

Usage:
    python scripts-causal/plot_gnn_test_metrics.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("results/results_causal_gcn",        "ABA-GCN"),
    ("results/results_causal_gat",        "ABA-GAT"),
    ("results/results_causal_gcn_scored", "ABA-GCN+"),
    ("results/results_causal_gat_scored", "ABA-GAT+"),
]

METRICS     = ["F1-Score", "Precision", "Recall", "Accuracy"]
METRIC_KEYS = ["f1",       "prec",      "rec",    "acc"]
COLORS      = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7"]

OUT_DIR = REPO_ROOT / "results" / "causal_gnn_metrics"


def load_test_overall(results_dir: str) -> dict:
    path = REPO_ROOT / results_dir / "results.json"
    with open(path) as f:
        data = json.load(f)
    return data["test"]["overall"]


def main():
    values = []
    labels = []
    for results_dir, label in MODELS:
        overall = load_test_overall(results_dir)
        row = [overall[k] for k in METRIC_KEYS]
        values.append(row)
        labels.append(label)

    values = np.array(values)  # shape (4 models, 4 metrics)

    fontsize = 16
    n_models  = len(MODELS)
    n_metrics = len(METRICS)
    width     = 0.18
    x         = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (label, color) in enumerate(zip(labels, COLORS)):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, values[i], width=width * 0.9,
               label=label, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=fontsize)
    ax.set_ylabel("Score", fontsize=fontsize)
    ax.set_title("GNN Models — Test Set Performance (Causal ABA)", fontsize=fontsize)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=fontsize - 2, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.yaxis.set_tick_params(labelsize=fontsize - 2)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    for i in range(n_models):
        for j in range(n_metrics):
            offset = (i - (n_models - 1) / 2) * width
            ax.text(x[j] + offset, values[i, j] + 0.012,
                    f"{values[i, j]:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "gnn_test_metrics.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Also print table
    print(f"\n{'Model':<16} " + "  ".join(f"{m:>10}" for m in METRICS))
    print("─" * (16 + 14 * n_metrics))
    for label, row in zip(labels, values):
        print(f"{label:<16} " + "  ".join(f"{v:>10.4f}" for v in row))


if __name__ == "__main__":
    main()
