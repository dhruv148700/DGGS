"""
Threshold tuning for a trained causal ABA model.

Reads model_type and all hyperparams from results.json, sweeps thresholds on the
validation set, picks the best F1 threshold, and saves results back to the same
results directory.

Usage:
    python scripts-causal/run_threshold_tuning.py \
        --results-dir results/results_causal_gcn_scored \
        --val-bin     splits_scored/val.bin
"""

import sys
import os
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dgl
import torch
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from scr.data_utils import set_seeds

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="Directory containing best_model.pt and results.json")
    p.add_argument("--val-bin", required=True,
                   help="Path to validation .bin file (e.g. splits/val.bin)")
    return p.parse_args()


def load_model(model_type, in_features, hidden_dim, embedding_dim, num_layers, dropout, device):
    if model_type == "gcn":
        from scr.GCN_learnable import GCNLearnableModel
        model = GCNLearnableModel(
            in_features=in_features,
            hidden_features=hidden_dim,
            out_features=1,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "gat":
        from scr.GAT_learnable import GATLearnableModel
        model = GATLearnableModel(
            in_features=in_features,
            hidden_features=hidden_dim,
            out_features=1,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def main():
    args = parse_args()
    results_dir = args.results_dir
    val_bin     = args.val_bin

    with open(os.path.join(results_dir, "results.json")) as f:
        meta = json.load(f)

    model_type   = meta["model_type"]
    hp           = meta["hyperparams"]
    in_features  = hp["in_features"]
    hidden_dim   = hp["hidden_dim"]
    embedding_dim= hp["embedding_dim"]
    num_layers   = hp["num_layers"]
    dropout      = hp["dropout"]
    batch_size   = meta.get("batch_size", 64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seeds(42)

    model = load_model(model_type, in_features, hidden_dim, embedding_dim,
                       num_layers, dropout, device)
    model.load_state_dict(
        torch.load(os.path.join(results_dir, "best_model.pt"), map_location=device)
    )
    model = model.to(device)
    model.eval()

    val_graphs, _ = dgl.load_graphs(val_bin)
    val_loader = GraphDataLoader(val_graphs, batch_size=batch_size,
                                 shuffle=False, drop_last=False, num_workers=0)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for g in val_loader:
            g = g.to(device)
            inputs = {
                "assmpt":     g.nodes["assmpt"].data["features"],
                "rule":       g.nodes["rule"].data["features"],
                "non_assmpt": g.nodes["non_assmpt"].data["features"],
            }
            labels  = g.nodes["assmpt"].data["label"]
            logits  = model(g, inputs)["assmpt"].squeeze(1)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_probs  = 1 / (1 + np.exp(-all_logits))

    print(f"Val size: {len(all_labels)}, positive ratio: {all_labels.mean():.4f}")

    thresholds = np.arange(0.0, 1.01, 0.05)
    rows = []

    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
    print("=" * 60)
    for t in thresholds:
        preds = (all_probs > t).astype(int)
        prec  = precision_score(all_labels, preds, zero_division=0)
        rec   = recall_score(all_labels, preds, zero_division=0)
        f1    = f1_score(all_labels, preds, zero_division=0)
        acc   = (preds == all_labels).mean()
        rows.append({"threshold": float(t), "precision": prec, "recall": rec,
                     "f1": f1, "accuracy": acc})
        print(f"{t:<12.3f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {acc:<12.4f}")
    print("=" * 60)

    best_idx = int(np.argmax([r["f1"] for r in rows]))
    best     = rows[best_idx]
    print(f"\nBest threshold: {best['threshold']:.3f}  F1={best['f1']:.4f}  "
          f"Prec={best['precision']:.4f}  Rec={best['recall']:.4f}")

    # PR curve
    fig, ax = plt.subplots(figsize=(8, 6))
    precs = [r["precision"] for r in rows]
    recs  = [r["recall"]    for r in rows]
    f1s   = [r["f1"]        for r in rows]
    thrs  = [r["threshold"] for r in rows]

    ax.plot(recs, precs, "b-o", linewidth=2, markersize=5, label="PR Curve")
    for recall, prec, f1v, thr in zip(recs, precs, f1s, thrs):
        ax.annotate(f"t={thr:.2f}\nF1={f1v:.3f}", xy=(recall, prec),
                    xytext=(4, 4), textcoords="offset points", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5))
    ax.plot(recs[best_idx], precs[best_idx], "r*", markersize=18,
            label=f"Best F1 (t={best['threshold']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Threshold Tuning — {model_type.upper()} ({in_features} features)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "threshold_tuning_pr_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"PR curve saved to {plot_path}")

    out = {
        "threshold_sweep": rows,
        "best_threshold":  best["threshold"],
        "best_f1":         best["f1"],
        "best_precision":  best["precision"],
        "best_recall":     best["recall"],
    }
    out_path = os.path.join(results_dir, "threshold_tuning_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
