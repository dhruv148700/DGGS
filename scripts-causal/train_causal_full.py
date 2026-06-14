"""
train_causal_full.py
────────────────────
Train GCN or GAT on the full stratified causal ABA dataset produced by
build_causal_splits_dataset.py + merge_graph_dataset.py.

Unlike train_causal.py (which used 3-fold cross-validation on a mini dataset),
this script uses the pre-computed train/val/test splits:
  - train  → gradient updates
  - val    → early stopping / model selection
  - test   → final held-out evaluation (reported once, at the end)

Results are broken down by n_nodes, probe_role, and graph_type so the thesis
results section can be populated directly from output.txt / results.json.

── USAGE ────────────────────────────────────────────────────────────────────

  python scripts-causal/train_causal_full.py
  python scripts-causal/train_causal_full.py --model-type gat
  python scripts-causal/train_causal_full.py --model-type gcn --batch-size 64
  python scripts-causal/train_causal_full.py --data-dir splits/ --out-dir results_full_gcn/

── BATCH SIZE GUIDANCE ──────────────────────────────────────────────────────

  The heterograph fed to the GNN has three node types (assmpt, non_assmpt,
  rule).  Total nodes per graph ≈ n_atoms + n_rules, so n_atoms is the right
  memory proxy (not n_assumptions, which is only one node type).

  Dataset n_atoms distribution (train):
    p50=915   p90=3,825   p99=6,388   max=7,000  (after 7k filter)

  Approximate total nodes in a batch (n_atoms proxy, excluding rule nodes):
    batch_size=32: p90 ~122K nodes  worst-case ~224K nodes
    batch_size=64: p90 ~245K nodes  worst-case ~448K nodes

  Safe defaults:
    GCN (hidden_dim=32) on A40 (48 GB) : --batch-size 64   (default)
    GCN (hidden_dim=32) on A30 (24 GB) : --batch-size 32
    GAT (hidden_dim=64) on A40 (48 GB) : --batch-size 32   (default)
    GAT (hidden_dim=64) on A30 (24 GB) : --batch-size 16
  Reduce if you see CUDA out-of-memory errors.
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import defaultdict

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scr"))

from scr.data_utils import set_seeds
from scr.GCN_learnable import GCNLearnableModel
from scr.GAT_learnable import GATLearnableModel

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─── Device ───────────────────────────────────────────────────────────────────

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", device)

# ─── Optimised hyperparameters (carried over from train_causal.py) ────────────

DEFAULTS = {
    "gcn": dict(
        patience=10,
        batch_size=64,        # safe on A40 (48GB); use 32 on A30 (24GB)
        pos_weight=0.94,
        threshold=0.4998769870504005,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=10,
        dropout=0.02943105695360959,
        lr=0.008622415975216019,
    ),
    "gat": dict(
        patience=30,
        batch_size=32,        # GAT hidden_dim=64 doubles memory vs GCN; use 16 on A30
        pos_weight=0.94,
        threshold=0.4578899746620344,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=10,
        dropout=0.2198191427741004,
        lr=0.006584746811018268,
    ),
}

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_split(data_dir: str, split: str) -> tuple:
    """Load a merged split's graphs and parallel metadata list.

    Returns (graphs, metadata_list) where metadata_list[i] is the manifest
    dict for graphs[i], carrying n_nodes, graph_type, probe_role, etc.
    """
    bin_path  = os.path.join(data_dir, f"{split}.bin")
    meta_path = os.path.join(data_dir, f"{split}_metadata.json")

    graphs, _ = dgl.load_graphs(bin_path)
    graphs = list(graphs)
    with open(meta_path) as fh:
        metadata = json.load(fh)

    assert len(graphs) == len(metadata), (
        f"Alignment error for {split}: {len(graphs)} graphs vs {len(metadata)} metadata entries"
    )
    log.info("Loaded %s: %d graphs", split, len(graphs))
    return graphs, metadata

# ─── Model construction ───────────────────────────────────────────────────────

def build_model(model_type: str, cfg: dict):
    in_features = cfg.get("in_features", 2)
    if model_type == "gcn":
        return GCNLearnableModel(
            in_features=in_features,
            hidden_features=cfg["hidden_dim"],
            out_features=1,
            embedding_dim=cfg["embedding_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    return GATLearnableModel(
        in_features=in_features,
        hidden_features=cfg["hidden_dim"],
        out_features=1,
        embedding_dim=cfg["embedding_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )

# ─── Evaluation ───────────────────────────────────────────────────────────────

def _metrics(labels, preds):
    """Compute accuracy, F1, precision, recall from flat numpy arrays."""
    acc  = (preds == labels).mean()
    f1   = f1_score(labels,   preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels,  preds, zero_division=0)
    return acc, f1, prec, rec


def evaluate(loader, model, threshold: float, criterion=None) -> tuple:
    """Fast evaluation returning aggregate metrics only.

    Used during training for val-loss monitoring and early stopping.
    Returns (avg_loss, accuracy, f1, precision, recall).
    avg_loss is None when criterion is None.
    """
    model.eval()
    total_loss, total_nodes = 0.0, 0
    all_preds, all_labels   = [], []

    with torch.no_grad():
        for g in loader:
            g      = g.to(device)
            inputs = {
                "assmpt":     g.nodes["assmpt"].data["features"],
                "rule":       g.nodes["rule"].data["features"],
                "non_assmpt": g.nodes["non_assmpt"].data["features"],
            }
            labels = g.nodes["assmpt"].data["label"]
            logits = model(g, inputs)["assmpt"].squeeze(1)

            if criterion is not None:
                total_loss += (criterion(logits, labels.float()) * labels.size(0)).item()

            preds = (torch.sigmoid(logits) > threshold).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_nodes += labels.size(0)

    avg_loss        = total_loss / total_nodes if criterion is not None else None
    all_preds       = np.array(all_preds)
    all_labels      = np.array(all_labels)
    acc, f1, prec, rec = _metrics(all_labels, all_preds)
    return avg_loss, acc, f1, prec, rec


def evaluate_with_breakdown(
    graphs: list,
    metadata: list,
    model,
    threshold: float,
    cfg: dict,
    batch_size: int,
) -> dict:
    """Full evaluation with per-group breakdown for the final test report.

    Runs inference on all graphs, then slices predictions by:
        n_nodes    : {3, 4, 5, 6}
        probe_role : {initial_full, easy_sat, boundary_sat, boundary_unsat}
        graph_type : {er, ba}

    Returns a nested dict:
        {
          'overall':      {'acc', 'f1', 'prec', 'rec', 'n_nodes', 'n_preds'},
          'by_n_nodes':   {3: {...}, 4: {...}, ...},
          'by_probe_role':{...},
          'by_graph_type':{...},
        }
    All metric dicts contain: acc, f1, prec, rec, n_graphs, n_assumptions.
    """
    # Non-shuffled loader so graph order matches metadata order
    loader = GraphDataLoader(graphs, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=0)

    model.eval()
    # Collect per-graph predictions by splitting batched logits by node count
    graph_preds:  list = []   # [np.array of preds per graph]
    graph_labels: list = []   # [np.array of labels per graph]

    graph_idx = 0
    with torch.no_grad():
        for batched_g in loader:
            batched_g = batched_g.to(device)
            inputs = {
                "assmpt":     batched_g.nodes["assmpt"].data["features"],
                "rule":       batched_g.nodes["rule"].data["features"],
                "non_assmpt": batched_g.nodes["non_assmpt"].data["features"],
            }
            labels_batch = batched_g.nodes["assmpt"].data["label"]
            logits_batch = model(batched_g, inputs)["assmpt"].squeeze(1)
            preds_batch  = (torch.sigmoid(logits_batch) > threshold).long()

            # Split back to individual graphs using unbatch
            individual = dgl.unbatch(batched_g)
            cursor = 0
            for g in individual:
                n = g.num_nodes("assmpt")
                graph_preds.append(preds_batch[cursor:cursor + n].cpu().numpy())
                graph_labels.append(labels_batch[cursor:cursor + n].cpu().numpy())
                cursor += n
            graph_idx += len(individual)

    def _group_metrics(indices):
        if not indices:
            return None
        p = np.concatenate([graph_preds[i]  for i in indices])
        l = np.concatenate([graph_labels[i] for i in indices])
        acc, f1, prec, rec = _metrics(l, p)
        return {
            "acc": round(float(acc),  4),
            "f1":  round(float(f1),   4),
            "prec": round(float(prec), 4),
            "rec":  round(float(rec),  4),
            "n_graphs":      len(indices),
            "n_assumptions": int(len(p)),
        }

    all_idx = list(range(len(metadata)))

    by_n_nodes    = {}
    by_probe_role = {}
    by_graph_type = {}

    for i, m in enumerate(metadata):
        by_n_nodes.setdefault(m["n_nodes"],    []).append(i)
        by_probe_role.setdefault(m["probe_role"], []).append(i)
        by_graph_type.setdefault(m["graph_type"], []).append(i)

    return {
        "overall":       _group_metrics(all_idx),
        "by_n_nodes":    {k: _group_metrics(v) for k, v in sorted(by_n_nodes.items())},
        "by_probe_role": {k: _group_metrics(v) for k, v in sorted(by_probe_role.items())},
        "by_graph_type": {k: _group_metrics(v) for k, v in sorted(by_graph_type.items())},
    }

# ─── Breakdown table formatter ────────────────────────────────────────────────

def print_breakdown(results: dict, split_name: str, log_path: str):
    """Print a formatted breakdown table to stdout and append to log_path."""
    lines = [
        "",
        f"{'='*65}",
        f"TEST RESULTS — {split_name.upper()}",
        f"{'='*65}",
    ]

    def row(label, m):
        if m is None:
            return f"  {label:<30}  (no data)"
        return (
            f"  {label:<30}  F1={m['f1']:.4f}  acc={m['acc']:.4f}  "
            f"prec={m['prec']:.4f}  rec={m['rec']:.4f}  "
            f"(n_graphs={m['n_graphs']})"
        )

    lines.append(row("overall", results["overall"]))
    lines.append("")
    lines.append("  by n_nodes:")
    for k, m in results["by_n_nodes"].items():
        lines.append(row(f"    n_nodes={k}", m))
    lines.append("")
    lines.append("  by probe_role:")
    for k, m in results["by_probe_role"].items():
        lines.append(row(f"    {k}", m))
    lines.append("")
    lines.append("  by graph_type:")
    for k, m in results["by_graph_type"].items():
        lines.append(row(f"    {k}", m))
    lines.append(f"{'='*65}")

    block = "\n".join(lines)
    print(block)
    with open(log_path, "a") as fh:
        fh.write(block + "\n")

# ─── Training loop ────────────────────────────────────────────────────────────

def train(
    train_graphs, train_meta,
    val_graphs,
    model_type: str,
    cfg: dict,
    epochs: int,
    out_dir: str,
    batch_size: int,
):
    """Train with early stopping on val loss.  Returns the best model."""
    model     = build_model(model_type, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.75, patience=5, verbose=True, min_lr=1e-6
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg["pos_weight"]]).to(device)
    )

    train_loader = GraphDataLoader(
        train_graphs, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0
    )
    val_loader = GraphDataLoader(
        val_graphs, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0
    )

    best_val_loss    = float("inf")
    best_state       = None
    patience_counter = 0
    best_epoch       = 0
    log_path         = os.path.join(out_dir, "output.txt")

    log.info(
        "Training: %d graphs  |  val: %d graphs  |  batch_size=%d  |  patience=%d",
        len(train_graphs), len(val_graphs), batch_size, cfg["patience"],
    )

    for epoch in range(epochs):
        # ── Training step ─────────────────────────────────────────────────────
        model.train()
        train_loss, total_nodes = 0.0, 0

        for g in train_loader:
            g = g.to(device)
            inputs = {
                "assmpt":     g.nodes["assmpt"].data["features"],
                "rule":       g.nodes["rule"].data["features"],
                "non_assmpt": g.nodes["non_assmpt"].data["features"],
            }
            labels = g.nodes["assmpt"].data["label"]
            logits = model(g, inputs)["assmpt"].squeeze(1)
            loss   = loss_fn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss  += loss.item() * labels.size(0)
            total_nodes += labels.size(0)

        avg_train = train_loss / total_nodes

        # ── Validation step ───────────────────────────────────────────────────
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(
            val_loader, model, cfg["threshold"], criterion=loss_fn
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        line = (
            f"Epoch {epoch:4d}: train_loss={avg_train:.4f}  "
            f"val_loss={val_loss:.4f}  f1={val_f1:.4f}  acc={val_acc:.4f}  "
            f"prec={val_prec:.4f}  rec={val_rec:.4f}  lr={lr:.6f}  "
            f"patience={patience_counter}"
        )
        log.info(line)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch       = epoch
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            msg = (
                f"Early stopping at epoch {epoch}  "
                f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})"
            )
            log.info(msg)
            with open(log_path, "a") as fh:
                fh.write(msg + "\n")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_val_loss

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", choices=["gcn", "gat"], default="gcn",
                   help="Model architecture (default: gcn)")
    p.add_argument("--epochs",     type=int, default=200,
                   help="Max training epochs (default: 200)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Graphs per batch — overrides model default. "
                        "Recommended: 64 on A40, 32 on A30")
    p.add_argument("--data-dir",   default=os.path.join(REPO_ROOT, "splits"),
                   help="Directory containing .bin and _metadata.json files "
                        "(default: splits/)")
    p.add_argument("--out-dir",    default=None,
                   help="Output directory (default: results_full_<model_type>/)")
    p.add_argument("--seed",       type=int, default=42,
                   help="Global random seed (default: 42)")
    p.add_argument("--in-features", type=int, default=2,
                   help="Node feature dimension: 2 for baseline (in_degree, out_degree), "
                        "3 when the dataset was built with --use-scores (default: 2)")
    # Hyperparameter overrides — if omitted, model-type defaults from DEFAULTS are used
    p.add_argument("--hidden-dim",    type=int,   default=None)
    p.add_argument("--embedding-dim", type=int,   default=None)
    p.add_argument("--num-layers",    type=int,   default=None)
    p.add_argument("--dropout",       type=float, default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--patience",      type=int,   default=None)
    p.add_argument("--pos-weight",    type=float, default=None)
    p.add_argument("--threshold",     type=float, default=None)
    args = p.parse_args()

    set_seeds(args.seed)

    cfg     = DEFAULTS[args.model_type].copy()
    out_dir = args.out_dir or os.path.join(REPO_ROOT, f"results_full_{args.model_type}")
    os.makedirs(out_dir, exist_ok=True)

    # CLI overrides — only applied when explicitly passed
    for key, val in [
        ("hidden_dim",    args.hidden_dim),
        ("embedding_dim", args.embedding_dim),
        ("num_layers",    args.num_layers),
        ("dropout",       args.dropout),
        ("lr",            args.lr),
        ("patience",      args.patience),
        ("pos_weight",    args.pos_weight),
        ("threshold",     args.threshold),
    ]:
        if val is not None:
            cfg[key] = val

    # CLI --batch-size overrides model default
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]

    log_path = os.path.join(out_dir, "output.txt")
    header   = (
        f"model={args.model_type}  epochs={args.epochs}  "
        f"batch_size={batch_size}  seed={args.seed}  in_features={args.in_features}\n"
        f"data_dir={args.data_dir}\n"
    )
    log.info(header)
    with open(log_path, "w") as fh:   # fresh log for this run
        fh.write(header + "\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_graphs, train_meta = load_split(args.data_dir, "train")
    val_graphs,   val_meta   = load_split(args.data_dir, "val")
    test_graphs,  test_meta  = load_split(args.data_dir, "test")

    size_msg = (
        f"Dataset sizes  —  "
        f"train: {len(train_graphs)}  val: {len(val_graphs)}  test: {len(test_graphs)}"
    )
    log.info(size_msg)
    with open(log_path, "a") as fh:
        fh.write(size_msg + "\n\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    cfg["in_features"] = args.in_features
    model, best_epoch, best_val_loss = train(
        train_graphs, train_meta,
        val_graphs,
        model_type=args.model_type,
        cfg=cfg,
        epochs=args.epochs,
        out_dir=out_dir,
        batch_size=batch_size,
    )

    # Save best model weights
    model_path = os.path.join(out_dir, "best_model.pt")
    torch.save(model.state_dict(), model_path)
    log.info("Best model saved to %s  (epoch %d, val_loss=%.4f)",
             model_path, best_epoch, best_val_loss)

    # ── Final test evaluation with breakdown ──────────────────────────────────
    log.info("Running final evaluation on test set ...")
    test_results = evaluate_with_breakdown(
        test_graphs, test_meta, model, cfg["threshold"], cfg, batch_size
    )
    print_breakdown(test_results, f"{args.model_type} / test", log_path)

    # Also run on train and val for reference (no breakdown needed)
    val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(
        GraphDataLoader(val_graphs, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0),
        model, cfg["threshold"],
    )
    val_summary = (
        f"\nVal (best model): f1={val_f1:.4f}  acc={val_acc:.4f}  "
        f"prec={val_prec:.4f}  rec={val_rec:.4f}\n"
    )
    log.info(val_summary.strip())
    with open(log_path, "a") as fh:
        fh.write(val_summary)

    # ── Persist results as JSON ───────────────────────────────────────────────
    results_json = {
        "model_type":    args.model_type,
        "seed":          args.seed,
        "batch_size":    batch_size,
        "best_epoch":    best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "val": {
            "f1":  round(val_f1,   4),
            "acc": round(val_acc,  4),
            "prec": round(val_prec, 4),
            "rec":  round(val_rec,  4),
        },
        "test": test_results,
        "hyperparams": {k: v for k, v in cfg.items()
                        if k not in ("batch_size",)},
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as fh:
        json.dump(results_json, fh, indent=2)
    log.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
