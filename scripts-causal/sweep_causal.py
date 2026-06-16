"""
sweep_causal.py
───────────────
W&B sweep agent for the causal ABA dataset.

Each SLURM job runs this script once (count=1). The W&B controller assigns
one hyperparameter config, this script trains the model and reports val_f1,
then the job exits.

Create the sweep once with:
    python -c "import wandb; sweep_id = wandb.sweep(<config>, project='gnn4aba-causal-sweep'); print(sweep_id)"

Then submit N parallel SLURM jobs (each runs one trial):
    sbatch --array=0-49 scripts-slurm/sweep_agent_a40.sh

Environment variables (set by the SLURM script):
    SWEEP_ID   : W&B sweep ID (required)
    MODEL_TYPE : gcn | gat  (default: gcn)
    DATA_DIR   : path to splits/  (default: splits/)
"""

import copy
import os
import sys

import dgl
import numpy as np
import torch
import torch.nn as nn
import wandb
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scr"))

from scr.data_utils import set_seeds
from scr.GCN_learnable import GCNLearnableModel
from scr.GAT_learnable import GATLearnableModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_split(data_dir, split):
    import json
    graphs, _ = dgl.load_graphs(os.path.join(data_dir, f"{split}.bin"))
    with open(os.path.join(data_dir, f"{split}_metadata.json")) as fh:
        metadata = json.load(fh)
    return list(graphs), metadata


def build_model(model_type, cfg):
    if model_type == "gcn":
        return GCNLearnableModel(
            in_features=cfg["in_features"],
            hidden_features=cfg["hidden_dim"],
            out_features=1,
            embedding_dim=cfg["embedding_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    return GATLearnableModel(
        in_features=cfg["in_features"],
        hidden_features=cfg["hidden_dim"],
        out_features=1,
        embedding_dim=cfg["embedding_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )


def evaluate(loader, model, threshold, criterion=None):
    model.eval()
    total_loss, total_nodes = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for g in loader:
            g = g.to(device)
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

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss   = total_loss / total_nodes if criterion is not None else None
    f1   = f1_score(all_labels,        all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels,    all_preds, zero_division=0)
    acc  = (all_preds == all_labels).mean()
    return avg_loss, acc, f1, prec, rec


def train_one_trial():
    run = wandb.init()
    cfg = dict(wandb.config)

    model_type = os.environ.get("MODEL_TYPE", "gcn")
    data_dir   = os.environ.get("DATA_DIR",   os.path.join(REPO_ROOT, "splits"))

    set_seeds(42)

    train_graphs, _ = load_split(data_dir, "train")
    val_graphs,   _ = load_split(data_dir, "val")

    batch_size = cfg["batch_size"]
    train_loader = GraphDataLoader(train_graphs, batch_size=batch_size,
                                   shuffle=True,  drop_last=False, num_workers=0)
    val_loader   = GraphDataLoader(val_graphs,   batch_size=batch_size,
                                   shuffle=False, drop_last=False, num_workers=0)

    model     = build_model(model_type, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.75, patience=5, min_lr=1e-6
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg["pos_weight"]]).to(device)
    )

    best_val_loss    = float("inf")
    best_val_f1      = 0.0
    best_state       = None
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
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
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(
            val_loader, model, cfg["threshold"], criterion=loss_fn
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "epoch":      epoch,
            "train_loss": avg_train,
            "val_loss":   val_loss,
            "val_f1":     val_f1,
            "val_acc":    val_acc,
            "val_prec":   val_prec,
            "val_rec":    val_rec,
            "lr":         lr,
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_val_f1      = val_f1
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    # Report the best val_f1 seen — this is what Bayesian optimisation ranks on
    wandb.summary["val_f1"]     = best_val_f1
    wandb.summary["val_loss"]   = best_val_loss
    wandb.summary["model_type"] = model_type
    run.finish()


if __name__ == "__main__":
    sweep_id   = os.environ.get("SWEEP_ID")
    model_type = os.environ.get("MODEL_TYPE", "gcn")
    project    = os.environ.get("WANDB_PROJECT", "gnn4aba-causal-sweep")

    if not sweep_id:
        raise ValueError("SWEEP_ID environment variable not set")

    wandb.agent(sweep_id, function=train_one_trial, project=project, count=1)
