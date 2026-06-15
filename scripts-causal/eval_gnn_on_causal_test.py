"""
eval_gnn_on_causal_test.py
──────────────────────────
Run a trained causal GNN on the synthetic test set and compute DGGS-style
tier-separation AUC metrics (same 3 views as scripts-sweep/compute_metrics.py).

Tier labels come from dataset/tier_labels.json (all 4021 test instances are
present).  Assumption-name → DGL-node-index mapping is recovered by re-parsing
the original .aba files (lightweight text parsing only, no extension building).
Pre-built test.bin graphs are used for GNN inference so features are identical
to those seen during training.

Usage:
    python scripts-causal/eval_gnn_on_causal_test.py \\
        --results-dir results/results_causal_gcn

    python scripts-causal/eval_gnn_on_causal_test.py \\
        --results-dir results/results_causal_gat_scored

Output:  results/causal_gnn_metrics/<model_label>.json
"""

import argparse
import json
import math
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import dgl
import torch


class _Timeout(Exception):
    pass


def _get_assmpt_mapping_safe(aba_path: str, timeout_s: int = 30):
    """Wraps get_assmpt_mapping with a SIGALRM timeout (Linux only)."""
    def _handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        result = get_assmpt_mapping(aba_path)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return result

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scr.dependency_graph import DependencyGraph
from scr.data_utils import reindex_nodes

TIER_ORDER = ["skeptical", "credulous", "rejected"]
TIER_PAIRS = [
    ("skeptical", "rejected"),
    ("skeptical", "credulous"),
    ("credulous",  "rejected"),
]
VIEW3_MIN_LABELED = 5


# ── Metric helpers (same as compute_metrics.py) ───────────────────────────────

def _auc(a: list, b: list) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    if not a or not b:
        return None
    scores = np.concatenate([np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)])
    labels = np.array([1] * len(a) + [0] * len(b))
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def _cohens_d(a: list, b: list) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr, b_arr = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    pooled_var = (
        ((len(a_arr) - 1) * a_arr.var(ddof=1) + (len(b_arr) - 1) * b_arr.var(ddof=1))
        / (len(a_arr) + len(b_arr) - 2)
    )
    sd = math.sqrt(float(pooled_var)) if pooled_var > 0 else 0.0
    return float(a_arr.mean() - b_arr.mean()) / sd if sd > 0 else None


def _r(v, d=6):
    return round(float(v), d) if v is not None else None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_type, in_features, hidden_dim, embedding_dim, num_layers, dropout):
    if model_type == "gcn":
        from scr.GCN_learnable import GCNLearnableModel
        return GCNLearnableModel(
            in_features=in_features, hidden_features=hidden_dim, out_features=1,
            embedding_dim=embedding_dim, num_layers=num_layers, dropout=dropout,
        )
    elif model_type == "gat":
        from scr.GAT_learnable import GATLearnableModel
        return GATLearnableModel(
            in_features=in_features, hidden_features=hidden_dim, out_features=1,
            embedding_dim=embedding_dim, num_layers=num_layers, dropout=dropout,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


# ── Assumption mapping from .aba (lightweight re-parse) ───────────────────────

def get_assmpt_mapping(aba_path: str) -> dict:
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(aba_path)
    dep_graph.create_dependency_graph()
    _, assmpt_mapping, _ = reindex_nodes(dep_graph)
    return assmpt_mapping


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="Directory containing best_model.pt and results.json")
    p.add_argument("--manifest", default=str(REPO_ROOT / "splits" / "test_manifest.json"))
    p.add_argument("--tier-labels", default=str(REPO_ROOT / "dataset" / "tier_labels.json"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "causal_gnn_metrics"))
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    with open(results_dir / "results.json") as f:
        meta = json.load(f)

    model_type   = meta["model_type"]
    hp           = meta["hyperparams"]
    in_features  = hp["in_features"]
    hidden_dim   = hp["hidden_dim"]
    embedding_dim= hp["embedding_dim"]
    num_layers   = hp["num_layers"]
    dropout      = hp["dropout"]

    # Scored models use splits_scored/test.bin, baseline models use splits/test.bin
    if in_features == 3:
        test_bin = str(REPO_ROOT / "splits_scored" / "test.bin")
        label_suffix = "scored"
    else:
        test_bin = str(REPO_ROOT / "splits" / "test.bin")
        label_suffix = ""

    model_label = results_dir.name  # e.g. "results_causal_gcn_scored"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {model_label}  ({model_type}, in_features={in_features})")
    print(f"Test bin: {test_bin}")

    model = load_model(model_type, in_features, hidden_dim, embedding_dim, num_layers, dropout)
    model.load_state_dict(torch.load(results_dir / "best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    print("Loading test graphs …")
    test_graphs, _ = dgl.load_graphs(test_bin)

    with open(args.manifest) as f:
        test_manifest = json.load(f)
    with open(args.tier_labels) as f:
        all_tiers: dict = json.load(f)

    assert len(test_graphs) == len(test_manifest), (
        f"Graph count mismatch: {len(test_graphs)} graphs vs {len(test_manifest)} manifest entries"
    )

    # Accumulation structures (mirrors compute_metrics.py)
    pooled: Dict[str, List[float]] = {t: [] for t in TIER_ORDER}
    per_abaf_aucs: Dict[Tuple, List[float]] = {pair: [] for pair in TIER_PAIRS}
    norm: Dict[str, List[float]] = {t: [] for t in TIER_ORDER}
    n_v3_abafs = n_v3_asms = 0

    n = len(test_manifest)
    n_skipped = 0
    for i, entry in enumerate(test_manifest):
        if i % 100 == 0:
            print(f"  [{i}/{n}] processing …", flush=True)

        # GNN inference on the pre-built test graph
        g = test_graphs[i].to(device)
        with torch.no_grad():
            inputs = {
                "assmpt":     g.nodes["assmpt"].data["features"],
                "rule":       g.nodes["rule"].data["features"],
                "non_assmpt": g.nodes["non_assmpt"].data["features"],
            }
            logits = model(g, inputs)["assmpt"].squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()

        # Re-parse .aba to get assumption-name → node-index mapping
        aba_path = os.path.join(REPO_ROOT, entry["abaf"])
        try:
            assmpt_mapping = _get_assmpt_mapping_safe(aba_path, timeout_s=30)
        except _Timeout:
            print(f"  [{i}] TIMEOUT parsing {aba_path} — skipping", flush=True)
            n_skipped += 1
            continue
        except Exception as e:
            print(f"  [{i}] ERROR {type(e).__name__}: {e} — {aba_path}", flush=True)
            n_skipped += 1
            continue

        instance_id = entry["instance_id"] + "_full"
        tier_map    = all_tiers.get(instance_id, {})

        abaf_t: Dict[str, List[float]] = {t: [] for t in TIER_ORDER}
        for name, idx in assmpt_mapping.items():
            tier = tier_map.get(name, "unknown")
            prob = float(probs[idx])
            if tier in TIER_ORDER:
                pooled[tier].append(prob)
                abaf_t[tier].append(prob)

        # View 2: per-ABAF AUC (independent per pair)
        for tier_a, tier_b in TIER_PAIRS:
            auc = _auc(abaf_t[tier_a], abaf_t[tier_b])
            if auc is not None:
                per_abaf_aucs[(tier_a, tier_b)].append(auc)

        # View 3: z-score normalised (≥5 labeled assumptions, sd > 0)
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

    print(f"Done processing {n} test instances (skipped {n_skipped}).")

    # ── Build output row (mirrors compute_metrics.py format) ─────────────────
    row: dict = {
        "config_id":   model_label,
        "model_type":  model_type,
        "in_features": in_features,
        "n_test":      n,
    }

    for t in TIER_ORDER:
        arr = np.array(pooled[t])
        row[f"{t}_n"]    = len(arr)
        row[f"{t}_mean"] = _r(arr.mean())      if len(arr) > 0 else None
        row[f"{t}_sd"]   = _r(arr.std(ddof=1)) if len(arr) > 1 else None

    for tier_a, tier_b in TIER_PAIRS:
        row[f"v1_auc_{tier_a}_vs_{tier_b}"]    = _r(_auc(pooled[tier_a], pooled[tier_b]))
        row[f"v1_cohend_{tier_a}_vs_{tier_b}"] = _r(_cohens_d(pooled[tier_a], pooled[tier_b]))

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

    row["v3_n_abafs"] = n_v3_abafs
    row["v3_n_asms"]  = n_v3_asms
    for tier_a, tier_b in TIER_PAIRS:
        row[f"v3_auc_{tier_a}_vs_{tier_b}"]    = _r(_auc(norm[tier_a], norm[tier_b]))
        row[f"v3_cohend_{tier_a}_vs_{tier_b}"] = _r(_cohens_d(norm[tier_a], norm[tier_b]))

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  {model_label}")
    print(f"{'─'*60}")
    for tier_a, tier_b in TIER_PAIRS:
        print(f"  V1 AUC  {tier_a}/{tier_b}: {row.get(f'v1_auc_{tier_a}_vs_{tier_b}')}")
    for tier_a, tier_b in TIER_PAIRS:
        print(f"  V2 med  {tier_a}/{tier_b}: {row.get(f'v2_auc_median_{tier_a}_vs_{tier_b}')} "
              f"(n={row.get(f'v2_n_abafs_{tier_a}_vs_{tier_b}')})")
    for tier_a, tier_b in TIER_PAIRS:
        print(f"  V3 AUC  {tier_a}/{tier_b}: {row.get(f'v3_auc_{tier_a}_vs_{tier_b}')} "
              f"(n_abafs={row.get('v3_n_abafs')})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_label}.json"
    with open(out_path, "w") as f:
        json.dump(row, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
