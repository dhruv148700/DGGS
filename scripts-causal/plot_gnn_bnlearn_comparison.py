"""
plot_gnn_bnlearn_comparison.py
──────────────────────────────
6-figure comparison of GNN variants against baselines / DGGS on bnlearn datasets.

Naming:
  ABA-GCN   — GCN, 2-feature graph (in-/out-degree only)
  ABA-GAT   — GAT, 2-feature graph
  ABA-GCN+  — GCN, 3-feature graph (adds CI-test score)
  ABA-GAT+  — GAT, 3-feature graph (adds CI-test score)

Plots (all NSID = Normalised SID, lower is better):
  1  gnn_nsid_aba_gcn_vs_baselines.pdf    — ABA-GCN  vs Random/FGS/NOTEARS-MLP/MPC/ABAPC
  2  gnn_nsid_aba_gat_vs_baselines.pdf    — ABA-GAT  vs baselines
  3  gnn_nsid_aba_gcn_p_vs_baselines.pdf  — ABA-GCN+ vs baselines
  4  gnn_nsid_aba_gat_p_vs_baselines.pdf  — ABA-GAT+ vs baselines
  5  gnn_nsid_gnns_vs_dggs.pdf            — all 4 GNNs + DGGS + DGGS(opt)
  6  gnn_runtime_all.pdf                  — log-scale runtime for 4 GNNs + DGGS + DGGS(opt)

Output directory: results/causal_recovery/
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from plotly import graph_objects as go

SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
ACD_DIR     = REPO_ROOT / "ArgCausalDisco"
RESULTS_DIR = ACD_DIR / "results"
FIGS_DIR    = REPO_ROOT / "results" / "causal_recovery"
JSON_DIR    = REPO_ROOT / "merged_results_bnlearn_eval"

sys.path.insert(0, str(ACD_DIR))
sys.path.insert(0, str(ACD_DIR / "utils"))

from plotting import double_bar_chart_plotly, sec_blue, sec_orange, main_purple, main_green

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASETS  = ["cancer", "earthquake", "survey"]
ARCS_MAP  = {"cancer": 4, "earthquake": 4, "survey": 6}
NODES_MAP = {"cancer": 5, "earthquake": 5, "survey": 6}
FONT_SIZE = 23

# ── GNN variant definitions ───────────────────────────────────────────────────
GNN_SPECS = [
    # (display_name, style_key, color, json_path)
    ("ABA-GCN",  "aba_gcn",   "#1f77b4", FIGS_DIR / "gnn_gcn.json"),
    ("ABA-GAT",  "aba_gat",   "#ff7f0e", FIGS_DIR / "gnn_gat.json"),
    ("ABA-GCN+", "aba_gcn_p", "#2ca02c", FIGS_DIR / "gnn_gcn_scored.json"),
    ("ABA-GAT+", "aba_gat_p", "#9467bd", FIGS_DIR / "gnn_gat_scored.json"),
]
GNN_ORDER = [label for label, _, _, _ in GNN_SPECS]

BASELINE_ORDER = ["Random", "FGS", "NOTEARS-MLP", "MPC", "ABAPC"]
DGGS_ORDER     = ["DGGS", "DGGS (opt)"]

# ── Style dictionaries ────────────────────────────────────────────────────────
NAMES_DICT = {
    "random":    "Random",
    "fgs":       "FGS",
    "nt":        "NOTEARS-MLP",
    "mpc":       "MPC",
    "abapc":     "ABAPC",
    "dggs":      "DGGS",
    "dggs_opt":  "DGGS (opt)",
    "aba_gcn":   "ABA-GCN",
    "aba_gat":   "ABA-GAT",
    "aba_gcn_p": "ABA-GCN+",
    "aba_gat_p": "ABA-GAT+",
}

COLORS_DICT = {
    "random":    "#7f7f7f",
    "fgs":       sec_orange,    # #b85c00 dark orange
    "nt":        main_purple,   # #9454c4 medium purple
    "mpc":       main_green,    # #379f9f teal-green
    "abapc":     sec_blue,      # #0085CA bright blue
    "dggs":      "#e377c2",     # pink
    "dggs_opt":  "#d62728",     # red
    # GNN colors chosen to avoid all 7 baseline hues above:
    "aba_gcn":   "#00b4d8",     # sky cyan   (≠ blue, ≠ teal-green)
    "aba_gat":   "#f48c06",     # amber      (≠ dark orange)
    "aba_gcn_p": "#5c4033",     # dark brown (≠ all)
    "aba_gat_p": "#70b244",     # lime green (≠ teal-green)
}

_REV_NAMES = {v: k for k, v in NAMES_DICT.items()}

def colour(method: str) -> str:
    return COLORS_DICT.get(_REV_NAMES.get(method, ""), "#333333")

# ── Load baselines from .npy ──────────────────────────────────────────────────
CPDAG_COLS = [
    "dataset", "model", "elapsed_mean", "elapsed_std", "nnz_mean", "nnz_std",
    "fdr_mean", "fdr_std", "tpr_mean", "tpr_std", "fpr_mean", "fpr_std",
    "precision_mean", "precision_std", "recall_mean", "recall_std",
    "F1_mean", "F1_std", "shd_mean", "shd_std",
    "SID_low_mean", "SID_low_std", "SID_high_mean", "SID_high_std",
]

def load_npy(fname):
    df = pd.DataFrame(
        np.load(RESULTS_DIR / fname, allow_pickle=True), columns=CPDAG_COLS)
    df["dataset"] = df["dataset"].astype(str)
    df["model"]   = df["model"].astype(str)
    return df

main_npy = load_npy("stored_results_bnlearn_50rep_cpdag.npy")
main_npy = main_npy[
    main_npy["dataset"].isin(DATASETS) &
    main_npy["model"].isin(["Random", "FGS", "NOTEARS-MLP", "ABAPC (Ours)"])
].copy()
main_npy["model"] = main_npy["model"].replace({"ABAPC (Ours)": "ABAPC"})

mpc_npy = load_npy("stored_results_bnlearn_50rep_mpc_cpdag.npy")
mpc_npy = mpc_npy[mpc_npy["dataset"].isin(DATASETS)].copy()

baselines = pd.concat([main_npy, mpc_npy], ignore_index=True)
baselines["n_edges"] = baselines["dataset"].map(ARCS_MAP).astype(float)
baselines["n_nodes"] = baselines["dataset"].map(NODES_MAP).astype(float)
for col in ["shd", "SID_low", "SID_high"]:
    baselines[f"p_{col}_mean"] = baselines[f"{col}_mean"].astype(float) / baselines["n_edges"]
    baselines[f"p_{col}_std"]  = baselines[f"{col}_std"].astype(float)  / baselines["n_edges"]

# ── Aggregate per-seed JSON runs into one row per dataset ─────────────────────
_SEED_METRICS = ["elapsed", "nnz", "fdr", "tpr", "fpr", "precision", "recall", "f1", "shd"]

def _agg_runs(runs, ds: str) -> dict:
    n_edges = ARCS_MAP[ds]
    row: dict = {"n_edges": float(n_edges), "n_nodes": float(NODES_MAP[ds])}
    for m in _SEED_METRICS:
        vals = np.array([r[m] for r in runs], dtype=float)
        row[f"{m}_mean"] = float(np.mean(vals))
        row[f"{m}_std"]  = float(np.std(vals, ddof=1))
    row["F1_mean"] = row.pop("f1_mean")
    row["F1_std"]  = row.pop("f1_std")
    for sid in ["sid_low_n", "sid_high_n"]:
        vals = np.array([r[sid] for r in runs], dtype=float)
        row[f"{sid}_mean"] = float(np.mean(vals))
        row[f"{sid}_std"]  = float(np.std(vals, ddof=1))
    row["p_shd_mean"]      = row["shd_mean"] / n_edges
    row["p_shd_std"]       = row["shd_std"]  / n_edges
    row["p_SID_low_mean"]  = row["sid_low_n_mean"]
    row["p_SID_low_std"]   = row["sid_low_n_std"]
    row["p_SID_high_mean"] = row["sid_high_n_mean"]
    row["p_SID_high_std"]  = row["sid_high_n_std"]
    return row

def load_json_method(path: Path, label: str, nested: bool = False) -> pd.DataFrame:
    """nested=True for GNN files (results under data['results'][ds])."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    for ds in DATASETS:
        runs = [r for r in (data["results"][ds] if nested else data[ds]) if r["status"] == "ok"]
        rows.append({"dataset": ds, "model": label, **_agg_runs(runs, ds)})
    return pd.DataFrame(rows)

# ── Load DGGS and GNN frames ──────────────────────────────────────────────────
dggs_df = pd.concat([
    load_json_method(JSON_DIR / "merged_reject_off.json", "DGGS",      nested=False),
    load_json_method(JSON_DIR / "merged_reject_on.json",  "DGGS (opt)", nested=False),
], ignore_index=True)

gnn_dfs: dict[str, pd.DataFrame] = {}
for label, _, _, path in GNN_SPECS:
    if not path.exists():
        print(f"Warning: {path} not found — skipping {label}")
        continue
    gnn_dfs[label] = load_json_method(path, label, nested=True)

# ── Prettify dataset labels ───────────────────────────────────────────────────
def prettify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = (
        df["dataset"].str.upper()
        + "<br> |V|=" + df["n_nodes"].astype(int).astype(str)
        + ", |E|="    + df["n_edges"].astype(int).astype(str)
    )
    return df

def fix_zero_nsid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_SID_low_mean"]  = df["p_SID_low_mean"].replace(0, 0.03)
    df["p_SID_high_mean"] = df["p_SID_high_mean"].replace(0, 0.03)
    return df

baselines = fix_zero_nsid(prettify(baselines))
dggs_df   = fix_zero_nsid(prettify(dggs_df))
gnn_dfs   = {k: fix_zero_nsid(prettify(v)) for k, v in gnn_dfs.items()}

# ── NSID plot: calls double_bar_chart_plotly ──────────────────────────────────
FIGS_DIR.mkdir(parents=True, exist_ok=True)

def plot_nsid(df: pd.DataFrame, methods: list, out_stem: str) -> None:
    # Passing a .pdf path: write_html saves HTML there, then write_image
    # overwrites it with PDF (plotly infers format from extension since there
    # is no '.html' substring to replace with '.jpeg').
    out_pdf = str(FIGS_DIR / f"{out_stem}.pdf")
    double_bar_chart_plotly(
        df, ["p_SID_low", "p_SID_high"], NAMES_DICT, COLORS_DICT, methods,
        save_figs=True, font_size=FONT_SIZE, output_name=out_pdf,
        range_y1=[0, 6], range_y2=[0, 6], rect_exp=0.01,
    )
    pdf = FIGS_DIR / f"{out_stem}.pdf"
    if pdf.exists():
        print(f"Saved → {pdf}")

# Plots 1–4: individual GNN vs 5 baselines
for label, key, _, _ in GNN_SPECS:
    if label not in gnn_dfs:
        continue
    df      = fix_zero_nsid(pd.concat([gnn_dfs[label], baselines], ignore_index=True))
    methods = [label] + BASELINE_ORDER
    print(f"\nPlot: NSID {label} vs baselines")
    plot_nsid(df, methods, f"gnn_nsid_{key}_vs_baselines")

# Plot 5: all 4 GNNs + DGGS + DGGS(opt)
available_gnns = [l for l in GNN_ORDER if l in gnn_dfs]
df5 = fix_zero_nsid(pd.concat(
    [gnn_dfs[l] for l in available_gnns] + [dggs_df],
    ignore_index=True,
))
print("\nPlot: NSID all GNNs vs DGGS")
plot_nsid(df5, available_gnns + DGGS_ORDER, "gnn_nsid_vs_dggs")

# ── Plot 6: Runtime (log-scale) — GNNs + DGGS only ──────────────────────────
all_runtime = pd.concat(
    [gnn_dfs[l] for l in available_gnns] + [dggs_df],
    ignore_index=True,
)
all_methods = available_gnns + DGGS_ORDER

datasets_pretty = list(dict.fromkeys(all_runtime["dataset"].tolist()))

fig = go.Figure()
for method in all_methods:
    df_m = all_runtime[all_runtime["model"] == method]
    x, y, err = [], [], []
    for ds in datasets_pretty:
        row = df_m[df_m["dataset"] == ds]
        if not row.empty:
            x.append(ds)
            y.append(float(row["elapsed_mean"].values[0]))
            err.append(float(row["elapsed_std"].values[0]))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        error_y=dict(type="data", array=err, visible=True, thickness=2),
        name=method, mode="lines+markers",
        line=dict(color=colour(method), width=2),
        marker=dict(size=8, color=colour(method)),
    ))

fig.update_yaxes(
    type="log",
    dtick=1,
    tickfont=dict(size=FONT_SIZE - 6),
    title=dict(text="Elapsed time (s)", font=dict(size=FONT_SIZE)),
)
fig.update_xaxes(title=dict(text="Dataset", font=dict(size=FONT_SIZE)))
fig.update_layout(
    template="plotly_white", width=1000, height=550,
    font=dict(size=FONT_SIZE, family="Serif", color="black"),
    legend=dict(orientation="v", xanchor="left", x=1.02, yanchor="middle", y=0.5),
    margin=dict(l=60, r=220, b=80, t=40),
)
runtime_path = FIGS_DIR / "gnn_runtime_all.pdf"
fig.write_image(str(runtime_path))
print(f"\nSaved → {runtime_path}")

print("\nDone.")
