#!/usr/bin/env python3
"""
analyse_convergence.py — Analysis of DGGS convergence experiment results.

Loads the PKL files produced by test_convergence.py (random and/or fixed
initialisation) and generates four figures saved to OUTPUT_DIR:

  Figure 1 — Overall convergence rate and average steps (by init type)
  Figure 2 — Convergence rate by generation parameters (s, n, r, b)
  Figure 3 — Flat vs non-flat convergence breakdown
  Figure 4 — Per-assumption strength trajectory for one framework

Usage
-----
    python scripts/analyse_convergence.py
    python scripts/analyse_convergence.py --trajectory path/to/file.aba
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # project root → scr/
sys.path.insert(0, str(_HERE))         # scripts/     → run_dggs

# ─── Config ──────────────────────────────────────────────────────────────────
PKL_RANDOM = Path("convergence_results_dggs/dggs_e3_d5_s5000_randinit.pkl")
PKL_FIXED  = Path("convergence_results_dggs/dggs_e3_d5_s5000.pkl")
OUTPUT_DIR = Path("figures/")
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
INIT_ORDER  = ["fixed", "random"]
INIT_LABELS = {"fixed": "Fixed (0.5)", "random": "Random [0,1]"}


# ─── Load ─────────────────────────────────────────────────────────────────────

def _load_pkl(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] {path} not found — skipping '{label}' init.")
        return pd.DataFrame()
    with open(path, "rb") as f:
        records = pickle.load(f)
    df = pd.DataFrame(records)
    df["init"] = label
    return df


def load_results() -> pd.DataFrame:
    frames = [
        _load_pkl(PKL_RANDOM, "random"),
        _load_pkl(PKL_FIXED,  "fixed"),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError("No PKL files found. Run test_convergence.py first.")
    df = pd.concat(frames, ignore_index=True)

    # mark entries that have full numerical results
    df["timeout"] = df["timeout"].fillna(False)
    df["oom"]     = df.get("oom", pd.Series(False, index=df.index)).fillna(False)
    df["usable"]  = ~df["timeout"] & ~df["oom"]
    df["non_flat"] = df["non_flat"].fillna(False)
    return df


# ─── Figure 1: Overall convergence rate and average steps ────────────────────

def fig1_overall(df: pd.DataFrame):
    usable = df[df["usable"]].copy()
    inits_present = [i for i in INIT_ORDER if i in usable["init"].values]

    summary = (
        usable.groupby("init", sort=False)
        .agg(
            total        = ("global_converged", "count"),
            n_converged  = ("global_converged", "sum"),
            avg_steps    = ("convergence_time", lambda x: x.dropna().mean()),
            median_steps = ("convergence_time", lambda x: x.dropna().median()),
        )
        .reindex(inits_present)
        .reset_index()
    )
    summary["pct_converged"] = 100 * summary["n_converged"] / summary["total"]
    summary["init_label"]    = summary["init"].map(INIT_LABELS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    bars1 = ax1.bar(
        summary["init_label"], summary["pct_converged"],
        color=sns.color_palette("muted", len(summary))
    )
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("% frameworks globally converged")
    ax1.set_xlabel("Initialisation")
    ax1.set_title("Global convergence rate")
    for bar, val in zip(bars1, summary["pct_converged"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(
        summary["init_label"], summary["avg_steps"],
        color=sns.color_palette("muted", len(summary))
    )
    ax2.set_ylabel("Iterations")
    ax2.set_xlabel("Initialisation")
    ax2.set_title("Average steps to convergence\n(converged frameworks only)")
    for bar, (avg, med) in zip(bars2, zip(summary["avg_steps"], summary["median_steps"])):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"μ={avg:.0f}\nmed={med:.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 1 — Overall DGGS convergence", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig1_overall_convergence.png")

    print("\n── Figure 1 summary ──────────────────────────────────")
    print(summary[["init_label","total","n_converged","pct_converged",
                   "avg_steps","median_steps"]].to_string(index=False))


# ─── Figure 2: Convergence by generation parameters ──────────────────────────

def fig2_by_params(df: pd.DataFrame):
    usable = df[df["usable"]].copy()
    inits_present = [i for i in INIT_ORDER if i in usable["init"].values]
    usable["init_label"] = usable["init"].map(INIT_LABELS)

    params = [
        ("s", "Sentences (s)"),
        ("n", "Negation density (n)"),
        ("r", "Rules per head (r)"),
        ("b", "Max body size (b)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (param, xlabel) in zip(axes, params):
        grouped = (
            usable.groupby(["init_label", param])
            .agg(pct=("global_converged", lambda x: 100 * x.mean()))
            .reset_index()
        )
        labels_present = [INIT_LABELS[i] for i in inits_present]
        sns.lineplot(
            data=grouped, x=param, y="pct",
            hue="init_label", hue_order=labels_present,
            marker="o", ax=ax
        )
        ax.set_ylim(0, 105)
        ax.set_ylabel("% globally converged")
        ax.set_xlabel(xlabel)
        ax.set_title(f"Convergence rate by {xlabel}")
        ax.legend(title="Init", fontsize=8)

    fig.suptitle("Figure 2 — Convergence rate by generation parameters",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig2_convergence_by_params.png")


# ─── Figure 3: Flat vs non-flat breakdown ────────────────────────────────────

def fig3_flat_nonflat(df: pd.DataFrame):
    usable = df[df["usable"]].copy()
    usable["structure"]  = usable["non_flat"].map({True: "Non-flat", False: "Flat"})
    usable["init_label"] = usable["init"].map(INIT_LABELS)
    inits_present        = [INIT_LABELS[i] for i in INIT_ORDER
                            if i in usable["init"].values]

    summary = (
        usable.groupby(["init_label", "structure"])
        .agg(
            total        = ("global_converged", "count"),
            n_converged  = ("global_converged", "sum"),
            avg_steps    = ("convergence_time", lambda x: x.dropna().mean()),
            avg_prop     = ("prop_converged", "mean"),
        )
        .reset_index()
    )
    summary["pct_converged"] = 100 * summary["n_converged"] / summary["total"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    sns.barplot(data=summary, x="structure", y="pct_converged",
                hue="init_label", hue_order=inits_present,
                ax=ax1, palette="muted")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("% globally converged")
    ax1.set_xlabel("")
    ax1.set_title("Convergence rate")
    ax1.legend(title="Init", fontsize=8)

    sns.barplot(data=summary, x="structure", y="avg_steps",
                hue="init_label", hue_order=inits_present,
                ax=ax2, palette="muted")
    ax2.set_ylabel("Average steps to convergence")
    ax2.set_xlabel("")
    ax2.set_title("Average steps (converged frameworks only)")
    ax2.legend(title="Init", fontsize=8)

    fig.suptitle("Figure 3 — Flat vs non-flat DGGS convergence", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig3_flat_nonflat.png")

    print("\n── Figure 3 summary ──────────────────────────────────")
    print(summary[["init_label","structure","total","n_converged",
                   "pct_converged","avg_steps","avg_prop"]].to_string(index=False))


# ─── Figure 4: Strength trajectory ───────────────────────────────────────────

def fig4_trajectory(df: pd.DataFrame, trajectory_file: str = None):
    from scr.dependency_graph import DependencyGraph
    from scr.ABAF import ABAF
    from run_dggs import _Index, initialise_state, step

    if trajectory_file:
        aba_path = Path(trajectory_file)
    else:
        # prefer a converged non-flat framework for visual interest
        candidates = df[df["usable"] & df["non_flat"] & df["global_converged"]]
        if candidates.empty:
            candidates = df[df["usable"] & df["global_converged"]]
        if candidates.empty:
            print("[WARN] No converged framework found — skipping trajectory plot.")
            return
        aba_path = Path(candidates.iloc[0]["file_path"])

    if not aba_path.exists():
        print(f"[WARN] {aba_path} not found — skipping trajectory plot.")
        return

    print(f"\nTrajectory: {aba_path.name}")

    dg = DependencyGraph()
    dg.create_from_file(str(aba_path))
    abaf = ABAF.from_dependency_graph(dg)   # fixed 0.5 tau for clean visualisation

    idx = _Index(abaf)
    asm, rule, claim = initialise_state(abaf)

    # run until convergence or 500 steps
    history = {a.name: [asm[a.name]] for a in abaf.assumptions}
    for _ in range(500):
        prev = dict(asm)
        asm, rule, claim = step(abaf, idx, asm, rule, claim)
        for name in asm:
            history[name].append(asm[name])
        if max(abs(asm[k] - prev[k]) for k in asm) < 1e-6:
            break

    fig, ax = plt.subplots(figsize=(10, 5))
    show_legend = len(history) <= 20
    for name, vals in history.items():
        ax.plot(vals, alpha=0.7, linewidth=1.2,
                label=name if show_legend else None)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Assumption strength σ")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Figure 4 — Strength trajectory\n{aba_path.name}")
    if show_legend:
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    else:
        ax.text(0.99, 0.01, f"{len(history)} assumptions",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray")

    fig.tight_layout()
    _save(fig, "fig4_trajectory.png")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    out = OUTPUT_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse DGGS convergence results.")
    parser.add_argument(
        "--trajectory", default=None, metavar="FILE",
        help="Path to a specific .aba file to use for the trajectory plot."
    )
    args = parser.parse_args()

    df = load_results()
    inits = df["init"].value_counts().to_dict()
    print(f"Loaded {len(df)} entries — {inits}")
    print(f"Usable (no timeout/OOM): {df['usable'].sum()}\n")

    fig1_overall(df)
    fig2_by_params(df)
    fig3_flat_nonflat(df)
    fig4_trajectory(df, trajectory_file=args.trajectory)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
