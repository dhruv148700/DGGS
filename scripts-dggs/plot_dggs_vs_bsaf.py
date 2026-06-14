"""
plot_dggs_vs_bsaf.py — Convergence comparison: DGGS vs BSAF.

Direct adaptation of GradualABA/convergence_results_v2/plots_v2.ipynb (Cell 2),
replacing BAF with DGGS.

Figure layout (matching Figure 3 of the GradualABA paper):
  - 2 vertically stacked axes: top = global convergence rate, bottom = avg steps
  - X-axis: 4 groups = [Constant×Product, Constant×Minimum,
                        Uniform×Product, Uniform×Minimum]
  - 4 bars per group: DF-QuAD (BSAF), QE (BSAF), lin (DGGS), quad (DGGS)
  - Grey shaded boxes above x-axis annotate the two τ-init regions

Run from project root:
  python scripts-dggs/plot_dggs_vs_bsaf.py [--save]
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 25

BSAF_DIR = Path("convergence_results_bsaf")
DGGS_DIR = Path("convergence_results_dggs")
OUT_DIR  = Path("convergence_results_dggs/plots")

# ── scenario index ─────────────────────────────────────────────────────────
# (init_label, body, system, path)
scenario_files = [
    ("Constant",    "Product", "(BSAF)", BSAF_DIR / "convergence_results_to10m_nf_atm_e3_d5_s5000_prod.pkl"),
    ("Constant",    "Minimum", "(BSAF)", BSAF_DIR / "convergence_results_to10m_nf_atm_e3_d5_s5000_min.pkl"),
    ("Constant",    "Product", "(DGGS)", DGGS_DIR / "dggs_e3_d5_s5000_body-prod_inf-lin.pkl",
                                         DGGS_DIR / "dggs_e3_d5_s5000_body-prod_inf-quad.pkl"),
    ("Constant",    "Minimum", "(DGGS)", DGGS_DIR / "dggs_e3_d5_s5000_body-min_inf-lin.pkl",
                                         DGGS_DIR / "dggs_e3_d5_s5000_body-min_inf-quad.pkl"),
    ("Uniform(0,1)", "Product", "(BSAF)", BSAF_DIR / "convergence_results_to10m_nf_atm_e3_d5_s5000_randinitall_prod.pkl"),
    ("Uniform(0,1)", "Minimum", "(BSAF)", BSAF_DIR / "convergence_results_to10m_nf_atm_e3_d5_s5000_randinitall_min.pkl"),
    ("Uniform(0,1)", "Product", "(DGGS)", DGGS_DIR / "dggs_e3_d5_s5000_body-prod_inf-lin_randinit.pkl",
                                          DGGS_DIR / "dggs_e3_d5_s5000_body-prod_inf-quad_randinit.pkl"),
    ("Uniform(0,1)", "Minimum", "(DGGS)", DGGS_DIR / "dggs_e3_d5_s5000_body-min_inf-lin_randinit.pkl",
                                          DGGS_DIR / "dggs_e3_d5_s5000_body-min_inf-quad_randinit.pkl"),
]

models = ["DF-QuAD (BSAF)", "QE (BSAF)", "DF-QuAD (DGGS)", "QE (DGGS)"]
inits  = ["Constant", "Uniform(0,1)"]
aggs   = ["Product", "Minimum"]

colors = {
    "DF-QuAD (BSAF)": "#009988",
    "QE (BSAF)":      "#EE7733",
    "DF-QuAD (DGGS)": "#4477AA",
    "QE (DGGS)":      "#AA3377",
}


# ── helpers ───────────────────────────────────────────────────────────────

def _mean_time(ct):
    if isinstance(ct, dict):
        vals = [v for v in ct.values() if v is not None]
        return sum(vals) / len(vals) if vals else None
    if ct is None:
        return None
    return float(ct)


def load_bsaf_stats(path: Path) -> dict:
    """Load a BSAF pkl; return {model_name: (rate, mean_time)}."""
    runs = pickle.load(open(path, "rb"))
    good = [r for r in runs if not r.get("timeout", False)]
    by_model = defaultdict(list)
    for r in good:
        by_model[r["model"]].append(r)

    stats = {}
    for model, mrs in by_model.items():
        N = len(mrs)
        rate = sum(r.get("global_converged", False) for r in mrs) / N if N else 0.0
        times = [t for r in mrs for t in [_mean_time(r.get("convergence_time"))] if t is not None]
        stats[model] = (rate, sum(times) / len(times) if times else 0.0)
    return stats


def load_dggs_stats(lin_path: Path, quad_path: Path) -> dict:
    """Load two DGGS pkls (lin + quad); return {model_name: (rate, mean_time)}."""
    stats = {}
    for path, label in [(lin_path, "DF-QuAD (DGGS)"), (quad_path, "QE (DGGS)")]:
        runs = pickle.load(open(path, "rb"))
        good = [r for r in runs if not r.get("timeout", False)]
        N = len(good)
        rate = sum(r.get("global_converged", False) for r in good) / N if N else 0.0
        times = [t for r in good for t in [_mean_time(r.get("convergence_time"))] if t is not None]
        stats[label] = (rate, sum(times) / len(times) if times else 0.0)
    return stats


def annotate_tau_regions(ax, spans, labels,
                         y_base=1.02, height=0.10,
                         facecolor="lightgrey", alpha=0.5, fontsize=25):
    for (xmin, xmax), txt in zip(spans, labels):
        ax.axvspan(xmin, xmax, ymin=y_base, ymax=y_base + height,
                   color=facecolor, alpha=alpha, clip_on=False)
        ax.text(
            (xmin + xmax) / 2,
            (y_base + y_base + height) / 2,
            txt,
            transform=ax.get_xaxis_transform(),
            ha="center", va="center", fontsize=fontsize,
        )


# ── load all scenarios ────────────────────────────────────────────────────

raw = {}
for entry in scenario_files:
    init, agg, sys_label = entry[0], entry[1], entry[2]
    if sys_label == "(BSAF)":
        raw[(init, agg, sys_label)] = load_bsaf_stats(entry[3])
    else:
        raw[(init, agg, sys_label)] = load_dggs_stats(entry[3], entry[4])

# ── build value arrays ────────────────────────────────────────────────────

conv_rates = {(i, a): [] for i in inits for a in aggs}
conv_times = {(i, a): [] for i in inits for a in aggs}

for i in inits:
    for a in aggs:
        for m in models:
            sys_label = "(BSAF)" if "BSAF" in m else "(DGGS)"
            r, t = raw[(i, a, sys_label)].get(m, (0.0, 0.0))
            conv_rates[(i, a)].append(r)
            conv_times[(i, a)].append(t)

# ── plot ──────────────────────────────────────────────────────────────────

n     = len(models)
width = 0.8 / n
x     = np.arange(len(inits) * len(aggs))

fig, (ax_r, ax_t) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

for idx, m in enumerate(models):
    off   = (idx - (n - 1) / 2) * width
    rates = [conv_rates[(i, a)][idx] for i in inits for a in aggs]
    times = [conv_times[(i, a)][idx] for i in inits for a in aggs]

    ax_r.bar(x + off, rates, width, color=colors[m], label=m, alpha=0.6)
    ax_t.bar(x + off, times, width, color=colors[m], label=m, alpha=0.6)

    for xi, v in zip(x, rates):
        ax_r.text(xi + off, v + 0.005, f"{100*v:.0f}%",
                  ha="center", va="bottom", fontsize=FONTSIZE, rotation=90)
    for xi, v in zip(x, times):
        ax_t.text(xi + off, v + 0.5, f"{v:.1f}",
                  ha="center", va="bottom", fontsize=FONTSIZE, rotation=90)

# axes styling — extra headroom so rotated labels never clip
all_times = [t for vals in conv_times.values() for t in vals]
max_time  = max(all_times) if all_times else 1.0

ax_r.set_ylabel("% Converged", fontsize=FONTSIZE)
ax_r.set_ylim(0, 1.45)  # ~45% headroom above 1.0 for rotated "100%" labels
for s in ["top", "right"]:
    ax_r.spines[s].set_visible(False)

ax_t.set_ylim(0, max_time * 1.45)  # headroom for tallest rotated label
ax_t.set_ylabel("Avg Number of Steps", fontsize=FONTSIZE)
ax_t.set_xticks(x)
ax_t.set_xticklabels(aggs * len(inits), fontsize=FONTSIZE)
ax_t.set_xlabel("Attack/Supp & Base-Score Aggregation", fontsize=FONTSIZE)
for s in ["top", "right"]:
    ax_t.spines[s].set_visible(False)

for ax in [ax_r, ax_t]:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(FONTSIZE)

# legend
h, l = ax_t.get_legend_handles_labels()
l = [lab.replace("DF-QuAD", "DfQ") for lab in l]
fig.legend(h, l, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.06), fontsize=FONTSIZE)

# τ region annotations
region_spans = [
    (-0.42, 1.43),
    ( 1.57, 3.44),
]
annotate_tau_regions(
    ax_r, region_spans,
    labels=[r"$\tau(a)=0.5$", r"$\tau(a)\sim U(0,1)$"],
    y_base=1.02, height=0.12,
    facecolor="lightgrey", alpha=0.5, fontsize=FONTSIZE,
)

plt.tight_layout(rect=[0, 0.05, 1, 0.92])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save to convergence_results_dggs/plots/ instead of showing")
    args = parser.parse_args()

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "dggs_vs_bsaf.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()
