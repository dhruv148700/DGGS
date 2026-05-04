#!/usr/bin/env python3
"""
compare_dggs_bsaf.py — Pairwise DGGS vs BSAF convergence comparison.

For each .aba file present in both datasets (1,348 files):
  - Compares convergence outcomes (who converges, who doesn't)
  - When both converge, compares per-assumption final strengths
  - Prints a few illustrative exact scenarios (best/worst/typical cases)
  - Prints general trends binned by structural features (s, n, r, b, non_flat)

Usage:
    python scripts/compare_dggs_bsaf.py
"""

import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── File registry ─────────────────────────────────────────────────────────────
DGGS_FILES = {
    "fixed": ROOT / "convergence_results_dggs/dggs_e3_d5_s5000.pkl",
    "rand":  ROOT / "convergence_results_dggs/dggs_e3_d5_s5000_randinit.pkl",
}
BSAF_FILES = {
    ("min",  "fixed"): ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_min.pkl",
    ("prod", "fixed"): ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_prod.pkl",
    ("min",  "rand"):  ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_randinitall_min.pkl",
    ("prod", "rand"):  ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_randinitall_prod.pkl",
}
BSAF_MODELS = ["DF-QuAD (BSAF)", "QE (BSAF)"]

SEP  = "=" * 78
SEP2 = "-" * 78


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_pkl(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def index_dggs(records: list[dict]) -> dict[str, dict]:
    """filename → entry"""
    return {e["file"]: e for e in records}


def index_bsaf(records: list[dict]) -> dict[tuple[str, str], dict]:
    """(filename, model) → entry"""
    return {(e["file"], e["model"]): e for e in records}


# ── Per-file comparison ───────────────────────────────────────────────────────

def strength_diff(dggs_entry: dict, bsaf_entry: dict) -> dict:
    """
    Compute per-assumption and aggregate strength differences.
    Only considers assumptions present in both entries.
    Returns dict with keys: per_assumption, mean_abs, max_abs, mean_dggs, mean_bsaf, direction
    """
    df = dggs_entry.get("final_strengths", {})
    bf = bsaf_entry.get("final_strengths", {})
    common = sorted(set(df) & set(bf))
    if not common:
        return {}
    diffs = {k: df[k] - bf[k] for k in common}
    abs_diffs = [abs(v) for v in diffs.values()]
    mean_dggs = np.mean([df[k] for k in common])
    mean_bsaf = np.mean([bf[k] for k in common])
    return {
        "per_assumption": diffs,
        "mean_abs":  np.mean(abs_diffs),
        "max_abs":   np.max(abs_diffs),
        "mean_dggs": mean_dggs,
        "mean_bsaf": mean_bsaf,
        "direction": "DGGS_higher" if mean_dggs > mean_bsaf else "BSAF_higher",
    }


def build_comparison_df(dggs_idx: dict, bsaf_idx: dict, bsaf_model: str) -> pd.DataFrame:
    """
    Join DGGS and one BSAF model on filename. Return a flat DataFrame with
    one row per matched file (only non-timeout DGGS entries; BSAF timeout handled).
    """
    rows = []
    for fname, de in dggs_idx.items():
        be = bsaf_idx.get((fname, bsaf_model))
        if be is None:
            continue  # file not in BSAF (was timed-out and skipped)

        d_timeout = de.get("timeout", False)
        b_timeout = be.get("timeout", False)
        d_conv    = de.get("global_converged", False) and not d_timeout
        b_conv    = be.get("global_converged", False) and not b_timeout

        if b_conv and d_conv:
            outcome = "both"
        elif d_conv and not b_conv:
            outcome = "dggs_only"
        elif b_conv and not d_conv:
            outcome = "bsaf_only"
        else:
            outcome = "neither"

        # strength diff only when both converge and neither timed out
        sd = {}
        if outcome == "both":
            sd = strength_diff(de, be)

        row = {
            "file":           fname,
            "s":              de["s"],
            "n":              de["n"],
            "r":              de["r"],
            "b":              de["b"],
            "num_assumptions": de["num_assumptions"],
            "non_flat":       de["non_flat"],
            "d_timeout":      d_timeout,
            "b_timeout":      b_timeout,
            "d_conv":         d_conv,
            "b_conv":         b_conv,
            "outcome":        outcome,
            "mean_abs_diff":  sd.get("mean_abs", np.nan),
            "max_abs_diff":   sd.get("max_abs",  np.nan),
            "mean_dggs":      sd.get("mean_dggs", np.nan),
            "mean_bsaf":      sd.get("mean_bsaf", np.nan),
            "direction":      sd.get("direction", ""),
            # store raw entries for scenario printing
            "_de":            de,
            "_be":            be,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ── Scenario printer ──────────────────────────────────────────────────────────

def print_scenario(title: str, row: pd.Series, bsaf_label: str):
    de = row["_de"]
    be = row["_be"]
    print(f"\n  >>> {title}")
    print(f"      File : {row['file']}")
    print(f"      Params: s={row['s']}  n={row['n']}  r={row['r']}  b={row['b']}"
          f"  assumptions={row['num_assumptions']}  non_flat={row['non_flat']}")
    print(f"      Init  : DGGS={'random' if de.get('initial_strengths',{}).get('a0',0.5)!=0.5 else 'fixed 0.5'}  |  BSAF label: {bsaf_label}")
    print(f"      Outcome: DGGS_converged={row['d_conv']}  BSAF_converged={row['b_conv']}")

    df_s = de.get("final_strengths", {})
    bf_s = be.get("final_strengths", {})
    common = sorted(set(df_s) & set(bf_s))
    if common:
        print(f"\n      {'Assumption':<14} {'Initial':>9}  {'DGGS final':>11}  {'BSAF final':>11}  {'Δ(D-B)':>9}")
        print(f"      {'-'*14}  {'-'*9}  {'-'*11}  {'-'*11}  {'-'*9}")
        init_s = de.get("initial_strengths", {})
        for k in common:
            iv  = init_s.get(k, float("nan"))
            dv  = df_s[k]
            bv  = bf_s[k]
            print(f"      {k:<14}  {iv:9.4f}  {dv:11.4f}  {bv:11.4f}  {dv-bv:+9.4f}")
        if not np.isnan(row["mean_abs_diff"]):
            print(f"\n      Mean |Δ| = {row['mean_abs_diff']:.4f}   Max |Δ| = {row['max_abs_diff']:.4f}"
                  f"   Direction: {row['direction']}")
    print()


# ── Trend analysis ────────────────────────────────────────────────────────────

def trend_table(df: pd.DataFrame, col: str, label: str):
    """Group by a structural feature and show convergence rates + mean strength diff."""
    both_df = df[df["outcome"] == "both"]
    rows = []
    for val, grp in df.groupby(col, sort=True):
        both_grp = both_df[both_df[col] == val]
        n        = len(grp)
        d_rate   = grp["d_conv"].mean() * 100
        b_rate   = grp["b_conv"].mean() * 100
        advantage = d_rate - b_rate
        mean_diff = both_grp["mean_abs_diff"].mean() if len(both_grp) else np.nan
        direction_mode = (
            both_grp["direction"].mode()[0]
            if len(both_grp) > 0 and both_grp["direction"].notna().any()
            else "—"
        )
        rows.append({
            label:          val,
            "n_files":      n,
            "DGGS_conv%":   f"{d_rate:.1f}",
            "BSAF_conv%":   f"{b_rate:.1f}",
            "Δconv% (D-B)": f"{advantage:+.1f}",
            "mean|Δstrength|": f"{mean_diff:.4f}" if not np.isnan(mean_diff) else "—",
            "DGGS_direction": direction_mode,
        })
    tbl = pd.DataFrame(rows).set_index(label)
    print(tbl.to_string())
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # ── 1. Load all pkl files ──────────────────────────────────────────────────
    print("Loading pkl files...")
    dggs_data = {k: index_dggs(load_pkl(v)) for k, v in DGGS_FILES.items()}
    bsaf_data = {k: index_bsaf(load_pkl(v)) for k, v in BSAF_FILES.items()}
    print("Done.\n")

    # ── 2. Build all comparison DataFrames ────────────────────────────────────
    # comparisons[(init, agg, model_short)] = df
    comparisons = {}
    for (agg, init), bsaf_idx in bsaf_data.items():
        dggs_idx = dggs_data[init]
        for bsaf_model in BSAF_MODELS:
            model_short = bsaf_model.split(" ")[0]  # "DF-QuAD" or "QE"
            key = (init, agg, model_short)
            comparisons[key] = build_comparison_df(dggs_idx, bsaf_idx, bsaf_model)

    # ── 3. Aggregate overview table ───────────────────────────────────────────
    print(SEP)
    print("AGGREGATE CONVERGENCE OVERVIEW  (DGGS vs each BSAF variant × model)")
    print(SEP)

    agg_rows = []
    for (init, agg, model), df in sorted(comparisons.items()):
        oc = df["outcome"].value_counts()
        n  = len(df)
        both      = oc.get("both",      0)
        dggs_only = oc.get("dggs_only", 0)
        bsaf_only = oc.get("bsaf_only", 0)
        neither   = oc.get("neither",   0)

        d_rate = df["d_conv"].mean() * 100
        b_rate = df["b_conv"].mean() * 100
        mean_diff = df.loc[df["outcome"] == "both", "mean_abs_diff"].mean()

        agg_rows.append({
            "init":        init,
            "agg":         agg,
            "BSAF_model":  model,
            "n_files":     n,
            "both%":       f"{both/n*100:.1f}",
            "dggs_only%":  f"{dggs_only/n*100:.1f}",
            "bsaf_only%":  f"{bsaf_only/n*100:.1f}",
            "neither%":    f"{neither/n*100:.1f}",
            "DGGS_conv%":  f"{d_rate:.1f}",
            "BSAF_conv%":  f"{b_rate:.1f}",
            "Δconv%(D-B)": f"{d_rate-b_rate:+.1f}",
            "mean|Δstr|":  f"{mean_diff:.4f}" if not np.isnan(mean_diff) else "—",
        })

    overview = pd.DataFrame(agg_rows).set_index(["init", "agg", "BSAF_model"])
    print(overview.to_string())
    print()

    # ── 4. Illustrative exact scenarios ───────────────────────────────────────
    # Use fixed-init, min-aggregation, DF-QuAD as the main lens
    ref_key = ("fixed", "min", "DF-QuAD")
    ref_df  = comparisons[ref_key]
    ref_label = "BSAF/min/fixed — DF-QuAD"

    print(SEP)
    print(f"ILLUSTRATIVE EXACT SCENARIOS  [{ref_label}]")
    print("(Chosen to show DGGS best, worst, most-divergent, and typical cases)")
    print(SEP)

    # Scenario A: DGGS converges, BSAF doesn't — pick smallest framework (most interpretable)
    dggs_wins = ref_df[ref_df["outcome"] == "dggs_only"].copy()
    if len(dggs_wins):
        row = dggs_wins.sort_values("num_assumptions").iloc[0]
        print_scenario("A — DGGS converges, DF-QuAD(BSAF) does NOT", row, ref_label)
    else:
        print("  [No cases where DGGS converges and BSAF/DF-QuAD does not]\n")

    # Scenario B: BSAF converges, DGGS doesn't — pick smallest framework
    bsaf_wins = ref_df[ref_df["outcome"] == "bsaf_only"].copy()
    if len(bsaf_wins):
        row = bsaf_wins.sort_values("num_assumptions").iloc[0]
        print_scenario("B — DF-QuAD(BSAF) converges, DGGS does NOT", row, ref_label)
    else:
        print("  [No cases where BSAF/DF-QuAD converges and DGGS does not]\n")

    # Scenario C: Both converge, max mean |Δ strength| (most divergent values)
    both = ref_df[ref_df["outcome"] == "both"].copy()
    if len(both):
        row = both.sort_values("mean_abs_diff", ascending=False).iloc[0]
        print_scenario("C — Both converge, LARGEST per-assumption strength divergence", row, ref_label)

    # Scenario D: Both converge, min mean |Δ strength| — representative "agreement"
    if len(both):
        row = both.sort_values("mean_abs_diff").iloc[0]
        print_scenario("D — Both converge, SMALLEST divergence (near-agreement)", row, ref_label)

    # Scenario E: DGGS converges, QE(BSAF) does NOT — show it differs from DF-QuAD picture
    ref_qe  = comparisons[("fixed", "min", "QE")]
    qe_wins_dggs = ref_qe[ref_qe["outcome"] == "dggs_only"]
    if len(qe_wins_dggs):
        row = qe_wins_dggs.sort_values("num_assumptions").iloc[0]
        print_scenario("E — DGGS converges, QE(BSAF/min/fixed) does NOT", row, "BSAF/min/fixed — QE")

    # ── 5. General trend analysis ─────────────────────────────────────────────
    print(SEP)
    print(f"GENERAL TRENDS — DGGS vs DF-QuAD(BSAF) | fixed init, min agg")
    print("(Convergence rate gap and mean strength divergence, binned by framework feature)")
    print(SEP)

    for col, lbl in [
        ("s",            "framework_size_s"),
        ("n",            "noise_n"),
        ("r",            "rules_r"),
        ("b",            "branching_b"),
        ("non_flat",     "non_flat"),
        ("num_assumptions", "num_assumptions"),
    ]:
        print(f"\n  By {lbl}:")
        trend_table(ref_df, col, lbl)

    # ── 6. Cross-model trend: is the DGGS advantage consistent across models? ──
    print(SEP)
    print("DGGS CONVERGENCE ADVANTAGE vs ALL BSAF MODELS  (fixed init only)")
    print("Shows Δconv% = DGGS_conv% − BSAF_conv% for each agg × model pairing")
    print(SEP)
    fixed_rows = []
    for (init, agg, model), df in comparisons.items():
        if init != "fixed":
            continue
        d_rate = df["d_conv"].mean() * 100
        b_rate = df["b_conv"].mean() * 100
        fixed_rows.append({
            "agg": agg, "model": model,
            "DGGS_conv%":  f"{d_rate:.1f}",
            "BSAF_conv%":  f"{b_rate:.1f}",
            "Δconv%(D-B)": f"{d_rate - b_rate:+.1f}",
        })
    print(pd.DataFrame(fixed_rows).set_index(["agg", "model"]).to_string())
    print()

    # ── 7. Direction of strength difference ───────────────────────────────────
    print(SEP)
    print("STRENGTH DIRECTION SUMMARY — when both converge")
    print("Does DGGS tend to produce higher or lower final strengths than BSAF?")
    print(SEP)
    for (init, agg, model), df in sorted(comparisons.items()):
        both = df[df["outcome"] == "both"]
        if len(both) == 0:
            continue
        higher = (both["direction"] == "DGGS_higher").sum()
        lower  = (both["direction"] == "BSAF_higher").sum()
        mean_d_str = both["mean_dggs"].mean()
        mean_b_str = both["mean_bsaf"].mean()
        mean_diff  = both["mean_abs_diff"].mean()
        print(f"  [{init:5s} / {agg:4s} / {model:8s}]"
              f"  DGGS_higher: {higher:4d}  BSAF_higher: {lower:4d}"
              f"  mean_DGGS_str: {mean_d_str:.4f}  mean_BSAF_str: {mean_b_str:.4f}"
              f"  mean|Δ|: {mean_diff:.4f}")
    print()


if __name__ == "__main__":
    run()
