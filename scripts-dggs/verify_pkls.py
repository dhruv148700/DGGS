#!/usr/bin/env python3
"""
verify_pkls.py — Field presence, type, and sanity check for all convergence pkl files.

Checks all 6 pkl files (4 BSAF + 2 DGGS) and reports:
  - field presence and types per file
  - counts of missing / wrong-type fields
  - high-level summary: length, models, timeout rate, convergence rate

Usage:
    python scripts/verify_pkls.py
"""

import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ── File registry ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "BSAF/min/fixed":    ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_min.pkl",
    "BSAF/prod/fixed":   ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_prod.pkl",
    "BSAF/min/rand":     ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_randinitall_min.pkl",
    "BSAF/prod/rand":    ROOT / "convergence_results_bsaf/convergence_results_to10m_nf_atm_e3_d5_s5000_randinitall_prod.pkl",
    "DGGS/fixed":        ROOT / "convergence_results_dggs/dggs_e3_d5_s5000.pkl",
    "DGGS/rand":         ROOT / "convergence_results_dggs/dggs_e3_d5_s5000_randinit.pkl",
}

# ── Expected schema ────────────────────────────────────────────────────────────
# (field_name, expected_python_type, nullable)
# convergence_time differs between BSAF (dict) and DGGS (int) — handled separately

COMMON_FIELDS = [
    ("file",              str,   False),
    ("file_path",         str,   False),
    ("model",             str,   False),
    ("s",                 (int, float), False),
    ("n",                 (int, float), False),
    ("a",                 (int, float), False),
    ("r",                 (int, float), False),
    ("b",                 (int, float), False),
    ("num_assumptions",   int,   False),
    ("num_rules",         int,   False),
    ("num_sentences",     int,   False),
    ("non_flat",          bool,  False),
    ("initial_strengths", dict,  False),
    ("final_strengths",   dict,  False),
    ("global_converged",  bool,  False),
    ("prop_converged",    float, False),
    ("per_arg",           dict,  False),
    ("convergence_time",  None,  False),   # type checked separately
    ("timeout",           bool,  False),
]

BSAF_CONV_TIME_TYPE = dict
DGGS_CONV_TIME_TYPE = (int, float)   # scalar step count


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: Path, label: str):
    if not path.exists():
        print(f"[MISSING FILE] {label}: {path}")
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        print(f"[WRONG TOP-LEVEL TYPE] {label}: expected list, got {type(data)}")
        return []
    return data


def check_entry(entry: dict, label: str, idx: int, is_bsaf: bool) -> list[str]:
    """Return list of issue strings for this entry (empty = clean)."""
    issues = []
    for field, expected_type, nullable in COMMON_FIELDS:
        val = entry.get(field, "__MISSING__")

        if val == "__MISSING__":
            issues.append(f"field '{field}' missing")
            continue

        if val is None:
            if not nullable:
                issues.append(f"field '{field}' is None (not nullable)")
            continue

        if field == "convergence_time":
            conv_type = BSAF_CONV_TIME_TYPE if is_bsaf else DGGS_CONV_TIME_TYPE
            if not isinstance(val, conv_type):
                issues.append(
                    f"convergence_time: expected {'dict' if is_bsaf else 'scalar'}, "
                    f"got {type(val).__name__}"
                )
            continue

        if expected_type is not None and not isinstance(val, expected_type):
            issues.append(f"field '{field}': expected {expected_type}, got {type(val).__name__}")

    # value-level checks
    fs = entry.get("final_strengths", {})
    ins = entry.get("initial_strengths", {})
    pa = entry.get("per_arg", {})

    if isinstance(fs, dict) and isinstance(ins, dict):
        if set(fs.keys()) != set(ins.keys()):
            issues.append("final_strengths and initial_strengths have different keys")
        for k, v in fs.items():
            if not isinstance(v, (int, float)):
                issues.append(f"final_strengths[{k}] is not numeric: {type(v)}")
            elif not (0.0 <= v <= 1.0):
                issues.append(f"final_strengths[{k}]={v:.4f} out of [0,1]")

    if isinstance(pa, dict) and isinstance(fs, dict):
        if set(pa.keys()) != set(fs.keys()):
            issues.append("per_arg keys do not match final_strengths keys")

    pc = entry.get("prop_converged")
    if pc is not None and not (0.0 <= pc <= 1.0):
        issues.append(f"prop_converged={pc} out of [0,1]")

    return issues


# ── Per-file audit ─────────────────────────────────────────────────────────────

def audit(label: str, data: list[dict], is_bsaf: bool) -> dict:
    total = len(data)
    issue_count = 0
    field_issues = defaultdict(int)

    for i, entry in enumerate(data):
        issues = check_entry(entry, label, i, is_bsaf)
        if issues:
            issue_count += 1
            for iss in issues:
                # bucket by field name (first word after "field '" or raw)
                key = iss.split("'")[1] if "'" in iss else iss.split(":")[0]
                field_issues[key] += 1

    models      = sorted(set(e.get("model", "?") for e in data))
    n_timeout   = sum(1 for e in data if e.get("timeout", False))
    n_converged = sum(1 for e in data if e.get("global_converged", False) and not e.get("timeout", False))
    n_usable    = total - n_timeout
    prop_conv   = n_converged / n_usable if n_usable else float("nan")

    # per-model convergence rate
    model_conv = {}
    for m in models:
        sub = [e for e in data if e.get("model") == m and not e.get("timeout", False)]
        mc  = sum(1 for e in sub if e.get("global_converged", False))
        model_conv[m] = f"{mc}/{len(sub)} ({mc/len(sub)*100:.1f}%)" if sub else "n/a"

    return {
        "label":       label,
        "total":       total,
        "models":      ", ".join(models),
        "n_timeout":   n_timeout,
        "n_usable":    n_usable,
        "n_converged": n_converged,
        "conv_rate":   f"{prop_conv*100:.1f}%",
        "n_dirty":     issue_count,
        "field_issues": dict(field_issues),
        "model_conv":  model_conv,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    results = []
    for label, path in FILES.items():
        is_bsaf = label.startswith("BSAF")
        print(f"\nLoading {label} ...")
        data = load(path, label)
        if not data:
            continue
        res = audit(label, data, is_bsaf)
        results.append(res)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    rows = []
    for r in results:
        rows.append({
            "dataset":        r["label"],
            "entries":        r["total"],
            "models":         r["models"],
            "timeouts":       r["n_timeout"],
            "usable":         r["n_usable"],
            "globally_conv":  r["n_converged"],
            "conv_rate":      r["conv_rate"],
            "dirty_entries":  r["n_dirty"],
        })

    df = pd.DataFrame(rows).set_index("dataset")
    print(df.to_string())

    # ── Field issue breakdown ─────────────────────────────────────────────────
    any_issues = False
    for r in results:
        if r["field_issues"]:
            any_issues = True
            print(f"\n  Field issues in [{r['label']}]:")
            for field, count in sorted(r["field_issues"].items(), key=lambda x: -x[1]):
                print(f"    {field:30s}  {count} entries affected")

    if not any_issues:
        print("\n  All entries pass field checks.")

    # ── Per-model convergence ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-MODEL CONVERGENCE RATES")
    print("=" * 80)
    for r in results:
        print(f"\n  [{r['label']}]")
        for model, rate in r["model_conv"].items():
            print(f"    {model:30s}  {rate}")

    # ── Schema diff BSAF vs DGGS ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SCHEMA NOTES")
    print("=" * 80)
    print("  convergence_time  BSAF → dict[assumption→step]  |  DGGS → scalar int")
    print("  num_sentences     may differ between BSAF and DGGS for the same .aba file")
    print("  DGGS has one model per entry; BSAF has two (DF-QuAD, QE) interleaved")


if __name__ == "__main__":
    run()
